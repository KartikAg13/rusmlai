//! [`DecisionTreeClassifier`] and [`DecisionTreeRegressor`], and the [`Tree`] they learn.
//!
//! Both grow one tree on the histogram engine (`models/hist.rs`): each feature is put into at
//! most `max_bins` buckets once, and every node finds its split from the sums per bucket. A
//! feature with at most `max_bins` distinct values has one bucket per value, so the tree is
//! the one scikit-learn's exact search finds; ties between equally good splits go to the
//! first feature, where scikit-learn picks one at random.

use std::io::{self, Write};

use indexmap::IndexMap;
use polars::prelude::{Column, DataFrame, DataType, IdxCa, IdxSize, NamedFrom, Series};
use polars_core::runtime::THREAD_POOL;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    dataset::{Dataset, split::Rng},
    error::OrFatal,
    metadata::models::CachedRun,
    models::{
        Classifier, Model, ModelError, Regressor, bad_learned, check, class_labels,
        design::{CHUNK, column, prepared},
        hist::{Bins, Growth, Grown, Rules, Target, feature, grow},
        label_target, numeric_target,
    },
    preprocess::one_of,
};

/// One node, in 16 bytes: a walk reads one of them per level, and a node's left child is the
/// next one, often in the same cache line.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Node {
    pub threshold: f64,
    /// The feature × 2, plus 1 when a missing value goes left; [`LEAF`] for a leaf.
    pub split: u32,
    /// The right child; 0 for a leaf.
    pub right: u32,
}

/// [`Node::split`] of a leaf.
pub(crate) const LEAF: u32 = u32::MAX;

/// The bit of [`Node::split`] that marks a split on a categorical feature: the categories in
/// a set go left. Its `threshold` is then the place of that set in [`Tree`]'s sets.
pub(crate) const CATEGORICAL: u32 = 1 << 31;

/// The categories that go left at a categorical split: bit `c` for the category of code `c`.
pub(crate) type Left = [u64; 4];

impl Node {
    /// The feature a split reads.
    pub(crate) fn feature(&self) -> usize {
        ((self.split & !CATEGORICAL) >> 1) as usize
    }

    /// Whether the node splits a categorical feature into two sets of categories.
    pub(crate) fn is_categorical(&self) -> bool {
        self.split != LEAF && self.split & CATEGORICAL != 0
    }
}

/// Rows that walk a tree together: while one waits for a node from memory, the others move.
const LANES: usize = 16;

/// The nodes of a fitted tree, like scikit-learn's `tree_`.
///
/// Node 0 is the root. A node is followed by its left subtree, so the left child of node `i`
/// is node `i + 1` (scikit-learn's `children_left`), and `children_right()[i]` is its right
/// child. A row goes left when its value of `feature()[i]` is at most `threshold()[i]`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Tree {
    nodes: Vec<Node>,
    value: Vec<f64>,
    n_node_samples: Vec<u32>,
    /// The categories that go left at each categorical split.
    sets: Vec<Left>,
}

/// The next `len` bytes of `bytes`, or the error of a weights file that is cut short.
fn take<'a>(bytes: &mut &'a [u8], len: usize) -> Result<&'a [u8], ModelError> {
    if bytes.len() < len {
        return Err(bad_learned("the weights file is cut short".into()));
    }
    let (head, tail) = bytes.split_at(len);
    *bytes = tail;
    Ok(head)
}

impl Tree {
    pub(crate) fn from_parts(nodes: Vec<Node>, value: Vec<f64>, n_node_samples: Vec<u32>) -> Self {
        Self {
            nodes,
            value,
            n_node_samples,
            sets: Vec::new(),
        }
    }

    /// The tree with the category sets of its categorical splits.
    pub(crate) fn with_sets(mut self, sets: Vec<Left>) -> Self {
        self.sets = sets;
        self
    }

    /// The values of the nodes, to set a leaf's own (boosting's absolute and quantile losses).
    pub(crate) fn value_mut(&mut self) -> &mut [f64] {
        &mut self.value
    }

    /// The nodes, splits and leaves together.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// The leaves.
    pub fn n_leaves(&self) -> usize {
        self.nodes.iter().filter(|node| node.split == LEAF).count()
    }

    /// The splits on the longest way from the root to a leaf.
    pub fn max_depth(&self) -> usize {
        let mut depth: Vec<usize> = vec![0; self.node_count()];
        for (i, node) in self.nodes.iter().enumerate() {
            if node.split != LEAF {
                depth[i + 1] = depth[i] + 1;
                depth[node.right as usize] = depth[i] + 1;
            }
        }
        depth.into_iter().max().unwrap_or(0)
    }

    /// The feature each node splits on, as its place in `feature_names_in_`; -1 for a leaf.
    pub fn feature(&self) -> Vec<i32> {
        let feature = |node: &Node| if node.split == LEAF { -1 } else { node.feature() as i32 };
        self.nodes.iter().map(feature).collect()
    }

    /// The split value of each node (0 for a leaf). `f64::MAX` sends every value left and
    /// only the missing ones right, where scikit-learn has infinity. NaN at a categorical
    /// split, which sends a set of categories left ([`Tree::left_categories`]).
    pub fn threshold(&self) -> Vec<f64> {
        let threshold = |node: &Node| if node.is_categorical() { f64::NAN } else { node.threshold };
        self.nodes.iter().map(threshold).collect()
    }

    /// 1 where a node splits a categorical feature into two sets of categories.
    pub fn is_categorical(&self) -> Vec<u8> {
        self.nodes.iter().map(|node| u8::from(node.is_categorical())).collect()
    }

    /// The codes of the categories that go left at node `i` (their places among the
    /// feature's sorted categories), or `None` when node `i` is not a categorical split.
    pub fn left_categories(&self, i: usize) -> Option<Vec<u32>> {
        let node: &Node = self.nodes.get(i).filter(|node| node.is_categorical())?;
        let set: &Left = &self.sets[node.threshold as usize];
        Some((0..256).filter(|c| set[c >> 6] >> (c & 63) & 1 == 1).map(|c| c as u32).collect())
    }

    /// The right child of each node; 0 for a leaf.
    pub fn children_right(&self) -> Vec<u32> {
        self.nodes.iter().map(|node| node.right).collect()
    }

    /// 1 where a missing value goes left, 0 where it goes right (and for a leaf).
    pub fn missing_go_to_left(&self) -> Vec<u8> {
        let left = |node: &Node| u8::from(node.split != LEAF && node.split & 1 == 1);
        self.nodes.iter().map(left).collect()
    }

    /// What each node predicts: for a classifier the training rows of each class (one number
    /// per class, node after node; their weights in a forest), for a regressor the mean of
    /// the target.
    pub fn value(&self) -> &[f64] {
        &self.value
    }

    /// The training rows in each node.
    pub fn n_node_samples(&self) -> &[u32] {
        &self.n_node_samples
    }

    /// Whether node `i` is a leaf.
    pub(crate) fn is_leaf(&self, i: usize) -> bool {
        self.nodes[i].split == LEAF
    }

    /// The numbers per node in `value`.
    pub(crate) fn outputs(&self) -> usize {
        self.value.len() / self.node_count().max(1)
    }

    /// An error unless the tree is for `features` features and `outputs` numbers per node,
    /// and every child comes after its parent: then a walk from the root always ends in a
    /// leaf and never leaves the lists, whatever a cache file says.
    pub(crate) fn check(&self, features: usize, outputs: usize) -> Result<(), ModelError> {
        let n: usize = self.node_count();
        if n == 0 || self.n_node_samples.len() != n || self.value.len() != n * outputs {
            return Err(bad_learned(format!(
                "a tree of {n} nodes whose lists have other lengths"
            )));
        }
        for (i, node) in self.nodes.iter().enumerate() {
            let (f, right): (usize, usize) = (node.feature(), node.right as usize);
            if node.split != LEAF && !(f < features && right > i + 1 && right < n) {
                return Err(bad_learned(format!(
                    "node {i} splits on feature {f} with its right child at {right}, in a tree \
                     of {n} nodes for {features} features"
                )));
            }
            let set: f64 = node.threshold;
            if node.is_categorical() && !(set >= 0.0 && set.fract() == 0.0 && (set as usize) < self.sets.len()) {
                return Err(bad_learned(format!(
                    "node {i} reads category set {set} of {}",
                    self.sets.len()
                )));
            }
        }
        Ok(())
    }

    /// Writes the tree as bytes: the node count, then the numbers per node with the count of
    /// category sets in its top 32 bits, then the nodes, the values, the rows and the sets,
    /// all little-endian. A tree with no categorical split writes what it wrote before there
    /// were sets. A full tree on 630,000 rows is 1.2 million nodes: 34 MB here, 50 MB and
    /// 300 ms as JSON.
    pub(crate) fn write(&self, out: &mut dyn Write) -> io::Result<()> {
        out.write_all(&(self.node_count() as u64).to_le_bytes())?;
        let sets: u64 = (self.sets.len() as u64) << 32;
        out.write_all(&(self.outputs() as u64 | sets).to_le_bytes())?;
        let mut bytes: Vec<u8> = Vec::with_capacity(self.node_count() * 16);
        for node in &self.nodes {
            bytes.extend_from_slice(&node.threshold.to_le_bytes());
            bytes.extend_from_slice(&node.split.to_le_bytes());
            bytes.extend_from_slice(&node.right.to_le_bytes());
        }
        out.write_all(&bytes)?;
        bytes.clear();
        bytes.extend(self.value.iter().flat_map(|v| v.to_le_bytes()));
        out.write_all(&bytes)?;
        bytes.clear();
        bytes.extend(self.n_node_samples.iter().flat_map(|n| n.to_le_bytes()));
        bytes.extend(self.sets.iter().flatten().flat_map(|bits| bits.to_le_bytes()));
        out.write_all(&bytes)
    }

    /// Reads one tree from the front of `bytes`, which [`Tree::write`] made. The lengths are
    /// checked against the bytes that are there before anything is allocated.
    pub(crate) fn read(bytes: &mut &[u8]) -> Result<Self, ModelError> {
        let header: Vec<usize> = take(bytes, 16)?
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| u64::from_le_bytes(*b) as usize)
            .collect();
        let (n, outputs, sets): (usize, usize, usize) =
            (header[0], header[1] & 0xFFFF_FFFF, header[1] >> 32);
        let sizes: Option<(usize, usize)> = n.checked_mul(16).zip(
            n.checked_mul(outputs)
                .and_then(|values| values.checked_mul(8)),
        );
        let Some((nodes, values)) = sizes else {
            return Err(bad_learned(format!("a tree of {n} nodes")));
        };
        // A node is 16 bytes: the threshold, then the split, then the right child.
        let nodes: Vec<Node> = take(bytes, nodes)?
            .as_chunks::<16>()
            .0
            .iter()
            .map(|b| {
                let bits: u128 = u128::from_le_bytes(*b);
                Node {
                    threshold: f64::from_bits(bits as u64),
                    split: (bits >> 64) as u32,
                    right: (bits >> 96) as u32,
                }
            })
            .collect();
        let value: Vec<f64> = take(bytes, values)?
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| f64::from_le_bytes(*b))
            .collect();
        let n_node_samples: Vec<u32> = take(bytes, n * 4)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect();
        let words: Vec<u64> = take(bytes, sets * 32)?
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| u64::from_le_bytes(*b))
            .collect();
        let sets: Vec<Left> = words.as_chunks::<4>().0.to_vec();
        Ok(Self::from_parts(nodes, value, n_node_samples).with_sets(sets))
    }

    /// The leaf of every row of `columns`, on the Polars pool.
    pub(crate) fn leaves(&self, columns: &[&[f64]], leaves: &mut [u32]) {
        THREAD_POOL.install(|| {
            leaves
                .par_chunks_mut(CHUNK)
                .enumerate()
                .for_each(|(k, part)| self.walk(columns, k * CHUNK, part))
        });
    }

    /// The leaf of each row from `first` on, into `leaves`. [`LANES`] rows walk together, one
    /// step each in turn, so the wait for one row's next node is used by the others. A
    /// categorical feature's column holds category codes (NaN for a missing or unknown one).
    pub(crate) fn walk(&self, columns: &[&[f64]], first: usize, leaves: &mut [u32]) {
        match self.sets.is_empty() {
            true => self.walk_with::<false>(columns, first, leaves),
            false => self.walk_with::<true>(columns, first, leaves),
        }
    }

    /// [`Tree::walk`], with the test of a categorical split only in a tree that has one.
    fn walk_with<const SETS: bool>(&self, columns: &[&[f64]], first: usize, leaves: &mut [u32]) {
        for (block, lanes) in leaves.chunks_mut(LANES).enumerate() {
            let row: usize = first + block * LANES;
            lanes.fill(0);
            let mut walking: bool = true;
            while walking {
                walking = false;
                for (k, at) in lanes.iter_mut().enumerate() {
                    let node: Node = self.nodes[*at as usize];
                    if node.split == LEAF {
                        continue;
                    }
                    let x: f64 = columns[node.feature()][row + k];
                    *at = if self.goes_left::<SETS>(node, x) { *at + 1 } else { node.right };
                    walking = true;
                }
            }
        }
    }

    /// Whether a row whose feature is `x` goes left at `node`.
    #[inline(always)]
    fn goes_left<const SETS: bool>(&self, node: Node, x: f64) -> bool {
        if x.is_nan() {
            node.split & 1 == 1
        } else if SETS && node.split & CATEGORICAL != 0 {
            let (set, code): (&Left, usize) = (&self.sets[node.threshold as usize], x as usize);
            code < 256 && set[code >> 6] >> (code & 63) & 1 == 1
        } else {
            x <= node.threshold
        }
    }

    /// A symmetric tree's shape (every node of a level splits alike, as CatBoost grows them):
    /// the split of each level from the root, and for each leaf number (bit `l` set: right at
    /// level `l`) the leaf's node. `None` for any other tree.
    pub(crate) fn symmetric(&self) -> Option<(Vec<Node>, Vec<u32>)> {
        let mut levels: Vec<Node> = Vec::new();
        while self.nodes.get(levels.len())?.split != LEAF {
            levels.push(self.nodes[levels.len()]);
            if levels.len() > 20 {
                return None;
            }
        }
        let leaf_of: Vec<u32> = (0..1usize << levels.len())
            .map(|bits| {
                let mut at: usize = 0;
                for (l, level) in levels.iter().enumerate() {
                    let node: Node = self.nodes[at];
                    if node.split != level.split || node.threshold.to_bits() != level.threshold.to_bits() {
                        return None;
                    }
                    at = if bits >> l & 1 == 1 { node.right as usize } else { at + 1 };
                }
                (self.nodes[at].split == LEAF).then_some(at as u32)
            })
            .collect::<Option<_>>()?;
        Some((levels, leaf_of))
    }

    /// [`Tree::walk`] of a symmetric tree of shape `(levels, leaf_of)` ([`Tree::symmetric`]):
    /// one pass over the rows per level, each adding the side the rows take as one bit.
    pub(crate) fn walk_symmetric(&self, (levels, leaf_of): &(Vec<Node>, Vec<u32>), columns: &[&[f64]], first: usize, leaves: &mut [u32]) {
        leaves.fill(0);
        for (l, node) in levels.iter().enumerate() {
            let x: &[f64] = &columns[node.feature()][first..first + leaves.len()];
            match node.is_categorical() {
                true => leaves.iter_mut().zip(x).for_each(|(bits, x)| *bits |= u32::from(!self.goes_left::<true>(*node, *x)) << l),
                false => {
                    // Branch-free: NaN fails `x <= t`, so it goes right unless missing values go left.
                    let (t, nan_left): (f64, bool) = (node.threshold, node.split & 1 == 1);
                    #[allow(clippy::neg_cmp_op_on_partial_ord)]
                    for (bits, x) in leaves.iter_mut().zip(x) {
                        *bits |= u32::from(!(*x <= t) & !(nan_left & x.is_nan())) << l;
                    }
                }
            }
        }
        leaves.iter_mut().for_each(|bits| *bits = leaf_of[*bits as usize]);
    }
}

/// The bytes of nodes that stay in the CPU cache next to a chunk of rows.
const CACHED: usize = 1 << 20;

/// For each row of `dataset`, `width` numbers: `add(t, value, out)` adds what the row's leaf
/// in tree `t` holds (`Tree::value` of that leaf) to the row's numbers. `names` are the
/// features, in the order of `fit`; nulls and NaN are missing values. Chunks of rows run on
/// the Polars pool.
///
/// Small trees (a boosted model) stay in the CPU cache together, so each chunk of rows walks
/// them all. Large trees (a forest of full trees) do not: there every chunk walks one tree,
/// then every chunk the next, so that all cores read the same tree while it is in the cache.
pub(crate) fn sum_leaves(
    trees: &[Tree],
    dataset: &Dataset,
    names: &[String],
    width: usize,
    add: impl Fn(usize, &[f64], &mut [f64]) + Sync,
) -> Result<Vec<f64>, ModelError> {
    let series: Vec<Series> = THREAD_POOL.install(|| {
        names
            .par_iter()
            .map(|name| feature(dataset, name))
            .collect::<Result<_, _>>()
    })?;
    let columns: Vec<&[f64]> = series.iter().map(column).collect();
    Ok(sum_leaves_on(trees, &columns, dataset.height(), width, add))
}

/// Rows a task reads through bytes ([`Quantized`]) for every tree of a group: their bytes and
/// sides stay in the core's first cache. Other trees walk blocks of `CHUNK` rows (4,096: 512
/// made forests 2% to 3% slower to predict).
const ROWS: usize = 512;

/// The code of a missing value in [`Quantized`]: above every border.
const NAN_CODE: u8 = u8::MAX;

/// Whether a row of byte `code` goes right at a level on border `j`: `x > t` is `code > j`, and a
/// missing value goes right unless it goes left there.
#[inline(always)]
fn right(code: u8, j: u8, nan_left: bool) -> bool {
    (code > j) & !(nan_left & (code == NAN_CODE))
}

/// Symmetric trees on numbers, read as CatBoost reads them: each block of rows gets one byte per
/// feature, the number of the trees' thresholds on that feature below its value, and a level
/// on threshold `j` sends right the rows whose byte is above `j` (16 rows per instruction,
/// against 2 for the comparison of floats: 4.5× faster on one core, 2026-11-21).
pub(crate) struct Quantized {
    /// Per feature, the distinct thresholds of every level that splits on it, ascending.
    borders: Vec<Vec<f64>>,
    /// Per tree, per level: the feature, the place of its threshold in `borders`, and whether
    /// a missing value goes left.
    levels: Vec<Vec<(usize, u8, bool)>>,
    /// Per tree of 8 levels or fewer and one output, its value for each pattern of sides, so
    /// that a row's byte of sides is its value's place (256 places, so no byte needs a bounds
    /// check); `None` for the other trees.
    by_sides: Vec<Option<Box<[f64; 256]>>>,
}

impl Quantized {
    /// `None` unless every tree is symmetric with numbers only, and no feature has more than
    /// 254 thresholds (a byte holds them and the missing code).
    pub(crate) fn new(trees: &[Tree], shapes: &[Option<(Vec<Node>, Vec<u32>)>], features: usize) -> Option<Self> {
        let mut borders: Vec<Vec<f64>> = vec![Vec::new(); features];
        for shape in shapes {
            for node in &shape.as_ref()?.0 {
                if node.is_categorical() {
                    return None;
                }
                borders[node.feature()].push(node.threshold);
            }
        }
        for b in &mut borders {
            b.sort_unstable_by(f64::total_cmp);
            b.dedup();
            if b.len() >= usize::from(NAN_CODE) {
                return None;
            }
        }
        let place = |node: &Node| {
            let b: &[f64] = &borders[node.feature()];
            b.partition_point(|t| *t < node.threshold) as u8
        };
        let levels: Vec<Vec<(usize, u8, bool)>> = shapes
            .iter()
            .map(|shape| shape.as_ref().map_or_else(Vec::new, |(levels, _)| {
                levels.iter().map(|node| (node.feature(), place(node), node.split & 1 == 1)).collect()
            }))
            .collect();
        let by_sides: Vec<Option<Box<[f64; 256]>>> = trees
            .iter()
            .zip(shapes)
            .map(|(tree, shape)| match shape {
                Some((levels, leaf_of)) if levels.len() <= 8 && tree.outputs() == 1 => {
                    let mut values: Box<[f64; 256]> = Box::new([0.0; 256]);
                    values.iter_mut().zip(leaf_of).for_each(|(v, leaf)| *v = tree.value[*leaf as usize]);
                    Some(values)
                }
                _ => None,
            })
            .collect();
        Some(Quantized { borders, levels, by_sides })
    }

    /// The bytes of `n` rows from `first`, feature after feature, into `codes`.
    fn codes(&self, columns: &[&[f64]], first: usize, n: usize, codes: &mut [u8]) {
        for (f, borders) in self.borders.iter().enumerate() {
            if borders.is_empty() {
                continue;
            }
            for (code, x) in codes[f * n..(f + 1) * n].iter_mut().zip(&columns[f][first..first + n]) {
                // A NaN is below no threshold and above none: its own code.
                *code = if x.is_nan() { NAN_CODE } else { borders.partition_point(|t| t < x) as u8 };
            }
        }
    }

    /// The leaf of each row in tree `t` of shape `leaf_of`, from the bytes of [`Quantized::codes`].
    /// A tree of 8 levels or fewer (CatBoost's default is 6) keeps a row's sides in a byte, so a
    /// level handles 16 rows per instruction; a deeper one in 4 bytes.
    fn leaves(&self, t: usize, leaf_of: &[u32], codes: &[u8], sides: &mut [u8], leaves: &mut [u32]) {
        self.sides(t, codes, sides, leaves.len());
        if self.levels[t].len() <= 8 {
            leaves.iter_mut().zip(sides.iter()).for_each(|(leaf, bits)| *leaf = leaf_of[usize::from(*bits)]);
            return;
        }
        let n: usize = leaves.len();
        leaves.fill(0);
        for (l, &(f, j, nan_left)) in self.levels[t].iter().enumerate() {
            for (bits, code) in leaves.iter_mut().zip(&codes[f * n..(f + 1) * n]) {
                *bits |= u32::from(right(*code, j, nan_left)) << l;
            }
        }
        leaves.iter_mut().for_each(|bits| *bits = leaf_of[*bits as usize]);
    }

    /// The sides of `n` rows in tree `t` of 8 levels or fewer, one bit per level, into `sides`.
    fn sides(&self, t: usize, codes: &[u8], sides: &mut [u8], n: usize) {
        if self.levels[t].len() > 8 {
            return;
        }
        let sides: &mut [u8] = &mut sides[..n];
        sides.fill(0);
        for (l, &(f, j, nan_left)) in self.levels[t].iter().enumerate() {
            // A set bit rather than a shift (SSE2 shifts no bytes), and a loop per missing side.
            let (bit, codes): (u8, &[u8]) = (1 << l, &codes[f * n..(f + 1) * n]);
            match nan_left {
                false => sides.iter_mut().zip(codes).for_each(|(b, c)| *b |= if *c > j { bit } else { 0 }),
                true => sides.iter_mut().zip(codes).for_each(|(b, c)| *b |= if (*c > j) & (*c != NAN_CODE) { bit } else { 0 }),
            }
        }
    }
}

/// The most trees [`sum_leaves_on`] adds in one pass over a block of rows.
const RUN: usize = 8;

/// Adds the values of trees `at..at + R` to each of `n` rows of `part`, tree after tree within
/// a row, from their sides (`ROWS` bytes per tree in `sides`).
fn add_run<const R: usize>(
    at: usize,
    q: &Quantized,
    sides: &[u8],
    n: usize,
    part: &mut [f64],
    add: &impl Fn(usize, &[f64], &mut [f64]),
) {
    let tables: [&[f64; 256]; R] = std::array::from_fn(|r| q.by_sides[at + r].as_deref().expect("a run of tables"));
    let sides: [&[u8]; R] = std::array::from_fn(|r| &sides[r * ROWS..r * ROWS + n]);
    for (i, cell) in part[..n].iter_mut().enumerate() {
        for r in 0..R {
            add(at + r, std::slice::from_ref(&tables[r][usize::from(sides[r][i])]), std::slice::from_mut(cell));
        }
    }
}

/// [`sum_leaves`] on `rows` rows of `columns`, one per feature in the order of `fit`.
pub(crate) fn sum_leaves_on(
    trees: &[Tree],
    columns: &[&[f64]],
    rows: usize,
    width: usize,
    add: impl Fn(usize, &[f64], &mut [f64]) + Sync,
) -> Vec<f64> {
    let mut out: Vec<f64> = vec![0.0; rows * width];
    let nodes: usize = trees.iter().map(Tree::node_count).sum();
    let together: usize = if nodes * 16 <= CACHED { trees.len().max(1) } else { 1 };
    let shapes: Vec<Option<(Vec<Node>, Vec<u32>)>> = trees.iter().map(Tree::symmetric).collect();
    let quantized: Option<Quantized> = Quantized::new(trees, &shapes, columns.len());
    let block: usize = if quantized.is_some() { ROWS } else { CHUNK };
    for (g, group) in trees.chunks(together).enumerate() {
        THREAD_POOL.install(|| {
            out.par_chunks_mut(block * width)
                .enumerate()
                .for_each_init(
                    || (vec![0u32; block], vec![0u8; block * columns.len()], vec![0u8; RUN * ROWS]),
                    |(leaves, codes, sides), (k, part)| {
                        let leaves: &mut [u32] = &mut leaves[..part.len() / width];
                        let n: usize = leaves.len();
                        if let Some(q) = &quantized {
                            q.codes(columns, k * block, n, codes);
                        }
                        let mut t: usize = 0;
                        while t < group.len() {
                            let tree: &Tree = &group[t];
                            let at: usize = g * together + t;
                            if let Some(q) = &quantized
                                && width == 1
                                && q.by_sides[at].is_some()
                            {
                                // One number per row: a row's byte of sides is its value's place.
                                // Up to `RUN` trees in one pass, each row adding their values in
                                // tree order, so the sums are those of one tree at a time.
                                let run: usize = (at..at + RUN.min(group.len() - t))
                                    .take_while(|u| q.by_sides[*u].is_some())
                                    .count();
                                for r in 0..run {
                                    q.sides(at + r, &codes[..n * columns.len()], &mut sides[r * ROWS..], n);
                                }
                                let sides: &[u8] = sides;
                                match run {
                                    8 => add_run::<8>(at, q, sides, n, part, &add),
                                    7 => add_run::<7>(at, q, sides, n, part, &add),
                                    6 => add_run::<6>(at, q, sides, n, part, &add),
                                    5 => add_run::<5>(at, q, sides, n, part, &add),
                                    4 => add_run::<4>(at, q, sides, n, part, &add),
                                    3 => add_run::<3>(at, q, sides, n, part, &add),
                                    2 => add_run::<2>(at, q, sides, n, part, &add),
                                    _ => add_run::<1>(at, q, sides, n, part, &add),
                                }
                                t += run;
                                continue;
                            }
                            t += 1;
                            match (&quantized, &shapes[at]) {
                                (Some(q), Some((_, leaf_of))) => q.leaves(at, leaf_of, &codes[..n * columns.len()], sides, leaves),
                                (None, Some(shape)) => tree.walk_symmetric(shape, columns, k * block, leaves),
                                (_, None) => tree.walk(columns, k * block, leaves),
                            }
                            let outputs: usize = tree.outputs();
                            if (width, outputs) == (1, 1) {
                                // One number per row (regression, two classes): no slices to cut.
                                for (leaf, cell) in leaves.iter().zip(part.iter_mut()) {
                                    add(at, std::slice::from_ref(&tree.value[*leaf as usize]), std::slice::from_mut(cell));
                                }
                                continue;
                            }
                            for (leaf, cell) in leaves.iter().zip(part.chunks_mut(width)) {
                                let value: &[f64] = &tree.value[*leaf as usize * outputs..][..outputs];
                                add(at, value, cell);
                            }
                        }
                    },
                )
        });
    }
    out
}

/// Adds a leaf's share of each class to a row's sums: one tree's vote.
pub(crate) fn add_shares(_: usize, counts: &[f64], sums: &mut [f64]) {
    let total: f64 = counts.iter().sum();
    for (sum, count) in sums.iter_mut().zip(counts) {
        *sum += count / total;
    }
}

/// The class with the largest sum of each row (the first one on a tie, like scikit-learn),
/// as labels of the target's type.
pub(crate) fn pick_classes(
    sums: &[f64],
    classes: &[String],
    target: &str,
    dtype: &DataType,
) -> Result<Series, ModelError> {
    let k: usize = classes.len();
    let picks: Vec<IdxSize> = sums
        .chunks(k)
        .map(|row| (1..k).fold(0, |best, c| if row[c] > row[best] { c } else { best }) as IdxSize)
        .collect();
    let labels: Series = class_labels(target, classes, dtype)?;
    Ok(labels.take(&IdxCa::from_vec("".into(), picks))?)
}

/// The sums of each row divided by `trees`, as one column per class.
pub(crate) fn share_columns(
    dataset: &Dataset,
    sums: &[f64],
    classes: &[String],
    trees: usize,
) -> Result<Dataset, ModelError> {
    let k: usize = classes.len();
    let columns: Vec<Column> = classes
        .iter()
        .enumerate()
        .map(|(c, class)| {
            let share = sums.iter().skip(c).step_by(k).map(|sum| sum / trees as f64);
            Series::new(class.into(), share.collect::<Vec<f64>>()).into()
        })
        .collect();
    Ok(dataset.derived(DataFrame::new_infer_height(columns)?))
}

/// How many features a node may split on: scikit-learn's `max_features`. A whole number is a
/// count (`3`), a number with a point a share of the features (`0.5`, and `1.0` is all of
/// them), and `"sqrt"` or `"log2"` is that function of the number of features.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MaxFeatures {
    /// This many features.
    Count(usize),
    /// This share of the features, above 0 and at most 1.
    Share(f64),
    /// `"sqrt"` or `"log2"`.
    Name(String),
}

impl From<usize> for MaxFeatures {
    fn from(count: usize) -> Self {
        Self::Count(count)
    }
}

impl From<f64> for MaxFeatures {
    fn from(share: f64) -> Self {
        Self::Share(share)
    }
}

impl From<&str> for MaxFeatures {
    fn from(name: &str) -> Self {
        Self::Name(name.into())
    }
}

impl MaxFeatures {
    /// Panics at the caller's line unless the value is one that scikit-learn takes.
    #[track_caller]
    pub(crate) fn checked(self) -> Self {
        let (ok, shown): (bool, String) = match &self {
            Self::Count(count) => (*count >= 1, count.to_string()),
            Self::Share(share) => (*share > 0.0 && *share <= 1.0, share.to_string()),
            Self::Name(name) => (name == "sqrt" || name == "log2", name.clone()),
        };
        let expected: &str = "a count of 1 or more, a share above 0.0 up to 1.0, \"sqrt\" or \"log2\"";
        check(ok, "max_features", shown, expected);
        self
    }

    /// The count for `features` features, 1 at least, like scikit-learn.
    fn count(&self, features: usize) -> usize {
        let count: usize = match self {
            Self::Count(count) => *count,
            Self::Share(share) => (share * features as f64) as usize,
            Self::Name(name) if name == "log2" => (features as f64).log2() as usize,
            Self::Name(_) => (features as f64).sqrt() as usize,
        };
        count.clamp(1, features)
    }
}

/// How much each class counts: scikit-learn's `class_weight`. `"balanced"` gives each class
/// the same total weight, rows ÷ (classes × rows of the class), so a rare class counts as
/// much as a frequent one. A forest also takes `"balanced_subsample"`, the same from the rows
/// each tree drew. Pairs such as `[("yes", 5.0), ("no", 1.0)]` give your own weights; a class
/// you leave out has weight 1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ClassWeight {
    /// `"balanced"` or `"balanced_subsample"`.
    Name(String),
    /// A weight per class.
    Weights(IndexMap<String, f64>),
}

impl From<&str> for ClassWeight {
    fn from(name: &str) -> Self {
        Self::Name(name.into())
    }
}

impl<const N: usize> From<[(&str, f64); N]> for ClassWeight {
    fn from(weights: [(&str, f64); N]) -> Self {
        Self::Weights(weights.iter().map(|(class, w)| (class.to_string(), *w)).collect())
    }
}

impl ClassWeight {
    /// Panics at the caller's line unless the value is one that scikit-learn takes; only a
    /// `forest` draws rows, so only it takes `"balanced_subsample"`.
    #[track_caller]
    pub(crate) fn checked(self, forest: bool) -> Self {
        let (ok, shown): (bool, String) = match &self {
            Self::Name(name) => (
                name == "balanced" || (forest && name == "balanced_subsample"),
                name.clone(),
            ),
            // Not every class 0: a tree would have no weight to split, a forest no row to draw.
            Self::Weights(weights) => (
                weights.values().all(|w| w.is_finite() && *w >= 0.0)
                    && (weights.is_empty() || weights.values().any(|w| *w > 0.0)),
                format!("{weights:?}"),
            ),
        };
        let expected: &str = if forest {
            "\"balanced\", \"balanced_subsample\", or a weight of 0 or more per class, not all 0"
        } else {
            "\"balanced\", or a weight of 0 or more per class, not all 0"
        };
        check(ok, "class_weight", shown, expected);
        self
    }

    /// Whether the weights come from the rows each tree drew.
    pub(crate) fn per_tree(&self) -> bool {
        matches!(self, Self::Name(name) if name == "balanced_subsample")
    }

    /// The weight of each class of `classes`. `y` is the class of each row, and `counts` how
    /// often each row counts (`None`: once).
    pub(crate) fn of_classes(
        &self,
        classes: &[String],
        y: &[u32],
        counts: Option<&[f32]>,
    ) -> Result<Vec<f32>, ModelError> {
        Ok(self.exact(classes, y, counts)?.into_iter().map(|w| w as f32).collect())
    }

    /// [`ClassWeight::of_classes`] in f64, as boosting needs them: scikit-learn's balanced
    /// weights give a weighted share of exactly ½.
    pub(crate) fn exact(
        &self,
        classes: &[String],
        y: &[u32],
        counts: Option<&[f32]>,
    ) -> Result<Vec<f64>, ModelError> {
        match self {
            Self::Weights(weights) => {
                if let Some(class) = weights.keys().find(|class| !classes.contains(class)) {
                    return Err(ModelError::UnknownClass {
                        class: class.clone(),
                        classes: classes.to_vec(),
                    });
                }
                let weight = |class: &String| weights.get(class).copied().unwrap_or(1.0);
                Ok(classes.iter().map(weight).collect())
            }
            Self::Name(_) => {
                let mut rows: Vec<f64> = vec![0.0; classes.len()];
                for (i, class) in y.iter().enumerate() {
                    rows[*class as usize] += counts.map_or(1.0, |c| f64::from(c[i]));
                }
                let total: f64 = rows.iter().sum();
                let present: f64 = rows.iter().filter(|r| **r > 0.0).count() as f64;
                let weight = |r: &f64| if *r > 0.0 { total / (present * r) } else { 0.0 };
                Ok(rows.iter().map(weight).collect())
            }
        }
    }
}

/// The weight of each row from the weight of its class, scikit-learn's
/// `compute_sample_weight(class_weight, y)`: `"balanced"` gives a row of a class with n rows the
/// weight rows ÷ (classes × n), so every class weighs the same in total; pairs such as
/// `[("yes", 5.0), ("no", 1.0)]` give your own (a class left out weighs 1). Give the weights to
/// a metric or to a model's `fit` with weights, for a balanced accuracy or a rare class that
/// counts.
///
/// ```
/// use rusmlai::polars::prelude::*;
/// use rusmlai::prelude::*;
/// let y = Series::new("y".into(), ["a", "a", "a", "b"]);
/// assert_eq!(compute_sample_weight("balanced", &y), [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 2.0]);
/// assert_eq!(compute_sample_weight([("b", 4.0)], &y), [1.0, 1.0, 1.0, 4.0]);
/// ```
///
/// Python:
///
/// ```python
/// import numpy as np
/// from sklearn.utils.class_weight import compute_sample_weight
/// y = ["a", "a", "a", "b"]
/// assert np.allclose(compute_sample_weight("balanced", y), [2 / 3, 2 / 3, 2 / 3, 2.0])
/// assert compute_sample_weight({"b": 4.0}, y).tolist() == [1.0, 1.0, 1.0, 4.0]
/// ```
#[track_caller]
pub fn compute_sample_weight(class_weight: impl Into<ClassWeight>, y: &Series) -> Vec<f64> {
    try_compute_sample_weight(class_weight, y).or_fatal()
}

/// [`compute_sample_weight`] that returns the error: nulls in `y`, a name other than
/// `"balanced"`, or a weight for a class `y` doesn't have.
pub fn try_compute_sample_weight(class_weight: impl Into<ClassWeight>, y: &Series) -> Result<Vec<f64>, ModelError> {
    let class_weight: ClassWeight = class_weight.into();
    if matches!(&class_weight, ClassWeight::Name(name) if name != "balanced") {
        return Err(ModelError::InvalidSetting {
            setting: "class_weight",
            value: format!("{class_weight:?}"),
            expected: "\"balanced\", or a weight per class",
        });
    }
    let frame: DataFrame = DataFrame::new(y.len(), vec![y.clone().into()])?;
    let (classes, codes) = crate::models::label_target(&Dataset::new(frame, ""), y.name())?;
    let codes: Vec<u32> = codes.f64()?.into_no_null_iter().map(|c| c as u32).collect();
    let weights: Vec<f64> = class_weight.exact(&classes, &codes, None)?;
    Ok(codes.iter().map(|c| weights[*c as usize]).collect())
}

/// The weight of each row: its class's weight, times how often the row counts.
pub(crate) fn row_weights(of_classes: &[f32], y: &[u32], counts: Option<&[f32]>) -> Vec<f32> {
    let weight = |(i, class): (usize, &u32)| {
        of_classes[*class as usize] * counts.map_or(1.0, |c| c[i])
    };
    y.iter().enumerate().map(weight).collect()
}

/// The settings of a tree, with scikit-learn's names.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Params {
    pub criterion: String,
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub max_features: MaxFeatures,
    pub random_state: u64,
    pub min_impurity_decrease: f64,
    pub max_leaf_nodes: Option<usize>,
    pub max_bins: usize,
    /// A classifier's only, and written only when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub class_weight: Option<ClassWeight>,
    /// `"best"` or `"random"` (Extra Trees); written only when random, so the settings of
    /// older runs read unchanged.
    #[serde(default = "best_splitter", skip_serializing_if = "is_best_splitter")]
    pub splitter: String,
}

fn best_splitter() -> String {
    "best".into()
}

fn is_best_splitter(splitter: &str) -> bool {
    splitter == "best"
}

impl Params {
    pub fn new(criterion: &str, max_features: MaxFeatures) -> Self {
        Self {
            criterion: criterion.into(),
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features,
            random_state: 42,
            min_impurity_decrease: 0.0,
            max_leaf_nodes: None,
            max_bins: 255,
            class_weight: None,
            splitter: best_splitter(),
        }
    }

    /// The settings of one tree on `features` features, whose nodes draw from `seed`.
    pub fn growth(&self, features: usize, seed: u64) -> Growth {
        Growth {
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            min_impurity_decrease: self.min_impurity_decrease,
            max_features: self.max_features.count(features),
            seed,
            max_leaf_nodes: self.max_leaf_nodes,
            features: None,
            by_level: 1.0,
            by_node: 1.0,
            rules: Rules::Sklearn,
            random_splits: self.splitter == "random",
        }
    }
}

/// The learned values that are small, as the model cache stores them in JSON. The trees are
/// bytes ([`Tree::write`]).
#[derive(Serialize, Deserialize)]
pub(crate) struct Learned {
    /// Empty for a regressor.
    pub classes_: Vec<String>,
    pub feature_names_in_: Vec<String>,
    pub feature_importances_: Vec<f64>,
    /// The target's type, so a classifier's `predict` gives labels of that type.
    pub dtype: DataType,
    /// A forest's out-of-bag score, when it was asked for.
    #[serde(default)]
    pub oob_score_: Option<f64>,
}

/// A cached record of a tree model: its small values, and its trees read from `weights`,
/// each checked against the features and the classes. An error changes nothing.
pub(crate) fn load(
    learned: Value,
    weights: &[u8],
    classifier: bool,
) -> Result<(Learned, Vec<Tree>), ModelError> {
    let l: Learned = serde_json::from_value(learned)?;
    let features: usize = l.feature_names_in_.len();
    let outputs: usize = if classifier { l.classes_.len() } else { 1 };
    if (classifier && outputs < 2) || l.feature_importances_.len() != features {
        return Err(bad_learned(format!(
            "{outputs} classes and {} importances for {features} features",
            l.feature_importances_.len()
        )));
    }
    Ok((l, read_trees(weights, features, outputs)?))
}

/// The trees in `weights`, one after the other, each checked: for `features` features and
/// `outputs` numbers per node.
pub(crate) fn read_trees(
    mut weights: &[u8],
    features: usize,
    outputs: usize,
) -> Result<Vec<Tree>, ModelError> {
    let mut trees: Vec<Tree> = Vec::new();
    while !weights.is_empty() {
        let tree: Tree = Tree::read(&mut weights)?;
        tree.check(features, outputs)?;
        trees.push(tree);
    }
    if trees.is_empty() {
        return Err(bad_learned("no tree in the weights file".into()));
    }
    Ok(trees)
}

/// The settings every tree model shares, as setters on `$model`, whose tree settings are
/// `self.$params`.
macro_rules! tree_settings {
    ($model:ident, $($params:ident).+) => {
        impl $model {
            /// The most splits between the root and a leaf. Default: no limit, so a tree
            /// grows until every leaf is pure or can't be split.
            #[track_caller]
            pub fn max_depth(mut self, max_depth: usize) -> Self {
                check(max_depth > 0, "max_depth", max_depth, "1 or more");
                self.$($params).+.max_depth = Some(max_depth);
                self
            }

            /// The fewest rows a node needs to be split. Default 2. A count of rows:
            /// scikit-learn's share of the rows (a float) is not taken.
            #[track_caller]
            pub fn min_samples_split(mut self, min_samples_split: usize) -> Self {
                let name: &str = "min_samples_split";
                check(min_samples_split >= 2, name, min_samples_split, "2 or more");
                self.$($params).+.min_samples_split = min_samples_split;
                self
            }

            /// The fewest rows on each side of a split. Default 1. A count of rows, like
            /// [`Self::min_samples_split`].
            #[track_caller]
            pub fn min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
                let name: &str = "min_samples_leaf";
                check(min_samples_leaf >= 1, name, min_samples_leaf, "1 or more");
                self.$($params).+.min_samples_leaf = min_samples_leaf;
                self
            }

            /// How many features each node may split on, drawn anew for each node: a count
            /// (`3`), a share of the features (`0.5`; `1.0` is all of them), `"sqrt"` or
            /// `"log2"`. Like scikit-learn, a node looks at more features while every one it
            /// drew has a single value among its rows.
            #[track_caller]
            pub fn max_features(mut self, max_features: impl Into<MaxFeatures>) -> Self {
                self.$($params).+.max_features = max_features.into().checked();
                self
            }

            /// The seed of every draw: the features of each node, and in a forest the rows
            /// of each tree. Default 42, so every run gives the same model.
            pub fn random_state(mut self, random_state: u64) -> Self {
                self.$($params).+.random_state = random_state;
                self
            }

            /// A node is split only when that takes at least this much off the impurity,
            /// weighted by the node's share of the rows. Default 0.
            #[track_caller]
            pub fn min_impurity_decrease(mut self, min_impurity_decrease: f64) -> Self {
                crate::models::at_least_zero("min_impurity_decrease", min_impurity_decrease);
                self.$($params).+.min_impurity_decrease = min_impurity_decrease;
                self
            }

            /// The most leaves, 2 or more. The tree then grows leaf by leaf: the leaf whose
            /// split takes most off the impurity is split first, like scikit-learn's.
            /// Default: no limit.
            #[track_caller]
            pub fn max_leaf_nodes(mut self, max_leaf_nodes: usize) -> Self {
                check(max_leaf_nodes >= 2, "max_leaf_nodes", max_leaf_nodes, "2 or more");
                self.$($params).+.max_leaf_nodes = Some(max_leaf_nodes);
                self
            }

            /// The most buckets per feature, 2 to 65,535. Default 255. A feature with at most
            /// this many distinct values is split exactly, like scikit-learn's tree; one with
            /// more is split between quantile buckets, like `HistGradientBoosting*`
            /// (`max_bins`). scikit-learn's trees and forests have no such setting: set it to
            /// at least a feature's distinct values to split that feature as they do, at the
            /// cost of larger histograms. A text feature has at most 255 categories.
            #[track_caller]
            pub fn max_bins(mut self, max_bins: usize) -> Self {
                check((2..=65_535).contains(&max_bins), "max_bins", max_bins, "2 to 65,535");
                self.$($params).+.max_bins = max_bins;
                self
            }
        }
    };
}
pub(crate) use tree_settings;

/// What a classifier's criterion may be, checked at the caller's line.
#[track_caller]
pub(crate) fn class_criterion(criterion: &str) -> String {
    one_of("criterion", criterion, &["gini", "entropy", "log_loss"]).or_fatal()
}

/// What a regressor's criterion may be, checked at the caller's line.
#[track_caller]
pub(crate) fn number_criterion(criterion: &str) -> String {
    one_of("criterion", criterion, &["squared_error"]).or_fatal()
}

/// The classes of a classifier's target and what a tree sums for it.
pub(crate) fn class_target(
    dataset: &Dataset,
    target: &str,
    criterion: &str,
) -> Result<(Vec<String>, Target), ModelError> {
    let (classes, codes): (Vec<String>, Series) = label_target(dataset, target)?;
    let task: Target = Target::Classes {
        y: column(&codes).iter().map(|code| *code as u32).collect(),
        classes: classes.len(),
        entropy: criterion != "gini",
    };
    Ok((classes, task))
}

/// What a tree sums for a regressor's target, after the checks every regressor makes.
pub(crate) fn number_target(dataset: &Dataset, target: &str) -> Result<Target, ModelError> {
    numeric_target(dataset, target)?;
    let (y, _): (Series, f64) = prepared(dataset, target)?;
    Ok(Target::Regression {
        y: column(&y).to_vec(),
    })
}

/// Predicts a class by asking one yes / no question about a feature at a time, like
/// scikit-learn's `DecisionTreeClassifier`.
///
/// `fit` grows a tree: each node splits its rows on the feature and value that make the two
/// sides purest (`criterion`: `"gini"`, or `"entropy"` / `"log_loss"`, which are the same),
/// until a leaf has one class only or a setting stops it. `predict` gives the most frequent
/// class of the leaf a row ends in (the first one on a tie), and `predict_proba` the share of
/// each class there. The target can be text, integers or booleans, like every classifier's.
///
/// Features are numbers or booleans; encode text with `OrdinalEncoder`, which a tree handles
/// better than one-hot columns. Nulls and NaN need no imputer: at each split the missing
/// values go to the side that fits best, like scikit-learn's.
///
/// How it differs from scikit-learn:
///
/// - A feature with more than `max_bins` (255) distinct values is split between 255 quantile
///   buckets, not between every pair of values. With fewer the tree is scikit-learn's.
/// - Ties between equally good splits go to the first feature and the lowest value;
///   scikit-learn draws one with `random_state`. So this tree is the same on every run.
///   `random_state` (default 42) only seeds the features each node draws when
///   `max_features` is fewer than all of them, and those draws are not scikit-learn's.
/// - With a `class_weight` that is not whole numbers, scikit-learn also splits some nodes that
///   are already pure, because their impurity rounds to a little above 0. The predictions are
///   the same, the tree has fewer nodes.
///
/// scikit-learn settings with no setter yet: `splitter="random"`, `min_weight_fraction_leaf`,
/// `ccp_alpha` and `monotonic_cst`.
///
/// ```
/// use rusmlai::prelude::*;
/// let ds = Dataset::new(
///     df!(
///         "hours" => [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
///         "sleep" => [8.0, 4.0, 7.0, 5.0, 4.5, 8.0, 6.0, 7.5, 5.0, 6.5],
///         "passed" => ["no", "no", "no", "yes", "no", "yes", "no", "yes", "yes", "yes"],
///     )
///     .unwrap(),
///     "",
/// );
/// let mut model = DecisionTreeClassifier::new();
/// model.fit(&ds, "passed");
/// assert_eq!(model.classes_, ["no", "yes"]);
/// assert_eq!(model.score(&ds, "passed"), 1.0); // a full tree learns its rows by heart
///
/// // One question only: hours <= 1.75?
/// let mut stump = DecisionTreeClassifier::new().max_depth(1);
/// stump.fit(&ds, "passed");
/// assert_eq!((stump.tree_.feature()[0], stump.tree_.threshold()[0]), (0, 1.75));
/// assert_eq!(stump.get_n_leaves(), 2);
/// assert_eq!(stump.feature_importances_, [1.0, 0.0]);
/// assert_eq!(stump.score(&ds, "passed"), 0.8);
/// let proba = stump.predict_proba(&ds); // one column per class
/// assert_eq!(proba.to_list("yes")[9], 5.0 / 7.0);
/// ```
///
/// Python:
///
/// ```python
/// import pandas as pd
/// from sklearn.tree import DecisionTreeClassifier
/// ds = pd.DataFrame({
///     "hours": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
///     "sleep": [8.0, 4.0, 7.0, 5.0, 4.5, 8.0, 6.0, 7.5, 5.0, 6.5],
///     "passed": ["no", "no", "no", "yes", "no", "yes", "no", "yes", "yes", "yes"],
/// })
/// x, y = ds[["hours", "sleep"]], ds["passed"]
/// model = DecisionTreeClassifier().fit(x, y)
/// assert model.classes_.tolist() == ["no", "yes"]
/// assert model.score(x, y) == 1.0
///
/// stump = DecisionTreeClassifier(max_depth=1).fit(x, y)
/// assert (stump.tree_.feature[0], stump.tree_.threshold[0]) == (0, 1.75)
/// assert stump.get_n_leaves() == 2
/// assert stump.feature_importances_.tolist() == [1.0, 0.0]
/// assert stump.score(x, y) == 0.8
/// proba = stump.predict_proba(x)
/// assert proba[9, 1] == 5.0 / 7.0
/// ```
#[derive(Debug, Clone)]
pub struct DecisionTreeClassifier {
    pub(crate) params: Params,
    /// The classes `fit` saw, sorted: the order of `predict_proba`'s columns.
    pub classes_: Vec<String>,
    /// The feature columns `fit` saw.
    pub feature_names_in_: Vec<String>,
    /// How many features `fit` saw.
    pub n_features_in_: usize,
    /// How much of the impurity the splits on each feature took off, as shares that sum to 1.
    pub feature_importances_: Vec<f64>,
    /// The fitted tree.
    pub tree_: Tree,
    dtype: Option<DataType>,
    target: Option<String>,
    run: Option<CachedRun>,
}

impl Default for DecisionTreeClassifier {
    fn default() -> Self {
        Self {
            params: Params::new("gini", MaxFeatures::Share(1.0)),
            classes_: Vec::new(),
            feature_names_in_: Vec::new(),
            n_features_in_: 0,
            feature_importances_: Vec::new(),
            tree_: Tree::default(),
            dtype: None,
            target: None,
            run: None,
        }
    }
}

tree_settings!(DecisionTreeClassifier, params);

impl DecisionTreeClassifier {
    /// `"best"` (the default) splits each node at the best threshold of the features drawn;
    /// `"random"` at a threshold drawn between each feature's smallest and largest value in
    /// the node, keeping the best of those (scikit-learn's `ExtraTree`).
    #[track_caller]
    pub fn splitter(mut self, splitter: &str) -> Self {
        self.params.splitter = crate::preprocess::one_of("splitter", splitter, &["best", "random"]).or_fatal();
        self
    }

    /// A full tree: gini, no depth limit, every feature at every node.
    pub fn new() -> Self {
        Self::default()
    }

    /// What a split makes smaller: `"gini"` (the default), or `"entropy"` / `"log_loss"`,
    /// two names of the same measure.
    #[track_caller]
    pub fn criterion(mut self, criterion: &str) -> Self {
        self.params.criterion = class_criterion(criterion);
        self
    }

    /// How much each class counts: `"balanced"`, so that a rare class counts as much as a
    /// frequent one, or your own weights, `[("yes", 5.0), ("no", 1.0)]`. Default: every row
    /// counts once.
    #[track_caller]
    pub fn class_weight(mut self, class_weight: impl Into<ClassWeight>) -> Self {
        self.params.class_weight = Some(class_weight.into().checked(false));
        self
    }

    /// The depth of the fitted tree: the splits on the longest way to a leaf.
    pub fn get_depth(&self) -> usize {
        self.tree_.max_depth()
    }

    /// The leaves of the fitted tree.
    pub fn get_n_leaves(&self) -> usize {
        self.tree_.n_leaves()
    }

    /// The sum of each class's share in each row's leaf, or the error of a model that is not
    /// fitted.
    fn shares(&self, dataset: &Dataset) -> Result<Vec<f64>, ModelError> {
        if self.target.is_none() {
            return Err(ModelError::NotFitted {
                model: "DecisionTreeClassifier",
            });
        }
        let (tree, names) = (std::slice::from_ref(&self.tree_), &self.feature_names_in_);
        sum_leaves(tree, dataset, names, self.classes_.len(), add_shares)
    }
}

impl DecisionTreeClassifier {
    /// Grows the tree; `weights` counts each row that many times.
    fn train(&mut self, dataset: &Dataset, target: &str, weights: Option<&[f32]>) -> Result<(), ModelError> {
        let p: &Params = &self.params;
        let (classes, task): (Vec<String>, Target) = class_target(dataset, target, &p.criterion)?;
        let bins: Bins = Bins::try_new(dataset, target, p.max_bins)?;
        let seed: u64 = Rng::new(p.random_state).next_u64();
        // A class weight makes a weight per row, on top of the weights the caller gave.
        let weighted: Option<Vec<f32>> = match (&p.class_weight, &task) {
            (Some(class_weight), Target::Classes { y, .. }) => {
                let of_classes: Vec<f32> = class_weight.of_classes(&classes, y, None)?;
                Some(row_weights(&of_classes, y, weights))
            }
            _ => None,
        };
        let weights: Option<&[f32]> = weighted.as_deref().or(weights);
        let grown: Grown = grow(&bins, &task, &p.growth(bins.names.len(), seed), weights, None);
        self.dtype = Some(dataset.column(target)?.dtype().clone());
        self.n_features_in_ = bins.names.len();
        (self.feature_names_in_, self.classes_) = (bins.names, classes);
        (self.tree_, self.feature_importances_) = (grown.tree, grown.importances);
        self.target = Some(target.into());
        Ok(())
    }
}

impl Model for DecisionTreeClassifier {
    fn target(&self) -> Option<&str> {
        self.target.as_deref()
    }

    fn try_train(&mut self, dataset: &Dataset, target: &str) -> Result<(), ModelError> {
        self.train(dataset, target, None)
    }

    fn try_train_weighted(
        &mut self,
        dataset: &Dataset,
        target: &str,
        weights: &[f32],
    ) -> Option<Result<(), ModelError>> {
        Some(self.train(dataset, target, Some(weights)))
    }

    fn params(&self) -> Value {
        serde_json::to_value(&self.params).unwrap_or_default()
    }

    fn learned(&self) -> Value {
        serde_json::to_value(Learned {
            classes_: self.classes_.clone(),
            feature_names_in_: self.feature_names_in_.clone(),
            feature_importances_: self.feature_importances_.clone(),
            dtype: self.dtype.clone().unwrap_or(DataType::String),
            oob_score_: None,
        })
        .unwrap_or_default()
    }

    fn write_weights(&self, out: &mut dyn Write) -> io::Result<()> {
        self.tree_.write(out)
    }

    fn try_set_params(&mut self, params: Value) -> Result<(), ModelError> {
        self.params = serde_json::from_value(params)?;
        Ok(())
    }

    fn try_evaluate(&self, dataset: &Dataset, target: &str, scoring: &str) -> Result<f64, ModelError> {
        crate::models::evaluate_classifier(self, dataset, target, scoring)
    }

    fn try_predict_method(&self, dataset: &Dataset, method: &str) -> Result<Dataset, ModelError> {
        crate::models::predict_classifier(self, dataset, method)
    }

    fn is_classifier(&self) -> bool {
        true
    }

    fn try_load(&mut self, learned: Value, weights: &[u8], target: &str) -> Result<(), ModelError> {
        let (l, mut trees): (Learned, Vec<Tree>) = load(learned, weights, true)?;
        let (Some(tree), true) = (trees.pop(), trees.is_empty()) else {
            return Err(bad_learned("more than one tree".into()));
        };
        self.n_features_in_ = l.feature_names_in_.len();
        (self.feature_names_in_, self.feature_importances_) =
            (l.feature_names_in_, l.feature_importances_);
        (self.classes_, self.dtype, self.tree_) = (l.classes_, Some(l.dtype), tree);
        self.target = Some(target.into());
        Ok(())
    }

    fn run(&self) -> Option<&CachedRun> {
        self.run.as_ref()
    }

    fn set_run(&mut self, run: Option<CachedRun>) {
        self.run = run;
    }
}

impl Classifier for DecisionTreeClassifier {
    /// The most frequent class of each row's leaf; the first one on a tie, like scikit-learn.
    fn try_predict(&self, dataset: &Dataset) -> Result<Series, ModelError> {
        pick_classes(
            &self.shares(dataset)?,
            &self.classes_,
            self.target.as_deref().unwrap_or_default(),
            self.dtype.as_ref().unwrap_or(&DataType::String),
        )
    }

    /// The share of each class among the training rows of each row's leaf.
    fn try_predict_proba(&self, dataset: &Dataset) -> Result<Dataset, ModelError> {
        share_columns(dataset, &self.shares(dataset)?, &self.classes_, 1)
    }
}

/// Predicts a number by asking one yes / no question about a feature at a time, like
/// scikit-learn's `DecisionTreeRegressor`.
///
/// `fit` grows a tree: each node splits its rows on the feature and value that leave the
/// smallest squared error (`criterion = "squared_error"`), until a leaf has one target value
/// only or a setting stops it. `predict` gives the mean target of the leaf a row ends in.
///
/// Features, missing values, the buckets (`max_bins`) and ties are those of
/// [`DecisionTreeClassifier`]. One more difference from scikit-learn: a node whose targets
/// are all equal is always a leaf. scikit-learn can split such a node when the targets are
/// not whole numbers, because their variance then rounds to a little above 0; the predictions
/// are the same, the tree has fewer nodes.
///
/// scikit-learn settings with no setter yet: the criteria `"friedman_mse"`,
/// `"absolute_error"` and `"poisson"`, `splitter="random"`, `min_weight_fraction_leaf`,
/// `ccp_alpha` and `monotonic_cst`.
///
/// ```
/// use rusmlai::prelude::*;
/// let ds = Dataset::new(
///     df!(
///         "area" => [50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0],
///         "rooms" => [1, 2, 2, 3, 3, 3, 4, 4],
///         "price" => [150.0, 180.0, 200.0, 260.0, 270.0, 310.0, 400.0, 420.0],
///     )
///     .unwrap(),
///     "",
/// );
/// let mut model = DecisionTreeRegressor::new();
/// model.fit(&ds, "price");
/// assert_eq!(model.score(&ds, "price"), 1.0); // a full tree learns its rows by heart
/// assert_eq!(model.get_n_leaves(), 8);
///
/// // Two questions: four leaves, each predicts the mean price of its rows.
/// let mut small = DecisionTreeRegressor::new().max_depth(2);
/// small.fit(&ds, "price");
/// assert_eq!(small.get_n_leaves(), 4);
/// let predicted = small.predict(&ds);
/// assert_eq!(predicted.f64().unwrap().get(3), Some(265.0)); // (260 + 270) / 2
/// assert!((small.score(&ds, "price") - 0.977_822_457_807_835_3).abs() < 1e-12);
/// ```
///
/// Python:
///
/// ```python
/// import pandas as pd
/// from sklearn.tree import DecisionTreeRegressor
/// ds = pd.DataFrame({
///     "area": [50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0],
///     "rooms": [1, 2, 2, 3, 3, 3, 4, 4],
///     "price": [150.0, 180.0, 200.0, 260.0, 270.0, 310.0, 400.0, 420.0],
/// })
/// x, y = ds[["area", "rooms"]], ds["price"]
/// model = DecisionTreeRegressor().fit(x, y)
/// assert model.score(x, y) == 1.0
/// assert model.get_n_leaves() == 8
///
/// small = DecisionTreeRegressor(max_depth=2).fit(x, y)
/// assert small.get_n_leaves() == 4
/// predicted = small.predict(x)
/// assert predicted[3] == 265.0
/// assert abs(small.score(x, y) - 0.9778224578078353) < 1e-12
/// ```
#[derive(Debug, Clone)]
pub struct DecisionTreeRegressor {
    params: Params,
    /// The feature columns `fit` saw.
    pub feature_names_in_: Vec<String>,
    /// How many features `fit` saw.
    pub n_features_in_: usize,
    /// How much of the squared error the splits on each feature took off, as shares that sum
    /// to 1.
    pub feature_importances_: Vec<f64>,
    /// The fitted tree.
    pub tree_: Tree,
    target: Option<String>,
    run: Option<CachedRun>,
}

impl Default for DecisionTreeRegressor {
    fn default() -> Self {
        Self {
            params: Params::new("squared_error", MaxFeatures::Share(1.0)),
            feature_names_in_: Vec::new(),
            n_features_in_: 0,
            feature_importances_: Vec::new(),
            tree_: Tree::default(),
            target: None,
            run: None,
        }
    }
}

tree_settings!(DecisionTreeRegressor, params);

impl DecisionTreeRegressor {
    /// `"best"` (the default) splits each node at the best threshold of the features drawn;
    /// `"random"` at a threshold drawn between each feature's smallest and largest value in
    /// the node, keeping the best of those (scikit-learn's `ExtraTree`).
    #[track_caller]
    pub fn splitter(mut self, splitter: &str) -> Self {
        self.params.splitter = crate::preprocess::one_of("splitter", splitter, &["best", "random"]).or_fatal();
        self
    }

    /// A full tree: squared error, no depth limit, every feature at every node.
    pub fn new() -> Self {
        Self::default()
    }

    /// What a split makes smaller: `"squared_error"`, the only one so far.
    #[track_caller]
    pub fn criterion(mut self, criterion: &str) -> Self {
        self.params.criterion = number_criterion(criterion);
        self
    }

    /// The depth of the fitted tree: the splits on the longest way to a leaf.
    pub fn get_depth(&self) -> usize {
        self.tree_.max_depth()
    }

    /// The leaves of the fitted tree.
    pub fn get_n_leaves(&self) -> usize {
        self.tree_.n_leaves()
    }
}

impl DecisionTreeRegressor {
    /// Grows the tree; `weights` counts each row that many times.
    fn train(&mut self, dataset: &Dataset, target: &str, weights: Option<&[f32]>) -> Result<(), ModelError> {
        let p: &Params = &self.params;
        let task: Target = number_target(dataset, target)?;
        let bins: Bins = Bins::try_new(dataset, target, p.max_bins)?;
        let seed: u64 = Rng::new(p.random_state).next_u64();
        let grown: Grown = grow(&bins, &task, &p.growth(bins.names.len(), seed), weights, None);
        self.n_features_in_ = bins.names.len();
        self.feature_names_in_ = bins.names;
        (self.tree_, self.feature_importances_) = (grown.tree, grown.importances);
        self.target = Some(target.into());
        Ok(())
    }
}

impl Model for DecisionTreeRegressor {
    fn target(&self) -> Option<&str> {
        self.target.as_deref()
    }

    fn try_train(&mut self, dataset: &Dataset, target: &str) -> Result<(), ModelError> {
        self.train(dataset, target, None)
    }

    fn try_train_weighted(
        &mut self,
        dataset: &Dataset,
        target: &str,
        weights: &[f32],
    ) -> Option<Result<(), ModelError>> {
        Some(self.train(dataset, target, Some(weights)))
    }

    fn params(&self) -> Value {
        serde_json::to_value(&self.params).unwrap_or_default()
    }

    fn learned(&self) -> Value {
        serde_json::to_value(Learned {
            classes_: Vec::new(),
            feature_names_in_: self.feature_names_in_.clone(),
            feature_importances_: self.feature_importances_.clone(),
            dtype: DataType::Float64,
            oob_score_: None,
        })
        .unwrap_or_default()
    }

    fn write_weights(&self, out: &mut dyn Write) -> io::Result<()> {
        self.tree_.write(out)
    }

    fn try_set_params(&mut self, params: Value) -> Result<(), ModelError> {
        self.params = serde_json::from_value(params)?;
        Ok(())
    }

    fn try_evaluate(&self, dataset: &Dataset, target: &str, scoring: &str) -> Result<f64, ModelError> {
        crate::models::evaluate_regressor(self, dataset, target, scoring)
    }

    fn try_predict_method(&self, dataset: &Dataset, method: &str) -> Result<Dataset, ModelError> {
        crate::models::predict_regressor(self, dataset, method)
    }

    fn try_load(&mut self, learned: Value, weights: &[u8], target: &str) -> Result<(), ModelError> {
        let (l, mut trees): (Learned, Vec<Tree>) = load(learned, weights, false)?;
        let (Some(tree), true) = (trees.pop(), trees.is_empty()) else {
            return Err(bad_learned("more than one tree".into()));
        };
        self.n_features_in_ = l.feature_names_in_.len();
        (self.feature_names_in_, self.feature_importances_) =
            (l.feature_names_in_, l.feature_importances_);
        self.tree_ = tree;
        self.target = Some(target.into());
        Ok(())
    }

    fn run(&self) -> Option<&CachedRun> {
        self.run.as_ref()
    }

    fn set_run(&mut self, run: Option<CachedRun>) {
        self.run = run;
    }
}

impl Regressor for DecisionTreeRegressor {
    /// The mean target of each row's leaf.
    fn try_predict(&self, dataset: &Dataset) -> Result<Series, ModelError> {
        let target: &String = self.target.as_ref().ok_or(ModelError::NotFitted {
            model: "DecisionTreeRegressor",
        })?;
        let (tree, names) = (std::slice::from_ref(&self.tree_), &self.feature_names_in_);
        let predicted: Vec<f64> = sum_leaves(tree, dataset, names, 1, |_, mean, sum| sum[0] += mean[0])?;
        Ok(Series::new(target.into(), predicted))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use polars::df;

    use super::*;

    /// 400 rows from whole-number formulas, the same as `table()` in
    /// `analysis/2026-11-16/tree-bench/parity.py`, which gave every scikit-learn 1.9.1 value
    /// below. Each of those trees is the same for 20 values of `random_state`: no ties.
    pub(crate) fn table() -> Dataset {
        let i: Vec<i64> = (0..400).collect();
        let a: Vec<f64> = i.iter().map(|i| (i * 37 % 101) as f64).collect();
        let b: Vec<f64> = i.iter().map(|i| (i * 53 % 17) as f64).collect();
        let c: Vec<f64> = i.iter().map(|i| (i * 11 % 7) as f64).collect();
        let score: Vec<f64> = (0..400)
            .map(|k| a[k] + 3.0 * b[k] - 5.0 * c[k] + (i[k] * 29 % 13) as f64)
            .collect();
        let class: Vec<i64> = score.iter().map(|s| ((s / 40.0).floor() as i64).rem_euclid(3)).collect();
        let y: Vec<f64> = score.iter().map(|s| s / 4.0).collect();
        Dataset::new(
            df!("a" => a, "b" => b, "c" => c, "class" => class, "y" => y).unwrap(),
            "",
        )
    }

    /// `table()` where every third `a` and every fifth `b` is missing.
    fn with_missing() -> Dataset {
        let ds: Dataset = table();
        let blank = |name: &str, every: usize| -> Vec<Option<f64>> {
            let values = ds.to_list(name).into_iter().enumerate();
            values.map(|(i, v)| (i % every != 0).then_some(v)).collect()
        };
        let mut frame: DataFrame = ds.dataframe.clone();
        frame.with_column(Series::new("a".into(), blank("a", 3)).into()).unwrap();
        frame.with_column(Series::new("b".into(), blank("b", 5)).into()).unwrap();
        Dataset::new(frame, "")
    }

    /// The tree's splits against scikit-learn's: (feature, threshold, rows) of every node.
    fn assert_tree(tree: &Tree, feature: &[i32], threshold: &[f64], samples: &[u32]) {
        assert_eq!(tree.feature(), feature);
        assert_eq!(tree.threshold(), threshold);
        assert_eq!(tree.n_node_samples(), samples);
    }

    /// Each of `got` within `eps` of `want`'s, for the tests of every model.
    pub(crate) fn assert_close(got: &[f64], want: &[f64], eps: f64) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < eps, "{got:?}, want {want:?}");
        }
    }

    fn classifier(model: DecisionTreeClassifier, ds: &Dataset) -> DecisionTreeClassifier {
        let mut model: DecisionTreeClassifier = model;
        model.fit(&ds.drop(&["y"]), "class");
        model
    }

    fn regressor(model: DecisionTreeRegressor, ds: &Dataset) -> DecisionTreeRegressor {
        let mut model: DecisionTreeRegressor = model;
        model.fit(&ds.drop(&["class"]), "y");
        model
    }

    #[test]
    fn the_classifier_grows_scikit_learn_s_tree() {
        let ds: Dataset = table();
        let features: Dataset = ds.drop(&["y"]);
        let gini = classifier(DecisionTreeClassifier::new().max_depth(3), &ds);
        assert_tree(
            &gini.tree_,
            &[0, 0, 1, -1, -1, 1, -1, -1, 1, 0, -1, -1, 0, -1, -1],
            &[50.5, 24.5, 8.5, 0.0, 0.0, 4.5, 0.0, 0.0, 6.5, 78.0, 0.0, 0.0, 90.5, 0.0, 0.0],
            &[400, 203, 100, 53, 47, 103, 29, 74, 197, 81, 45, 36, 116, 93, 23],
        );
        assert_eq!(gini.tree_.children_right(), [8, 5, 4, 0, 0, 7, 0, 0, 12, 11, 0, 0, 14, 0, 0]);
        assert_eq!(gini.tree_.missing_go_to_left(), [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]);
        assert_eq!(gini.tree_.value()[..9], [114.0, 159.0, 127.0, 88.0, 101.0, 14.0, 64.0, 29.0, 7.0]);
        assert_close(&gini.feature_importances_, &[0.7021526676677854, 0.2978473323322146, 0.0], 1e-12);
        assert_eq!(gini.score(&features, "class"), 0.755);
        assert_eq!((gini.get_depth(), gini.get_n_leaves(), gini.tree_.node_count()), (3, 8, 15));
        assert_eq!(gini.predict(&features).dtype(), &DataType::Int64);

        for name in ["entropy", "log_loss"] {
            let entropy = classifier(DecisionTreeClassifier::new().criterion(name).max_depth(3), &ds);
            assert_tree(
                &entropy.tree_,
                &[0, 1, 0, -1, -1, 0, -1, -1, 0, 1, -1, -1, 1, -1, -1],
                &[49.5, 7.5, 21.5, 0.0, 0.0, 25.5, 0.0, 0.0, 79.5, 7.5, 0.0, 0.0, 7.5, 0.0, 0.0],
                &[400, 199, 95, 40, 55, 104, 56, 48, 201, 118, 56, 62, 83, 37, 46],
            );
            assert_close(&entropy.feature_importances_, &[0.7101804865722318, 0.2898195134277683, 0.0], 1e-12);
            assert_eq!(entropy.score(&features, "class"), 0.7025);
        }
    }

    #[test]
    fn each_setting_stops_the_tree_where_scikit_learn_stops() {
        let ds: Dataset = table();
        let features: Dataset = ds.drop(&["y"]);
        let leaf = classifier(DecisionTreeClassifier::new().min_samples_leaf(20), &ds);
        assert_tree(
            &leaf.tree_,
            &[
                0, 0, 1, 1, -1, -1, 2, -1, -1, 1, -1, 2, -1, 1, -1, -1, 1, 0, 0, -1, -1, -1, 0, 0,
                -1, 1, -1, -1, -1,
            ],
            &[
                50.5, 24.5, 8.5, 4.5, 0.0, 0.0, 2.5, 0.0, 0.0, 4.5, 0.0, 2.5, 0.0, 10.5, 0.0, 0.0,
                6.5, 78.0, 65.5, 0.0, 0.0, 0.0, 90.5, 66.5, 0.0, 12.5, 0.0, 0.0, 0.0,
            ],
            &[
                400, 203, 100, 53, 30, 23, 47, 20, 27, 103, 29, 74, 33, 41, 20, 21, 197, 81, 45,
                25, 20, 36, 116, 93, 36, 57, 33, 24, 23,
            ],
        );
        assert_close(
            &leaf.feature_importances_,
            &[0.6484251877092363, 0.3023597541568698, 0.04921505813389395],
            1e-12,
        );
        assert_eq!(leaf.score(&features, "class"), 0.7725);

        let split = classifier(DecisionTreeClassifier::new().min_samples_split(150), &ds);
        assert_tree(
            &split.tree_,
            &[0, 0, -1, -1, 1, -1, -1],
            &[50.5, 24.5, 0.0, 0.0, 6.5, 0.0, 0.0],
            &[400, 203, 100, 103, 197, 81, 116],
        );

        let decrease = classifier(DecisionTreeClassifier::new().min_impurity_decrease(0.02), &ds);
        assert_tree(
            &decrease.tree_,
            &[0, 0, 1, -1, 2, -1, -1, 1, -1, -1, 1, 0, -1, -1, 0, -1, -1],
            &[
                50.5, 24.5, 8.5, 0.0, 3.5, 0.0, 0.0, 4.5, 0.0, 0.0, 6.5, 78.0, 0.0, 0.0, 90.5,
                0.0, 0.0,
            ],
            &[400, 203, 100, 53, 47, 28, 19, 103, 29, 74, 197, 81, 45, 36, 116, 93, 23],
        );

        // A full tree: scikit-learn's has ties, so its splits differ by seed, but every one
        // learns the rows by heart with 74 leaves.
        let full = classifier(DecisionTreeClassifier::new(), &ds);
        assert_eq!(full.score(&features, "class"), 1.0);
        assert_eq!((full.get_n_leaves(), full.tree_.node_count()), (74, 147));
    }

    #[test]
    fn the_regressor_grows_scikit_learn_s_tree() {
        let ds: Dataset = table();
        let features: Dataset = ds.drop(&["class"]);
        let model = regressor(DecisionTreeRegressor::new().max_depth(3), &ds);
        assert_tree(
            &model.tree_,
            &[0, 0, 1, -1, -1, 1, -1, -1, 1, 0, -1, -1, 0, -1, -1],
            &[49.5, 24.5, 8.5, 0.0, 0.0, 7.5, 0.0, 0.0, 8.5, 74.5, 0.0, 0.0, 75.5, 0.0, 0.0],
            &[400, 199, 100, 53, 47, 99, 47, 52, 201, 105, 52, 53, 96, 47, 49],
        );
        assert_close(
            model.tree_.value(),
            &[
                16.235,
                9.804020100502512,
                6.72,
                3.6556603773584904,
                10.175531914893616,
                12.919191919191919,
                9.702127659574469,
                15.826923076923077,
                22.601990049751244,
                19.478571428571428,
                16.240384615384617,
                22.65566037735849,
                26.018229166666668,
                22.8031914893617,
                29.102040816326532,
            ],
            1e-12,
        );
        assert_close(&model.feature_importances_, &[0.8311093245191086, 0.16889067548089148, 0.0], 1e-12);
        assert!((model.score(&features, "y") - 0.8240926174420184).abs() < 1e-12);
        // The same tree: no split below depth 3 takes 2.0 off the squared error.
        let decrease = regressor(DecisionTreeRegressor::new().min_impurity_decrease(2.0), &ds);
        assert_eq!(decrease.tree_.feature(), model.tree_.feature());

        let leaf = regressor(DecisionTreeRegressor::new().min_samples_leaf(25), &ds);
        assert_tree(
            &leaf.tree_,
            &[0, 0, 1, 0, -1, -1, -1, 1, -1, 0, -1, -1, 1, 0, 0, -1, -1, 2, -1, -1, 0, -1, -1],
            &[
                49.5, 24.5, 8.5, 12.5, 0.0, 0.0, 0.0, 7.5, 0.0, 36.5, 0.0, 0.0, 8.5, 74.5, 62.0,
                0.0, 0.0, 3.5, 0.0, 0.0, 75.5, 0.0, 0.0,
            ],
            &[
                400, 199, 100, 53, 28, 25, 47, 99, 47, 52, 25, 27, 201, 105, 52, 25, 27, 53, 28,
                25, 96, 47, 49,
            ],
        );
        assert!((leaf.score(&features, "y") - 0.8448728082980747).abs() < 1e-12);

        let full = regressor(DecisionTreeRegressor::new(), &ds);
        assert_eq!(full.score(&features, "y"), 1.0);
        assert_eq!(full.get_n_leaves(), 381);
    }

    #[test]
    fn missing_values_go_to_the_side_scikit_learn_sends_them() {
        let ds: Dataset = with_missing();
        let model = classifier(DecisionTreeClassifier::new().max_depth(3), &ds);
        assert_tree(
            &model.tree_,
            &[0, 0, 1, -1, -1, 0, -1, -1, 1, 2, -1, -1, 0, -1, -1],
            &[63.5, 24.5, 8.5, 0.0, 0.0, 49.5, 0.0, 0.0, 6.5, 4.5, 0.0, 0.0, 86.5, 0.0, 0.0],
            &[400, 303, 66, 43, 23, 237, 66, 171, 97, 49, 33, 16, 48, 25, 23],
        );
        assert_eq!(model.tree_.missing_go_to_left(), [1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0]);
        assert_close(
            &model.feature_importances_,
            &[0.7394181024118794, 0.17159619936189532, 0.08898569822622536],
            1e-12,
        );
        assert_eq!(model.score(&ds.drop(&["y"]), "class"), 0.6075);
        // A row with no `a` and no `b`: NaN and null are both missing.
        let row: Dataset = Dataset::new(
            df!("a" => [f64::NAN], "b" => [None::<f64>], "c" => [0.0]).unwrap(),
            "",
        );
        let proba: Dataset = model.predict_proba(&row);
        let shares: Vec<f64> = ["0", "1", "2"].iter().map(|c| proba.to_list(c)[0]).collect();
        assert_close(&shares, &[0.22807017543859648, 0.4444444444444444, 0.32748538011695905], 1e-12);

        let model = regressor(DecisionTreeRegressor::new().max_depth(3), &ds);
        assert_tree(
            &model.tree_,
            &[0, 0, 1, -1, -1, 1, -1, -1, 1, 0, -1, -1, 0, -1, -1],
            &[56.5, 31.5, 8.5, 0.0, 0.0, 7.5, 0.0, 0.0, 10.5, 79.5, 0.0, 0.0, 86.0, 0.0, 0.0],
            &[400, 285, 83, 56, 27, 202, 81, 121, 115, 86, 47, 39, 29, 17, 12],
        );
        assert_eq!(model.tree_.missing_go_to_left(), [1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0]);
        assert!((model.score(&ds.drop(&["class"]), "y") - 0.5731178852098175).abs() < 1e-12);
    }

    #[test]
    fn a_split_that_takes_nothing_off_is_still_made_and_equal_targets_are_a_leaf() {
        // XOR: no first split helps, the second ones do. scikit-learn fits it, so must we.
        let xor: Dataset = Dataset::new(
            df!("p" => [0, 0, 1, 1, 0, 0, 1, 1], "q" => [0, 1, 0, 1, 0, 1, 0, 1],
                "y" => ["a", "b", "b", "a", "a", "b", "b", "a"])
            .unwrap(),
            "",
        );
        let mut model: DecisionTreeClassifier = DecisionTreeClassifier::new();
        model.fit(&xor, "y");
        assert_eq!((model.score(&xor, "y"), model.tree_.node_count()), (1.0, 7));

        // 0.1 three times: Σy²/n − mean² is rounding noise, not a spread.
        let flat: Dataset = Dataset::new(
            df!("x" => [1.0, 2.0, 3.0, 4.0], "y" => [0.1, 0.1, 0.1, 0.7]).unwrap(),
            "",
        );
        let mut model: DecisionTreeRegressor = DecisionTreeRegressor::new();
        model.fit(&flat, "y");
        assert_eq!(model.tree_.feature(), [0, -1, -1]);
        assert_eq!(model.tree_.threshold()[0], 3.5);
        assert_eq!(model.feature_importances_, [1.0]);
    }

    #[test]
    fn many_distinct_values_are_split_between_quantile_buckets() {
        // 2,000 distinct values and a step at 1234.5: 255 buckets find it to within a bucket.
        let x: Vec<f64> = (0..2000).map(f64::from).collect();
        let y: Vec<f64> = x.iter().map(|x| if *x > 1234.5 { 5.0 } else { 1.0 }).collect();
        let ds: Dataset = Dataset::new(df!("x" => x, "y" => y).unwrap(), "");
        let mut model: DecisionTreeRegressor = DecisionTreeRegressor::new().max_depth(1);
        model.fit(&ds, "y");
        assert!((model.tree_.threshold()[0] - 1234.5).abs() <= 8.0, "{:?}", model.tree_.threshold());
        assert!(model.score(&ds, "y") > 0.98);
        // A full tree narrows it down to the bucket, never further.
        let mut full: DecisionTreeRegressor = DecisionTreeRegressor::new();
        full.fit(&ds, "y");
        assert!(full.get_n_leaves() <= 3);
        let mut coarse: DecisionTreeRegressor = DecisionTreeRegressor::new().max_bins(4);
        coarse.fit(&ds, "y");
        assert!(coarse.score(&ds, "y") < full.score(&ds, "y"));
        // A bucket per value: scikit-learn's exact threshold, and a perfect fit.
        let mut exact: DecisionTreeRegressor = DecisionTreeRegressor::new().max_bins(2000);
        exact.fit(&ds, "y");
        assert_eq!(exact.tree_.threshold()[0], 1234.5);
        assert_eq!(exact.score(&ds, "y"), 1.0);
        assert_eq!(exact.get_n_leaves(), 2);
    }

    #[test]
    fn errors_name_the_problem() {
        let fit = |ds: Dataset| {
            DecisionTreeClassifier::new()
                .try_train(&ds, "y")
                .unwrap_err()
                .to_string()
        };
        let text: Dataset = Dataset::new(df!("s" => ["u", "v"], "y" => ["a", "b"]).unwrap(), "");
        assert!(fit(text).starts_with("feature 's' is str, not a number"));
        let infinite: Dataset = Dataset::new(
            df!("x" => [1.0, f64::INFINITY], "y" => ["a", "b"]).unwrap(),
            "",
        );
        assert!(fit(infinite).starts_with("column 'x' has NaN or infinite values"));
        let alone: Dataset = Dataset::new(df!("y" => ["a", "b"]).unwrap(), "");
        assert!(fit(alone).starts_with("the dataset has no feature columns"));
        let ds: Dataset = table();
        assert!(matches!(
            DecisionTreeClassifier::new().try_predict(&ds),
            Err(ModelError::NotFitted { .. })
        ));
        assert!(matches!(
            DecisionTreeRegressor::new().try_predict(&ds),
            Err(ModelError::NotFitted { .. })
        ));
        // A regressor's target is a number.
        assert!(DecisionTreeRegressor::new().try_train(&with_missing(), "a").is_err());
    }

    #[test]
    #[should_panic(expected = "max_bins = 65536 is not valid")]
    fn a_bad_setting_panics_at_the_setter() {
        let _ = DecisionTreeRegressor::new().max_bins(65_536);
    }

    #[test]
    fn learned_values_round_trip_and_a_broken_tree_is_refused() {
        let ds: Dataset = table();
        let features: Dataset = ds.drop(&["y"]);
        let trained = classifier(DecisionTreeClassifier::new().max_depth(4), &ds);
        let mut weights: Vec<u8> = Vec::new();
        trained.write_weights(&mut weights).unwrap();
        assert_eq!(weights.len(), 16 + trained.tree_.node_count() * (16 + 3 * 8 + 4));
        let mut loaded: DecisionTreeClassifier = DecisionTreeClassifier::new();
        loaded.try_load(trained.learned(), &weights, "class").unwrap();
        assert_eq!(loaded.tree_, trained.tree_);
        assert_eq!(loaded.predict(&features), trained.predict(&features));
        let mut again: DecisionTreeClassifier = DecisionTreeClassifier::new();
        again.try_set_params(trained.params()).unwrap();
        assert_eq!(again.params(), trained.params());

        // Node 0 is bytes 16 to 32: its split at 24, its right child at 28. A right child
        // that points back would walk in a circle; one past the end would read outside the
        // lists; a feature the model doesn't have, too.
        for (at, value) in [(28, 0u32), (28, 99), (24, 3 << 1)] {
            let mut broken: Vec<u8> = weights.clone();
            broken[at..at + 4].copy_from_slice(&value.to_le_bytes());
            let mut model: DecisionTreeClassifier = DecisionTreeClassifier::new();
            let err: String = model
                .try_load(trained.learned(), &broken, "class")
                .unwrap_err()
                .to_string();
            assert!(err.contains("node 0 splits on feature"), "{err}");
            assert_eq!(model.tree_.node_count(), 0);
        }
        // Cut short, empty, a node count that the bytes can't hold, and two trees.
        let cut = |bytes: &[u8]| {
            DecisionTreeClassifier::new()
                .try_load(trained.learned(), bytes, "class")
                .unwrap_err()
                .to_string()
        };
        assert!(cut(&weights[..weights.len() - 1]).contains("cut short"));
        assert!(cut(&[]).contains("no tree"));
        let mut huge: Vec<u8> = weights.clone();
        huge[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(cut(&huge).contains("a tree of"));
        assert!(cut(&[weights.clone(), weights.clone()].concat()).contains("more than one tree"));
    }

    #[test]
    fn max_features_draws_per_node_and_the_seed_decides() {
        let ds: Dataset = table();
        let features: Dataset = ds.drop(&["y"]);
        let one = |seed: u64| classifier(DecisionTreeClassifier::new().max_features(1).random_state(seed), &ds);
        let (a, again, b) = (one(1), one(1), one(2));
        assert_eq!(a.tree_, again.tree_, "the same seed, the same tree");
        assert_ne!(a.tree_, b.tree_, "another seed, other draws");
        // One feature per node still learns the rows by heart, with a deeper tree.
        assert_eq!(a.score(&features, "class"), 1.0);
        let full = classifier(DecisionTreeClassifier::new(), &ds);
        assert!(a.tree_.node_count() > full.tree_.node_count());
        // Every way to say "all the features" is the tree without a draw.
        for all in [MaxFeatures::Count(3), MaxFeatures::Share(1.0), MaxFeatures::Count(9)] {
            let model = classifier(DecisionTreeClassifier::new().max_features(all), &ds);
            assert_eq!(model.tree_, full.tree_);
        }
        let counts = |m: MaxFeatures| [m.count(1), m.count(13), m.count(100)];
        assert_eq!(counts("sqrt".into()), [1, 3, 10]);
        assert_eq!(counts("log2".into()), [1, 3, 6]);
        assert_eq!(counts(0.5.into()), [1, 6, 50]);
        assert_eq!(counts(4.into()), [1, 4, 4]);
        let params: Value = one(1).params();
        assert_eq!((&params["max_features"], &params["random_state"]), (&Value::from(1), &Value::from(1)));
    }

    /// (rows, feature, threshold) of every node, sorted: a tree grown leaf by leaf has its
    /// nodes in another order than scikit-learn's, so compare what they are, not where.
    fn nodes(tree: &Tree) -> Vec<(u32, i32, f64)> {
        let (f, t) = (tree.feature(), tree.threshold());
        let mut nodes: Vec<(u32, i32, f64)> = (0..tree.node_count())
            .map(|i| (tree.n_node_samples()[i], f[i], t[i]))
            .collect();
        nodes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        nodes
    }

    #[test]
    fn max_leaf_nodes_grows_the_best_leaf_first_like_scikit_learn() {
        let ds: Dataset = table();
        let model = classifier(DecisionTreeClassifier::new().max_leaf_nodes(8), &ds);
        assert_eq!(model.get_n_leaves(), 8);
        let split = |rows, feature, threshold| (rows, feature, threshold);
        let leaf = |rows| (rows, -1, 0.0);
        assert_eq!(
            nodes(&model.tree_),
            [
                leaf(23), leaf(29), leaf(36), leaf(45), leaf(47), leaf(53), leaf(74),
                split(81, 0, 78.0), leaf(93), split(100, 1, 8.5), split(103, 1, 4.5),
                split(116, 0, 90.5), split(197, 1, 6.5), split(203, 0, 24.5), split(400, 0, 50.5),
            ]
        );
        assert_eq!(model.score(&ds.drop(&["y"]), "class"), 0.755);

        let model = regressor(DecisionTreeRegressor::new().max_leaf_nodes(10), &ds);
        assert_eq!(
            nodes(&model.tree_),
            [
                leaf(19), leaf(22), leaf(28), leaf(30), leaf(47), leaf(47), split(47, 2, 3.5),
                leaf(49), leaf(52), split(52, 2, 3.5), leaf(53), leaf(53), split(96, 0, 75.5),
                split(99, 1, 7.5), split(100, 1, 8.5), split(105, 0, 74.5), split(199, 0, 24.5),
                split(201, 1, 8.5), split(400, 0, 49.5),
            ]
        );
        assert!((model.score(&ds.drop(&["class"]), "y") - 0.8443496343183833).abs() < 1e-12);
        // Its nodes are in the usual order: a walk from the root ends in a leaf.
        model.tree_.check(3, 1).unwrap();
        // The second fit takes the buckets that the first one made.
        assert!(ds.try_column_info("a").unwrap().bins.get(255, 400).is_some());
    }

    #[test]
    fn class_weight_weighs_the_rows_of_each_class_like_scikit_learn() {
        let ds: Dataset = table();
        let features: Dataset = ds.drop(&["y"]);
        let balanced = classifier(
            DecisionTreeClassifier::new().class_weight("balanced").max_depth(3),
            &ds,
        );
        assert_eq!(
            balanced.tree_.feature(),
            [0, 1, 0, -1, -1, 0, -1, -1, 0, 1, -1, -1, 1, -1, -1]
        );
        assert_eq!(
            balanced.tree_.threshold(),
            [49.5, 7.5, 39.0, 0.0, 0.0, 15.5, 0.0, 0.0, 77.5, 7.5, 0.0, 0.0, 8.5, 0.0, 0.0]
        );
        // Each class weighs 400 / 3 in all.
        for weight in &balanced.tree_.value()[..3] {
            assert!((weight - 400.0 / 3.0).abs() < 1e-3, "{:?}", balanced.tree_.value());
        }
        assert_eq!(balanced.score(&features, "class"), 0.75);
        let first: Dataset = Dataset::new(features.dataframe().slice(0, 1), "");
        let proba: Dataset = balanced.predict_proba(&first);
        let shares: Vec<f64> = ["0", "1", "2"].iter().map(|c| proba.to_list(c)[0]).collect();
        let want: [f64; 3] = [0.8117192568073315, 0.10034265373675542, 0.08793808945591242];
        for (got, want) in shares.iter().zip(want) {
            assert!((got - want).abs() < 1e-6, "{shares:?}");
        }

        let weights = [("0", 1.0), ("1", 3.0), ("2", 0.5)];
        let weighted = classifier(DecisionTreeClassifier::new().class_weight(weights).max_depth(3), &ds);
        assert_eq!(
            weighted.tree_.feature(),
            [0, 0, 1, -1, -1, 0, -1, -1, 1, 2, -1, -1, 2, -1, -1]
        );
        assert_eq!(
            weighted.tree_.threshold(),
            [79.5, 24.5, 8.5, 0.0, 0.0, 65.5, 0.0, 0.0, 7.5, 4.5, 0.0, 0.0, 3.5, 0.0, 0.0]
        );
        assert_eq!(weighted.tree_.value()[..3], [114.0, 477.0, 63.5]);
        assert_eq!(weighted.score(&features, "class"), 0.6575);
        // The weights are settings, so the cache tells the runs apart.
        assert_eq!(weighted.params()["class_weight"]["1"], 3.0);
        assert!(DecisionTreeClassifier::new().params().get("class_weight").is_none());

        let unknown = DecisionTreeClassifier::new().class_weight([("7", 2.0)]);
        let err: String = classifier_err(unknown, &features);
        assert!(err.starts_with("'7' is not one of the classes"), "{err}");
    }

    fn classifier_err(model: DecisionTreeClassifier, ds: &Dataset) -> String {
        let mut model: DecisionTreeClassifier = model;
        model.try_train(ds, "class").unwrap_err().to_string()
    }

    #[test]
    #[should_panic(expected = "class_weight = balanced_subsample is not valid")]
    fn a_tree_draws_no_rows_so_it_takes_no_balanced_subsample() {
        let _ = DecisionTreeClassifier::new().class_weight("balanced_subsample");
    }

}
