//! [`CatBoostRegressor`] and [`CatBoostClassifier`]: gradient boosting with CatBoost's names,
//! defaults and trees.
//!
//! CatBoost grows symmetric (oblivious) trees: every node of one depth splits on the same
//! feature and border, so a tree of depth 6 is 6 splits and 64 leaves, and a row's leaf is the
//! 6 bits of its answers. A split is chosen for the whole level by CatBoost's `Cosine` score
//! over all the leaves. Leaves take one Newton step (the full hessian of the softmax for more
//! than two classes), with `l2_leaf_reg` added to the hessian. The buckets are the engine's
//! (`models/hist.rs`), the borders between them CatBoost's: the midpoint of two neighbouring
//! values, as a 32-bit float. A text column becomes CatBoost's ordered target statistics, or
//! one-hot splits when it has at most `one_hot_max_size` categories.
//!
//! A fitted tree is stored as an ordinary [`Tree`] (each level's split repeated in every node
//! of the level), so the cache, `predict` and the journal treat it as any other tree. A split
//! on a target statistic becomes a split on a set of categories: the ones whose statistic on
//! all the rows is at most the border go left.

use std::io::{self, Write};

use polars::prelude::{DataType, NamedFrom, Series};
use polars_core::runtime::THREAD_POOL;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    dataset::{Dataset, split::Rng},
    metadata::models::CachedRun,
    models::{
        Classifier, Model, ModelError, Regressor, bad_learned,
        boosting::{
            Boosted, Fitted, Learned, Library, Loss, Plan, boosted_model, categorical_columns, class_shares,
            fit_classifier, fit_regressor, mixed, scores_of,
        },
        check,
        hist::{Bins, Categories, Code, Codes, Rules, Step},
        tree::{CATEGORICAL, ClassWeight, LEAF, Left, Node, Tree, pick_classes, share_columns},
    },
};
use crate::models::{above_zero, at_least_zero, share};

/// Rows of one task of a histogram or a gradient pass. Fixed blocks, added in block order, so
/// a fit is the same on any number of threads.
const BLOCK: usize = 1 << 15;

/// Columns whose histograms fill in one pass over the rows: each row's leaf and sums are read
/// once for all of them.
const GROUP: usize = 4;

/// The buckets of a target statistic: CatBoost's 15 uniform borders on 0 to 1.
const CTR_LEVELS: usize = 16;

/// How rows are weighted for a tree (CatBoost's `bootstrap_type`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub(crate) enum Bootstrap {
    /// Every row once.
    No,
    /// Each row weighted (−ln u)^t, `bagging_temperature` t.
    Bayesian(f64),
    /// Each row kept with chance `subsample`.
    Bernoulli(f64),
    /// Minimal variance sampling: rows with larger gradients are kept more often, and weigh
    /// one over their chance, so that `subsample` of the rows are kept on average.
    Mvs(f64),
}

/// What CatBoost's trees need beyond a [`Plan`].
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Symmetric {
    pub depth: usize,
    pub l2: f64,
    pub random_strength: f64,
    pub bootstrap: Bootstrap,
    pub one_hot_max_size: usize,
    /// The share of the features each level draws (CatBoost's `rsm`).
    pub rsm: f64,
    /// `nan_mode="Max"`: a missing value is above every value, else below.
    pub nan_max: bool,
    /// `score_function="Cosine"` (the default), else `"L2"`.
    pub cosine: bool,
}

/// What a column of the tree's own view of the data is.
#[derive(Debug, Clone)]
enum Kind {
    /// A number's buckets.
    Number,
    /// A category against the rest: one-hot.
    OneHot(Code),
    /// An ordered target statistic: its value on all the rows per category code, and for a
    /// missing or unknown category.
    Ctr { all: Vec<f64>, unknown: f64 },
}

/// One column a level can split: a bucket per row, ordered, so that a border `t` sends the
/// rows above `t` right.
struct Column {
    /// The feature of the data it comes from.
    feature: usize,
    order: Vec<Code>,
    levels: usize,
    kind: Kind,
}

/// The split of one level.
#[derive(Debug, Clone, Copy)]
struct Chosen {
    column: usize,
    border: Code,
    score: f64,
}

/// Σ `term(row)` over `n` rows, in fixed blocks added in order: the same on any number of
/// threads (a plain parallel sum is not).
fn fixed_sum(n: usize, term: impl Fn(usize) -> f64 + Sync) -> f64 {
    let blocks: Vec<f64> = (0..n.div_ceil(BLOCK))
        .into_par_iter()
        .map(|b| (b * BLOCK..((b + 1) * BLOCK).min(n)).map(&term).sum())
        .collect();
    blocks.iter().sum()
}

/// A uniform number in [0, 1).
fn uniform(rng: &mut Rng) -> f64 {
    (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64
}

/// A standard normal number (Box-Muller).
fn normal(rng: &mut Rng) -> f64 {
    let u: f64 = uniform(rng).max(f64::MIN_POSITIVE);
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * uniform(rng)).cos()
}

/// CatBoost's border between two neighbouring values: their midpoint, as a 32-bit float.
fn midpoint(below: f64, above: f64) -> f64 {
    (below / 2.0 + above / 2.0) as f32 as f64
}

/// The ordered target statistics of a categorical feature with codes `codes` (`missing` for
/// a missing value) and `categories` categories: for each 0/1 target in `goods` and each
/// prior, (good rows before + prior) / (rows before + 1), the rows in the order of
/// `permutation`; then the share of the rows in the category (CatBoost's `Counter`). Gives
/// each statistic's column and its value on all the rows per category.
fn target_statistics(
    feature: usize,
    codes: &[Code],
    categories: usize,
    goods: &[Vec<bool>],
    permutation: &[u32],
) -> Vec<Column> {
    let bucket = |v: f64| ((v * CTR_LEVELS as f64).ceil() as i64 - 1).clamp(0, CTR_LEVELS as i64 - 1) as Code;
    let mut columns: Vec<Column> = Vec::new();
    let mut counts: Vec<f64> = vec![0.0; categories + 1];
    codes.iter().for_each(|code| counts[*code as usize] += 1.0);
    for good in goods {
        let mut good_all: Vec<f64> = vec![0.0; categories + 1];
        codes.iter().zip(good).for_each(|(code, g)| good_all[*code as usize] += f64::from(u8::from(*g)));
        for prior in [0.0, 0.5, 1.0] {
            let mut order: Vec<Code> = vec![0; codes.len()];
            let (mut seen, mut seen_good): (Vec<f64>, Vec<f64>) = (vec![0.0; categories + 1], vec![0.0; categories + 1]);
            for row in permutation {
                let (row, code): (usize, usize) = (*row as usize, codes[*row as usize] as usize);
                order[row] = bucket((seen_good[code] + prior) / (seen[code] + 1.0));
                seen[code] += 1.0;
                seen_good[code] += f64::from(u8::from(good[row]));
            }
            let all: Vec<f64> = (0..categories).map(|c| (good_all[c] + prior) / (counts[c] + 1.0)).collect();
            columns.push(Column { feature, order, levels: CTR_LEVELS, kind: Kind::Ctr { all, unknown: prior } });
        }
    }
    let most: f64 = counts[..categories].iter().cloned().fold(1.0, f64::max);
    let all: Vec<f64> = counts[..categories].iter().map(|c| c / most).collect();
    let order: Vec<Code> = codes.iter().map(|code| bucket(*all.get(*code as usize).unwrap_or(&0.0))).collect();
    columns.push(Column { feature, order, levels: CTR_LEVELS, kind: Kind::Ctr { all, unknown: 0.0 } });
    columns
}

/// The columns the trees split: each number's buckets, and for each categorical feature its
/// one-hot columns or its target statistics.
fn columns(bins: &Bins, loss: &Loss, s: &Symmetric, seed: u64) -> Vec<Column> {
    let n: usize = bins_rows(bins);
    // The 0/1 targets of the statistics: class 1 of two, each class but the first of more,
    // or for a number, above the median (CatBoost's one border of the target).
    let goods: Vec<Vec<bool>> = match loss {
        Loss::Binary(y, _) => vec![y.iter().map(|c| *c == 1).collect()],
        Loss::Classes(y, k, _) => (1..*k as u32).map(|class| y.iter().map(|c| *c == class).collect()).collect(),
        _ => {
            let y: &[f64] = match loss {
                Loss::Squared(y) => y,
                _ => &[],
            };
            let mut sorted: Vec<f64> = y.to_vec();
            sorted.sort_unstable_by(f64::total_cmp);
            let median: f64 = sorted.get(sorted.len() / 2).copied().unwrap_or(0.0);
            vec![y.iter().map(|v| *v > median).collect()]
        }
    };
    let mut permutation: Vec<u32> = (0..n as u32).collect();
    Rng::new(seed ^ 0xC7B0_057A_7157_1C5D).shuffle(&mut permutation);
    let mut out: Vec<Column> = Vec::new();
    for f in 0..bins.names.len() {
        let (codes, edges, missing): (&Codes, &[(f64, f64)], bool) = bins.binned(f);
        let codes: &[Code] = &codes.wide();
        let values: usize = edges.len();
        if bins.categories(f).is_some() {
            if values <= s.one_hot_max_size {
                for c in 0..values as Code {
                    let order: Vec<Code> = codes.iter().map(|code| Code::from(*code == c)).collect();
                    out.push(Column { feature: f, order, levels: 2, kind: Kind::OneHot(c) });
                }
            } else {
                out.extend(target_statistics(f, codes, values, &goods, &permutation));
            }
            continue;
        }
        let missing_code: Code = values as Code;
        let order: Vec<Code> = match (missing, s.nan_max) {
            (false, _) | (true, true) => codes.to_vec(),
            (true, false) => codes.iter().map(|code| if *code == missing_code { 0 } else { code + 1 }).collect(),
        };
        out.push(Column { feature: f, order, levels: values + usize::from(missing), kind: Kind::Number });
    }
    out
}

/// The rows the buckets were made from.
fn bins_rows(bins: &Bins) -> usize {
    bins.binned(0).0.len()
}

/// The tree's node for split `chosen`: a threshold on a number, or a set of categories that
/// go left. A missing value goes where CatBoost sends it.
fn node_of(bins: &Bins, column: &Column, border: Code, s: &Symmetric, sets: &mut Vec<Left>) -> Node {
    let f: usize = column.feature;
    let t: usize = border as usize;
    let set_node = |left: Left, missing_left: bool, sets: &mut Vec<Left>| {
        sets.push(left);
        let split: u32 = (f as u32) << 1 | u32::from(missing_left) | CATEGORICAL;
        Node { threshold: (sets.len() - 1) as f64, split, right: 0 }
    };
    let mut left: Left = [0; 4];
    match &column.kind {
        Kind::Number => {
            let (_, edges, missing): (&Codes, &[(f64, f64)], bool) = bins.binned(f);
            let (threshold, missing_left): (f64, bool) = match (missing, s.nan_max) {
                // Below every value: border 0 parts the missing values from the rest.
                (true, false) if t == 0 => (f64::NEG_INFINITY, true),
                (true, false) => (midpoint(edges[t - 1].1, edges[t].0), true),
                (true, true) if t + 1 == edges.len() => (f64::INFINITY, false),
                (_, nan_max) => (midpoint(edges[t].1, edges[t + 1].0), !nan_max),
            };
            Node { threshold, split: (f as u32) << 1 | u32::from(missing_left), right: 0 }
        }
        Kind::OneHot(c) => {
            let categories: usize = bins.binned(f).1.len();
            (0..categories).filter(|code| *code != *c as usize).for_each(|code| left[code >> 6] |= 1 << (code & 63));
            set_node(left, true, sets)
        }
        Kind::Ctr { all, unknown } => {
            let cut: f64 = (t + 1) as f64 / CTR_LEVELS as f64;
            for (code, v) in all.iter().enumerate() {
                if *v <= cut {
                    left[code >> 6] |= 1 << (code & 63);
                }
            }
            set_node(left, *unknown <= cut, sets)
        }
    }
}

/// The score of parting each leaf's rows into the sums `parts` (per leaf and side: Σwg per
/// output, then Σw): CatBoost's `Cosine` (Σ avg·Σwg over √Σ avg²·Σw) or `L2` (Σ avg·Σwg),
/// with avg = Σwg / (Σw + λ).
fn score(parts: &[f64], k: usize, l2: f64, cosine: bool) -> f64 {
    let (mut num, mut den): (f64, f64) = (0.0, 0.0);
    for part in parts.chunks(k + 1) {
        let w: f64 = part[k];
        if w <= 0.0 {
            continue;
        }
        for g in &part[..k] {
            let avg: f64 = g / (w + l2);
            num += avg * g;
            den += avg * avg * w;
        }
    }
    match cosine {
        true if den > 0.0 => num / den.sqrt(),
        true => 0.0,
        false => num,
    }
}

/// One tree's view of the rows: each row's leaf so far, and its Σwg per output then its weight
/// (`k + 1` numbers a row).
struct Rows<'a> {
    leaf: &'a [u32],
    stats: &'a [f64],
    k: usize,
}

/// The histograms of `columns` over `rows`, filled in one pass: per leaf (`leaf[row] & mask`)
/// and bucket, Σwg per output and Σw. A row's leaf and sums are read once for every column.
/// Blocks of rows, one task each, added in block order.
fn histograms(columns: &[&Column], rows: &[u32], of: &Rows, mask: u32, leaves: usize) -> Vec<Vec<f64>> {
    let Rows { leaf, stats, k } = *of;
    let s: usize = k + 1;
    let widths: Vec<usize> = columns.iter().map(|c| c.levels * s).collect();
    let empty = || -> Vec<Vec<f64>> { widths.iter().map(|w| vec![0.0; leaves * w]).collect() };
    let blocks: Vec<Vec<Vec<f64>>> = rows
        .par_chunks(BLOCK)
        .map(|rows| {
            let mut hists: Vec<Vec<f64>> = empty();
            let mut cells: Vec<(&[Code], usize, &mut [f64])> = columns
                .iter()
                .zip(&widths)
                .zip(hists.iter_mut())
                .map(|((c, w), h)| (c.order.as_slice(), *w, h.as_mut_slice()))
                .collect();
            // One output (regression, two classes) is two numbers a row: a constant the
            // compiler unrolls.
            match s {
                2 => fill(&mut cells, rows, leaf, mask, stats, 2),
                _ => fill(&mut cells, rows, leaf, mask, stats, s),
            }
            hists
        })
        .collect();
    let mut sums: Vec<Vec<f64>> = empty();
    for block in blocks {
        for (sum, hist) in sums.iter_mut().zip(block) {
            sum.iter_mut().zip(hist).for_each(|(a, b)| *a += b);
        }
    }
    sums
}

/// Adds each row's `s` sums into the cell of its leaf and bucket in every column's histogram.
#[inline(always)]
fn fill(cells: &mut [(&[Code], usize, &mut [f64])], rows: &[u32], leaf: &[u32], mask: u32, stats: &[f64], s: usize) {
    for row in rows {
        let row: usize = *row as usize;
        let l: usize = (leaf[row] & mask) as usize;
        let add: &[f64] = &stats[row * s..row * s + s];
        for (order, width, hist) in cells.iter_mut() {
            let at: usize = l * *width + order[row] as usize * s;
            hist[at..at + s].iter_mut().zip(add).for_each(|(cell, v)| *cell += v);
        }
    }
}

/// The histogram of a level of `leaves` leaves from the parents' (`leaves / 2` leaves) and the
/// right children's (indexed by parent): a left child is its parent minus its right sibling.
fn children(parent: &[f64], right: &[f64], width: usize, leaves: usize) -> Vec<f64> {
    let half: usize = leaves / 2;
    let mut hist: Vec<f64> = vec![0.0; leaves * width];
    for p in 0..half {
        let (from, to) = (p * width, (p + 1) * width);
        for i in 0..width {
            hist[from + i] = parent[from + i] - right[from + i];
        }
        hist[(p + half) * width..(p + half + 1) * width].copy_from_slice(&right[from..to]);
    }
    hist
}

/// The sums of each leaf (Σwg per output, then Σw) from a column's histogram.
fn leaf_totals(hist: &[f64], column: &Column, k: usize, leaves: usize) -> Vec<f64> {
    let (s, width): (usize, usize) = (k + 1, column.levels * (k + 1));
    let mut totals: Vec<f64> = vec![0.0; leaves * s];
    for leaf in 0..leaves {
        for level in 0..column.levels {
            for i in 0..s {
                totals[leaf * s + i] += hist[leaf * width + level * s + i];
            }
        }
    }
    totals
}

/// The best border of column `c` for the leaves in `hist`, with its score: every border
/// parts every leaf into its buckets up to the border and the rest.
fn best_border(hist: &[f64], column: &Column, k: usize, leaves: usize, l2: f64, cosine: bool) -> Option<(Code, f64)> {
    let (s, width): (usize, usize) = (k + 1, column.levels * (k + 1));
    let totals: Vec<f64> = leaf_totals(hist, column, k, leaves);
    let mut below: Vec<f64> = vec![0.0; leaves * s];
    let mut parts: Vec<f64> = vec![0.0; leaves * 2 * s];
    let mut best: Option<(Code, f64)> = None;
    for border in 0..column.levels.saturating_sub(1) {
        for leaf in 0..leaves {
            let at: usize = leaf * width + border * s;
            for i in 0..s {
                below[leaf * s + i] += hist[at + i];
                parts[leaf * 2 * s + i] = below[leaf * s + i];
                parts[leaf * 2 * s + s + i] = totals[leaf * s + i] - below[leaf * s + i];
            }
        }
        // A border that leaves every leaf whole still counts, as in CatBoost: under `"L2"`
        // it can score above every real split.
        let v: f64 = score(&parts, k, l2, cosine);
        if best.is_none_or(|(_, b)| v > b) {
            best = Some((border as Code, v));
        }
    }
    best
}

/// Each row's weight for one tree: the caller's weight times the bootstrap's.
fn bootstrap(base: &[f64], g: &[f64], k: usize, how: Bootstrap, seed: u64) -> Vec<f64> {
    let n: usize = base.len();
    let draw = |f: &(dyn Fn(usize, &mut Rng) -> f64 + Sync)| -> Vec<f64> {
        (0..n.div_ceil(BLOCK))
            .into_par_iter()
            .flat_map_iter(|b| {
                let mut rng: Rng = Rng::new(mixed(seed, b as u64));
                (b * BLOCK..((b + 1) * BLOCK).min(n)).map(|row| f(row, &mut rng)).collect::<Vec<f64>>()
            })
            .collect()
    };
    match how {
        Bootstrap::No => base.to_vec(),
        Bootstrap::Bayesian(t) => draw(&|row, rng| base[row] * (-uniform(rng).max(f64::MIN_POSITIVE).ln()).powf(t)),
        Bootstrap::Bernoulli(share) => draw(&|row, rng| if uniform(rng) < share { base[row] } else { 0.0 }),
        Bootstrap::Mvs(share) => {
            // Each row's gradient size; the cut μ where Σ min(1, size / μ) is share × rows.
            let size: Vec<f64> = (0..n).map(|r| g[r * k..r * k + k].iter().map(|v| v * v).sum::<f64>().sqrt()).collect();
            let want: f64 = share * n as f64;
            let top: f64 = size.iter().cloned().fold(0.0, f64::max);
            if top <= 0.0 || fixed_sum(n, |r| size[r]) / top >= want {
                return base.to_vec(); // too few rows with a gradient: keep all
            }
            let mu: f64 = mvs_cut(size.clone(), want);
            draw(&|row, rng| {
                let p: f64 = (size[row] / mu).min(1.0);
                if p > 0.0 && uniform(rng) < p { base[row] / p } else { 0.0 }
            })
        }
    }
}

/// The μ where Σ min(1, size / μ) = `want`, exactly, in time linear in the rows (as CatBoost
/// finds it): halve the undecided sizes around their median; sizes at or above μ count 1, the
/// rest size / μ.
fn mvs_cut(mut v: Vec<f64>, want: f64) -> f64 {
    let n: usize = v.len();
    // v[..lo] are below μ (their sum `small`), v[hi..] at or above it.
    let (mut lo, mut hi, mut small): (usize, usize, f64) = (0, n, 0.0);
    while hi - lo > 1 {
        let mid: usize = (lo + hi) / 2;
        v[lo..hi].select_nth_unstable_by(mid - lo, f64::total_cmp);
        let p: f64 = v[mid];
        let below: f64 = small + v[lo..mid].iter().sum::<f64>();
        match p <= 0.0 || (n - mid) as f64 + below / p > want {
            true => (lo, small) = (mid, below), // μ above p
            false => hi = mid,                   // μ at or below p
        }
    }
    // One size may be left undecided: below μ if it fits under the μ it gives.
    let mu: f64 = (small + v[lo..hi].iter().sum::<f64>()) / (want - (n - hi) as f64);
    let mu: f64 = match v[lo..hi].first() {
        Some(last) if *last > mu => small / (want - (n - lo) as f64),
        _ => mu,
    };
    // Never above a size that counts 1 (a zero divisor when exactly `want` sizes do). Fewer
    // than `want` rows with a gradient: every one of them counts 1.
    let mu: f64 = v[hi..].iter().fold(mu, |m, s| m.min(*s));
    match mu > 0.0 {
        true => mu,
        false => v.iter().filter(|s| **s > 0.0).fold(f64::INFINITY, |m, s| m.min(*s)),
    }
}

/// Solves (a + l2 I) x = b for a small symmetric matrix a (k × k), by Cholesky.
fn solve(mut a: Vec<f64>, b: &[f64], l2: f64) -> Vec<f64> {
    let k: usize = b.len();
    (0..k).for_each(|i| a[i * k + i] += l2);
    for j in 0..k {
        let d: f64 = (a[j * k + j] - (0..j).map(|p| a[j * k + p] * a[j * k + p]).sum::<f64>()).max(f64::MIN_POSITIVE).sqrt();
        a[j * k + j] = d;
        for i in j + 1..k {
            a[i * k + j] = (a[i * k + j] - (0..j).map(|p| a[i * k + p] * a[j * k + p]).sum::<f64>()) / d;
        }
    }
    let mut y: Vec<f64> = vec![0.0; k];
    for i in 0..k {
        y[i] = (b[i] - (0..i).map(|p| a[i * k + p] * y[p]).sum::<f64>()) / a[i * k + i];
    }
    for i in (0..k).rev() {
        y[i] = (y[i] - (i + 1..k).map(|p| a[p * k + i] * y[p]).sum::<f64>()) / a[i * k + i];
    }
    y
}

/// The gradients (der1, toward a lower loss) per row and output at scores `f`, and for the
/// leaves the probabilities (classes).
fn gradients(loss: &Loss, f: &[f64], k: usize, g: &mut [f64], p: &mut [f64]) {
    g.par_chunks_mut(BLOCK * k).zip(p.par_chunks_mut(BLOCK * k)).enumerate().for_each(|(b, (g, p))| {
        for (i, (g, p)) in g.chunks_mut(k).zip(p.chunks_mut(k)).enumerate() {
            let row: usize = b * BLOCK + i;
            match loss {
                Loss::Squared(y) => g[0] = y[row] - f[row],
                Loss::Binary(y, _) => {
                    p[0] = 1.0 / (1.0 + (-f[row]).exp());
                    g[0] = f64::from(y[row]) - p[0];
                }
                Loss::Classes(y, _, _) => {
                    let z: &[f64] = &f[row * k..row * k + k];
                    let top: f64 = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let sum: f64 = z.iter().map(|v| (v - top).exp()).sum();
                    for c in 0..k {
                        p[c] = (z[c] - top).exp() / sum;
                        g[c] = f64::from(u8::from(y[row] as usize == c)) - p[c];
                    }
                }
                _ => unreachable!("CatBoost's losses are checked at fit"),
            }
        }
    });
}

/// A symmetric tree of `splits` (one per level, level 0 at the root) as a [`Tree`] of
/// `outputs` trees' worth of leaf values: the tree for output `c` holds `values[leaf * k + c]`.
fn oblivious(splits: &[Node], sets: &[Left], values: &[f64], counts: &[u32], k: usize, c: usize) -> Tree {
    let depth: usize = splits.len();
    let (mut nodes, mut value, mut samples): (Vec<Node>, Vec<f64>, Vec<u32>) = (Vec::new(), Vec::new(), Vec::new());
    // Depth first, left first: (level, the leaf bits so far, the parent waiting for its right
    // child's place).
    let mut stack: Vec<(usize, usize, Option<usize>)> = vec![(0, 0, None)];
    while let Some((level, at, parent)) = stack.pop() {
        if let Some(parent) = parent {
            nodes[parent].right = nodes.len() as u32;
        }
        if level == depth {
            nodes.push(Node { threshold: 0.0, split: LEAF, right: 0 });
            value.push(values[at * k + c]);
            samples.push(counts[at]);
            continue;
        }
        let me: usize = nodes.len();
        nodes.push(splits[level]);
        value.push(0.0);
        // The rows below: every leaf whose first `level` bits are `at`'s.
        samples.push((0..1usize << (depth - level)).map(|rest| counts[at | rest << level]).sum());
        stack.push((level + 1, at | 1 << level, Some(me)));
        stack.push((level + 1, at, None));
    }
    Tree::from_parts(nodes, value, samples).with_sets(sets.to_vec())
}

/// CatBoost's boosting (`boosting_type="Plain"`): `p.rounds` symmetric trees of depth
/// `s.depth` on the buckets. `weights` is a weight per row (`None`: each counts once).
pub(crate) fn boost_symmetric(bins: &Bins, loss: &Loss, p: &Plan, s: &Symmetric, weights: Option<&[f64]>) -> Boosted {
    THREAD_POOL.install(|| boost_on(bins, loss, p, s, weights))
}

fn boost_on(bins: &Bins, loss: &Loss, p: &Plan, s: &Symmetric, weights: Option<&[f64]>) -> Boosted {
    let n: usize = bins_rows(bins);
    let k: usize = loss.outputs();
    let features: usize = bins.names.len();
    // A class's weight (`scale_pos_weight`) weighs its rows, in the start too.
    let mut base: Vec<f64> = weights.map_or_else(|| vec![1.0; n], <[f64]>::to_vec);
    if let Loss::Binary(y, labels) = loss {
        base.iter_mut().zip(y.iter()).for_each(|(w, c)| *w *= labels[*c as usize]);
    }
    let all: Vec<u32> = (0..n as u32).collect();
    let baseline: Vec<f64> = match p.base_score {
        Some(score) => vec![score; k],
        None => loss.baseline(&all, Some(&base), Rules::Sklearn),
    };
    let cols: Vec<Column> = columns(bins, loss, s, p.seed);
    let base_total: f64 = fixed_sum(n, |r| base[r]);
    let mut f: Vec<f64> = (0..n * k).map(|i| baseline[i % k]).collect();
    let (mut g, mut probs): (Vec<f64>, Vec<f64>) = (vec![0.0; n * k], vec![0.0; n * k]);
    let mut boosted: Boosted = Boosted {
        trees: Vec::new(),
        baseline,
        train_score: Vec::new(),
        validation_score: Vec::new(),
        n_iter: 0,
        importances: vec![0.0; features],
    };
    let mut rng: Rng = Rng::new(mixed(p.seed, 0x5EED));
    for round in 0..p.rounds {
        gradients(loss, &f, k, &mut g, &mut probs);
        let w: Vec<f64> = bootstrap(&base, &g, k, s.bootstrap, mixed(p.seed, round as u64 + 1));
        let total: f64 = fixed_sum(n, |r| w[r]);
        // λ scaled by the mean weight: the bootstrap's for the scores, the rows' own for the
        // leaves.
        let l2: f64 = s.l2 * total / n as f64;
        let leaf_l2: f64 = s.l2 * base_total / n as f64;
        // CatBoost's noise on the scores: its size follows the gradients, and it fades as the
        // model grows, measured in trees times the learning rate, past ln(rows).
        let noise: f64 = match s.random_strength > 0.0 {
            true => {
                let spread: f64 = (fixed_sum(n, |r| g[r * k..r * k + k].iter().map(|v| v * v).sum::<f64>()) / n as f64).sqrt();
                let left: f64 = ((n as f64).ln() - round as f64 * p.step.rate).exp();
                s.random_strength * spread * left / (1.0 + left)
            }
            false => 0.0,
        };
        // Σwg per row, and the rows with a weight: the only ones the histograms read.
        let stats: Vec<f64> = w
            .par_iter()
            .enumerate()
            .flat_map_iter(|(r, w)| g[r * k..r * k + k].iter().map(move |v| w * v).chain([*w]))
            .collect();
        let active: Vec<u32> = (0..n as u32).into_par_iter().filter(|r| w[*r as usize] > 0.0).collect();
        let mut leaf: Vec<u32> = vec![0; n];
        let mut splits: Vec<Node> = Vec::new();
        let mut sets: Vec<Left> = Vec::new();
        // The last level's histogram of each column, for the next by subtraction.
        let mut hists: Vec<Option<Vec<f64>>> = vec![None; cols.len()];
        let mut right: Vec<u32> = Vec::new();
        for d in 0..s.depth {
            let leaves: usize = 1 << d;
            let pool: Vec<usize> = match s.rsm < 1.0 {
                true => {
                    let mut picked: Vec<usize> = (0..cols.len()).collect();
                    rng.shuffle(&mut picked);
                    picked.truncate(((s.rsm * cols.len() as f64).ceil() as usize).max(1));
                    picked.sort_unstable();
                    picked
                }
                false => (0..cols.len()).collect(),
            };
            let of: Rows = Rows { leaf: &leaf, stats: &stats, k };
            // Each column's histogram and its best border; columns in groups of `GROUP`, each
            // group's histograms in one pass over the rows: from all the rows, or from the right
            // children's rows and the last level's histograms.
            type Made = (Vec<f64>, Option<(Code, f64)>);
            let made: Vec<Made> = pool
                .par_chunks(GROUP)
                .flat_map_iter(|group| {
                    let (old, new): (Vec<usize>, Vec<usize>) = group.iter().partition(|c| d > 0 && hists[**c].is_some());
                    let refs = |cs: &[usize]| -> Vec<&Column> { cs.iter().map(|c| &cols[*c]).collect() };
                    let mut hist: Vec<(usize, Vec<f64>)> = Vec::new();
                    if !new.is_empty() {
                        hist.extend(new.iter().copied().zip(histograms(&refs(&new), &active, &of, u32::MAX, leaves)));
                    }
                    if !old.is_empty() {
                        let half: u32 = (leaves / 2) as u32;
                        let of_right: Vec<Vec<f64>> = histograms(&refs(&old), &right, &of, half - 1, leaves / 2);
                        hist.extend(old.iter().zip(of_right).map(|(c, r)| {
                            let parent: &[f64] = hists[*c].as_deref().unwrap_or_default();
                            (*c, children(parent, &r, cols[*c].levels * (k + 1), leaves))
                        }));
                    }
                    hist.sort_unstable_by_key(|(c, _)| *c);
                    hist.into_iter().map(|(c, hist)| {
                        let best: Option<(Code, f64)> = best_border(&hist, &cols[c], k, leaves, l2, s.cosine);
                        (hist, best)
                    })
                })
                .collect();
            hists.iter_mut().for_each(|h| *h = None);
            let mut chosen: Option<(Chosen, f64)> = None;
            for (c, (hist, found)) in pool.iter().zip(made) {
                if let Some((border, raw)) = found {
                    let v: f64 = if noise > 0.0 { raw + noise * normal(&mut rng) } else { raw };
                    if chosen.is_none_or(|(best, _)| v > best.score) {
                        chosen = Some((Chosen { column: *c, border, score: v }, raw));
                    }
                }
                hists[*c] = Some(hist);
            }
            let Some((chosen, raw)) = chosen else { break };
            let column: &Column = &cols[chosen.column];
            // The gain over the leaves as they were, for the importances.
            let totals: Vec<f64> = leaf_totals(hists[chosen.column].as_deref().unwrap_or_default(), column, k, leaves);
            boosted.importances[column.feature] += (raw - score(&totals, k, l2, s.cosine)).max(0.0);
            let border: Code = chosen.border;
            leaf.par_iter_mut().zip(column.order.par_iter()).for_each(|(l, o)| *l |= u32::from(*o > border) << d);
            right = active.par_iter().copied().filter(|r| column.order[*r as usize] > border).collect();
            splits.push(node_of(bins, column, border, s, &mut sets));
        }
        // The leaves: one Newton step each, on every row at its own weight (the bootstrap only
        // chooses the splits, as in CatBoost), from blocks of rows added in order.
        let leaves: usize = 1 << splits.len();
        let all_rows: Vec<u32> = (0..n as u32).collect();
        let sums: Vec<(Vec<f64>, Vec<f64>)> = all_rows
            .par_chunks(BLOCK)
            .map(|rows| {
                let (mut sum_g, mut sum_h): (Vec<f64>, Vec<f64>) = (vec![0.0; leaves * k], vec![0.0; leaves * k * k]);
                for row in rows {
                    let (row, at): (usize, usize) = (*row as usize, leaf[*row as usize] as usize);
                    let weight: f64 = base[row];
                    if weight == 0.0 {
                        continue;
                    }
                    for c in 0..k {
                        sum_g[at * k + c] += weight * g[row * k + c];
                    }
                    match loss {
                        Loss::Squared(_) => sum_h[at] += weight,
                        Loss::Binary(..) => sum_h[at] += weight * probs[row] * (1.0 - probs[row]),
                        _ => {
                            let pr: &[f64] = &probs[row * k..row * k + k];
                            for a in 0..k {
                                for b in 0..k {
                                    let d: f64 = if a == b { pr[a] } else { 0.0 };
                                    sum_h[at * k * k + a * k + b] += weight * (d - pr[a] * pr[b]);
                                }
                            }
                        }
                    }
                }
                (sum_g, sum_h)
            })
            .collect();
        let (mut sum_g, mut sum_h): (Vec<f64>, Vec<f64>) = (vec![0.0; leaves * k], vec![0.0; leaves * k * k]);
        for (bg, bh) in sums {
            sum_g.iter_mut().zip(bg).for_each(|(s, b)| *s += b);
            sum_h.iter_mut().zip(bh).for_each(|(s, b)| *s += b);
        }
        let mut counts: Vec<u32> = vec![0; leaves];
        leaf.iter().for_each(|l| counts[*l as usize] += 1);
        let mut values: Vec<f64> = vec![0.0; leaves * k];
        for at in 0..leaves {
            let step: Vec<f64> = match k {
                1 if sum_h[at] > 0.0 || sum_g[at] != 0.0 => vec![sum_g[at] / (sum_h[at] + leaf_l2)],
                1 => vec![0.0],
                _ => solve(sum_h[at * k * k..(at + 1) * k * k].to_vec(), &sum_g[at * k..(at + 1) * k], leaf_l2),
            };
            for c in 0..k {
                values[at * k + c] = p.step.rate * step[c];
            }
        }
        f.par_chunks_mut(k).zip(leaf.par_iter()).for_each(|(f, l)| {
            for c in 0..k {
                f[c] += values[*l as usize * k + c];
            }
        });
        for c in 0..k {
            boosted.trees.push(oblivious(&splits, &sets, &values, &counts, k, c));
        }
        boosted.n_iter += 1;
    }
    let sum: f64 = boosted.importances.iter().sum();
    if sum > 0.0 {
        boosted.importances.iter_mut().for_each(|share| *share /= sum);
    }
    boosted
}

/// CatBoost's automatic learning rate for `rows` rows and `iterations` rounds: exp(a ln rows +
/// b ln iterations + c), at most 0.5, rounded to 6 digits as CatBoost rounds it. The constants
/// were fitted to CatBoost 1.2's choices (`analysis/2026-11-20/catboost/lr.py`), which they
/// reproduce on every probe.
fn auto_learning_rate(loss: &str, rows: usize, iterations: usize) -> f64 {
    let (a, b, c): (f64, f64, f64) = match loss {
        "Logloss" => (0.427, -0.917, -1.1905895),
        "MultiClass" => (0.051, -0.845, 2.948052),
        _ => (0.158, -0.813, 1.329005),
    };
    let lr: f64 = (a * (rows as f64).ln() + b * (iterations as f64).ln() + c).exp().min(0.5);
    ((lr * 1e6).round() / 1e6) as f32 as f64
}

/// CatBoost's settings, with its names and defaults (catboost 1.2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CatBoostParams {
    /// `None`: `"RMSE"`, `"Logloss"` or `"MultiClass"`, by the model and the classes.
    loss_function: Option<String>,
    iterations: usize,
    /// `None`: CatBoost's automatic rate.
    learning_rate: Option<f64>,
    depth: usize,
    l2_leaf_reg: f64,
    border_count: usize,
    random_strength: f64,
    /// `None`: `"MVS"`, or `"Bayesian"` for more than two classes.
    bootstrap_type: Option<String>,
    bagging_temperature: f64,
    subsample: f64,
    random_seed: u64,
    one_hot_max_size: usize,
    /// `None`: `true` for `"RMSE"`, `false` for the classifiers.
    boost_from_average: Option<bool>,
    rsm: f64,
    nan_mode: String,
    score_function: String,
    scale_pos_weight: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    class_weights: Option<ClassWeight>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    cat_features: Vec<String>,
}

impl Default for CatBoostParams {
    fn default() -> Self {
        Self {
            loss_function: None,
            iterations: 1000,
            learning_rate: None,
            depth: 6,
            l2_leaf_reg: 3.0,
            border_count: 254,
            random_strength: 1.0,
            bootstrap_type: None,
            bagging_temperature: 1.0,
            subsample: 0.8,
            random_seed: 0,
            one_hot_max_size: 2,
            boost_from_average: None,
            rsm: 1.0,
            nan_mode: "Min".into(),
            score_function: "Cosine".into(),
            scale_pos_weight: 1.0,
            class_weights: None,
            cat_features: Vec::new(),
        }
    }
}

const REGRESSION: &str = "\"RMSE\"";
const CLASSIFICATION: &str = "\"Logloss\" (two classes) or \"MultiClass\"";

impl CatBoostParams {
    /// The loss's name, for `classes` classes (0 for a regressor).
    fn loss(&self, classes: usize) -> &str {
        match (self.loss_function.as_deref(), classes) {
            (Some(name), _) => name,
            (None, 0) => "RMSE",
            (None, 2) => "Logloss",
            (None, _) => "MultiClass",
        }
    }

    fn bootstrap(&self, classes: usize) -> Bootstrap {
        match self.bootstrap_type.as_deref() {
            Some("No") => Bootstrap::No,
            Some("Bayesian") => Bootstrap::Bayesian(self.bagging_temperature),
            Some("Bernoulli") => Bootstrap::Bernoulli(self.subsample),
            Some(_) => Bootstrap::Mvs(self.subsample),
            None if classes > 2 => Bootstrap::Bayesian(self.bagging_temperature),
            None => Bootstrap::Mvs(self.subsample),
        }
    }

    /// The plan for `rows` rows and `classes` classes (0 for a regressor).
    fn plan_for(&self, rows: usize, classes: usize) -> Plan {
        let loss: &str = self.loss(classes);
        // CatBoost keeps the rate as a 32-bit float.
        let rate: f64 = self.learning_rate.map_or_else(|| auto_learning_rate(loss, rows, self.iterations), |r| r as f32 as f64);
        let from_average: bool = self.boost_from_average.unwrap_or(loss == "RMSE");
        Plan {
            rounds: self.iterations,
            step: Step::new(self.l2_leaf_reg, rate),
            max_depth: Some(self.depth),
            max_leaf_nodes: None,
            min_samples_leaf: 1,
            max_features: 1.0,
            by_tree: 1.0,
            by_level: 1.0,
            by_node: 1.0,
            subsample: 1.0,
            subsample_freq: 0,
            rules: Rules::Sklearn,
            seed: self.random_seed,
            early_stopping: false,
            validation_fraction: 0.1,
            n_iter_no_change: 10,
            tol: 0.0,
            base_score: (!from_average).then_some(0.0),
            symmetric: Some(Symmetric {
                depth: self.depth,
                l2: self.l2_leaf_reg,
                random_strength: self.random_strength,
                bootstrap: self.bootstrap(classes),
                one_hot_max_size: self.one_hot_max_size,
                rsm: self.rsm,
                nan_max: self.nan_mode == "Max",
                cosine: self.score_function == "Cosine",
            }),
        }
    }
}

impl Library for CatBoostParams {
    fn plan(&self, rows: usize) -> Plan {
        self.plan_for(rows, 0)
    }

    fn classifier_plan(&self, rows: usize, classes: usize) -> Plan {
        self.plan_for(rows, classes)
    }

    fn max_bins(&self) -> usize {
        self.border_count + 1
    }

    fn categorical(&self, dataset: &Dataset, target: &str) -> Result<Vec<String>, ModelError> {
        let names: Option<&[String]> = (!self.cat_features.is_empty()).then_some(&self.cat_features[..]);
        categorical_columns(dataset, target, names)
    }

    fn regression<'a>(&self, y: &'a [f64]) -> Result<Loss<'a>, ModelError> {
        match self.loss(0) {
            "RMSE" => Ok(Loss::Squared(y)),
            other => Err(ModelError::InvalidSetting { setting: "loss_function", value: other.into(), expected: REGRESSION }),
        }
    }

    fn classification<'a>(&self, y: &'a [u32], classes: usize) -> Result<Loss<'a>, ModelError> {
        match (self.loss(classes), classes) {
            ("Logloss", 2) => Ok(Loss::Binary(y, [1.0, self.scale_pos_weight])),
            ("MultiClass", k) if k > 2 => Ok(Loss::Classes(y, k, 1.0)),
            (other, _) => Err(ModelError::InvalidSetting { setting: "loss_function", value: other.into(), expected: CLASSIFICATION }),
        }
    }

    fn class_weight(&self) -> Option<&ClassWeight> {
        self.class_weights.as_ref()
    }

    fn log_link(&self) -> bool {
        false
    }
}

macro_rules! catboost_settings {
    ($model:ident) => {
        impl $model {
            /// The rounds of trees. Default 1000. `n_estimators` is the same.
            #[track_caller]
            pub fn iterations(mut self, iterations: usize) -> Self {
                check(iterations > 0, "iterations", iterations, "1 or more");
                self.params.iterations = iterations;
                self
            }

            /// [`Self::iterations`], by scikit-learn's name.
            #[track_caller]
            pub fn n_estimators(self, n_estimators: usize) -> Self {
                self.iterations(n_estimators)
            }

            /// How much of each tree's step is taken, above 0. Default: CatBoost's choice
            /// from the rows and the rounds (`learning_rate_` after `fit`).
            #[track_caller]
            pub fn learning_rate(mut self, learning_rate: f64) -> Self {
                above_zero("learning_rate", learning_rate);
                self.params.learning_rate = Some(learning_rate);
                self
            }

            /// The splits of every tree, 1 to 16: a tree has 2^depth leaves. Default 6.
            #[track_caller]
            pub fn depth(mut self, depth: usize) -> Self {
                check((1..=16).contains(&depth), "depth", depth, "1 to 16");
                self.params.depth = depth;
                self
            }

            /// [`Self::depth`], by scikit-learn's name.
            #[track_caller]
            pub fn max_depth(self, max_depth: usize) -> Self {
                self.depth(max_depth)
            }

            /// The L2 penalty on the leaf values, added to each leaf's hessian. Default 3.
            #[track_caller]
            pub fn l2_leaf_reg(mut self, l2_leaf_reg: f64) -> Self {
                at_least_zero("l2_leaf_reg", l2_leaf_reg);
                self.params.l2_leaf_reg = l2_leaf_reg;
                self
            }

            /// [`Self::l2_leaf_reg`], by XGBoost's name.
            #[track_caller]
            pub fn reg_lambda(self, reg_lambda: f64) -> Self {
                self.l2_leaf_reg(reg_lambda)
            }

            /// The most borders of a number, 1 to 65,534. Default 254. A feature with more
            /// distinct values gets quantile buckets (CatBoost's are `GreedyLogSum`).
            #[track_caller]
            pub fn border_count(mut self, border_count: usize) -> Self {
                check((1..=65_534).contains(&border_count), "border_count", border_count, "1 to 65,534");
                self.params.border_count = border_count;
                self
            }

            /// How much random noise is added to the split scores, 0 for none. Default 1.
            /// The noise follows the gradients' size and fades as the model grows.
            #[track_caller]
            pub fn random_strength(mut self, random_strength: f64) -> Self {
                at_least_zero("random_strength", random_strength);
                self.params.random_strength = random_strength;
                self
            }

            /// How each tree weighs the rows: `"MVS"` (the default for a regressor and two
            /// classes: rows with larger gradients kept more often), `"Bayesian"` (the
            /// default for more classes: each row weighted (−ln u)^`bagging_temperature`),
            /// `"Bernoulli"` (each row kept with chance `subsample`) or `"No"`.
            #[track_caller]
            pub fn bootstrap_type(mut self, bootstrap_type: &str) -> Self {
                let ok: bool = matches!(bootstrap_type, "MVS" | "Bayesian" | "Bernoulli" | "No");
                check(ok, "bootstrap_type", bootstrap_type, "\"MVS\", \"Bayesian\", \"Bernoulli\" or \"No\"");
                self.params.bootstrap_type = Some(bootstrap_type.into());
                self
            }

            /// The power of `"Bayesian"`'s weights: 0 weighs every row once. Default 1.
            #[track_caller]
            pub fn bagging_temperature(mut self, bagging_temperature: f64) -> Self {
                at_least_zero("bagging_temperature", bagging_temperature);
                self.params.bagging_temperature = bagging_temperature;
                self
            }

            /// The share of the rows `"MVS"` and `"Bernoulli"` keep. Default 0.8.
            #[track_caller]
            pub fn subsample(mut self, subsample: f64) -> Self {
                share("subsample", subsample);
                self.params.subsample = subsample;
                self
            }

            /// The seed of the bootstrap, the noise and the order of the target statistics.
            /// Default 0, CatBoost's.
            pub fn random_seed(mut self, random_seed: u64) -> Self {
                self.params.random_seed = random_seed;
                self
            }

            /// [`Self::random_seed`], by scikit-learn's name.
            pub fn random_state(self, random_state: u64) -> Self {
                self.random_seed(random_state)
            }

            /// A text column of at most this many categories is split one category against
            /// the rest; one of more becomes target statistics. Default 2.
            pub fn one_hot_max_size(mut self, one_hot_max_size: usize) -> Self {
                self.params.one_hot_max_size = one_hot_max_size;
                self
            }

            /// The columns to treat as categories. Default: every text column.
            pub fn cat_features(mut self, cat_features: &[&str]) -> Self {
                self.params.cat_features = cat_features.iter().map(|name| name.to_string()).collect();
                self
            }

            /// `true` starts every prediction from the constant with the lowest loss, `false`
            /// from 0. Default: `true` for `"RMSE"`, `false` for the classifiers.
            pub fn boost_from_average(mut self, boost_from_average: bool) -> Self {
                self.params.boost_from_average = Some(boost_from_average);
                self
            }

            /// The share of the features each level of a tree draws. Default 1.
            #[track_caller]
            pub fn rsm(mut self, rsm: f64) -> Self {
                share("rsm", rsm);
                self.params.rsm = rsm;
                self
            }

            /// [`Self::rsm`], by XGBoost's and LightGBM's name.
            #[track_caller]
            pub fn colsample_bylevel(self, colsample_bylevel: f64) -> Self {
                self.rsm(colsample_bylevel)
            }

            /// Where a missing number goes: `"Min"` (the default: below every value) or
            /// `"Max"` (above every value).
            #[track_caller]
            pub fn nan_mode(mut self, nan_mode: &str) -> Self {
                check(matches!(nan_mode, "Min" | "Max"), "nan_mode", nan_mode, "\"Min\" or \"Max\"");
                self.params.nan_mode = nan_mode.into();
                self
            }

            /// How a split is scored: `"Cosine"` (the default) or `"L2"`.
            #[track_caller]
            pub fn score_function(mut self, score_function: &str) -> Self {
                let ok: bool = matches!(score_function, "Cosine" | "L2");
                check(ok, "score_function", score_function, "\"Cosine\" or \"L2\"");
                self.params.score_function = score_function.into();
                self
            }

            /// The learning rate of the fit: the one set, or CatBoost's choice for the rows
            /// and rounds `fit` saw. 0 before `fit`.
            pub fn learning_rate_(&self) -> f64 {
                let rows: usize = self.estimators_.first().map_or(0, |tree| tree.n_node_samples()[0] as usize);
                match (self.params.learning_rate, rows) {
                    (Some(rate), _) => rate,
                    (None, 0) => 0.0,
                    (None, rows) => auto_learning_rate(self.params.loss(self.class_count()), rows, self.params.iterations),
                }
            }
        }
    };
}

boosted_model! {
    regressor
    /// Gradient boosting with CatBoost's names, defaults and trees: `catboost.CatBoostRegressor`.
    ///
    /// 1000 symmetric trees of depth 6 on the squared error (`"RMSE"`), from the mean, with
    /// CatBoost's automatic learning rate. Each tree weighs the rows by minimal variance
    /// sampling (`"MVS"`, 0.8 of them) and adds noise to the split scores
    /// (`random_strength`). With `random_strength(0.0)` and `bootstrap_type("No")`, and at
    /// most 254 distinct values per feature, the trees are CatBoost's, split for split, and
    /// the predictions agree to 1e-6. The bootstrap and the noise draw from rusmlai's own
    /// generator, so with them the trees differ from CatBoost's as two seeds of CatBoost do.
    ///
    /// Text columns (or `cat_features`) become CatBoost's ordered target statistics
    /// (`Borders` with priors 0, 0.5 and 1 on the target above its median, and `Counter`),
    /// on one seeded order of the rows, or one-hot splits up to `one_hot_max_size`
    /// categories. A split on a statistic is stored as a split on a set of categories, so a
    /// column holds at most 255 categories.
    ///
    /// CatBoost settings with no setter: `thread_count`, `verbose`, `logging_level`,
    /// `train_dir`, `allow_writing_files`, `task_type`, `used_ram_limit` and the other
    /// system settings. Not yet: losses other than `"RMSE"`, `grow_policy` (`"Depthwise"`,
    /// `"Lossguide"`), `boosting_type="Ordered"`, combinations of categories
    /// (`max_ctr_complexity` is 1), `GreedyLogSum` borders for a feature with more than 254
    /// values, early stopping (it needs an `eval_set`), `leaf_estimation_iterations` above 1,
    /// `monotone_constraints`, text and embedding features.
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
    /// let mut model = CatBoostRegressor::new().iterations(200).depth(2);
    /// model.fit(&ds, "price");
    /// assert_eq!(model.n_iter_, 200);
    /// assert!(model.score(&ds, "price") > 0.95);
    ///
    /// // CatBoost's trees, split for split: no noise and no bootstrap.
    /// let mut exact = CatBoostRegressor::new()
    ///     .iterations(50)
    ///     .learning_rate(0.3)
    ///     .random_strength(0.0)
    ///     .bootstrap_type("No");
    /// exact.fit(&ds, "price");
    /// assert_eq!(exact.learning_rate_(), 0.3);
    /// assert!(exact.score(&ds, "price") > 0.99);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// from catboost import CatBoostRegressor
    /// ds = pd.DataFrame({
    ///     "area": [50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0],
    ///     "rooms": [1, 2, 2, 3, 3, 3, 4, 4],
    ///     "price": [150.0, 180.0, 200.0, 260.0, 270.0, 310.0, 400.0, 420.0],
    /// })
    /// x, y = ds[["area", "rooms"]], ds["price"]
    /// model = CatBoostRegressor(iterations=200, depth=2, verbose=0, allow_writing_files=False).fit(x, y)
    /// assert model.tree_count_ == 200
    /// assert model.score(x, y) > 0.95
    ///
    /// exact = CatBoostRegressor(iterations=50, learning_rate=0.3, random_strength=0, bootstrap_type="No",
    ///                           verbose=0, allow_writing_files=False).fit(x, y)
    /// assert abs(exact.learning_rate_ - 0.3) < 1e-7
    /// assert exact.score(x, y) > 0.99
    /// ```
    pub struct CatBoostRegressor(CatBoostParams = CatBoostParams::default());
}

catboost_settings!(CatBoostRegressor);

impl CatBoostRegressor {
    fn class_count(&self) -> usize {
        0
    }

    /// 1000 symmetric trees of depth 6 on the squared error, with CatBoost's learning rate.
    pub fn new() -> Self {
        Self::default()
    }

    /// What the trees lower: `"RMSE"`, the only loss so far.
    #[track_caller]
    pub fn loss_function(mut self, loss_function: &str) -> Self {
        check(loss_function == "RMSE", "loss_function", loss_function, REGRESSION);
        self.params.loss_function = Some(loss_function.into());
        self
    }
}

boosted_model! {
    classifier
    /// Gradient boosting with CatBoost's names, defaults and trees: `catboost.CatBoostClassifier`.
    ///
    /// Two classes boost the log loss (`"Logloss"`, from 0, rows weighed by `"MVS"`); more
    /// classes boost the softmax (`"MultiClass"`, rows weighed by `"Bayesian"`): one
    /// symmetric tree per round whose leaves hold a value per class, each leaf one Newton
    /// step with the full hessian of the softmax, as CatBoost's. The trees are
    /// [`CatBoostRegressor`]'s; `estimators_` holds one tree per class and round, the
    /// classes' trees of a round alike but for their values. `class_weights` weighs each
    /// row by its class (`"Balanced"` is CatBoost's `auto_class_weights`), and
    /// `scale_pos_weight` the rows of the second of two classes.
    ///
    /// CatBoost settings with no setter: those of [`CatBoostRegressor`]. Not yet: the same,
    /// and the losses other than `"Logloss"` and `"MultiClass"`.
    ///
    /// ```
    /// use rusmlai::prelude::*;
    /// let ds = Dataset::new(
    ///     df!(
    ///         "hours" => [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5],
    ///         "passed" => [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    ///     )
    ///     .unwrap(),
    ///     "",
    /// );
    /// let mut model = CatBoostClassifier::new().iterations(100);
    /// model.fit(&ds, "passed");
    /// assert_eq!(model.classes_, ["0", "1"]);
    /// assert_eq!(model.score(&ds, "passed"), 1.0);
    /// // 16 rows: CatBoost's learning rate is small (0.0146), so 100 rounds stay cautious.
    /// assert!(model.predict_proba(&ds).to_list("1")[15] > 0.6);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// from catboost import CatBoostClassifier
    /// ds = pd.DataFrame({
    ///     "hours": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5],
    ///     "passed": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    /// })
    /// x, y = ds[["hours"]], ds["passed"]
    /// model = CatBoostClassifier(iterations=100, verbose=0, allow_writing_files=False).fit(x, y)
    /// assert model.classes_.tolist() == [0, 1]
    /// assert model.score(x, y) == 1.0
    /// assert model.predict_proba(x)[15, 1] > 0.6
    /// ```
    pub struct CatBoostClassifier(CatBoostParams = CatBoostParams::default());
}

catboost_settings!(CatBoostClassifier);

impl CatBoostClassifier {
    fn class_count(&self) -> usize {
        self.classes_.len()
    }

    /// 1000 symmetric trees of depth 6 on the log loss, with CatBoost's learning rate.
    pub fn new() -> Self {
        Self::default()
    }

    /// `"Logloss"` (two classes) or `"MultiClass"`. Default: by the classes.
    #[track_caller]
    pub fn loss_function(mut self, loss_function: &str) -> Self {
        let ok: bool = matches!(loss_function, "Logloss" | "MultiClass");
        check(ok, "loss_function", loss_function, CLASSIFICATION);
        self.params.loss_function = Some(loss_function.into());
        self
    }

    /// How much each class counts: your own weights, `[("yes", 5.0), ("no", 1.0)]`.
    #[track_caller]
    pub fn class_weights(mut self, class_weights: impl Into<ClassWeight>) -> Self {
        self.params.class_weights = Some(class_weights.into().checked(false));
        self
    }

    /// `"Balanced"`: every class counts as much in total (scikit-learn's `"balanced"`).
    #[track_caller]
    pub fn auto_class_weights(mut self, auto_class_weights: &str) -> Self {
        check(auto_class_weights == "Balanced", "auto_class_weights", auto_class_weights, "\"Balanced\"");
        self.params.class_weights = Some(ClassWeight::from("balanced").checked(false));
        self
    }

    /// How much more a row of the second of two classes counts, above 0. Default 1.
    #[track_caller]
    pub fn scale_pos_weight(mut self, scale_pos_weight: f64) -> Self {
        above_zero("scale_pos_weight", scale_pos_weight);
        self.params.scale_pos_weight = scale_pos_weight;
        self
    }
}

/// `save_model` / `try_save_model` on a CatBoost-named model.
macro_rules! catboost_export {
    ($model:ident, $columns:literal, $target:literal) => {
        impl $model {
            /// Writes the fitted model as CatBoost's own JSON model, which Python reads with
            /// `CatBoostRegressor().load_model(path, format="json")` (or `CatBoostClassifier`)
            /// and which predicts what this model predicts. (catboost:
            /// `model.save_model("model.json", format="json")`) Not yet: a model with
            /// categorical features. Panics with the error's message at your line.
            ///
            /// ```
            /// use rusmlai::prelude::*;
            #[doc = concat!("let ds = read_csv(\"tests/fixtures/data/students.csv\").select(&", $columns, ").dropna(&[]);")]
            #[doc = concat!("let mut model = ", stringify!($model), "::new().iterations(5);")]
            #[doc = concat!("model.fit(&ds, \"", $target, "\");")]
            #[doc = concat!("let path = std::env::temp_dir().join(\"rusmlai-doc-", stringify!($model), ".json\");")]
            /// model.save_model(&path);
            /// assert!(std::fs::read_to_string(&path).unwrap().contains("oblivious_trees"));
            /// ```
            ///
            /// Edge case: a model that was never fitted has nothing to write:
            /// ```
            /// use rusmlai::prelude::*;
            #[doc = concat!("let model = ", stringify!($model), "::new();")]
            /// let path = std::env::temp_dir().join("rusmlai-doc-unfitted");
            /// assert!(matches!(model.try_save_model(&path), Err(ModelError::NotFitted { .. })));
            /// ```
            ///
            /// Python:
            ///
            /// ```python
            /// import tempfile
            /// import pandas as pd
            /// from catboost import CatBoostClassifier, CatBoostRegressor
            /// ds = pd.read_csv("tests/fixtures/data/students.csv")[["age", "cgpa", "passed"]].dropna()
            /// kw = dict(iterations=5, verbose=0, allow_writing_files=False)
            /// for model, y in [(CatBoostRegressor(**kw), "cgpa"), (CatBoostClassifier(**kw), "passed")]:
            ///     x = ds.drop(columns=[y] if y == "passed" else [y, "passed"])
            ///     model.fit(x, ds[y].astype(int) if y == "passed" else ds[y])
            ///     path = tempfile.mkdtemp() + "/model.json"
            ///     model.save_model(path, format="json")
            ///     assert "oblivious_trees" in open(path).read()
            /// ```
            #[track_caller]
            pub fn save_model(&self, path: impl AsRef<std::path::Path>) {
                crate::error::OrFatal::or_fatal(self.try_save_model(path))
            }

            /// [`Self::save_model`] that returns the error.
            pub fn try_save_model(&self, path: impl AsRef<std::path::Path>) -> Result<(), ModelError> {
                if self.target.is_none() {
                    return Err(ModelError::NotFitted { model: stringify!($model) });
                }
                let classes: &[String] = self.export_classes();
                let model = crate::models::export::CatBoostModel {
                    trees: &self.estimators_,
                    baseline: &self.baseline,
                    names: &self.feature_names_in_,
                    loss: self.params.loss(classes.len()),
                    classes,
                    nan_max: self.params.nan_mode == "Max",
                };
                crate::models::export::catboost_json(&model, path.as_ref())
            }
        }
    };
}

impl CatBoostRegressor {
    fn export_classes(&self) -> &[String] {
        &[]
    }
}

impl CatBoostClassifier {
    fn export_classes(&self) -> &[String] {
        &self.classes_
    }
}

catboost_export!(CatBoostRegressor, r#"["age", "cgpa"]"#, "cgpa");
catboost_export!(CatBoostClassifier, r#"["age", "cgpa", "passed"]"#, "passed");

#[cfg(test)]
mod tests {
    use polars::prelude::{DataFrame, IntoColumn};

    use super::*;
    use crate::models::xgboost::tests::{Want, assert_like, data, features};

    /// Every 7th `a` missing, and `side`: "left" in every 3rd row, else "right".
    fn extended(ds: &Dataset) -> (Dataset, Dataset) {
        let mut a: Vec<Option<f64>> = ds.to_list("a").into_iter().map(Some).collect();
        a.iter_mut().step_by(7).for_each(|v| *v = None);
        let side: Vec<&str> = (0..ds.height()).map(|i| if i % 3 == 0 { "left" } else { "right" }).collect();
        let mut nan: DataFrame = ds.dataframe().clone();
        nan.with_column(Series::new("a".into(), a).into_column()).unwrap();
        let mut sided: DataFrame = ds.dataframe().clone();
        sided.with_column(Series::new("side".into(), side).into_column()).unwrap();
        (Dataset::new(nan, ""), Dataset::new(sided.drop("c").unwrap(), ""))
    }


    /// Against CatBoost's, from `analysis/2026-11-20/catboost/parity.py`.
    #[test]
    fn regressors_match_catboost() {
        let ds: Dataset = data();
        let (nan, sided): (Dataset, Dataset) = extended(&ds);
        let r = || CatBoostRegressor::new().iterations(20).random_strength(0.0).bootstrap_type("No");
        let cases: [(&str, CatBoostRegressor, &Dataset, Want); 7] = [
            ("rmse", r().learning_rate(0.3), &ds, (0, 16.237824742516842, [1.5643377630986457, 6.988282346103343, 21.434259408821895, 2.2793158962483577])),
            ("shallow", r().learning_rate(0.5).depth(3).l2_leaf_reg(1.0), &ds, (0, 16.234549047380494, [1.129271096355465, 7.609426690376068, 21.591934786975944, 1.6725759475446313])),
            ("l2 score", r().learning_rate(0.3).score_function("L2"), &ds, (0, 16.231724208891148, [1.6501305935134134, 7.043612195546302, 21.313333397837372, 2.0835460610402645])),
            ("auto rate", r(), &ds, (0, 16.231983969972326, [0.8080542977157297, 6.737860735802009, 21.730517298845133, 2.1450519410453825])),
            ("nan min", r().learning_rate(0.3), &nan, (0, 16.229062517656303, [11.811531838834316, 6.430475099726932, 21.048432653974707, 1.7975081922685607])),
            ("nan max", r().learning_rate(0.3).nan_mode("Max"), &nan, (0, 16.236374836823735, [12.029211880574156, 7.056987849312501, 21.649711463336665, 2.0039093793567826])),
            ("one hot", r().learning_rate(0.3), &sided, (0, 16.23260731744268, [-0.8569853958819067, 7.916902619214433, 19.039669425912223, 4.674728105340087])),
        ];
        let mut failed: Vec<String> = Vec::new();
        for (name, mut model, ds, want) in cases {
            let x: Dataset = features(ds, "y");
            model.fit(&x, "y");
            let preds: Vec<f64> = model.predict(&x).f64().unwrap().into_no_null_iter().collect();
            if let Err(e) = std::panic::catch_unwind(|| assert_like(name, 0, &preds, want, 1e-5)) {
                failed.push(e.downcast_ref::<String>().cloned().unwrap_or_default());
            }
        }
        assert!(failed.is_empty(), "{}", failed.join("\n"));
    }

    #[test]
    fn classifiers_match_catboost() {
        let ds: Dataset = data();
        let c = || CatBoostClassifier::new().iterations(20).random_strength(0.0).bootstrap_type("No");
        let cases: [(&str, CatBoostClassifier, &str, Want); 5] = [
            ("logloss", c().learning_rate(0.3), "hit", (0, 0.40838883122100283, [0.06667798624952358, 0.16477980170319198, 0.2760616644895883, 0.03744673412848724])),
            ("logloss auto", c(), "hit", (0, 0.40888912479423195, [0.08672549206395073, 0.22141849671033093, 0.2723244918005648, 0.0479178160637963])),
            ("from average", c().learning_rate(0.3).boost_from_average(true), "hit", (0, 0.40511426367525305, [0.048833279919030886, 0.15641779150537788, 0.2492514598447607, 0.034703404124250097])),
            ("scale_pos_weight", c().learning_rate(0.3).scale_pos_weight(3.0), "hit", (0, 0.4966611244370101, [0.0799759195135493, 0.2765918883216167, 0.5332104688951703, 0.04363211858789076])),
            ("multiclass", c().learning_rate(0.3), "class", (0, 0.3333333333333333, [0.9146621353470988, 0.043281162652601485, 0.042056702000299616, 0.871636513446626])),
        ];
        let mut failed: Vec<String> = Vec::new();
        for (name, mut model, target, want) in cases {
            let x: Dataset = features(&ds, target);
            model.fit(&x, target);
            let proba: Dataset = model.predict_proba(&x);
            let preds: Vec<f64> = match target {
                "hit" => proba.to_list("1"),
                _ => {
                    let columns: Vec<Vec<f64>> = model.classes_.iter().map(|c| proba.to_list(c)).collect();
                    (0..x.height()).flat_map(|r| columns.iter().map(move |c| c[r])).collect()
                }
            };
            if let Err(e) = std::panic::catch_unwind(|| assert_like(name, 0, &preds, want, 1e-5)) {
                failed.push(e.downcast_ref::<String>().cloned().unwrap_or_default());
            }
        }
        assert!(failed.is_empty(), "{}", failed.join("\n"));
    }

    #[test]
    fn the_defaults_draw_from_the_seed_and_a_fit_round_trips_through_the_cache_record() {
        let ds: Dataset = data();
        let (_, sided): (Dataset, Dataset) = extended(&ds);
        let x: Dataset = features(&sided, "y");
        let fit = |seed: u64| {
            let mut model = CatBoostRegressor::new().iterations(30).random_seed(seed);
            model.fit(&x, "y");
            model
        };
        let (a, b, other): (CatBoostRegressor, CatBoostRegressor, CatBoostRegressor) = (fit(7), fit(7), fit(8));
        let preds = |m: &CatBoostRegressor| -> Vec<f64> { m.predict(&x).f64().unwrap().into_no_null_iter().collect() };
        assert_eq!(preds(&a), preds(&b));
        assert_ne!(preds(&a), preds(&other));
        assert!(a.score(&x, "y") > 0.9, "{}", a.score(&x, "y"));
        assert_eq!(a.learning_rate_(), 0.5);

        let mut weights: Vec<u8> = Vec::new();
        a.write_weights(&mut weights).unwrap();
        let mut loaded: CatBoostRegressor = CatBoostRegressor::new().iterations(30).random_seed(7);
        loaded.try_load(a.learned(), &weights, "y").unwrap();
        assert_eq!(preds(&loaded), preds(&a));
        assert!(loaded.try_load(a.learned(), &weights[..weights.len() - 3], "y").is_err());
    }

    #[test]
    fn the_mvs_cut_is_the_one_bisection_finds() {
        let mut rng: Rng = Rng::new(7);
        for (n, share) in [(1usize, 0.5), (2, 0.5), (10, 0.8), (1000, 0.8), (1000, 0.1), (5000, 0.99)] {
            // Ties and zeros too: sizes from a few values.
            let size: Vec<f64> = (0..n).map(|i| if i % 3 == 0 { (rng.next_u64() % 5) as f64 } else { uniform(&mut rng) }).collect();
            let want: f64 = share * n as f64;
            let kept = |mu: f64| -> f64 { size.iter().map(|s| (s / mu).min(1.0)).sum() };
            let top: f64 = size.iter().cloned().fold(0.0, f64::max);
            if kept(top) >= want {
                continue;
            }
            let mu: f64 = super::mvs_cut(size.clone(), want);
            if size.iter().filter(|s| **s > 0.0).count() as f64 <= want {
                assert!(size.iter().all(|s| *s == 0.0 || *s >= mu), "n {n}: a row with a gradient is not kept");
                continue;
            }
            assert!((kept(mu) - want).abs() < 1e-9 * n as f64, "n {n}: kept {} want {want}", kept(mu));
        }
    }

    #[test]
    fn many_categories_become_target_statistics_split_as_sets() {
        let ds: Dataset = data();
        // `grade`: 10 categories that follow the target, so a split on their statistic wins.
        let y: Vec<f64> = ds.to_list("y");
        let grade: Vec<String> = y.iter().map(|v| format!("g{}", (*v / 4.0) as i64 % 10)).collect();
        let mut frame: DataFrame = features(&ds, "y").dataframe().clone();
        frame.with_column(Series::new("grade".into(), grade).into_column()).unwrap();
        let x: Dataset = Dataset::new(frame.select(["grade", "y"]).unwrap(), "");
        let mut model = CatBoostRegressor::new().iterations(50).random_strength(0.0).bootstrap_type("No");
        model.fit(&x, "y");
        assert_eq!(model.is_categorical_, [true]);
        assert!(model.estimators_.iter().any(|t| t.is_categorical().contains(&1)));
        assert!(model.score(&x, "y") > 0.5, "{}", model.score(&x, "y"));
    }

    #[test]
    fn a_symmetric_tree_walked_level_by_level_lands_in_the_same_leaves() {
        let (nan, _): (Dataset, Dataset) = extended(&data());
        let mut model = CatBoostRegressor::new().iterations(20);
        model.fit(&nan, "y");
        let columns: Vec<Vec<f64>> = model.feature_names_in_.iter().map(|f| nan.to_list(f)).collect();
        assert!(columns.iter().any(|c| c.iter().any(|v| v.is_nan())));
        let refs: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();
        for tree in &model.estimators_ {
            let shape = tree.symmetric().expect("CatBoost's trees are symmetric");
            let (mut walked, mut by_levels): (Vec<u32>, Vec<u32>) = (vec![0; nan.height()], vec![0; nan.height()]);
            tree.walk(&refs, 0, &mut walked);
            tree.walk_symmetric(&shape, &refs, 0, &mut by_levels);
            assert_eq!(walked, by_levels);
        }
        // Through bytes (`tree::Quantized`), as `predict` reads them: the same sums.
        let trees: &[Tree] = &model.estimators_;
        let shapes: Vec<_> = trees.iter().map(Tree::symmetric).collect();
        assert!(crate::models::tree::Quantized::new(trees, &shapes, refs.len()).is_some());
        let sums: Vec<f64> = crate::models::tree::sum_leaves_on(trees, &refs, nan.height(), 1, |_, v, out| out[0] += v[0]);
        let mut want: Vec<f64> = vec![0.0; nan.height()];
        for tree in trees {
            let mut leaves: Vec<u32> = vec![0; nan.height()];
            tree.walk(&refs, 0, &mut leaves);
            want.iter_mut().zip(&leaves).for_each(|(w, leaf)| *w += tree.value()[*leaf as usize]);
        }
        assert_eq!(sums, want);
    }

    #[test]
    #[should_panic(expected = "loss_function")]
    fn an_unknown_loss_panics_at_the_setter() {
        let _ = CatBoostRegressor::new().loss_function("MAE");
    }
}
