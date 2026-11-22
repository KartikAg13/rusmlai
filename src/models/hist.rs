//! The histogram tree engine that every tree model shares: [`Bins`] (each feature in at most
//! `max_bins` buckets, 255 by default, stored as `u8`, or `u16` above 255), [`Target`] (what a node's rows are summed into) and [`grow`]
//! (one [`Tree`] from the bins).
//!
//! A feature with at most `max_bins` distinct values gets one bucket per value, so its splits
//! are the exact ones (scikit-learn's `DecisionTree*`). Any other feature gets quantile
//! buckets, like LightGBM and scikit-learn's `HistGradientBoosting*`.
//!
//! A node finds its split from one histogram per feature: the sums of its rows per bucket. A
//! large node keeps its histograms, so that one child is built from its rows and the other is
//! the parent minus that child. A small node reads only the buckets its rows are in. The two
//! children of a large node grow as two tasks on the Polars pool.
//!
//! A forest bins once and grows every tree from the same [`Bins`]: a tree then differs by its
//! row weights (a bootstrap sample is a count per row, not a copy of the rows) and by the
//! features each node may split on (`max_features`). Boosting grows each tree on the
//! gradients of the rounds before ([`Target::Gradients`]), leaf by leaf (`max_leaf_nodes`).

use std::{collections::HashMap, sync::Arc};

use polars::prelude::{DataType, NamedFrom, Series};
use polars_core::runtime::THREAD_POOL;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    dataset::{Dataset, split::Rng},
    models::{
        ModelError,
        design::{column, floats},
        tree::{CATEGORICAL, LEAF, Left, Node, Tree},
    },
};

/// Rows that give a feature's quantile buckets, like LightGBM and scikit-learn.
const SAMPLE: usize = 200_000;

/// Children with this many rows each grow as two tasks; below it a task costs more than it
/// saves.
const SPAWN: usize = 4096;

/// Rows × features from which the histograms of one node are built one feature per task.
const WIDE: usize = 1 << 18;

/// The same for a tree that grows leaf by leaf: its nodes wait for each other, so only the
/// features of a node can run at once.
const LEAF_WIDE: usize = 1 << 14;

/// Rows from which a node puts its rows in two parts, or copies their targets, in blocks of
/// this many, one task each.
const BLOCK: usize = 1 << 14;

/// The most features a round of boosting fills in one pass over the rows
/// ([`Grower::fill_gradients`]). On one thread a fit of S6E1 took 17.2 G CPU cycles with one
/// feature per pass, 14.2 G with 2, 13.2 G with 3 and 12.8 G with 4.
const GROUP: usize = 4;

/// "No node" / "no bin".
const NONE: usize = usize::MAX;

/// How a round of boosting turns a leaf's sums into its value, and which splits it takes: the
/// settings every boosting library shares, under each library's own names in its model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Step {
    /// The L2 penalty on a leaf's value (scikit-learn's `l2_regularization`, XGBoost's and
    /// LightGBM's `reg_lambda`).
    pub l2: f64,
    /// The L1 penalty: Σg moves this much towards 0 before it counts (`reg_alpha`).
    pub l1: f64,
    /// The learning rate.
    pub rate: f64,
    /// The least a split must take off the loss (LightGBM's `min_split_gain`).
    pub min_gain: f64,
    /// The least Σh on each side of a split (scikit-learn's `min_hessian_to_split`, 1e-3;
    /// XGBoost's and LightGBM's `min_child_weight`).
    pub min_hessian: f64,
    /// The most a leaf's value may move before the learning rate, 0 for no limit (XGBoost's
    /// and LightGBM's `max_delta_step`).
    pub max_delta: f64,
    /// LightGBM scores a split as its children's gains added (`GetLeafGain`), not as the
    /// node's loss minus theirs: the same order, but two mirror splits then tie exactly.
    pub rules: Rules,
}

impl Step {
    /// The rules of scikit-learn's histogram boosting with this L2 penalty and rate.
    pub(crate) fn new(l2: f64, rate: f64) -> Self {
        Self { l2, l1: 0.0, rate, min_gain: 0.0, min_hessian: 1e-3, max_delta: 0.0, rules: Rules::Sklearn }
    }

    /// A leaf's value before the learning rate: the Newton step, at most `max_delta` away
    /// from 0.
    fn weight(&self, g: f64, h: f64) -> f64 {
        let w: f64 = -shrunk(g, self.l1) / (h + self.l2 + 1e-15);
        if self.max_delta > 0.0 { w.clamp(-self.max_delta, self.max_delta) } else { w }
    }

    /// What a leaf of sums `g`, `h` costs. Without `max_delta`, scikit-learn's
    /// `_loss_from_value(compute_node_value(..))` operation for operation, so that two splits
    /// that tie in theory tie or differ as in scikit-learn (with gradients of ±1 exact ties are
    /// common). With it, the loss at the capped value, as XGBoost computes it:
    /// 2 (G w + l1 |w|) + (H + l2) w².
    fn cost(&self, g: f64, h: f64) -> f64 {
        if self.max_delta > 0.0 {
            let w: f64 = self.weight(g, h);
            return 2.0 * (g * w + self.l1 * w.abs()) + (h + self.l2) * w * w;
        }
        let g: f64 = shrunk(g, self.l1);
        g * (-g / (h + self.l2 + 1e-15))
    }

    /// LightGBM's `GetLeafGain`: minus the cost, with no 1e-15 in the denominator.
    fn leaf_gain(&self, g: f64, h: f64) -> f64 {
        if self.max_delta > 0.0 {
            return -self.cost(g, h);
        }
        let g: f64 = shrunk(g, self.l1);
        g * g / (h + self.l2)
    }
}


/// `name` as `Float64` in one chunk, with NaN for each null: a tree's missing value.
pub(crate) fn feature(dataset: &Dataset, name: &str) -> Result<Series, ModelError> {
    let series: Series = floats(dataset, name)?;
    if series.null_count() == 0 {
        return Ok(series);
    }
    let filled: Vec<f64> = series
        .f64()?
        .iter()
        .map(|value| value.unwrap_or(f64::NAN))
        .collect();
    Ok(Series::new(name.into(), filled))
}

/// Whether a column holds text (or Polars categories): such a feature can only be
/// categorical.
pub(crate) fn is_text(dtype: &DataType) -> bool {
    matches!(dtype, DataType::String | DataType::Categorical(..) | DataType::Enum(..))
}

/// The categories of a categorical feature, sorted (numbers ascending, text by its bytes):
/// a category's code is its place, as in scikit-learn's `HistGradientBoosting*`, which encode
/// them with an `OrdinalEncoder`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub(crate) enum Categories {
    Numbers(Vec<f64>),
    Text(Vec<String>),
}

impl Categories {
    /// The code of each row of column `name`: the place of its category, NaN for a missing
    /// value or a category `fit` never saw (scikit-learn takes those as missing too).
    pub(crate) fn codes(&self, dataset: &Dataset, name: &str) -> Result<Vec<f64>, ModelError> {
        let dtype: DataType = dataset.column(name)?.dtype().clone();
        let mistyped = |was: &'static str| ModelError::CategoryType {
            column: name.into(),
            dtype: dtype.to_string(),
            was,
        };
        match self {
            Categories::Numbers(known) => {
                if is_text(&dtype) {
                    return Err(mistyped("numbers"));
                }
                let series: Series = feature(dataset, name)?;
                let code = |value: &f64| {
                    let value: f64 = value + 0.0; // -0.0 is 0.0
                    known.binary_search_by(|k| k.total_cmp(&value)).map_or(f64::NAN, |at| at as f64)
                };
                Ok(column(&series).iter().map(code).collect())
            }
            Categories::Text(known) => {
                if !is_text(&dtype) {
                    return Err(mistyped("text"));
                }
                let places: HashMap<&str, usize> =
                    known.iter().enumerate().map(|(i, name)| (name.as_str(), i)).collect();
                let text: Series = text(dataset, name)?;
                let code = |value: Option<&str>| {
                    value.and_then(|v| places.get(v)).map_or(f64::NAN, |at| *at as f64)
                };
                Ok(text.str()?.iter().map(code).collect())
            }
        }
    }
}

/// A text or categorical column as Polars `String`.
fn text(dataset: &Dataset, name: &str) -> Result<Series, ModelError> {
    let source = dataset.column(name)?;
    Ok(source.as_materialized_series().cast(&DataType::String)?)
}

/// A row's bucket while a feature is binned: two bytes, so a feature can have up to 65,535
/// buckets (`max_bins`). [`Codes`] stores them in one byte when `max_bins` is at most 255.
pub(crate) type Code = u16;

/// What a hot loop reads a bucket from: `u8` or `u16`.
pub(crate) trait Bucket: Copy + Into<usize> + Send + Sync {}
impl Bucket for u8 {}
impl Bucket for u16 {}

/// The buckets of a feature's rows. One byte each when `max_bins` is at most 255 (every code,
/// the missing one too, is then at most 255), else two: a fit reads half the bytes, 4% to 15%
/// fewer cycles (2026-11-21). Every feature of one [`Bins`] has the same `max_bins`, so the
/// same width.
pub(crate) enum Codes {
    Narrow(Vec<u8>),
    Wide(Vec<u16>),
}

/// Runs `$body` with `$c` bound to the codes as a slice of their own width.
macro_rules! by_width {
    ($codes:expr, |$c:ident| $body:expr) => {
        match $codes {
            Codes::Narrow($c) => $body,
            Codes::Wide($c) => $body,
        }
    };
}

impl Codes {
    fn new(codes: Vec<Code>, max_bins: usize) -> Self {
        match max_bins <= 255 {
            true => Codes::Narrow(codes.into_iter().map(|code| code as u8).collect()),
            false => Codes::Wide(codes),
        }
    }

    pub(crate) fn len(&self) -> usize {
        by_width!(self, |c| c.len())
    }

    /// The bucket of row `i`.
    pub(crate) fn get(&self, i: usize) -> usize {
        by_width!(self, |c| c[i].into())
    }

    /// Every code as two bytes, for code outside the hot loops (CatBoost's columns).
    pub(crate) fn wide(&self) -> std::borrow::Cow<'_, [Code]> {
        match self {
            Codes::Narrow(c) => c.iter().map(|code| Code::from(*code)).collect(),
            Codes::Wide(c) => std::borrow::Cow::Borrowed(c),
        }
    }
}

/// One feature in buckets.
pub(crate) struct Binned {
    /// The bucket of each row; a missing value has the code after the last bucket.
    pub(crate) codes: Codes,
    /// The smallest and the largest value in each bucket.
    edges: Vec<(f64, f64)>,
    missing: bool,
    /// A text feature's categories, sorted: bucket `i` is the `i`-th of them.
    categories: Option<Vec<String>>,
}

/// A text column in buckets, one per category in sorted order; `None` for more than
/// `max_bins` categories, with how many there are.
fn bin_text(series: &Series, max_bins: usize) -> Result<Binned, usize> {
    let most: usize = max_bins.min(255); // a category split is a set of 256 bits
    let values = series.str().expect("text() makes String columns");
    let mut places: HashMap<&str, Code> = HashMap::new();
    let mut names: Vec<&str> = Vec::new();
    let mut codes: Vec<Code> = Vec::with_capacity(values.len());
    let mut missing: bool = false;
    for value in values.iter() {
        let Some(value) = value else {
            missing = true;
            codes.push(Code::MAX);
            continue;
        };
        let place: Code = match places.get(value) {
            Some(place) => *place,
            None => {
                if names.len() == most {
                    let more: usize = values.iter().flatten().collect::<std::collections::HashSet<_>>().len();
                    return Err(more);
                }
                places.insert(value, names.len() as Code);
                names.push(value);
                (names.len() - 1) as Code
            }
        };
        codes.push(place);
    }
    let mut order: Vec<usize> = (0..names.len()).collect();
    order.sort_unstable_by(|a, b| names[*a].cmp(names[*b]));
    let mut rank: Vec<Code> = vec![0; names.len()];
    for (r, place) in order.iter().enumerate() {
        rank[*place] = r as Code;
    }
    let last: Code = names.len() as Code;
    codes.iter_mut().for_each(|code| *code = if *code == Code::MAX { last } else { rank[*code as usize] });
    Ok(Binned {
        codes: Codes::new(codes, max_bins),
        edges: (0..names.len()).map(|i| (i as f64, i as f64)).collect(),
        missing,
        categories: Some(order.iter().map(|place| names[*place].to_string()).collect()),
    })
}

/// `values` in at most `max_bins` buckets; `None` when a value is infinite.
fn bin(values: &[f64], max_bins: usize) -> Option<Binned> {
    if values.iter().any(|value| value.is_infinite()) {
        return None;
    }
    // One pass gives each value the place of its first appearance among the distinct values,
    // through a small hash table (a binary search per value took three times as long). It
    // stops at the first distinct value too many.
    // At least twice as many slots as distinct values, a power of two: 512 for 255 buckets.
    let slots: usize = (2 * max_bins + 2).next_power_of_two().max(512);
    let shift: u32 = 64 - slots.trailing_zeros();
    const EMPTY: u64 = u64::MAX; // a NaN, and no NaN is a key
    const MISSING: Code = Code::MAX; // max_bins is below 65,535, so no place is this
    let mut keys: Vec<u64> = vec![EMPTY; slots];
    let mut places: Vec<Code> = vec![0; slots];
    let mut distinct: Vec<f64> = Vec::with_capacity(max_bins);
    let mut codes: Vec<Code> = Vec::with_capacity(values.len());
    let mut exact: bool = true;
    let (mut last, mut place): (u64, Code) = (EMPTY, 0);
    'values: for &value in values {
        if value.is_nan() {
            codes.push(MISSING);
            continue;
        }
        let bits: u64 = (value + 0.0).to_bits(); // -0.0 is 0.0
        if bits != last {
            let hash: u64 = (bits ^ (bits >> 29)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut slot: usize = (hash >> shift) as usize;
            loop {
                if keys[slot] == bits {
                    place = places[slot];
                    break;
                }
                if keys[slot] == EMPTY {
                    if distinct.len() == max_bins {
                        exact = false;
                        break 'values;
                    }
                    place = distinct.len() as Code;
                    (keys[slot], places[slot]) = (bits, place);
                    distinct.push(value);
                    break;
                }
                slot = (slot + 1) % slots;
            }
            last = bits;
        }
        codes.push(place);
    }
    if exact {
        // Each distinct value is a bucket of its own: the places become ranks.
        let mut order: Vec<usize> = (0..distinct.len()).collect();
        order.sort_unstable_by(|a, b| distinct[*a].total_cmp(&distinct[*b]));
        let mut rank: Vec<Code> = vec![0; distinct.len()];
        for (r, place) in order.iter().enumerate() {
            rank[*place] = r as Code;
        }
        let mut missing: bool = false;
        for code in &mut codes {
            missing |= *code == MISSING;
            *code = if *code == MISSING { distinct.len() as Code } else { rank[*code as usize] };
        }
        let edges: Vec<(f64, f64)> = order.iter().map(|p| (distinct[*p], distinct[*p])).collect();
        return Some(Binned {
            codes: Codes::new(codes, max_bins),
            edges,
            missing,
            categories: None,
        });
    }
    // More distinct values than buckets: quantiles of every k-th value, so each bucket holds
    // about the same number of rows, and the last one is everything above the last top.
    let step: usize = values.len().div_ceil(SAMPLE);
    let mut sample: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| !value.is_nan())
        .step_by(step)
        .collect();
    sample.sort_unstable_by(f64::total_cmp);
    let mut tops: Vec<f64> = (1..max_bins)
        .map(|i| sample[i * sample.len() / max_bins])
        .collect();
    tops.dedup();
    let bins: usize = tops.len() + 1;
    let mut edges: Vec<(f64, f64)> = vec![(f64::INFINITY, f64::NEG_INFINITY); bins];
    let mut missing: bool = false;
    let codes: Vec<Code> = values
        .iter()
        .map(|&value| {
            if value.is_nan() {
                missing = true;
                return bins as Code;
            }
            let at: usize = tops.partition_point(|top| *top < value);
            let edge: &mut (f64, f64) = &mut edges[at];
            *edge = (edge.0.min(value), edge.1.max(value));
            at as Code
        })
        .collect();
    Some(Binned {
        codes: Codes::new(codes, max_bins),
        edges,
        missing,
        categories: None,
    })
}

/// The features of a dataset (every column but the target) in buckets: the only form of the
/// data a tree reads while it grows. Nulls and NaN are missing values, which get a bucket of
/// their own.
pub(crate) struct Bins {
    /// The feature names, in column order (scikit-learn's `feature_names_in_`).
    pub names: Vec<String>,
    features: Vec<Arc<Binned>>,
    /// Whether each feature is categorical: a split sends a set of its categories left.
    categorical: Vec<bool>,
    /// Where each feature's buckets start in a histogram, and at the end the total.
    offsets: Vec<usize>,
    /// The most buckets of one feature.
    widest: usize,
}

impl Bins {
    /// Bins every feature, one per task on the Polars pool. `max_bins` is 2 to 255. A column
    /// keeps its buckets (`ColumnInfo::bins`), so the next tree model on the same dataset, or
    /// on a `select` of it, makes none: a bag's copies, the candidates of a search, a refit.
    pub fn try_new(dataset: &Dataset, target: &str, max_bins: usize) -> Result<Self, ModelError> {
        Self::try_with(dataset, target, max_bins, &[])
    }

    /// [`Bins::try_new`] where the features named in `categorical` are categorical: a bucket
    /// per category (at most `max_bins`), in sorted order. A text feature must be one of them.
    pub fn try_with(
        dataset: &Dataset,
        target: &str,
        max_bins: usize,
        categorical: &[String],
    ) -> Result<Self, ModelError> {
        let names: Vec<String> = dataset
            .columns()
            .into_iter()
            .filter(|name| name != target)
            .collect();
        if names.is_empty() {
            return Err(ModelError::NoFeatures {
                target: target.into(),
            });
        }
        let rows: usize = dataset.height();
        let features: Vec<Arc<Binned>> = THREAD_POOL.install(|| {
            names
                .par_iter()
                .map(|name| {
                    let is_categorical: bool = categorical.contains(name);
                    let text: bool = is_text(dataset.column(name)?.dtype());
                    let too_many = |categories: usize| ModelError::TooManyCategories {
                        column: name.clone(),
                        categories,
                        max_bins,
                    };
                    if text && !is_categorical {
                        let _ = feature(dataset, name)?; // text is not a number: that error
                    }
                    let cache = &dataset.try_column_info(name)?.bins;
                    let binned: Arc<Binned> = match cache.get(max_bins, rows) {
                        Some(binned) => binned,
                        // A category split is a set of 256 bits: at most 255 categories.
                        None if text && is_categorical => {
                            let binned: Binned = bin_text(&self::text(dataset, name)?, max_bins)
                                .map_err(too_many)?;
                            cache.set(max_bins, binned)
                        }
                        None => {
                            let series: Series = feature(dataset, name)?;
                            let binned: Binned = bin(column(&series), max_bins)
                                .ok_or_else(|| ModelError::NotFinite { column: name.clone() })?;
                            cache.set(max_bins, binned)
                        }
                    };
                    // Numbers as categories: a bucket per distinct value, or there are more
                    // values than buckets.
                    let exact: bool = binned.edges.iter().all(|(low, high)| low == high);
                    if is_categorical && !(exact && binned.edges.len() <= max_bins.min(255)) {
                        return Err(too_many(binned.edges.len().max(max_bins.min(255) + 1)));
                    }
                    Ok(binned)
                })
                .collect::<Result<_, ModelError>>()
        })?;
        let categorical: Vec<bool> = names.iter().map(|name| categorical.contains(name)).collect();
        let mut offsets: Vec<usize> = vec![0];
        for binned in &features {
            let slots: usize = binned.edges.len() + usize::from(binned.missing);
            offsets.push(offsets[offsets.len() - 1] + slots);
        }
        let widest: usize = offsets.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0);
        Ok(Self {
            names,
            features,
            categorical,
            offsets,
            widest,
        })
    }

    /// LightGBM's `min_data_in_bin`: in a feature with a bucket per value, neighbouring values
    /// share a bucket until it holds `min_rows` rows, and the last bucket takes the rest
    /// (LightGBM's `GreedyFindBin`). Categories and quantile buckets stay as they are.
    pub(crate) fn merged(mut self, min_rows: usize) -> Self {
        if min_rows <= 1 {
            return self;
        }
        for f in 0..self.names.len() {
            let b: &Binned = &self.features[f];
            let n: usize = b.edges.len();
            if self.categorical[f] || n < 2 || b.edges.iter().any(|(low, high)| low != high) {
                continue;
            }
            let mut counts: Vec<usize> = vec![0; n + 1];
            by_width!(&b.codes, |c| c.iter().for_each(|code| counts[usize::from(*code)] += 1));
            let mut group: Vec<Code> = vec![0; n + 1];
            let mut edges: Vec<(f64, f64)> = Vec::new();
            let (mut low, mut run): (f64, usize) = (b.edges[0].0, 0);
            for i in 0..n {
                group[i] = edges.len() as Code;
                run += counts[i];
                if i + 1 < n && run >= min_rows {
                    edges.push((low, b.edges[i].1));
                    (low, run) = (b.edges[i + 1].0, 0);
                }
            }
            edges.push((low, b.edges[n - 1].1));
            if edges.len() == n {
                continue;
            }
            group[n] = edges.len() as Code; // the missing values' code
            let codes: Codes = match &b.codes {
                Codes::Narrow(c) => Codes::Narrow(c.iter().map(|code| group[usize::from(*code)] as u8).collect()),
                Codes::Wide(c) => Codes::Wide(c.iter().map(|code| group[usize::from(*code)]).collect()),
            };
            self.features[f] = Arc::new(Binned { codes, edges, missing: b.missing, categories: None });
        }
        let mut offsets: Vec<usize> = vec![0];
        for binned in &self.features {
            let slots: usize = binned.edges.len() + usize::from(binned.missing);
            offsets.push(offsets[offsets.len() - 1] + slots);
        }
        self.widest = offsets.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0);
        self.offsets = offsets;
        self
    }

    /// Feature `f`'s bucket of each row (a missing value's code is the bucket count), the
    /// smallest and largest value of each bucket, and whether any value is missing: what a
    /// tree grown outside this engine reads (CatBoost's symmetric trees, `models/catboost.rs`).
    pub(crate) fn binned(&self, f: usize) -> (&Codes, &[(f64, f64)], bool) {
        let b: &Binned = &self.features[f];
        (&b.codes, &b.edges, b.missing)
    }

    /// The categories of feature `f`, when it is categorical.
    pub(crate) fn categories(&self, f: usize) -> Option<Categories> {
        let binned: &Binned = &self.features[f];
        self.categorical[f].then(|| match &binned.categories {
            Some(names) => Categories::Text(names.clone()),
            None => Categories::Numbers(binned.edges.iter().map(|(value, _)| *value).collect()),
        })
    }

    /// The code of each row of feature `f`: a categorical feature's category codes, NaN for
    /// a missing value. What a tree walks on for a categorical feature.
    pub(crate) fn codes(&self, f: usize) -> Vec<f64> {
        let binned: &Binned = &self.features[f];
        let missing: usize = binned.edges.len();
        let wide: std::borrow::Cow<'_, [Code]> = binned.codes.wide();
        wide.iter().map(|code| if usize::from(*code) == missing { f64::NAN } else { f64::from(*code) }).collect()
    }
}

/// What a tree predicts, and so what a histogram sums per bucket.
pub(crate) enum Target {
    /// The class of each row, 0 to `classes - 1`. A bucket holds the weight of each class.
    Classes {
        y: Vec<u32>,
        classes: usize,
        /// `true` for entropy (log loss), `false` for gini.
        entropy: bool,
    },
    /// The target of each row. A bucket holds Σw, Σwy and Σwy².
    Regression { y: Vec<f64> },
    /// A round of boosting: the gradient and the hessian of the loss at each row. A bucket
    /// holds its rows, Σg and Σh, and a leaf's value is one Newton step, −Σg / (Σh + `l2`),
    /// times `rate`. An empty `h` means that every hessian is 1 (squared error): a bucket then
    /// holds its rows and Σg, and Σh is the rows.
    Gradients {
        g: Vec<f32>,
        h: Vec<f32>,
        step: Step,
    },
}

/// Σg moved `l1` towards 0, and 0 within `l1` of it: the L1 penalty on a leaf's value.
fn shrunk(g: f64, l1: f64) -> f64 {
    if l1 == 0.0 { g } else { g.signum() * (g.abs() - l1).max(0.0) }
}

fn xlnx(x: f64) -> f64 {
    if x > 0.0 { x * x.ln() } else { 0.0 }
}

impl Target {
    /// The sums per bucket.
    fn stats(&self) -> usize {
        match self {
            Target::Classes { classes, .. } => *classes,
            // With every hessian 1, Σh is the rows: no third sum.
            Target::Gradients { h, .. } if h.is_empty() => 2,
            Target::Regression { .. } | Target::Gradients { .. } => 3,
        }
    }

    /// Σh of gradient sums.
    fn hessian(&self, sums: &[f64]) -> f64 {
        match self {
            Target::Gradients { h, .. } if h.is_empty() => sums[0],
            _ => sums[2],
        }
    }

    pub fn rows(&self) -> usize {
        match self {
            Target::Classes { y, .. } => y.len(),
            Target::Regression { y } => y.len(),
            Target::Gradients { g, .. } => g.len(),
        }
    }

    /// The weight in these sums: the rows, when rows have no weights.
    fn weight(&self, sums: &[f64]) -> f64 {
        match self {
            Target::Classes { .. } => sums.iter().sum(),
            Target::Regression { .. } | Target::Gradients { .. } => sums[0],
        }
    }

    /// Gini, entropy in bits, or the variance, like scikit-learn's `tree_.impurity`.
    fn impurity(&self, sums: &[f64]) -> f64 {
        let n: f64 = self.weight(sums);
        match self {
            Target::Classes { entropy: false, .. } => {
                1.0 - sums.iter().map(|c| (c / n) * (c / n)).sum::<f64>()
            }
            Target::Classes { entropy: true, .. } => {
                (xlnx(n) - sums.iter().map(|c| xlnx(*c)).sum::<f64>())
                    / (n * std::f64::consts::LN_2)
            }
            Target::Regression { .. } => sums[2] / n - (sums[1] / n) * (sums[1] / n),
            Target::Gradients { .. } => 0.0,
        }
    }

    /// What a node costs: its impurity times its weight, or for gradients the loss that its
    /// Newton step leaves, −G² / (H + l2). A split takes off the cost of the node minus the
    /// costs of its two sides.
    fn cost(&self, sums: &[f64]) -> f64 {
        match self {
            Target::Gradients { step, .. } => step.cost(sums[1], self.hessian(sums)),
            _ => self.weight(sums) * self.impurity(sums),
        }
    }

    /// Whether every row has the same target, so there is nothing to split.
    fn pure(&self, sums: &[f64]) -> bool {
        match self {
            Target::Classes { .. } => sums.contains(&self.weight(sums)),
            // scikit-learn's rule is a variance of at most ε. Σy²/n − mean² of equal values
            // that are not whole numbers is rounding noise of about ε · mean², which that rule
            // reads as a spread and splits on, so here the noise counts as no spread.
            Target::Regression { .. } => {
                self.impurity(sums) <= f64::EPSILON * (1.0 + 16.0 * sums[2] / sums[0])
            }
            Target::Gradients { .. } => false,
        }
    }

    /// Whether a split that takes nothing off is still made. scikit-learn's trees make it,
    /// because the splits below it may take something off; its boosting does not.
    fn splits_on_nothing(&self) -> bool {
        !matches!(self, Target::Gradients { .. })
    }

    /// The score of a split into `left` and the rest of `total`: higher is a larger drop of the
    /// impurity. scikit-learn's `proxy_impurity_improvement`.
    fn proxy(&self, left: &[f64], total: &[f64]) -> f64 {
        match self {
            Target::Classes { entropy: false, .. } => {
                let (mut sq_l, mut sq_r, mut n_l, mut n_r) = (0.0, 0.0, 0.0, 0.0);
                for (l, t) in left.iter().zip(total) {
                    let r: f64 = t - l;
                    sq_l += l * l;
                    sq_r += r * r;
                    n_l += l;
                    n_r += r;
                }
                sq_l / n_l + sq_r / n_r
            }
            Target::Classes { entropy: true, .. } => {
                let (mut sum, mut n_l, mut n_r) = (0.0, 0.0, 0.0);
                for (l, t) in left.iter().zip(total) {
                    let r: f64 = t - l;
                    sum += xlnx(*l) + xlnx(r);
                    n_l += l;
                    n_r += r;
                }
                sum - xlnx(n_l) - xlnx(n_r)
            }
            Target::Regression { .. } => {
                let (n_r, sum_r): (f64, f64) = (total[0] - left[0], total[1] - left[1]);
                left[1] * left[1] / left[0] + sum_r * sum_r / n_r
            }
            // scikit-learn's `_split_gain`: the node's loss minus its children's.
            Target::Gradients { .. } => {
                let h_l: f64 = self.hessian(left);
                let h_r: f64 = self.hessian(total) - h_l;
                let Target::Gradients { step, .. } = self else { unreachable!() };
                if h_l < step.min_hessian || h_r < step.min_hessian {
                    return f64::NEG_INFINITY;
                }
                if step.rules == Rules::Lightgbm {
                    return step.leaf_gain(left[1], h_l) + step.leaf_gain(total[1] - left[1], h_r);
                }
                let right: [f64; 3] = [total[0] - left[0], total[1] - left[1], h_r];
                let right: &[f64] = match self.stats() {
                    2 => &right[..2],
                    _ => &right,
                };
                self.cost(total) - self.cost(left) - self.cost(right)
            }
        }
    }

    /// Adds a node's value to `out`: the weight of each class, the mean, or the Newton step.
    fn value(&self, sums: &[f64], out: &mut Vec<f64>) {
        match self {
            Target::Classes { .. } => out.extend_from_slice(sums),
            Target::Regression { .. } => out.push(sums[1] / sums[0]),
            Target::Gradients { step, .. } => out.push(step.rate * step.weight(sums[1], self.hessian(sums))),
        }
    }
}

/// When a node stops splitting and what it may split on: scikit-learn's settings of the same
/// names.
#[derive(Clone)]
pub(crate) struct Growth {
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub min_impurity_decrease: f64,
    /// The features each node draws; the number of features means all, in column order.
    pub max_features: usize,
    /// Seeds each node's draw.
    pub seed: u64,
    /// The most leaves. The tree then grows leaf by leaf: the leaf whose split takes most off
    /// is split first. `None` is no limit, and the tree grows subtree by subtree.
    pub max_leaf_nodes: Option<usize>,
    /// The features this tree may split on, ascending (XGBoost's `colsample_bytree`,
    /// LightGBM's `feature_fraction`); `None` is every feature.
    pub features: Option<Vec<u32>>,
    /// The share of the tree's features that each depth draws (XGBoost's
    /// `colsample_bylevel`), and of those, that each node draws (`colsample_bynode`,
    /// LightGBM's `feature_fraction_bynode`); 1 is all.
    pub by_level: f64,
    pub by_node: f64,
    /// Whose rules a split follows where the libraries differ.
    pub rules: Rules,
    /// scikit-learn's `splitter="random"` (Extra Trees): each feature drawn offers one split,
    /// at a threshold drawn uniformly between its smallest and largest value in the node, not
    /// its best one.
    pub random_splits: bool,
}

/// The library whose split rules a tree follows where they differ: where a threshold lies, how
/// a split counts rows, how many features a share is. The gain and the leaf values are the
/// same in all three.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) enum Rules {
    /// The threshold is halfway between the values on both sides, and rows are counted.
    #[default]
    Sklearn,
    /// A row goes left when its value is below the smallest value on the right; a node
    /// without missing values sends them right.
    Xgboost,
    /// The threshold is halfway between the smallest value on the right and the value before
    /// it, and a side has as many rows as its share of the hessian says (LightGBM keeps no
    /// counts).
    Lightgbm,
}

impl Rules {
    /// How many of `n` features a share `share` draws: XGBoost's `max(1, ⌊share n⌋)`, LightGBM's
    /// `max(round(share n), min(2, n))`.
    pub(crate) fn count(self, n: usize, share: f64) -> usize {
        let count: usize = match self {
            Rules::Lightgbm => ((share * n as f64 + 0.5) as usize).max(n.min(2)),
            _ => ((share * n as f64) as usize).max(1),
        };
        count.min(n)
    }
}

/// Whether two LightGBM gains are equal but for the order of their sums.
fn ties(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-12 * a.abs().max(b.abs())
}

/// `take` features of `pool` drawn without repeats from `rng`, in column order.
pub(crate) fn draw_features(pool: &[u32], take: usize, rng: &mut Rng) -> Vec<u32> {
    let mut pool: Vec<u32> = pool.to_vec();
    for i in 0..take.min(pool.len()) {
        let j: usize = i + (rng.next_u64() % (pool.len() - i) as u64) as usize;
        pool.swap(i, j);
    }
    pool.truncate(take);
    pool.sort_unstable();
    pool
}

/// The nodes of a subtree in the order they were made: a node, its left subtree, its right
/// subtree. While it grows, `Node::right` is how far ahead the right child is, so two
/// subtrees join without a fix.
#[derive(Default)]
struct Nodes {
    nodes: Vec<Node>,
    value: Vec<f64>,
    samples: Vec<u32>,
    /// The impurity a split took off, weighted by its share of the rows; 0 for a leaf.
    decrease: Vec<f64>,
    /// The categories that go left at each categorical split ([`Tree`]'s sets).
    sets: Vec<Left>,
}

impl Nodes {
    fn append(&mut self, mut other: Nodes) {
        // The other subtree's categorical splits name their sets from its own first set.
        let first: f64 = self.sets.len() as f64;
        for node in other.nodes.iter_mut().filter(|node| node.is_categorical()) {
            node.threshold += first;
        }
        self.sets.append(&mut other.sets);
        self.nodes.append(&mut other.nodes);
        self.value.append(&mut other.value);
        self.samples.append(&mut other.samples);
        self.decrease.append(&mut other.decrease);
    }

    /// Adds a leaf of `rows` rows with these sums, and gives its place.
    fn leaf(&mut self, target: &Target, sums: &[f64], rows: usize) -> usize {
        self.nodes.push(Node {
            threshold: 0.0,
            split: LEAF,
            right: 0,
        });
        target.value(sums, &mut self.value);
        self.samples.push(rows as u32);
        self.decrease.push(0.0);
        self.nodes.len() - 1
    }
}

/// A node that waits for its turn: rows `lo..hi` of its task's slice.
struct Work {
    lo: usize,
    hi: usize,
    depth: usize,
    /// The parent, when this is its right child.
    right_of: usize,
    hist: Option<Vec<f64>>,
}

/// The best split of a node so far.
#[derive(Clone)]
struct Best {
    /// Whether a feature offered a split at all: it has two values among the node's rows.
    tried: bool,
    proxy: f64,
    feature: usize,
    /// The last bucket that goes left, and the first that goes right ([`NONE`]: only the
    /// missing values go right).
    cut: usize,
    next: usize,
    missing_left: bool,
    rows_left: f64,
    left: Vec<f64>,
    /// At a categorical split, the buckets that go left (the missing values' bucket too, when
    /// they go left); `None` at a split between two buckets.
    set: Option<Left>,
    /// The drawn threshold of a random split; NaN for a split at the midpoint.
    random: f64,
}

/// The running sums of one scan: the buckets so far, and the same plus the missing values.
struct Scan {
    left: Vec<f64>,
    with_missing: Vec<f64>,
    /// The buckets with rows, in order (boosting's second pass).
    present: Vec<usize>,
}

/// The target and the weight of each row of a node, in the order of its rows. A histogram
/// reads a node's rows once per feature: with these copies in line, only the bucket of a row
/// is looked up by its row number, not its target and weight too.
#[derive(Default)]
struct Along {
    class: Vec<u32>,
    value: Vec<f64>,
    weight: Vec<f32>,
    gradient: Vec<f32>,
    hessian: Vec<f32>,
}

/// What one task needs to search and split its nodes, made once.
struct Scratch {
    /// The rows that go right, while a node's rows are put in two parts.
    spare: Vec<u32>,
    /// The histogram of one feature of a node that keeps none.
    small: Vec<f64>,
    /// Histograms to use again.
    pool: Vec<Vec<f64>>,
    along: Along,
    /// The tree's features in column order, between two draws, and the swaps of the last draw.
    order: Vec<u32>,
    drawn: Vec<usize>,
    /// The features of the depth of the node searched last, when each depth draws its own.
    level: Vec<u32>,
    level_of: usize,
    scan: Scan,
    /// The best split of the node searched last, and the sums of its right side.
    best: Best,
    right: Vec<f64>,
}

/// The buckets marked in `seen`, ascending.
fn marked(seen: [u64; 4]) -> impl Iterator<Item = usize> {
    seen.into_iter().enumerate().flat_map(|(k, mut word)| {
        std::iter::from_fn(move || {
            (word != 0).then(|| {
                let bit: usize = word.trailing_zeros() as usize;
                word &= word - 1;
                k * 64 + bit
            })
        })
    })
}

/// The open leaves of a tree that grows leaf by leaf, by what their split takes off: a heap
/// that is Python's `heapq`, step for step. Boosting (`python`) pops leaves of equal gain in
/// the order scikit-learn's grower does; a tree pops the older one first.
#[derive(Default)]
struct Queue {
    heap: Vec<(f64, usize)>,
    python: bool,
}

impl Queue {
    /// `heapq`'s `<`: a larger gain first (scikit-learn's `TreeNode.__lt__`).
    fn before(&self, a: (f64, usize), b: (f64, usize)) -> bool {
        a.0 > b.0 || (!self.python && a.0 == b.0 && a.1 < b.1)
    }

    fn push(&mut self, item: (f64, usize)) {
        self.heap.push(item);
        self.sift_down(0, self.heap.len() - 1);
    }

    fn pop(&mut self) -> Option<(f64, usize)> {
        let last: (f64, usize) = self.heap.pop()?;
        if self.heap.is_empty() {
            return Some(last);
        }
        let top: (f64, usize) = std::mem::replace(&mut self.heap[0], last);
        self.sift_up(0);
        Some(top)
    }

    /// `heapq._siftdown`.
    fn sift_down(&mut self, start: usize, mut pos: usize) {
        let item: (f64, usize) = self.heap[pos];
        while pos > start {
            let parent: usize = (pos - 1) >> 1;
            if !self.before(item, self.heap[parent]) {
                break;
            }
            self.heap[pos] = self.heap[parent];
            pos = parent;
        }
        self.heap[pos] = item;
    }

    /// `heapq._siftup`.
    fn sift_up(&mut self, mut pos: usize) {
        let (end, start, item): (usize, usize, (f64, usize)) = (self.heap.len(), pos, self.heap[pos]);
        let mut child: usize = 2 * pos + 1;
        while child < end {
            if child + 1 < end && !self.before(self.heap[child], self.heap[child + 1]) {
                child += 1;
            }
            self.heap[pos] = self.heap[child];
            pos = child;
            child = 2 * pos + 1;
        }
        self.heap[pos] = item;
        self.sift_down(start, pos);
    }
}

/// A leaf of a tree that grows leaf by leaf, with the split it would get.
struct Open {
    lo: usize,
    hi: usize,
    depth: usize,
    hist: Option<Vec<f64>>,
    best: Best,
    right: Vec<f64>,
    decrease: f64,
}

/// A tree that grows leaf by leaf: its nodes in the order they were made, each split with
/// the places of its two children, and the leaves that can still be split.
#[derive(Default)]
struct Loose {
    nodes: Nodes,
    children: Vec<(usize, usize)>,
    open: Vec<Option<Open>>,
    /// The open leaves by what their split takes off; of two equal ones, the older first.
    queue: Queue,
}

struct Grower<'a> {
    bins: &'a Bins,
    target: &'a Target,
    growth: &'a Growth,
    /// The weight of each row of the data; `None` when every row counts once. With weights, a
    /// bucket also holds its rows, after the sums: the `min_samples_*` settings count rows.
    weights: Option<&'a [f32]>,
    /// Sums per bucket, and numbers per bucket (the sums, and the rows when rows have weights).
    stats: usize,
    width: usize,
    /// The weight of the whole tree.
    weight: f64,
    /// See [`SPAWN`]; a test sets it to compare the paths.
    spawn: usize,
    /// `false` makes every node read its rows (the small-node path), for the same test.
    dense: bool,
}

impl Grower<'_> {
    /// Whether a node of `rows` rows keeps histograms of every feature: when it has about as
    /// many row values as buckets, and its split may be on at least half of the features. A
    /// node that draws fewer reads its rows for those features only.
    fn is_dense(&self, rows: usize) -> bool {
        let features: usize = self.bins.names.len();
        self.dense
            && 2 * self.growth.max_features >= features
            && rows * features >= self.bins.offsets[features]
    }

    /// The rows in a bucket.
    #[inline]
    fn rows_of(&self, cell: &[f64]) -> f64 {
        match (self.weights, self.target) {
            (Some(_), _) => cell[self.stats],
            (None, Target::Classes { .. }) => cell.iter().sum(),
            (None, _) => cell[0],
        }
    }

    /// Whether a split of score `proxy` beats the best so far. XGBoost keeps a split's gain
    /// as an f32 (`loss_chg`), so two gains that differ in the last bits of an f64 tie, and
    /// the first feature wins.
    /// LightGBM's gains of a split and of its mirror differ only by the order its sums were
    /// added in: rusmlai takes gains within 1e-12 of each other as equal, so that the first
    /// feature wins, as in LightGBM's own ties.
    #[inline]
    fn beats(&self, proxy: f64, best: f64) -> bool {
        match self.growth.rules {
            Rules::Xgboost => proxy as f32 > best as f32,
            Rules::Lightgbm => proxy > best && !ties(proxy, best),
            Rules::Sklearn => proxy > best,
        }
    }

    fn is_leaf(&self, rows: usize, depth: usize, cell: &[f64]) -> bool {
        let g: &Growth = self.growth;
        g.max_depth.is_some_and(|max| depth >= max)
            || rows < g.min_samples_split
            || rows < 2 * g.min_samples_leaf
            || self.target.pure(&cell[..self.stats])
    }

    fn scratch(&self, rows: usize) -> Scratch {
        let w: usize = self.width;
        Scratch {
            spare: vec![0; rows],
            small: vec![0.0; self.bins.widest * w],
            pool: Vec::new(),
            along: Along::default(),
            order: (self.growth.features.clone())
                .unwrap_or_else(|| (0..self.bins.names.len() as u32).collect()),
            drawn: Vec::new(),
            level: Vec::new(),
            level_of: NONE,
            scan: Scan {
                left: vec![0.0; w],
                with_missing: vec![0.0; w],
                present: Vec::new(),
            },
            best: Best {
                tried: false,
                proxy: f64::NEG_INFINITY,
                feature: 0,
                cut: 0,
                next: 0,
                missing_left: false,
                rows_left: 0.0,
                left: vec![0.0; w],
                set: None,
                random: f64::NAN,
            },
            right: vec![0.0; w],
        }
    }

    /// Adds one row to the sums in `cell`.
    #[inline]
    fn add_row(&self, row: usize, cell: &mut [f64]) {
        let w: f64 = self.weights.map_or(1.0, |w| f64::from(w[row]));
        match self.target {
            Target::Classes { y, .. } => cell[y[row] as usize] += w,
            Target::Regression { y } => {
                cell[0] += w;
                cell[1] += w * y[row];
                cell[2] += w * y[row] * y[row];
            }
            Target::Gradients { g, h, .. } => {
                cell[0] += 1.0;
                cell[1] += f64::from(g[row]);
                if let Some(h) = h.get(row) {
                    cell[2] += f64::from(*h);
                }
            }
        }
        if self.weights.is_some() {
            cell[self.stats] += 1.0;
        }
    }

    /// The sums of `rows`: the root's. Added in blocks of [`BLOCK`] rows, one task each, then
    /// block by block, so the sums are the same on any number of threads.
    fn total(&self, rows: &[u32]) -> Vec<f64> {
        let block = |rows: &[u32]| -> Vec<f64> {
            let mut sums: Vec<f64> = vec![0.0; self.width];
            for &row in rows {
                self.add_row(row as usize, &mut sums);
            }
            sums
        };
        let blocks: Vec<Vec<f64>> = rows.par_chunks(BLOCK).map(block).collect();
        let mut total: Vec<f64> = vec![0.0; self.width];
        for sums in blocks {
            total.iter_mut().zip(sums).for_each(|(t, s)| *t += s);
        }
        total
    }

    /// Whether `rows` are every row of the data, in order, without weights: then a histogram
    /// reads the target itself, not a copy along the rows.
    fn is_whole(&self, rows: &[u32]) -> bool {
        self.weights.is_none() && rows.len() == self.target.rows()
    }

    /// Copies the target and the weight of each of `rows` into `along`, in their order.
    fn gather(&self, rows: &[u32], along: &mut Along) {
        fn copy<T: Copy + Send + Sync>(to: &mut Vec<T>, from: &[T], rows: &[u32], blocks: bool) {
            if blocks {
                rows.par_iter().map(|row| from[*row as usize]).collect_into_vec(to);
            } else {
                to.clear();
                to.extend(rows.iter().map(|row| from[*row as usize]));
            }
        }
        let blocks: bool = rows.len() >= BLOCK;
        match self.target {
            Target::Classes { y, .. } => copy(&mut along.class, y, rows, blocks),
            Target::Regression { y } => copy(&mut along.value, y, rows, blocks),
            Target::Gradients { g, h, .. } => {
                copy(&mut along.gradient, g, rows, blocks);
                if !h.is_empty() {
                    copy(&mut along.hessian, h, rows, blocks);
                }
            }
        }
        if let Some(weights) = self.weights {
            copy(&mut along.weight, weights, rows, blocks);
        }
    }

    /// Adds `rows` to the histogram `h` of feature `f`; `along` holds their targets and
    /// weights ([`Grower::gather`]), unless every row is in `rows`, in order, and none has a
    /// weight ([`Grower::is_whole`]). `mark` sees each row's bucket.
    #[inline]
    fn fill(&self, f: usize, rows: &[u32], along: &Along, h: &mut [f64], mark: impl FnMut(usize)) {
        by_width!(&self.bins.features[f].codes, |codes| self.fill_on(codes, rows, along, h, mark))
    }

    /// [`Grower::fill`] on the codes of one width.
    #[inline]
    fn fill_on<C: Bucket>(
        &self,
        codes: &[C],
        rows: &[u32],
        along: &Along,
        h: &mut [f64],
        mut mark: impl FnMut(usize),
    ) {
        // Every row in order (the root): no row index to follow.
        let whole: bool = self.is_whole(rows);
        match (self.target, self.weights.is_some()) {
            (Target::Classes { y, classes, .. }, false) if whole => {
                for (code, class) in codes.iter().zip(y) {
                    let bin: usize = (*code).into();
                    mark(bin);
                    h[bin * classes + *class as usize] += 1.0;
                }
            }
            (Target::Regression { y }, false) if whole => {
                for (code, value) in codes.iter().zip(y) {
                    let bin: usize = (*code).into();
                    mark(bin);
                    let cell: &mut [f64] = &mut h[bin * 3..bin * 3 + 3];
                    cell[0] += 1.0;
                    cell[1] += value;
                    cell[2] += value * value;
                }
            }
            (Target::Classes { classes, .. }, false) => {
                for (row, class) in rows.iter().zip(&along.class) {
                    let bin: usize = codes[*row as usize].into();
                    mark(bin);
                    h[bin * classes + *class as usize] += 1.0;
                }
            }
            (Target::Regression { .. }, false) => {
                for (row, value) in rows.iter().zip(&along.value) {
                    let bin: usize = codes[*row as usize].into();
                    mark(bin);
                    let cell: &mut [f64] = &mut h[bin * 3..bin * 3 + 3];
                    cell[0] += 1.0;
                    cell[1] += value;
                    cell[2] += value * value;
                }
            }
            (Target::Classes { classes, .. }, true) => {
                let targets = along.class.iter().zip(&along.weight);
                for (row, (class, weight)) in rows.iter().zip(targets) {
                    let bin: usize = codes[*row as usize].into();
                    mark(bin);
                    let cell: &mut [f64] = &mut h[bin * (classes + 1)..(bin + 1) * (classes + 1)];
                    cell[*class as usize] += f64::from(*weight);
                    cell[*classes] += 1.0;
                }
            }
            (Target::Regression { .. }, true) => {
                let targets = along.value.iter().zip(&along.weight);
                for (row, (value, weight)) in rows.iter().zip(targets) {
                    let bin: usize = codes[*row as usize].into();
                    mark(bin);
                    let w: f64 = f64::from(*weight);
                    let cell: &mut [f64] = &mut h[bin * 4..bin * 4 + 4];
                    cell[0] += w;
                    cell[1] += w * value;
                    cell[2] += w * value * value;
                    cell[3] += 1.0;
                }
            }
            // Boosting has no row weights. With one hessian for every row, Σh is the rows.
            (Target::Gradients { g, h: hessians, .. }, _) if whole && hessians.is_empty() => {
                for (code, gradient) in codes.iter().zip(g) {
                    let bin: usize = (*code).into();
                    mark(bin);
                    h[bin * 2] += 1.0;
                    h[bin * 2 + 1] += f64::from(*gradient);
                }
            }
            (Target::Gradients { g, h: hessians, .. }, _) if whole => {
                for (code, (gradient, hessian)) in codes.iter().zip(g.iter().zip(hessians)) {
                    let bin: usize = (*code).into();
                    mark(bin);
                    let cell: &mut [f64] = &mut h[bin * 3..bin * 3 + 3];
                    cell[0] += 1.0;
                    cell[1] += f64::from(*gradient);
                    cell[2] += f64::from(*hessian);
                }
            }
            (Target::Gradients { h: hessians, .. }, _) if hessians.is_empty() => {
                for (row, gradient) in rows.iter().zip(&along.gradient) {
                    let bin: usize = codes[*row as usize].into();
                    mark(bin);
                    h[bin * 2] += 1.0;
                    h[bin * 2 + 1] += f64::from(*gradient);
                }
            }
            (Target::Gradients { .. }, _) => {
                let targets = along.gradient.iter().zip(&along.hessian);
                for (row, (gradient, hessian)) in rows.iter().zip(targets) {
                    let bin: usize = codes[*row as usize].into();
                    mark(bin);
                    let cell: &mut [f64] = &mut h[bin * 3..bin * 3 + 3];
                    cell[0] += 1.0;
                    cell[1] += f64::from(*gradient);
                    cell[2] += f64::from(*hessian);
                }
            }
        }
    }

    /// The histograms of every feature for `rows`. Each bucket adds its rows in row order,
    /// whatever the number of threads.
    fn histogram(&self, rows: &[u32], along: &mut Along, pool: &mut Vec<Vec<f64>>) -> Vec<f64> {
        let (w, features): (usize, usize) = (self.width, self.bins.names.len());
        if !self.is_whole(rows) {
            self.gather(rows, along);
        }
        let along: &Along = along;
        let mut h: Vec<f64> = pool.pop().unwrap_or_default();
        h.clear();
        h.resize(self.bins.offsets[features] * w, 0.0);
        let mut parts: Vec<&mut [f64]> = Vec::with_capacity(features);
        let mut rest: &mut [f64] = &mut h;
        for f in 0..features {
            let slots: usize = self.bins.offsets[f + 1] - self.bins.offsets[f];
            let (part, tail) = std::mem::take(&mut rest).split_at_mut(slots * w);
            parts.push(part);
            rest = tail;
        }
        let wide: usize = match self.growth.max_leaf_nodes {
            Some(_) => LEAF_WIDE,
            None => WIDE,
        };
        // Boosting: a few features per pass over the rows (`fill_gradients`). More per pass is
        // less work, fewer per pass more tasks: enough groups for every thread.
        let group: usize = features.div_ceil(rayon::current_num_threads()).clamp(1, GROUP);
        if matches!(self.target, Target::Gradients { .. }) {
            let mut groups: Vec<(usize, Vec<&mut [f64]>)> = Vec::new();
            for (f, part) in parts.into_iter().enumerate() {
                if f % group == 0 {
                    groups.push((f, Vec::new()));
                }
                groups.last_mut().expect("a group per GROUP features").1.push(part);
            }
            let fill = |(first, mut parts): (usize, Vec<&mut [f64]>)| {
                let mut own: Vec<Vec<f64>> = parts.iter().map(|part| vec![0.0; part.len()]).collect();
                self.fill_gradients(first, rows, along, &mut own);
                for (part, own) in parts.iter_mut().zip(&own) {
                    part.copy_from_slice(own);
                }
            };
            if rows.len() * features >= wide {
                groups.into_par_iter().for_each(fill);
            } else {
                groups.into_iter().for_each(fill);
            }
        } else if rows.len() * features >= wide {
            parts.into_par_iter().enumerate().for_each(|(f, part)| {
                // A buffer of its own while it fills: the histograms of features with few
                // buckets share cache lines, and tasks writing one line slow each other.
                let mut own: Vec<f64> = vec![0.0; part.len()];
                self.fill(f, rows, along, &mut own, |_| {});
                part.copy_from_slice(&own);
            });
        } else {
            for (f, part) in parts.into_iter().enumerate() {
                self.fill(f, rows, along, part, |_| {});
            }
        }
        h
    }

    /// The histograms of the features from `first`, one per entry of `hists`, of a round of
    /// boosting, in one pass over `rows`: a row's gradient is read once for all of them, and the
    /// adds of different features do not wait for each other. A feature with few buckets gets
    /// the same bucket row after row, and alone each add would wait for the one before. Each
    /// bucket still adds its rows in row order, so the sums are those of [`Grower::fill`].
    fn fill_gradients(&self, first: usize, rows: &[u32], along: &Along, hists: &mut [Vec<f64>]) {
        let Target::Gradients { g, h, .. } = self.target else {
            unreachable!("fill_gradients is for boosting");
        };
        let features = &self.bins.features[first..first + hists.len()];
        match &features[0].codes {
            Codes::Narrow(_) => {
                let codes: Vec<&[u8]> = features.iter().map(|b| narrow(&b.codes)).collect();
                self.fill_gradients_on(&codes, rows, along, hists, g, h)
            }
            Codes::Wide(_) => {
                let codes: Vec<&[u16]> = features.iter().map(|b| wide(&b.codes)).collect();
                self.fill_gradients_on(&codes, rows, along, hists, g, h)
            }
        }
    }

    /// [`Grower::fill_gradients`] on the codes of one width.
    fn fill_gradients_on<C: Bucket>(
        &self,
        codes: &[&[C]],
        rows: &[u32],
        along: &Along,
        hists: &mut [Vec<f64>],
        g: &[f32],
        h: &[f32],
    ) {
        let w: usize = self.width;
        if self.is_whole(rows) {
            let rows = g.iter().enumerate();
            match h.is_empty() {
                true => add_group(codes, hists, w, rows.map(|(r, g)| (r, *g, 1.0))),
                false => add_group(codes, hists, w, rows.zip(h).map(|((r, g), h)| (r, *g, *h))),
            }
        } else {
            let rows = rows.iter().map(|row| *row as usize).zip(&along.gradient);
            match h.is_empty() {
                true => add_group(codes, hists, w, rows.map(|(r, g)| (r, *g, 1.0))),
                false => {
                    let rows = rows.zip(&along.hessian).map(|((r, g), h)| (r, *g, *h));
                    add_group(codes, hists, w, rows)
                }
            }
        }
    }

    /// Offers one split to `best`: `left` goes left, the rest of `total` right.
    #[allow(clippy::too_many_arguments)]
    fn consider(
        &self,
        (feature, cut, next): (usize, usize, usize),
        missing_left: bool,
        left: &[f64],
        rows_left: f64,
        total: &[f64],
        rows: f64,
        best: &mut Best,
        random: f64,
    ) {
        let min_leaf: f64 = self.growth.min_samples_leaf as f64;
        best.tried = true;
        if rows_left < min_leaf || rows - rows_left < min_leaf {
            return;
        }
        let proxy: f64 = self
            .target
            .proxy(&left[..self.stats], &total[..self.stats]);
        // LightGBM scans a feature from the right: of equal splits, the highest wins.
        let higher: bool = self.growth.rules == Rules::Lightgbm && ties(proxy, best.proxy) && feature == best.feature;
        if self.beats(proxy, best.proxy) || higher {
            (best.proxy, best.feature, best.cut, best.next) = (proxy, feature, cut, next);
            (best.missing_left, best.rows_left) = (missing_left, rows_left);
            best.left.copy_from_slice(left);
            (best.set, best.random) = (None, random);
        }
    }

    /// [`Grower::scan`] for boosting, step for step scikit-learn's histogram boosting. First
    /// from the left, with the missing values on the right: a split lies on the bucket
    /// boundary after the last bucket with rows on the left. Then, when the node has missing
    /// values, from the right with them on the left, on the boundary before the first bucket
    /// with rows on the right; such a split wins only when it is better. A node with no missing
    /// value sends one right when the feature had some at fit, else to the larger side.
    #[allow(clippy::too_many_arguments)]
    fn scan_boosting(
        &self,
        f: usize,
        h: &[f64],
        bins: impl Iterator<Item = usize>,
        total: &[f64],
        rows: f64,
        sums: &mut Scan,
        best: &mut Best,
    ) {
        let w: usize = self.width;
        let binned: &Binned = &self.bins.features[f];
        let real: usize = binned.edges.len();
        let missing: bool = h.get(real * w..(real + 1) * w).is_some_and(|cell| self.rows_of(cell) > 0.0);
        sums.present.clear();
        sums.left.fill(0.0);
        let (mut rows_left, mut prev): (f64, usize) = (0.0, NONE);
        // LightGBM keeps no counts: a bucket has (its Σh) × rows / (node's Σh) rows, rounded,
        // and the left side the node's rows minus the right side's buckets.
        let lightgbm: bool = self.growth.rules == Rules::Lightgbm;
        let factor: f64 = rows / self.target.hessian(total);
        let rounded = |cell: &[f64]| (self.target.hessian(cell) * factor + 0.5).floor();
        let all_rounded: f64 = if lightgbm { h.chunks(w).map(rounded).sum() } else { 0.0 };
        let mut left_rounded: f64 = 0.0;
        for bin in bins {
            if bin >= real {
                break;
            }
            let cell: &[f64] = &h[bin * w..(bin + 1) * w];
            let count: f64 = self.rows_of(cell);
            if count == 0.0 {
                continue;
            }
            if prev != NONE {
                let counted: f64 = if lightgbm { rows - (all_rounded - left_rounded) } else { rows_left };
                let (larger, at): (bool, (usize, usize, usize)) = match self.growth.rules {
                    Rules::Sklearn => (!missing && !binned.missing && rows_left > rows - rows_left, (f, prev, prev + 1)),
                    Rules::Xgboost => (false, (f, prev, prev + 1)),
                    Rules::Lightgbm => (!missing && !binned.missing && rows_left > rows - rows_left, (f, bin - 1, bin)),
                };
                self.consider(at, larger, &sums.left, counted, total, rows, best, f64::NAN);
            }
            sums.left.iter_mut().zip(cell).for_each(|(sum, value)| *sum += value);
            rows_left += count;
            left_rounded += if lightgbm { rounded(cell) } else { 0.0 };
            prev = bin;
            sums.present.push(bin);
        }
        if !missing || prev == NONE {
            return;
        }
        // Every value left, only the missing ones right.
        self.consider((f, prev, NONE), false, &sums.left, rows_left, total, rows, best, f64::NAN);
        // From the right: the buckets from the `j`-th present one on go right, the rest and
        // the missing values left (the left sums are the node's minus the right's, as in
        // scikit-learn).
        let right: &mut Vec<f64> = &mut sums.with_missing;
        right.fill(0.0);
        let mut rows_right: f64 = 0.0;
        for j in (0..sums.present.len()).rev() {
            let bin: usize = sums.present[j];
            let cell: &[f64] = &h[bin * w..(bin + 1) * w];
            right.iter_mut().zip(cell).for_each(|(sum, value)| *sum += value);
            rows_right += self.rows_of(cell);
            if bin == 0 {
                break;
            }
            sums.left.iter_mut().zip(total.iter().zip(right.iter())).for_each(|(l, (t, r))| *l = t - r);
            self.consider((f, bin - 1, bin), true, &sums.left, rows - rows_right, total, rows, best, f64::NAN);
        }
    }

    /// Offers every split of categorical feature `f` into two sets of categories to `best`,
    /// from its histogram `h`, like scikit-learn's histogram boosting and LightGBM. The
    /// categories with enough support (10 rows, or that much hessian) are sorted by
    /// Σg / (Σh + 10); the splits are the first `i` of that order on the left, or the last
    /// `i`, up to half of them. A category with little support always goes right; the missing
    /// values are one more category.
    #[allow(clippy::too_many_arguments)]
    fn scan_categories(
        &self,
        f: usize,
        h: &[f64],
        bins: impl Iterator<Item = usize>,
        total: &[f64],
        rows: f64,
        sums: &mut Scan,
        best: &mut Best,
    ) {
        const SUPPORT: f64 = 10.0;
        let w: usize = self.width;
        let missing: usize = self.bins.features[f].edges.len();
        // scikit-learn's `support_factor`: 1 when every hessian is 1.
        let factor: f64 = rows / self.target.hessian(total);
        let mut order: Vec<(f64, usize)> = Vec::new();
        for bin in bins {
            let cell: &[f64] = &h[bin * w..(bin + 1) * w];
            let hessian: f64 = self.target.hessian(cell);
            if self.rows_of(cell) > 0.0 && hessian * factor >= SUPPORT {
                order.push((cell[1] / (hessian + SUPPORT), bin));
            }
        }
        if order.len() <= 1 {
            return;
        }
        // The missing bucket is last of the buckets, so a tie keeps it last.
        order.sort_by(|a, b| a.0.total_cmp(&b.0));
        let n: usize = order.len();
        for forward in [true, false] {
            let middle: usize = if forward { n.div_ceil(2) } else { n.div_ceil(2) - 1 };
            sums.left.fill(0.0);
            let (mut rows_left, mut set): (f64, Left) = (0.0, [0; 4]);
            for i in 0..middle {
                let bin: usize = order[if forward { i } else { n - 1 - i }].1;
                let cell: &[f64] = &h[bin * w..(bin + 1) * w];
                sums.left.iter_mut().zip(cell).for_each(|(sum, value)| *sum += value);
                rows_left += self.rows_of(cell);
                set[bin >> 6] |= 1 << (bin & 63);
                let min_leaf: f64 = self.growth.min_samples_leaf as f64;
                best.tried = true;
                if rows_left < min_leaf || rows - rows_left < min_leaf {
                    continue;
                }
                let proxy: f64 = self.target.proxy(&sums.left[..self.stats], &total[..self.stats]);
                if self.beats(proxy, best.proxy) {
                    (best.proxy, best.feature, best.rows_left) = (proxy, f, rows_left);
                    // A feature with no missing value at fit sends one to the larger side.
                    best.missing_left = match self.bins.features[f].missing {
                        true => set[missing >> 6] >> (missing & 63) & 1 == 1,
                        false => rows_left > rows - rows_left,
                    };
                    best.left.copy_from_slice(&sums.left);
                    best.random = f64::NAN;
                best.set = Some(set);
                }
            }
        }
    }

    /// Offers every split of feature `f` to `best`, from its histogram `h`. `bins` gives the
    /// buckets to read, ascending. A split lies between two buckets that have rows; missing
    /// values are tried on both sides, and on the right alone, like scikit-learn.
    #[allow(clippy::too_many_arguments)]
    fn scan(
        &self,
        f: usize,
        h: &[f64],
        bins: impl Iterator<Item = usize>,
        total: &[f64],
        rows: f64,
        sums: &mut Scan,
        best: &mut Best,
    ) {
        if self.bins.categorical[f] {
            return self.scan_categories(f, h, bins, total, rows, sums, best);
        }
        if matches!(self.target, Target::Gradients { .. }) {
            return self.scan_boosting(f, h, bins, total, rows, sums, best);
        }
        if self.growth.random_splits {
            return self.scan_random(f, h, bins, total, rows, sums, best);
        }
        let w: usize = self.width;
        let real: usize = self.bins.features[f].edges.len();
        let missing: Option<&[f64]> = h
            .get(real * w..(real + 1) * w)
            .filter(|cell| self.rows_of(cell) > 0.0);
        let rows_missing: f64 = missing.map_or(0.0, |cell| self.rows_of(cell));
        sums.left.fill(0.0);
        if let Some(cell) = missing {
            sums.with_missing.copy_from_slice(cell);
        }
        let mut rows_left: f64 = 0.0;
        let mut prev: usize = NONE;
        for bin in bins {
            if bin >= real {
                break;
            }
            let cell: &[f64] = &h[bin * w..(bin + 1) * w];
            let count: f64 = self.rows_of(cell);
            if count == 0.0 {
                continue;
            }
            if prev != NONE {
                let at: (usize, usize, usize) = (f, prev, bin);
                if missing.is_some() {
                    self.consider(at, false, &sums.left, rows_left, total, rows, best, f64::NAN);
                    let with: f64 = rows_left + rows_missing;
                    self.consider(at, true, &sums.with_missing, with, total, rows, best, f64::NAN);
                } else {
                    // No missing value here: one met later goes to the larger side.
                    let larger: bool = rows_left > rows - rows_left;
                    self.consider(at, larger, &sums.left, rows_left, total, rows, best, f64::NAN);
                }
            }
            for (sum, value) in sums.left.iter_mut().zip(cell) {
                *sum += value;
            }
            if missing.is_some() {
                for (sum, value) in sums.with_missing.iter_mut().zip(cell) {
                    *sum += value;
                }
            }
            rows_left += count;
            prev = bin;
        }
        if missing.is_some() && prev != NONE {
            let at: (usize, usize, usize) = (f, prev, NONE);
            self.consider(at, false, &sums.left, rows_left, total, rows, best, f64::NAN);
        }
    }

    /// [`Grower::scan`] for Extra Trees, scikit-learn's `RandomSplitter`: one threshold drawn
    /// uniformly between the smallest and the largest value of feature `f` among the node's
    /// rows (a feature with one value offers nothing), the missing values to a side drawn at
    /// random. The draw is seeded by the tree's seed, the feature and the node's sums, so the
    /// tree is the same on any number of threads.
    #[allow(clippy::too_many_arguments)]
    fn scan_random(
        &self,
        f: usize,
        h: &[f64],
        bins: impl Iterator<Item = usize>,
        total: &[f64],
        rows: f64,
        sums: &mut Scan,
        best: &mut Best,
    ) {
        let w: usize = self.width;
        let edges: &[(f64, f64)] = &self.bins.features[f].edges;
        let real: usize = edges.len();
        let present: &mut Vec<usize> = &mut sums.present;
        present.clear();
        present.extend(bins.take_while(|b| *b < real).filter(|b| self.rows_of(&h[b * w..(b + 1) * w]) > 0.0));
        if present.len() < 2 {
            return;
        }
        let mut key: u64 = self.growth.seed ^ (f as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        for value in total {
            key = (key ^ value.to_bits()).wrapping_mul(0x0100_0000_01B3).rotate_left(29);
        }
        let mut rng: Rng = Rng::new(key);
        let (low, high): (f64, f64) = (edges[present[0]].0, edges[present[present.len() - 1]].1);
        let unit: f64 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let mut t: f64 = low + unit * (high - low);
        if t >= high {
            t = low;
        }
        // The last bucket with every value at or below t, never the last bucket with rows.
        let at: usize = present.partition_point(|b| edges[*b].1 <= t).clamp(1, present.len() - 1) - 1;
        let (cut, next): (usize, usize) = (present[at], present[at + 1]);
        let inside: bool = edges[cut].1 <= t && t < edges[next].0;
        sums.left.fill(0.0);
        let mut rows_left: f64 = 0.0;
        for b in &present[..=at] {
            let cell: &[f64] = &h[b * w..(b + 1) * w];
            rows_left += self.rows_of(cell);
            for (sum, value) in sums.left.iter_mut().zip(cell) {
                *sum += value;
            }
        }
        let missing: Option<&[f64]> = h.get(real * w..(real + 1) * w).filter(|cell| self.rows_of(cell) > 0.0);
        let missing_left: bool = match missing {
            Some(cell) if rng.next_u64() & 1 == 1 => {
                rows_left += self.rows_of(cell);
                for (sum, value) in sums.left.iter_mut().zip(cell) {
                    *sum += value;
                }
                true
            }
            Some(_) => false,
            None => rows_left > rows - rows_left,
        };
        let threshold: f64 = if inside { t } else { f64::NAN };
        self.consider((f, cut, next), missing_left, &sums.left, rows_left, total, rows, best, threshold);
    }

    /// Offers every split of feature `f` to `best`, from the rows: their histogram is made in
    /// `h`, read and cleared. A node with fewer rows than buckets marks the buckets its rows
    /// are in and touches only those, so a node of 10 rows never reads 255 buckets.
    #[allow(clippy::too_many_arguments)]
    fn search_rows(
        &self,
        f: usize,
        rows: &[u32],
        along: &Along,
        total: &[f64],
        h: &mut [f64],
        sums: &mut Scan,
        best: &mut Best,
    ) {
        let w: usize = self.width;
        let slots: usize = self.bins.offsets[f + 1] - self.bins.offsets[f];
        if let ([a, b], false) = (rows, self.bins.categorical[f]) {
            // Two rows, as in half of the nodes of a full tree: no histogram. The one split
            // is between them, with the row of the lower bucket on the left (a missing value
            // has the highest), and it scores the same on every feature.
            let binned: &Binned = &self.bins.features[f];
            let (a, b): (usize, usize) = (*a as usize, *b as usize);
            let (low, cut, next): (usize, usize, usize) =
                match (binned.codes.get(a), binned.codes.get(b)) {
                    (x, y) if x < y => (a, x, y),
                    (x, y) if x > y => (b, y, x),
                    _ => return,
                };
            let next: usize = if next == binned.edges.len() { NONE } else { next };
            sums.left.fill(0.0);
            self.add_row(low, &mut sums.left);
            self.consider((f, cut, next), false, &sums.left, 1.0, total, 2.0, best, f64::NAN);
        } else if rows.len() >= slots {
            let h: &mut [f64] = &mut h[..slots * w];
            self.fill(f, rows, along, h, |_| {});
            self.scan(f, h, 0..slots, total, rows.len() as f64, sums, best);
            h.fill(0.0);
        } else {
            let mut seen: [u64; 4] = [0; 4];
            self.fill(f, rows, along, h, |bin| seen[bin >> 6] |= 1 << (bin & 63));
            // One bucket only: nothing to split on, as for most features of a small node.
            if seen.iter().map(|word| word.count_ones()).sum::<u32>() > 1 {
                self.scan(f, h, marked(seen), total, rows.len() as f64, sums, best);
            }
            for bin in marked(seen) {
                h[bin * w..(bin + 1) * w].fill(0.0);
            }
        }
    }

    /// The threshold of a split: halfway between the largest value on the left and the
    /// smallest on the right, as scikit-learn computes it.
    fn threshold(&self, best: &Best) -> f64 {
        if best.random.is_finite() {
            return best.random;
        }
        if best.next == NONE {
            // Every value goes left and only the missing ones right: any value passes.
            return f64::MAX;
        }
        let edges: &[(f64, f64)] = &self.bins.features[best.feature].edges;
        let (below, above): (f64, f64) = (edges[best.cut].1, edges[best.next].0);
        if matches!(self.target, Target::Gradients { .. }) && self.growth.rules == Rules::Xgboost {
            // A row goes left below the smallest value on the right: at most the float before it.
            return above.next_down();
        }
        if matches!(self.target, Target::Gradients { .. }) {
            // scikit-learn's bucket boundary: the mean of the two neighbouring values.
            let mean: f64 = (below + above) / 2.0;
            return if mean.is_finite() { mean } else { below / 2.0 + above / 2.0 };
        }
        let middle: f64 = below / 2.0 + above / 2.0;
        if middle == above || middle.is_infinite() {
            below
        } else {
            middle
        }
    }

    /// Moves the rows that go left to the front of `rows`, in order, and returns how many.
    /// A large node does it in blocks, one task each, then joins the blocks' parts in order.
    fn partition(&self, rows: &mut [u32], spare: &mut [u32], best: &Best) -> usize {
        let binned: &Binned = &self.bins.features[best.feature];
        match best.set {
            Some(set) => {
                let in_set = move |code: usize| set[code >> 6] >> (code & 63) & 1 == 1;
                by_width!(&binned.codes, |codes| partition_by(codes, rows, spare, in_set))
            }
            None => {
                let missing: usize = binned.edges.len();
                let (cut, missing_left): (usize, bool) = (best.cut, best.missing_left);
                let below = move |code: usize| (code <= cut) | ((code == missing) & missing_left);
                by_width!(&binned.codes, |codes| partition_by(codes, rows, spare, below))
            }
        }
    }

    /// Finds the best split of the node that holds `rows`, into `s.best`, with the sums of
    /// its right side in `s.right`. `place` is where its rows are among the rows of the whole
    /// tree, which names a node whatever task it is in. Gives what the split takes off,
    /// scikit-learn's impurity decrease weighted by the node's share of the weight, or `None`
    /// with the gain itself (what the split takes off the loss, unweighted), or `None`
    /// when the node stays a leaf: no split leaves `min_samples_leaf` rows on both sides, or
    /// the best one takes off too little. A split that takes off nothing is still made in a
    /// classification or regression tree, like scikit-learn's, because the splits below it may.
    fn search(
        &self,
        rows: &[u32],
        place: (usize, usize),
        depth: usize,
        total: &[f64],
        hist: Option<&[f64]>,
        s: &mut Scratch,
    ) -> Option<(f64, f64)> {
        let w: usize = self.width;
        let n: f64 = rows.len() as f64;
        let Scratch {
            small,
            along,
            order,
            drawn,
            level,
            level_of,
            scan,
            best,
            right,
            ..
        } = s;
        (best.tried, best.proxy) = (false, f64::NEG_INFINITY);
        // Two rows need no histogram, so none of this.
        if hist.is_none() && rows.len() > 2 {
            self.gather(rows, along);
        }
        let along: &Along = along;
        let mut visit = |f: usize, best: &mut Best| match hist {
            Some(h) => {
                let (from, to) = (self.bins.offsets[f] * w, self.bins.offsets[f + 1] * w);
                self.scan(f, &h[from..to], 0..(to - from) / w, total, n, scan, best);
            }
            None => self.search_rows(f, rows, along, total, small, scan, best),
        };
        // The features this node may split on: the tree's, of those a draw per depth, then a
        // draw per node.
        let g: &Growth = self.growth;
        if g.by_level < 1.0 && *level_of != depth {
            let mut rng: Rng = Rng::new(g.seed ^ (depth as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            *level = draw_features(order, g.rules.count(order.len(), g.by_level), &mut rng);
            *level_of = depth;
        }
        let pool: &mut Vec<u32> = if g.by_level < 1.0 { level } else { order };
        let features: usize = pool.len();
        let take: usize = match g.by_node < 1.0 {
            true => g.rules.count(features, g.by_node),
            false => features,
        };
        let take: usize = take.min(g.max_features);
        if take >= features
            && let (Some(h), Some(_)) = (hist, self.growth.max_leaf_nodes)
        {
            // A tree that grows leaf by leaf waits for each node, so its features are read
            // as tasks. Each finds its best split; the first of the best wins, as in order.
            let start: &Best = best;
            let found: Vec<Best> = (pool.par_iter())
                .map(|f| {
                    let f: usize = *f as usize;
                    let mut sums: Scan = Scan {
                        left: vec![0.0; w],
                        with_missing: vec![0.0; w],
                        present: Vec::new(),
                    };
                    let mut own: Best = start.clone();
                    let (from, to) = (self.bins.offsets[f] * w, self.bins.offsets[f + 1] * w);
                    self.scan(f, &h[from..to], 0..(to - from) / w, total, n, &mut sums, &mut own);
                    own
                })
                .collect();
            for own in found {
                let tried: bool = best.tried || own.tried;
                if self.beats(own.proxy, best.proxy) {
                    *best = own;
                }
                best.tried = tried;
            }
        } else if take >= features {
            for f in pool.iter() {
                visit(*f as usize, best);
            }
        } else {
            // A draw without repeats, seeded by the node, so the tree is the same on any
            // number of threads. Like scikit-learn's trees, it goes on past `max_features`
            // while every feature drawn so far has one value only among the node's rows.
            let key: u64 = (place.0 as u64) << 32 | place.1 as u64;
            let mut rng: Rng = Rng::new(self.growth.seed.wrapping_add(key));
            drawn.clear();
            for i in 0..features {
                let j: usize = i + (rng.next_u64() % (features - i) as u64) as usize;
                pool.swap(i, j);
                drawn.push(j);
                visit(pool[i] as usize, best);
                // Boosting draws exactly `take`, as scikit-learn's histogram boosting.
                let exact: bool = matches!(self.target, Target::Gradients { .. });
                if i + 1 >= take && (best.tried || exact) {
                    break;
                }
            }
            for (i, j) in drawn.iter().enumerate().rev() {
                pool.swap(i, *j);
            }
        }
        if best.proxy == f64::NEG_INFINITY {
            return None;
        }
        for ((r, t), l) in right.iter_mut().zip(total).zip(&best.left) {
            *r = t - l;
        }
        let cost = |cell: &[f64]| self.target.cost(&cell[..self.stats]);
        let gain: f64 = cost(total) - cost(&best.left) - cost(right);
        let decrease: f64 = gain / self.weight;
        let enough: bool = match self.target {
            // LightGBM's rule: the children's gains above the node's plus `min_split_gain`.
            Target::Gradients { step, .. } if step.rules == Rules::Lightgbm => {
                best.proxy > step.leaf_gain(total[1], self.target.hessian(total)) + step.min_gain
            }
            // Boosting's rule (scikit-learn's): the gain itself, above 0.
            Target::Gradients { step, .. } => gain > step.min_gain,
            _ => {
                decrease + f64::EPSILON >= self.growth.min_impurity_decrease
                    && (decrease > 0.0 || self.target.splits_on_nothing())
            }
        };
        enough.then_some((decrease, gain))
    }

    /// The histograms of the two sides of a node whose `rows` were just put in two parts at
    /// `mid`: the smaller side's from its rows, the larger one's as the node's own (`parent`)
    /// minus that. Only when the larger side will read them.
    #[allow(clippy::too_many_arguments)]
    fn side_hists(
        &self,
        rows: &[u32],
        mid: usize,
        depth: usize,
        parent: Option<Vec<f64>>,
        (left, right): (&[f64], &[f64]),
        along: &mut Along,
        pool: &mut Vec<Vec<f64>>,
    ) -> (Option<Vec<f64>>, Option<Vec<f64>>) {
        let Some(mut parent) = parent else {
            return (None, None);
        };
        let left_small: bool = mid <= rows.len() - mid;
        let (few, many, many_sums) = if left_small {
            (&rows[..mid], rows.len() - mid, right)
        } else {
            (&rows[mid..], mid, left)
        };
        if !self.is_dense(many) || self.is_leaf(many, depth + 1, many_sums) {
            pool.push(parent);
            return (None, None);
        }
        let side: Vec<f64> = self.histogram(few, along, pool);
        for (p, c) in parent.iter_mut().zip(&side) {
            *p -= c;
        }
        if left_small {
            (Some(side), Some(parent))
        } else {
            (Some(parent), Some(side))
        }
    }

    /// Makes node `at` of `nodes` the split in `best` and puts the node's rows in two parts;
    /// gives where the right part starts. A categorical split keeps its set of categories
    /// (without the missing values' bucket: a missing value follows `missing_left`).
    fn split(&self, nodes: &mut Nodes, at: usize, rows: &mut [u32], spare: &mut [u32], best: &Best) -> usize {
        let node: &mut Node = &mut nodes.nodes[at];
        node.split = (best.feature as u32) << 1 | u32::from(best.missing_left);
        match best.set {
            Some(mut set) => {
                let missing: usize = self.bins.features[best.feature].edges.len();
                set[missing >> 6] &= !(1 << (missing & 63));
                node.split |= CATEGORICAL;
                node.threshold = nodes.sets.len() as f64;
                nodes.sets.push(set);
            }
            None => node.threshold = self.threshold(best),
        }
        self.partition(rows, spare, best)
    }

    /// Grows the subtree of the node that holds `rows`, whose sums are `sums`. `base` is the
    /// place of `rows` among the rows of the whole tree. The nodes wait on a stack, not on the
    /// call stack: a tree can be thousands of levels deep.
    fn subtree(
        &self,
        rows: &mut [u32],
        base: usize,
        sums: &[f64],
        depth: usize,
        hist: Option<Vec<f64>>,
    ) -> Nodes {
        let w: usize = self.width;
        let mut out: Nodes = Nodes::default();
        let mut s: Scratch = self.scratch(rows.len());
        // The sums of the waiting nodes, in stack order.
        let mut waiting: Vec<f64> = sums.to_vec();
        let mut stack: Vec<Work> = vec![Work {
            lo: 0,
            hi: rows.len(),
            depth,
            right_of: NONE,
            hist,
        }];
        let mut total: Vec<f64> = vec![0.0; w];
        while let Some(work) = stack.pop() {
            let Work { lo, hi, depth, .. } = work;
            let top: usize = waiting.len() - w;
            total.copy_from_slice(&waiting[top..]);
            waiting.truncate(top);
            let at: usize = out.leaf(self.target, &total[..self.stats], hi - lo);
            if work.right_of != NONE {
                out.nodes[work.right_of].right = (at - work.right_of) as u32;
            }
            if self.is_leaf(hi - lo, depth, &total) {
                s.pool.extend(work.hist);
                continue;
            }
            let hist: Option<Vec<f64>> = work.hist.or_else(|| {
                self.is_dense(hi - lo)
                    .then(|| self.histogram(&rows[lo..hi], &mut s.along, &mut s.pool))
            });
            let place: (usize, usize) = (base + lo, base + hi);
            let Some((decrease, _)) = self.search(&rows[lo..hi], place, depth, &total, hist.as_deref(), &mut s)
            else {
                s.pool.extend(hist);
                continue;
            };
            out.decrease[at] = decrease;
            let Scratch {
                spare,
                along,
                pool,
                best,
                right,
                ..
            } = &mut s;
            let mid: usize = lo + self.split(&mut out, at, &mut rows[lo..hi], spare, best);
            let sides: (&[f64], &[f64]) = (&best.left, right);
            let (left_hist, right_hist) =
                self.side_hists(&rows[lo..hi], mid - lo, depth, hist, sides, along, pool);
            if mid - lo >= self.spawn && hi - mid >= self.spawn {
                let (left_rows, right_rows) = rows[lo..hi].split_at_mut(mid - lo);
                let (left, right) = rayon::join(
                    || self.subtree(left_rows, base + lo, sides.0, depth + 1, left_hist),
                    || self.subtree(right_rows, base + mid, sides.1, depth + 1, right_hist),
                );
                out.nodes[at].right = 1 + left.nodes.len() as u32;
                out.append(left);
                out.append(right);
            } else {
                waiting.extend_from_slice(sides.1);
                waiting.extend_from_slice(sides.0);
                stack.push(Work {
                    lo: mid,
                    hi,
                    depth: depth + 1,
                    right_of: at,
                    hist: right_hist,
                });
                stack.push(Work {
                    lo,
                    hi: mid,
                    depth: depth + 1,
                    right_of: NONE,
                    hist: left_hist,
                });
            }
        }
        out
    }

    /// Adds the leaf that holds `rows[lo..hi]` to a tree that grows leaf by leaf, finds its
    /// split and puts it in the queue; gives the leaf's place.
    #[allow(clippy::too_many_arguments)]
    fn open(
        &self,
        rows: &[u32],
        (lo, hi): (usize, usize),
        depth: usize,
        sums: &[f64],
        hist: Option<Vec<f64>>,
        tree: &mut Loose,
        s: &mut Scratch,
    ) -> usize {
        let at: usize = tree.nodes.leaf(self.target, &sums[..self.stats], hi - lo);
        tree.children.push((NONE, NONE));
        tree.open.push(None);
        if self.is_leaf(hi - lo, depth, sums) {
            s.pool.extend(hist);
            return at;
        }
        let hist: Option<Vec<f64>> = hist.or_else(|| {
            self.is_dense(hi - lo)
                .then(|| self.histogram(&rows[lo..hi], &mut s.along, &mut s.pool))
        });
        match self.search(&rows[lo..hi], (lo, hi), depth, sums, hist.as_deref(), s) {
            Some((decrease, gain)) => {
                tree.queue.push((gain, at));
                tree.open[at] = Some(Open {
                    lo,
                    hi,
                    depth,
                    hist,
                    best: s.best.clone(),
                    right: s.right.clone(),
                    decrease,
                });
            }
            None => s.pool.extend(hist),
        }
        at
    }

    /// Grows a tree of at most `leaves` leaves, leaf by leaf: the open leaf whose split takes
    /// most off is split next, like scikit-learn's best-first trees and its boosting. One
    /// task; the histograms of a large node still fill one feature per task.
    fn leaf_wise(&self, rows: &mut [u32], sums: &[f64], leaves: usize) -> Nodes {
        let mut s: Scratch = self.scratch(rows.len());
        let mut tree: Loose = Loose::default();
        self.open(rows, (0, rows.len()), 0, sums, None, &mut tree, &mut s);
        let mut count: usize = 1;
        tree.queue.python = matches!(self.target, Target::Gradients { .. });
        while count < leaves
            && let Some((_, at)) = tree.queue.pop()
        {
            let Some(leaf) = tree.open[at].take() else {
                continue;
            };
            let Open { lo, hi, depth, .. } = leaf;
            tree.nodes.decrease[at] = leaf.decrease;
            let mid: usize = lo + self.split(&mut tree.nodes, at, &mut rows[lo..hi], &mut s.spare, &leaf.best);
            let sides: (&[f64], &[f64]) = (&leaf.best.left, &leaf.right);
            let (left_hist, right_hist) = self.side_hists(
                &rows[lo..hi],
                mid - lo,
                depth,
                leaf.hist,
                sides,
                &mut s.along,
                &mut s.pool,
            );
            let left: usize = self.open(rows, (lo, mid), depth + 1, sides.0, left_hist, &mut tree, &mut s);
            let right: usize = self.open(rows, (mid, hi), depth + 1, sides.1, right_hist, &mut tree, &mut s);
            tree.children[at] = (left, right);
            count += 1;
        }
        // The nodes in the order of every tree: a node, its left subtree, its right subtree.
        let outputs: usize = tree.nodes.value.len() / tree.nodes.nodes.len();
        let mut out: Nodes = Nodes::default();
        let mut stack: Vec<(usize, usize)> = vec![(0, NONE)];
        while let Some((from, right_of)) = stack.pop() {
            let at: usize = out.nodes.len();
            if right_of != NONE {
                out.nodes[right_of].right = (at - right_of) as u32;
            }
            out.nodes.push(tree.nodes.nodes[from]);
            out.value
                .extend_from_slice(&tree.nodes.value[from * outputs..][..outputs]);
            out.samples.push(tree.nodes.samples[from]);
            out.decrease.push(tree.nodes.decrease[from]);
            if tree.nodes.nodes[from].split != LEAF {
                stack.push((tree.children[from].1, at));
                stack.push((tree.children[from].0, NONE));
            }
        }
        // The nodes moved, the sets did not: a split still names its set by its place.
        out.sets = std::mem::take(&mut tree.nodes.sets);
        out
    }
}

/// The codes of a [`Bins`] whose `max_bins` is at most 255.
fn narrow(codes: &Codes) -> &[u8] {
    match codes {
        Codes::Narrow(c) => c,
        Codes::Wide(_) => unreachable!("every feature of one Bins has the same width"),
    }
}

/// The codes of a [`Bins`] whose `max_bins` is above 255.
fn wide(codes: &Codes) -> &[u16] {
    match codes {
        Codes::Wide(c) => c,
        Codes::Narrow(_) => unreachable!("every feature of one Bins has the same width"),
    }
}

/// Moves the rows of `rows` whose bucket in `codes` goes left to the front, in order, and
/// gives how many. A large node does it in blocks, one task each, then joins the blocks'
/// parts in order.
fn partition_by<C: Bucket>(
    codes: &[C],
    rows: &mut [u32],
    spare: &mut [u32],
    goes_left: impl Fn(usize) -> bool + Sync,
) -> usize {
    // No branch: each row is written to both sides and one counter moves. The left rows end
    // at the front of `rows`, the right ones at the front of `spare`.
    let part = |rows: &mut [u32], spare: &mut [u32]| -> (usize, usize) {
        let (mut left, mut right): (usize, usize) = (0, 0);
        for i in 0..rows.len() {
            let row: u32 = rows[i];
            let goes_left: bool = goes_left(codes[row as usize].into());
            rows[left] = row;
            spare[right] = row;
            left += usize::from(goes_left);
            right += usize::from(!goes_left);
        }
        (left, right)
    };
    if rows.len() < BLOCK {
        let (left, right): (usize, usize) = part(rows, spare);
        rows[left..].copy_from_slice(&spare[..right]);
        return left;
    }
    let n: usize = rows.len();
    let parts: Vec<(usize, usize)> = (rows.par_chunks_mut(BLOCK))
        .zip(spare[..n].par_chunks_mut(BLOCK))
        .map(|(rows, spare)| part(rows, spare))
        .collect();
    // A block's left rows only move towards the front, past blocks already moved.
    let mut left: usize = 0;
    for (b, (count, _)) in parts.iter().enumerate() {
        rows.copy_within(b * BLOCK..b * BLOCK + count, left);
        left += count;
    }
    let mut at: usize = left;
    for (b, (_, count)) in parts.iter().enumerate() {
        rows[at..at + count].copy_from_slice(&spare[b * BLOCK..b * BLOCK + count]);
        at += count;
    }
    left
}

/// Adds `rows` (row, gradient, hessian) to the histograms `hists`, `w` sums per bucket, of
/// the features whose buckets are `codes`, all in one pass ([`Grower::fill_gradients`]). Two
/// sums are the rows and Σg (every hessian 1); three add Σh.
fn add_group<C: Bucket>(
    codes: &[&[C]],
    hists: &mut [Vec<f64>],
    w: usize,
    rows: impl Iterator<Item = (usize, f32, f32)>,
) {
    fn run<C: Bucket, const N: usize, const W: usize>(
        codes: &[&[C]],
        hists: &mut [Vec<f64>],
        rows: impl Iterator<Item = (usize, f32, f32)>,
    ) {
        let codes: [&[C]; N] = codes.try_into().expect("N features");
        let mut each = hists.iter_mut();
        let hists: [&mut [f64]; N] =
            std::array::from_fn(|_| each.next().expect("N histograms").as_mut_slice());
        for (row, g, h) in rows {
            let (g, h): (f64, f64) = (f64::from(g), f64::from(h));
            for k in 0..N {
                let at: usize = codes[k][row].into() * W;
                let cell: &mut [f64] = &mut hists[k][at..at + W];
                cell[0] += 1.0;
                cell[1] += g;
                if W == 3 {
                    cell[2] += h;
                }
            }
        }
    }
    match (codes.len(), w) {
        (1, 2) => run::<C, 1, 2>(codes, hists, rows),
        (2, 2) => run::<C, 2, 2>(codes, hists, rows),
        (3, 2) => run::<C, 3, 2>(codes, hists, rows),
        (4, 2) => run::<C, 4, 2>(codes, hists, rows),
        (1, 3) => run::<C, 1, 3>(codes, hists, rows),
        (2, 3) => run::<C, 2, 3>(codes, hists, rows),
        (3, 3) => run::<C, 3, 3>(codes, hists, rows),
        (4, 3) => run::<C, 4, 3>(codes, hists, rows),
        _ => unreachable!("a group has 1 to 4 features, a bucket 2 or 3 sums"),
    }
}

/// A grown tree, scikit-learn's `feature_importances_` (the impurity each feature's splits
/// took off, as shares that sum to 1; all 0 for a tree that is one leaf), and the rows the
/// tree grew on, leaf after leaf in the order of the tree's leaves: the first
/// `n_node_samples` of them are the rows of the first leaf.
pub(crate) struct Grown {
    pub tree: Tree,
    pub importances: Vec<f64>,
    pub rows: Vec<u32>,
}

/// Grows one tree on the Polars pool. `weights` gives each row of the data its weight (a row
/// of weight 0 is left out); `None` counts every row once. `rows` are the rows to grow on, in
/// rising order; `None` is every row with a weight. The tree is the same on any number of
/// threads.
pub(crate) fn grow(
    bins: &Bins,
    target: &Target,
    growth: &Growth,
    weights: Option<&[f32]>,
    rows: Option<Vec<u32>>,
) -> Grown {
    grow_with(bins, target, growth, weights, rows, SPAWN, true)
}

fn grow_with(
    bins: &Bins,
    target: &Target,
    growth: &Growth,
    weights: Option<&[f32]>,
    rows: Option<Vec<u32>>,
    spawn: usize,
    dense: bool,
) -> Grown {
    let all = 0..target.rows() as u32;
    let mut rows: Vec<u32> = match (rows, weights) {
        (Some(rows), _) => rows,
        (None, Some(weights)) => all.filter(|row| weights[*row as usize] > 0.0).collect(),
        (None, None) => all.collect(),
    };
    let stats: usize = target.stats();
    let mut grower: Grower = Grower {
        bins,
        target,
        growth,
        weights,
        stats,
        width: stats + usize::from(weights.is_some()),
        weight: 0.0,
        spawn,
        dense,
    };
    let total: Vec<f64> = THREAD_POOL.install(|| grower.total(&rows));
    grower.weight = target.weight(&total[..stats]);
    let nodes: Nodes = THREAD_POOL.install(|| match growth.max_leaf_nodes {
        Some(leaves) => grower.leaf_wise(&mut rows, &total, leaves),
        None => grower.subtree(&mut rows, 0, &total, 0, None),
    });
    let mut importances: Vec<f64> = vec![0.0; bins.names.len()];
    for (node, decrease) in nodes.nodes.iter().zip(&nodes.decrease) {
        if node.split != LEAF {
            importances[node.feature()] += decrease;
        }
    }
    let sum: f64 = importances.iter().sum();
    if sum > 0.0 {
        importances.iter_mut().for_each(|share| *share /= sum);
    }
    // From "how far ahead" to the place of the right child.
    let mut placed: Vec<Node> = nodes.nodes;
    for (i, node) in placed.iter_mut().enumerate() {
        if node.split != LEAF {
            node.right += i as u32;
        }
    }
    Grown {
        tree: Tree::from_parts(placed, nodes.value, nodes.samples).with_sets(nodes.sets),
        importances,
        rows,
    }
}

#[cfg(test)]
mod tests {
    use polars::df;

    use super::*;

    #[test]
    fn few_values_get_a_bucket_each_and_many_get_quantiles() {
        let exact: Binned = bin(&[3.0, 1.0, f64::NAN, 2.0, 1.0, -0.0, 0.0], 255).unwrap();
        assert_eq!(*exact.codes.wide(), [3, 1, 4, 2, 1, 0, 0]);
        assert!(matches!(exact.codes, Codes::Narrow(_)));
        assert!(matches!(bin(&[1.0, 2.0], 256).unwrap().codes, Codes::Wide(_)));
        assert_eq!(exact.edges, [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]);
        assert!(exact.missing);
        // 255 distinct values still get a bucket each, also behind a hash collision.
        let many: Vec<f64> = (0..2000).map(|i| f64::from(i * 7 % 255) * 0.25 - 9.0).collect();
        let binned: Binned = bin(&many, 255).unwrap();
        assert_eq!((binned.edges.len(), binned.missing), (255, false));
        for (value, code) in many.iter().zip(binned.codes.wide().iter()) {
            assert_eq!(binned.edges[*code as usize], (*value, *value));
        }

        let values: Vec<f64> = (0..1000).map(f64::from).collect();
        let wide: Binned = bin(&values, 4).unwrap();
        assert_eq!(
            wide.edges,
            [(0.0, 250.0), (251.0, 500.0), (501.0, 750.0), (751.0, 999.0)]
        );
        assert_eq!(
            (wide.codes.get(250), wide.codes.get(251), wide.missing),
            (0, 1, false)
        );
        assert!(bin(&[1.0, f64::INFINITY], 255).is_none());
        // Only missing values: no bucket, and nothing to split on.
        assert_eq!(*bin(&[f64::NAN; 3], 255).unwrap().codes.wide(), [0; 3]);
    }

    /// 20,000 rows of 6 features with 3 to 300 distinct values and some nulls, and a class
    /// that depends on them with noise.
    fn noisy() -> (Dataset, Vec<u32>) {
        let mut rng: Rng = Rng::new(7);
        let n: usize = 20_000;
        let mut column = |levels: u64, nulls: bool| -> Vec<Option<f64>> {
            (0..n)
                .map(|_| {
                    let draw: u64 = rng.next_u64();
                    (!(nulls && draw.is_multiple_of(17))).then(|| ((draw >> 8) % levels) as f64)
                })
                .collect()
        };
        let (a, b, c) = (column(3, false), column(40, true), column(300, false));
        let (d, e, f) = (column(7, true), column(120, false), column(2, false));
        let y: Vec<u32> = (0..n)
            .map(|i| {
                let score: f64 =
                    a[i].unwrap() * 30.0 + b[i].unwrap_or(55.0) + c[i].unwrap() / 4.0
                        - e[i].unwrap() / 2.0
                        + (rng.next_u64() % 60) as f64;
                (score / 70.0) as u32 % 3
            })
            .collect();
        let frame = df!("a" => a, "b" => b, "c" => c, "d" => d, "e" => e, "f" => f).unwrap();
        (Dataset::new(frame, ""), y)
    }

    #[test]
    fn the_tree_is_the_same_from_histograms_from_rows_and_from_tasks() {
        let (ds, y) = noisy();
        let bins: Bins = Bins::try_new(&ds, "", 255).unwrap();
        // A bootstrap sample: each row 0 to 3 times.
        let mut rng: Rng = Rng::new(3);
        let counts: Vec<f32> = (0..20_000).map(|_| (rng.next_u64() % 4) as f32).collect();
        let in_bag: u32 = counts.iter().filter(|c| **c > 0.0).count() as u32;
        // (entropy, min_samples_leaf, max_features, weights): 4 of 6 features still keeps
        // histograms, 2 of 6 reads the rows.
        let cases = [
            (false, 1, 6, None),
            (true, 1, 6, None),
            (false, 5, 6, None),
            (false, 1, 6, Some(&counts[..])),
            (false, 2, 4, Some(&counts[..])),
            (true, 1, 2, None),
        ];
        for (entropy, min_samples_leaf, max_features, weights) in cases {
            let target: Target = Target::Classes {
                y: y.clone(),
                classes: 3,
                entropy,
            };
            let growth: Growth = Growth {
                max_depth: None,
                min_samples_split: 2,
                min_samples_leaf,
                min_impurity_decrease: 0.0,
                max_features,
                seed: 11,
                max_leaf_nodes: None,
                features: None,
                by_level: 1.0,
                by_node: 1.0,
                rules: Rules::Sklearn,
                random_splits: false,
            };
            // Class counts are whole numbers, so a histogram made by subtraction is exact and
            // every path must give the same tree.
            let grow = |spawn: usize, dense: bool| {
                grow_with(&bins, &target, &growth, weights, None, spawn, dense)
            };
            let usual: Grown = grow(SPAWN, true);
            let (one_task, from_rows, small_tasks) =
                (grow(usize::MAX, true), grow(usize::MAX, false), grow(64, true));
            assert!(usual.tree.node_count() > 1000);
            assert_eq!(usual.tree, one_task.tree);
            assert_eq!(usual.tree, from_rows.tree);
            assert_eq!(usual.tree, small_tasks.tree);
            assert_eq!(usual.importances, small_tasks.importances);
            usual.tree.check(6, 3).unwrap();
            // Every row ends in a leaf, and the leaves hold every row once.
            let tree: &Tree = &usual.tree;
            let leaves = (0..tree.node_count()).filter(|i| tree.feature()[*i] < 0);
            let rows: u32 = leaves.map(|i| tree.n_node_samples()[i]).sum();
            assert_eq!(rows, if weights.is_some() { in_bag } else { 20_000 });
            let least: u32 = *tree.n_node_samples().iter().min().unwrap();
            assert!(least as usize >= min_samples_leaf);
        }
    }

    #[test]
    fn a_weight_counts_like_that_many_copies_of_the_row() {
        // Rows 0 and 3 twice, row 2 not at all: the tree of the six copied rows.
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ds: Dataset = Dataset::new(df!("x" => x).unwrap(), "");
        let bins: Bins = Bins::try_new(&ds, "", 255).unwrap();
        let target: Target = Target::Regression {
            y: vec![1.0, 2.0, 99.0, 7.0, 8.0],
        };
        let growth: Growth = Growth {
            max_depth: Some(1),
            min_samples_split: 2,
            min_samples_leaf: 1,
            min_impurity_decrease: 0.0,
            max_features: 1,
            seed: 0,
            max_leaf_nodes: None,
            features: None,
            by_level: 1.0,
            by_node: 1.0,
            rules: Rules::Sklearn,
            random_splits: false,
        };
        let grown: Grown = grow(&bins, &target, &growth, Some(&[2.0, 1.0, 0.0, 2.0, 1.0]), None);
        // Copies: y = 1, 1, 2 | 7, 7, 8. The row left out has no say: the threshold is halfway
        // between x = 2 and x = 4, and the nodes count rows, not copies.
        assert_eq!(grown.tree.threshold()[0], 3.0);
        assert_eq!(grown.tree.n_node_samples(), [4, 2, 2]);
        let want: [f64; 3] = [26.0 / 6.0, 4.0 / 3.0, 22.0 / 3.0];
        for (got, want) in grown.tree.value().iter().zip(want) {
            assert!((got - want).abs() < 1e-12, "{:?}", grown.tree.value());
        }
    }

    #[test]
    fn boosting_fills_a_few_features_per_pass_with_the_sums_of_one_at_a_time() {
        let (ds, _) = noisy();
        let bins: Bins = Bins::try_new(&ds, "", 255).unwrap();
        let mut rng: Rng = Rng::new(9);
        let g: Vec<f32> = (0..20_000).map(|_| (rng.next_u64() % 1000) as f32 / 37.0 - 13.0).collect();
        let h: Vec<f32> = (0..20_000).map(|_| (rng.next_u64() % 100) as f32 / 100.0).collect();
        let all: Vec<u32> = (0..20_000).collect();
        let some: Vec<u32> = all.iter().copied().filter(|r| r % 3 != 0).collect();
        let features: usize = bins.names.len();
        for hessians in [Vec::new(), h] {
            let target: Target = Target::Gradients { g: g.clone(), h: hessians, step: Step::new(0.0, 0.1) };
            let growth: Growth = Growth {
                max_depth: None,
                min_samples_split: 2,
                min_samples_leaf: 1,
                min_impurity_decrease: 0.0,
                max_features: features,
                seed: 0,
                max_leaf_nodes: Some(31),
                features: None,
                by_level: 1.0,
                by_node: 1.0,
                rules: Rules::Sklearn,
                random_splits: false,
            };
            let stats: usize = target.stats();
            let grower: Grower = Grower {
                bins: &bins,
                target: &target,
                growth: &growth,
                weights: None,
                stats,
                width: stats,
                weight: 1.0,
                spawn: SPAWN,
                dense: true,
            };
            for rows in [&all, &some] {
                let mut along: Along = Along::default();
                grower.gather(rows, &mut along);
                let one: Vec<Vec<f64>> = (0..features)
                    .map(|f| {
                        let slots: usize = bins.offsets[f + 1] - bins.offsets[f];
                        let mut hist: Vec<f64> = vec![0.0; slots * stats];
                        grower.fill(f, rows, &along, &mut hist, |_| {});
                        hist
                    })
                    .collect();
                for group in 1..=4 {
                    for first in (0..features).step_by(group) {
                        let to: usize = (first + group).min(features);
                        let mut hists: Vec<Vec<f64>> = one[first..to].iter().map(|h| vec![0.0; h.len()]).collect();
                        grower.fill_gradients(first, rows, &along, &mut hists);
                        assert_eq!(hists, one[first..to], "group {group} from {first}");
                    }
                }
            }
        }
    }

    #[test]
    fn a_large_node_puts_its_rows_in_two_parts_in_order() {
        // Over three blocks: the root's rows split in tasks, then join in row order.
        let n: usize = 3 * BLOCK + 5;
        let x: Vec<f64> = (0..n).map(|i| ((i * 7919) % 10) as f64).collect();
        let ds: Dataset = Dataset::new(df!("x" => x.clone()).unwrap(), "");
        let bins: Bins = Bins::try_new(&ds, "", 255).unwrap();
        let target: Target = Target::Regression { y: x.clone() };
        for max_leaf_nodes in [None, Some(2)] {
            let growth: Growth = Growth {
                max_depth: Some(1),
                min_samples_split: 2,
                min_samples_leaf: 1,
                min_impurity_decrease: 0.0,
                max_features: 1,
                seed: 0,
                max_leaf_nodes,
                features: None,
                by_level: 1.0,
                by_node: 1.0,
                rules: Rules::Sklearn,
                random_splits: false,
            };
            let grown: Grown = grow(&bins, &target, &growth, None, None);
            let cut: f64 = grown.tree.threshold()[0];
            let x: &[f64] = &x;
            let side = |left: bool| (0..n as u32).filter(move |i| (x[*i as usize] <= cut) == left);
            let want: Vec<u32> = side(true).chain(side(false)).collect();
            assert_eq!(grown.rows, want);
        }
    }
}
