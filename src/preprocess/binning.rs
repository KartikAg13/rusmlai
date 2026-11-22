//! [`KBinsDiscretizer`] and [`QuantileTransformer`]: numbers cut into bins, and numbers mapped
//! through their quantiles, like scikit-learn's.
//!
//! Both learn from a sample of the rows, drawn as scikit-learn draws it (numpy's generator
//! seeded with `random_state`), and compute quantiles with numpy's own formulas, so the edges
//! and the quantiles are scikit-learn's.

use indexmap::IndexMap;
use polars::prelude::{
    ChunkApply, Column, DataType, Expr, Field, Float64Chunked, IntoColumn, IntoSeries, Schema, lit, when,
};

use crate::{
    dataset::Dataset,
    error::{OrFatal, warn},
    models::{check, folds::Mt19937},
    preprocess::{Exprs, PreprocessError, Transform, one_of, scaler::normal_ppf},
};

/// numpy's `linspace(start, stop, num)`.
fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 1 {
        return vec![start];
    }
    let step: f64 = (stop - start) / (num - 1) as f64;
    let mut out: Vec<f64> = (0..num).map(|i| i as f64 * step + start).collect();
    out[num - 1] = stop;
    out
}

/// numpy's `_lerp`: from the nearer end, so `t` of 0 or 1 gives an end exactly.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    let diff: f64 = b - a;
    if t >= 0.5 { b - diff * (1.0 - t) } else { a + diff * t }
}

/// numpy's `percentile(sorted, q)` for `q` in percent, with `method` `"linear"`,
/// `"inverted_cdf"` or `"averaged_inverted_cdf"`.
pub(crate) fn percentile(sorted: &[f64], q: f64, method: &str) -> f64 {
    let n: usize = sorted.len();
    let q: f64 = q / 100.0;
    let (index, gamma): (f64, f64) = match method {
        "linear" => {
            let v: f64 = (n - 1) as f64 * q;
            (v.floor(), v - v.floor())
        }
        _ => {
            let v: f64 = n as f64 * q - 1.0;
            let g: f64 = v - v.floor();
            let gamma: f64 = match (method, g == 0.0) {
                ("averaged_inverted_cdf", true) => 0.5,
                (_, true) => 0.0,
                _ => 1.0,
            };
            (v.floor(), gamma)
        }
    };
    let at = |i: f64| sorted[(i.max(0.0) as usize).min(n - 1)];
    lerp(at(index), at(index + 1.0), gamma)
}

/// A column's finite values, as `f64`, nulls and NaN left out.
fn values(dataset: &Dataset, name: &str) -> Result<Vec<f64>, PreprocessError> {
    let column = dataset.column(name)?.cast(&DataType::Float64)?;
    Ok(column.f64()?.iter().flatten().filter(|v| !v.is_nan()).collect())
}

/// `values` at the rows `rows`.
fn take(dataset: &Dataset, name: &str, rows: &[usize]) -> Result<Vec<f64>, PreprocessError> {
    let column = dataset.column(name)?.cast(&DataType::Float64)?;
    let column = column.f64()?;
    Ok(rows.iter().filter_map(|r| column.get(*r)).filter(|v| !v.is_nan()).collect())
}

fn sorted(mut values: Vec<f64>) -> Vec<f64> {
    values.sort_unstable_by(f64::total_cmp);
    values
}

/// A `Float64` output field named like the input.
fn float_field(_: &Schema, field: &Field) -> polars::prelude::PolarsResult<Field> {
    Ok(Field::new(field.name().clone(), DataType::Float64))
}

/// Cuts each number column into bins, scikit-learn's `KBinsDiscretizer`: `strategy`
/// `"quantile"` (the default: bins with about the same number of rows), `"uniform"` (bins of
/// the same width) or `"kmeans"` (bins around the centers of a 1-D k-means). `encode`
/// `"onehot"` (the default) gives one 0 / 1 column per bin, `{column}_{bin}.0` as
/// scikit-learn names them; `"ordinal"` gives the bin number in the column itself.
///
/// The quantile edges use numpy's `percentile` with `quantile_method`
/// (`"averaged_inverted_cdf"` by default, as in scikit-learn 1.9). On more rows than
/// `subsample` (200,000), the edges come from that many rows drawn with replacement, by
/// numpy's generator seeded with `random_state`, as scikit-learn draws them. Bins narrower
/// than 1e-8 are merged away, with a warning; a constant column becomes one bin. A value
/// below the first edge falls in the first bin, one above the last in the last. Nulls stay
/// null.
///
/// scikit-learn settings with no setter: `dtype` (always `Float64`). `n_bins` is one count for
/// every column. With `"kmeans"`, an empty cluster keeps its center (scikit-learn moves it to
/// a far point).
///
/// ```
/// use rusmlai::prelude::*;
/// let ds = Dataset::new(df!("a" => [1.0, 2.0, 3.0, 4.0, 10.0]).unwrap(), "");
/// let mut bins = KBinsDiscretizer::new().n_bins(3);
/// let out = bins.fit_transform(&ds);
/// assert_eq!(bins.bin_edges_["a"], [1.0, 2.0, 4.0, 10.0]);
/// assert_eq!(out.columns(), ["a_0.0", "a_1.0", "a_2.0"]);
/// assert_eq!(out.to_list("a_1.0"), [0.0, 1.0, 1.0, 0.0, 0.0]);
/// let ordinal = KBinsDiscretizer::new().n_bins(3).strategy("uniform").encode("ordinal").fit_transform(&ds);
/// assert_eq!(ordinal.to_list("a"), [0.0, 0.0, 0.0, 1.0, 2.0]);
/// ```
///
/// Python:
///
/// ```python
/// import pandas as pd
/// from sklearn.preprocessing import KBinsDiscretizer
/// ds = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 10.0]})
/// bins = KBinsDiscretizer(n_bins=3, encode="onehot-dense")
/// out = bins.fit_transform(ds)
/// assert bins.bin_edges_[0].tolist() == [1.0, 2.0, 4.0, 10.0]
/// assert bins.get_feature_names_out().tolist() == ["a_0.0", "a_1.0", "a_2.0"]
/// assert out[:, 1].tolist() == [0.0, 1.0, 1.0, 0.0, 0.0]
/// ordinal = KBinsDiscretizer(n_bins=3, strategy="uniform", encode="ordinal").fit_transform(ds)
/// assert ordinal[:, 0].tolist() == [0.0, 0.0, 0.0, 1.0, 2.0]
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KBinsDiscretizer {
    n_bins: usize,
    encode: String,
    strategy: String,
    quantile_method: String,
    subsample: Option<usize>,
    random_state: u64,
    /// Each column's bin edges, from its smallest to its largest value, learned by `fit`.
    pub bin_edges_: IndexMap<String, Vec<f64>>,
    /// Each column's number of bins, learned by `fit` (fewer than `n_bins` when bins merged).
    pub n_bins_: IndexMap<String, usize>,
}

impl Default for KBinsDiscretizer {
    fn default() -> Self {
        Self::new()
    }
}

impl KBinsDiscretizer {
    /// 5 quantile bins, one-hot, scikit-learn's defaults.
    pub fn new() -> Self {
        Self {
            n_bins: 5,
            encode: "onehot".into(),
            strategy: "quantile".into(),
            quantile_method: "averaged_inverted_cdf".into(),
            subsample: Some(200_000),
            random_state: 42,
            bin_edges_: IndexMap::new(),
            n_bins_: IndexMap::new(),
        }
    }

    /// The number of bins of each column, 2 or more. Default 5.
    #[track_caller]
    pub fn n_bins(mut self, n_bins: usize) -> Self {
        check(n_bins >= 2, "n_bins", n_bins, "2 bins or more");
        self.n_bins = n_bins;
        self
    }

    /// `"onehot"` (the default; also `"onehot-dense"`, the same here) or `"ordinal"`.
    #[track_caller]
    pub fn encode(mut self, encode: &str) -> Self {
        self.encode = one_of("encode", encode, &["onehot", "onehot-dense", "ordinal"]).or_fatal();
        self
    }

    /// `"quantile"` (the default), `"uniform"` or `"kmeans"`.
    #[track_caller]
    pub fn strategy(mut self, strategy: &str) -> Self {
        self.strategy = one_of("strategy", strategy, &["quantile", "uniform", "kmeans"]).or_fatal();
        self
    }

    /// numpy's percentile method for `"quantile"`: `"averaged_inverted_cdf"` (the default),
    /// `"inverted_cdf"` or `"linear"`.
    #[track_caller]
    pub fn quantile_method(mut self, method: &str) -> Self {
        let allowed = ["averaged_inverted_cdf", "inverted_cdf", "linear"];
        self.quantile_method = one_of("quantile_method", method, &allowed).or_fatal();
        self
    }

    /// Learn from at most this many rows, drawn with replacement; `None` uses every row.
    /// Default `Some(200_000)`.
    #[track_caller]
    pub fn subsample(mut self, subsample: Option<usize>) -> Self {
        check(subsample != Some(0), "subsample", 0, "1 row or more, or None");
        self.subsample = subsample;
        self
    }

    /// The seed of the subsample. Default 42.
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = random_state;
        self
    }

    /// The edges of one column.
    fn edges(&self, name: &str, values: Vec<f64>) -> Vec<f64> {
        // k-means sums in row order, like scikit-learn; the quantiles need the sorted values.
        let unsorted: Vec<f64> = if self.strategy == "kmeans" { values.clone() } else { Vec::new() };
        let sorted: Vec<f64> = sorted(values);
        let (Some(&min), Some(&max)) = (sorted.first(), sorted.last()) else {
            return vec![f64::NEG_INFINITY, f64::INFINITY];
        };
        if min == max {
            warn(format_args!("column '{name}' is constant and is replaced with 0"));
            return vec![f64::NEG_INFINITY, f64::INFINITY];
        }
        let k: usize = self.n_bins;
        let edges: Vec<f64> = match self.strategy.as_str() {
            "uniform" => return linspace(min, max, k + 1),
            "quantile" => (linspace(0.0, 100.0, k + 1).into_iter())
                .map(|q| percentile(&sorted, q, &self.quantile_method))
                .collect(),
            _ => {
                let centers: Vec<f64> = kmeans_1d(&unsorted, k);
                let mut edges: Vec<f64> = vec![min];
                edges.extend(centers.windows(2).map(|w| (w[0] + w[1]) * 0.5));
                edges.push(max);
                edges
            }
        };
        let mut kept: Vec<f64> = Vec::with_capacity(edges.len());
        for (i, edge) in edges.iter().enumerate() {
            if i == 0 || edge - edges[i - 1] > 1e-8 {
                kept.push(*edge);
            }
        }
        if kept.len() - 1 != k {
            warn(format_args!(
                "bins narrower than 1e-8 in column '{name}' are removed; consider fewer bins"
            ));
        }
        kept
    }
}

/// numpy's pairwise sum of a float64 array (8 running sums up to 128 values, halves above),
/// so a mean rounds as `np.mean` does.
fn pairwise_sum(x: &[f64]) -> f64 {
    let n: usize = x.len();
    if n < 8 {
        return x.iter().fold(0.0, |s, v| s + v);
    }
    if n <= 128 {
        let mut r: [f64; 8] = [0.0; 8];
        r.copy_from_slice(&x[..8]);
        let mut i: usize = 8;
        while i + 8 <= n {
            for (k, lane) in r.iter_mut().enumerate() {
                *lane += x[i + k];
            }
            i += 8;
        }
        let mut res: f64 = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for v in &x[i..] {
            res += v;
        }
        return res;
    }
    let mut half: usize = n / 2;
    half -= half % 8;
    pairwise_sum(&x[..half]) + pairwise_sum(&x[half..])
}

/// scikit-learn's 1-D `KMeans(n_clusters=k, init=uniform centers, n_init=1)`: Lloyd's rounds
/// on the values minus their mean, each value to the center with the smallest ‖c‖² − 2·x·c
/// (the first on a tie, so ties break as in scikit-learn), until the labels stop changing or
/// the centers move less than 1e-4 of the variance. Returns the sorted centers.
fn kmeans_1d(values: &[f64], k: usize) -> Vec<f64> {
    let n: usize = values.len();
    let (min, max) = values.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
    let mean: f64 = pairwise_sum(values) / n as f64;
    let x: Vec<f64> = values.iter().map(|v| v - mean).collect();
    let squares: Vec<f64> = x.iter().map(|v| v * v).collect();
    let tol: f64 = pairwise_sum(&squares) / n as f64 * 1e-4;
    let uniform: Vec<f64> = linspace(min, max, k + 1);
    let mut centers: Vec<f64> = uniform.windows(2).map(|w| (w[0] + w[1]) * 0.5 - mean).collect();
    let mut last: Vec<usize> = Vec::new();
    let mut labels: Vec<usize> = vec![0; n];
    for _ in 0..300 {
        for (label, xi) in labels.iter_mut().zip(&x) {
            let mut best: (usize, f64) = (0, f64::INFINITY);
            for (j, c) in centers.iter().enumerate() {
                let d: f64 = c * c - 2.0 * xi * c;
                if d < best.1 {
                    best = (j, d);
                }
            }
            *label = best.0;
        }
        let (mut sums, mut counts): (Vec<f64>, Vec<usize>) = (vec![0.0; k], vec![0; k]);
        for (label, xi) in labels.iter().zip(&x) {
            sums[*label] += xi;
            counts[*label] += 1;
        }
        let mut shift: f64 = 0.0;
        for j in 0..k {
            // ponytail: an empty cluster keeps its center; scikit-learn moves it to a far point.
            if counts[j] > 0 {
                let new: f64 = sums[j] / counts[j] as f64;
                shift += (new - centers[j]) * (new - centers[j]);
                centers[j] = new;
            }
        }
        if labels == last || shift <= tol {
            break;
        }
        last.clone_from(&labels);
    }
    let mut centers: Vec<f64> = centers.into_iter().map(|c| c + mean).collect();
    centers.sort_by(f64::total_cmp);
    centers
}

impl Transform for KBinsDiscretizer {
    fn try_fit(&mut self, dataset: &Dataset) -> Result<(), PreprocessError> {
        let n: usize = dataset.height();
        // One draw of rows for every column, with replacement, as `resample` makes it.
        let rows: Option<Vec<usize>> = self.subsample.filter(|s| n > *s).map(|s| {
            let mut rng: Mt19937 = Mt19937::new(self.random_state);
            (0..s).map(|_| rng.interval((n - 1) as u64) as usize).collect()
        });
        self.bin_edges_.clear();
        self.n_bins_.clear();
        for name in dataset.columns() {
            let values: Vec<f64> = match &rows {
                Some(rows) => take(dataset, &name, rows)?,
                None => values(dataset, &name)?,
            };
            let edges: Vec<f64> = self.edges(&name, values);
            self.n_bins_.insert(name.clone(), edges.len() - 1);
            self.bin_edges_.insert(name, edges);
        }
        Ok(())
    }

    fn exprs(&self, inputs: &Exprs) -> Result<Exprs, PreprocessError> {
        let mut out: Exprs = IndexMap::new();
        for (name, expr) in inputs {
            let edges: &Vec<f64> = self.bin_edges_.get(name).ok_or_else(|| PreprocessError::NotFitted {
                transformer: "KBinsDiscretizer",
                column: name.clone(),
            })?;
            let x: Expr = expr.clone().cast(DataType::Float64);
            // The number of inner edges at or below x: numpy's searchsorted(side="right").
            let inner: &[f64] = &edges[1..edges.len() - 1];
            let bin: Expr = inner.iter().fold(lit(0.0), |sum, edge| {
                sum + x.clone().gt_eq(lit(*edge)).cast(DataType::Float64)
            });
            let bin: Expr = when(x.clone().is_null()).then(lit(polars::prelude::NULL).cast(DataType::Float64)).otherwise(bin);
            if self.encode == "ordinal" {
                out.insert(name.clone(), bin);
                continue;
            }
            for k in 0..edges.len() - 1 {
                let one: Expr = bin.clone().eq(lit(k as f64)).cast(DataType::Float64);
                out.insert(format!("{name}_{k}.0"), one);
            }
        }
        Ok(out)
    }
}

/// Maps each number column through its quantiles, scikit-learn's `QuantileTransformer`: a
/// value becomes its rank among the training values, as a number from 0 to 1
/// (`output_distribution` `"uniform"`, the default), or that rank through the normal
/// distribution's quantile function (`"normal"`). It flattens any shape of distribution and
/// any outlier; a model then sees ranks, not distances.
///
/// `fit` keeps `n_quantiles` (1,000) quantiles of each column, from at most `subsample`
/// (10,000) rows drawn without replacement by numpy's generator seeded with `random_state`, as
/// scikit-learn draws them, computed with numpy's `nanpercentile`. `transform` interpolates
/// between them as scikit-learn does (the mean of the interpolations from both ends, so equal
/// quantiles map to their middle). Values outside the training range map to 0 or 1 (with
/// `"normal"`, to ±5.2, the normal quantile of 1e-7). Nulls stay null.
///
/// scikit-learn settings with no setter: `ignore_implicit_zeros` (sparse data only), `copy`.
///
/// ```
/// use rusmlai::prelude::*;
/// let ds = Dataset::new(df!("a" => [1.0, 2.0, 3.0, 4.0, 100.0]).unwrap(), "");
/// let mut q = QuantileTransformer::new();
/// let out = q.fit_transform(&ds);
/// assert_eq!(q.n_quantiles_, 5); // at most one per row
/// assert_eq!(out.to_list("a"), [0.0, 0.25, 0.5, 0.75, 1.0]); // ranks: the outlier is just last
/// let new = Dataset::new(df!("a" => [2.5, 52.0, 500.0]).unwrap(), "");
/// assert_eq!(q.transform(&new).to_list("a"), [0.375, 0.875, 1.0]);
/// ```
///
/// Python:
///
/// ```python
/// import pandas as pd
/// from sklearn.preprocessing import QuantileTransformer
/// ds = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 100.0]})
/// q = QuantileTransformer(n_quantiles=5)
/// out = q.fit_transform(ds)
/// assert q.n_quantiles_ == 5
/// assert out[:, 0].tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
/// new = pd.DataFrame({"a": [2.5, 52.0, 500.0]})
/// assert q.transform(new)[:, 0].tolist() == [0.375, 0.875, 1.0]
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QuantileTransformer {
    n_quantiles: usize,
    output_distribution: String,
    subsample: Option<usize>,
    random_state: u64,
    /// The quantiles actually kept: `n_quantiles`, or the rows when there are fewer.
    pub n_quantiles_: usize,
    /// Each column's quantiles at `references_`, learned by `fit`.
    pub quantiles_: IndexMap<String, Vec<f64>>,
    /// The levels of the quantiles, `n_quantiles_` steps from 0 to 1.
    pub references_: Vec<f64>,
}

impl Default for QuantileTransformer {
    fn default() -> Self {
        Self::new()
    }
}

/// numpy's `interp(x, xp, fp)` for increasing `xp`: the ends outside, the left point on an
/// exact hit.
fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n: usize = xp.len();
    if x < xp[0] {
        return fp[0];
    }
    if x > xp[n - 1] {
        return fp[n - 1];
    }
    // The last j with xp[j] <= x.
    let j: usize = xp.partition_point(|v| *v <= x).saturating_sub(1);
    if j == n - 1 || xp[j] == x {
        return fp[j];
    }
    let slope: f64 = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    let y: f64 = slope * (x - xp[j]) + fp[j];
    if y.is_nan() {
        let y: f64 = slope * (x - xp[j + 1]) + fp[j + 1];
        if y.is_nan() && fp[j] == fp[j + 1] { fp[j] } else { y }
    } else {
        y
    }
}

/// scikit-learn's bound for the normal output: ranks closer than this to 0 or 1 are clipped.
const BOUNDS_THRESHOLD: f64 = 1e-7;

impl QuantileTransformer {
    /// 1,000 quantiles, a uniform output, 10,000 rows sampled: scikit-learn's defaults.
    pub fn new() -> Self {
        Self {
            n_quantiles: 1000,
            output_distribution: "uniform".into(),
            subsample: Some(10_000),
            random_state: 42,
            n_quantiles_: 0,
            quantiles_: IndexMap::new(),
            references_: Vec::new(),
        }
    }

    /// The quantiles to keep, 1 or more. Default 1,000.
    #[track_caller]
    pub fn n_quantiles(mut self, n_quantiles: usize) -> Self {
        check(n_quantiles >= 1, "n_quantiles", n_quantiles, "1 or more");
        self.n_quantiles = n_quantiles;
        self
    }

    /// `"uniform"` (the default) or `"normal"`.
    #[track_caller]
    pub fn output_distribution(mut self, distribution: &str) -> Self {
        self.output_distribution = one_of("output_distribution", distribution, &["uniform", "normal"]).or_fatal();
        self
    }

    /// Learn from at most this many rows, drawn without replacement; `None` uses every row.
    /// Default `Some(10_000)`.
    #[track_caller]
    pub fn subsample(mut self, subsample: Option<usize>) -> Self {
        check(subsample.is_none_or(|s| s >= 1), "subsample", 0, "1 row or more, or None");
        self.subsample = subsample;
        self
    }

    /// The seed of the subsample. Default 42.
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = random_state;
        self
    }

    /// One value of one column, scikit-learn's `_transform_col` forward. `reversed` and `back`
    /// are the negated quantiles and references, last first, made once per column.
    fn map(x: f64, (quantiles, references): (&[f64], &[f64]), (reversed, back): (&[f64], &[f64]), normal: bool) -> f64 {
        if x.is_nan() {
            return x;
        }
        let (low, high): (f64, f64) = (quantiles[0], quantiles[quantiles.len() - 1]);
        let (lower, upper): (bool, bool) = if normal {
            (x - BOUNDS_THRESHOLD < low, x + BOUNDS_THRESHOLD > high)
        } else {
            (x == low, x == high)
        };
        let mut y: f64 = 0.5 * (interp(x, quantiles, references) - interp(-x, reversed, back));
        if upper {
            y = 1.0;
        }
        if lower {
            y = 0.0;
        }
        if normal {
            let spacing: f64 = f64::EPSILON; // numpy's spacing(1)
            let (min, max) = (normal_ppf(BOUNDS_THRESHOLD - spacing), normal_ppf(1.0 - (BOUNDS_THRESHOLD - spacing)));
            y = normal_ppf(y).clamp(min, max);
        }
        y
    }
}

impl Transform for QuantileTransformer {
    fn try_fit(&mut self, dataset: &Dataset) -> Result<(), PreprocessError> {
        let n: usize = dataset.height();
        if let Some(s) = self.subsample.filter(|s| self.n_quantiles > *s) {
            return Err(PreprocessError::InvalidOption {
                option: "n_quantiles",
                value: self.n_quantiles.to_string(),
                expected: format!("at most subsample ({s}) quantiles"),
            });
        }
        self.n_quantiles_ = self.n_quantiles.min(n).max(1);
        self.references_ = linspace(0.0, 1.0, self.n_quantiles_);
        let levels: Vec<f64> = self.references_.iter().map(|r| r * 100.0).collect();
        // One draw of rows for every column, without replacement, as `resample` makes it.
        let rows: Option<Vec<usize>> = self.subsample.filter(|s| *s < n).map(|s| {
            let mut rows: Vec<usize> = (0..n).collect();
            Mt19937::new(self.random_state).shuffle(&mut rows);
            rows.truncate(s);
            rows
        });
        self.quantiles_.clear();
        for name in dataset.columns() {
            let values: Vec<f64> = match &rows {
                Some(rows) => take(dataset, &name, rows)?,
                None => values(dataset, &name)?,
            };
            let sorted: Vec<f64> = sorted(values);
            if sorted.is_empty() {
                return Err(PreprocessError::InvalidOption {
                    option: "column",
                    value: name,
                    expected: "a column with at least one number".into(),
                });
            }
            let quantiles: Vec<f64> = levels.iter().map(|q| percentile(&sorted, *q, "linear")).collect();
            self.quantiles_.insert(name, quantiles);
        }
        Ok(())
    }

    fn exprs(&self, inputs: &Exprs) -> Result<Exprs, PreprocessError> {
        let normal: bool = self.output_distribution == "normal";
        inputs
            .iter()
            .map(|(name, expr)| {
                let quantiles: Vec<f64> = self.quantiles_.get(name).cloned().ok_or_else(|| PreprocessError::NotFitted {
                    transformer: "QuantileTransformer",
                    column: name.clone(),
                })?;
                let references: Vec<f64> = self.references_.clone();
                let reversed: Vec<f64> = quantiles.iter().rev().map(|q| -q).collect();
                let back: Vec<f64> = references.iter().rev().map(|r| -r).collect();
                let mapped: Expr = expr.clone().cast(DataType::Float64).map(
                    move |column: Column| {
                        let (ends, mirrored) = ((&quantiles[..], &references[..]), (&reversed[..], &back[..]));
                        let values: Float64Chunked = column
                            .f64()?
                            .apply_values(|x| QuantileTransformer::map(x, ends, mirrored, normal));
                        Ok(values.into_series().into_column())
                    },
                    float_field,
                );
                Ok((name.clone(), mapped))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_are_numpys() {
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 10.0];
        // np.percentile(x, [0, 20, 40, 60, 80, 100], method=...)
        let levels: Vec<f64> = linspace(0.0, 100.0, 6);
        let got = |m: &str| levels.iter().map(|q| percentile(&x, *q, m)).collect::<Vec<f64>>();
        assert_eq!(got("averaged_inverted_cdf"), [1.0, 1.5, 2.5, 3.5, 7.0, 10.0]);
        assert_eq!(got("inverted_cdf"), [1.0, 1.0, 2.0, 3.0, 4.0, 10.0]);
        assert_eq!(got("linear"), [1.0, 1.8, 2.6, 3.4, 5.200000000000001, 10.0]);
    }
    /// The rows of `analysis/2026-11-18/new-features/binning-parity.py`: skewed, with ties.
    fn rows() -> Dataset {
        let n: usize = 3000;
        let a: Vec<f64> = (0..n).map(|i| ((i * 37) % 1009) as f64 / 7.0 + (((i * 11) % 13) as f64).powi(2)).collect();
        let b: Vec<f64> = (0..n).map(|i| (((i * 53) % 211) as f64 / 10.0 * 10.0).round() / 10.0).collect();
        Dataset::new(polars::df!("a" => a, "b" => b).unwrap(), "")
    }

    fn close(got: &[f64], want: &[f64], eps: f64) {
        assert!(got.len() == want.len() && got.iter().zip(want).all(|(g, w)| g == w || (g - w).abs() < eps), "{got:?}\nwant {want:?}");
    }

    #[test]
    fn bin_edges_are_scikit_learn_s_with_its_subsample() {
        type Case<'a> = (&'a str, [&'a [f64]; 2], [f64; 2]);
        let cases: [Case; 3] = [
            ("quantile", [&[2.7142857142857144, 57.142857142857146, 98.0, 118.71428571428572, 144.57142857142856, 183.71428571428572, 288.0], &[0.0, 3.4, 6.2, 10.4, 13.7, 17.1, 21.0]], [7444.0, 7750.0]),
            ("uniform", [&[2.7142857142857144, 50.26190476190476, 97.8095238095238, 145.35714285714286, 192.9047619047619, 240.45238095238093, 288.0], &[0.0, 3.5, 7.0, 10.5, 14.0, 17.5, 21.0]], [6035.0, 7509.0]),
            ("kmeans", [&[2.7142857142857144, 46.1900080256822, 88.8306812014273, 130.10923870112882, 176.12022548514227, 229.97308853118716, 288.0], &[0.0, 3.405801583374568, 7.055688534278961, 10.55048202614379, 13.857647058823531, 17.45711161387632, 21.0]], [6742.0, 7495.0]),
        ];
        let ds: Dataset = rows();
        for (strategy, edges, sums) in cases {
            let mut bins = KBinsDiscretizer::new().n_bins(6).strategy(strategy).encode("ordinal").subsample(Some(500));
            let out: Dataset = bins.try_fit_transform(&ds).unwrap();
            close(&bins.bin_edges_["a"], edges[0], 1e-9);
            close(&bins.bin_edges_["b"], edges[1], 1e-9);
            let totals: Vec<f64> = ["a", "b"].iter().map(|c| out.to_list(c).iter().sum()).collect();
            close(&totals, &sums, 0.0);
        }
        let mut full = KBinsDiscretizer::new().n_bins(6).quantile_method("linear").encode("ordinal").subsample(None);
        full.try_fit(&ds).unwrap();
        close(&full.bin_edges_["a"], &[0.0, 56.42857142857143, 90.80952380952382, 120.0, 146.1904761904762, 184.88095238095244, 288.0], 1e-12);
        close(&full.bin_edges_["b"], &[0.0, 3.4, 6.9, 10.5, 14.0, 17.5, 21.0], 1e-12);
    }

    #[test]
    fn quantiles_are_scikit_learn_s_with_its_subsample() {
        let ds: Dataset = rows();
        let new: Dataset = Dataset::new(polars::df!("a" => [-5.0, 0.0, 77.7, 1e9], "b" => [0.0, 10.5, 21.0, 21.1]).unwrap(), "");
        let cases = [
            ("uniform", [0.0, 0.0, 0.494949494949495, 0.2892156862745098], [1436.0220121198313, 1550.4806299622553], [[0.0, 0.0, 0.24040871679760567, 1.0], [0.0, 0.5151515151515152, 1.0, 1.0]]),
            ("normal", [-5.199337582605575, -5.199337582605575, -0.012660076940314008, -0.5556774532742532], [-191.46295015251582, 182.64545566096774], [[-5.199337582605575, -5.199337582605575, -0.7049884345282852, 5.19933758270342], [-5.199337582605575, 0.0379883512302825, 5.19933758270342, 5.19933758270342]]),
        ];
        for (distribution, first, sums, unseen) in cases {
            let mut q = QuantileTransformer::new().n_quantiles(100).subsample(Some(700)).output_distribution(distribution);
            let out: Dataset = q.try_fit_transform(&ds).unwrap();
            close(&[q.quantiles_["a"][1], q.quantiles_["b"][50]], &[7.0692640692640705, 10.203030303030305], 1e-12);
            let (a, b) = (out.to_list("a"), out.to_list("b"));
            close(&[a[0], b[0], a[1], b[1]], &first, 1e-9);
            close(&[a.iter().sum(), b.iter().sum()], &sums, 1e-7);
            let mapped: Dataset = q.try_transform(&new).unwrap();
            close(&mapped.to_list("a"), &unseen[0], 1e-9);
            close(&mapped.to_list("b"), &unseen[1], 1e-9);
        }
    }

}
