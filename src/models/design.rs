//! [`DesignMatrix`]: the features and the target of a `fit`, checked once, as `f64` columns.

use faer::{Accum, Mat, MatRef, Par, linalg::matmul::matmul};
use polars::prelude::{DataType, Series};
use polars_core::runtime::THREAD_POOL;
use rayon::prelude::*;

use crate::{
    dataset::Dataset,
    models::{ModelError, numeric_target},
};

/// Rows in one chunk of a pass. 4,096 rows of 30 features are about 1 MB, which stays in the
/// CPU cache.
pub(crate) const CHUNK: usize = 4096;

/// Rows in one task of a parallel sum. The blocks are fixed and their sums are added in block
/// order, so the result is the same on any number of threads and on every machine.
const BLOCK: usize = 4 * CHUNK;

/// The features (every column but the target) and the target of a dataset, as `f64` columns
/// with no null, NaN or infinity. A column that is already `Float64` in one chunk is Polars'
/// own memory, not a copy. Kernels read the columns in chunks of [`CHUNK`] rows
/// ([`DesignMatrix::fold`]), so a fit never holds a second copy of the whole table. Every pass
/// runs on the Polars thread pool.
pub(crate) struct DesignMatrix {
    /// The feature names, in column order (scikit-learn's `feature_names_in_`).
    pub names: Vec<String>,
    columns: Vec<Series>,
    /// The sum of each feature, from the pass that checks it.
    sums: Vec<f64>,
    y: Series,
    y_sum: f64,
}

impl DesignMatrix {
    /// Checks every column once: numbers only, no nulls, no NaN or infinity.
    pub fn try_new(dataset: &Dataset, target: &str) -> Result<Self, ModelError> {
        numeric_target(dataset, target)?;
        let (y, y_sum): (Series, f64) = prepared(dataset, target)?;
        Self::try_with_target(dataset, target, y, y_sum)
    }

    /// The features of `dataset` (every column but `target`), checked, with `y` as the target:
    /// a classifier passes its class codes.
    pub fn try_with_target(
        dataset: &Dataset,
        target: &str,
        y: Series,
        y_sum: f64,
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
        let (columns, sums): (Vec<Series>, Vec<f64>) =
            prepared_all(dataset, &names)?.into_iter().unzip();
        Ok(Self {
            names,
            columns,
            sums,
            y,
            y_sum,
        })
    }

    pub fn rows(&self) -> usize {
        self.y.len()
    }

    /// The mean of every feature and of the target, from the sums of the check: no pass.
    pub fn means(&self) -> (Vec<f64>, f64) {
        let n: f64 = self.rows() as f64;
        (
            self.sums.iter().map(|sum| sum / n).collect(),
            self.y_sum / n,
        )
    }

    /// ‖y − y_mean‖², which coordinate descent needs for its stopping rule.
    pub fn y_norm2(&self, y_mean: f64) -> f64 {
        column(&self.y)
            .iter()
            .map(|y| (y - y_mean) * (y - y_mean))
            .sum()
    }

    /// The target, as `fit` checked it.
    pub fn y(&self) -> &[f64] {
        column(&self.y)
    }

    /// XᵀX and Xᵀy, with `x_means` taken from the features and `y_mean` from the target first.
    /// faer adds the product of each chunk on one thread, which was faster than faer's own
    /// threads on this tall, narrow shape.
    pub fn gram(&self, x_means: &[f64], y_mean: f64) -> (Mat<f64>, Mat<f64>) {
        let p: usize = self.names.len();
        let size: usize = CHUNK.min(self.rows().max(1));
        let target: &[f64] = self.y();
        let zero = || (Mat::zeros(p, p), Mat::zeros(p, 1), Mat::zeros(size, 1));
        let parts = self.fold(x_means, false, zero, |(g, v, y), start, part| {
            let len: usize = part.nrows();
            for (i, value) in target[start..start + len].iter().enumerate() {
                y[(i, 0)] = value - y_mean;
            }
            matmul(
                g.as_mut(),
                Accum::Add,
                part.transpose(),
                part,
                1.0,
                Par::Seq,
            );
            let rhs = y.as_ref().subrows(0, len);
            matmul(v.as_mut(), Accum::Add, part.transpose(), rhs, 1.0, Par::Seq);
        });
        let (mut g, mut v): (Mat<f64>, Mat<f64>) = (Mat::zeros(p, p), Mat::zeros(p, 1));
        for (block_g, block_v, _) in parts {
            g += block_g;
            v += block_v;
        }
        (g, v)
    }

    /// Runs `f(acc, start, x)` on every chunk of rows, where `x` is the chunk as a
    /// column-major `len × p` matrix of each feature minus its `offset`, plus a last column of
    /// ones when `ones`. Each task on the Polars pool takes one block of rows into its own
    /// `zero()`. The results come back in block order, so sums added in that order are the
    /// same on any number of threads and on every machine.
    pub fn fold<T: Send>(
        &self,
        offset: &[f64],
        ones: bool,
        zero: impl Fn() -> T + Sync,
        f: impl Fn(&mut T, usize, MatRef<'_, f64>) + Sync,
    ) -> Vec<T> {
        let (n, p): (usize, usize) = (self.rows(), self.names.len());
        let size: usize = CHUNK.min(n.max(1));
        let starts: Vec<usize> = (0..n).step_by(BLOCK).collect();
        THREAD_POOL.install(|| {
            starts
                .par_iter()
                .map_init(
                    || Mat::from_fn(size, p + usize::from(ones), |_, j| f64::from(j == p)),
                    |x, &from| {
                        let to: usize = (from + BLOCK).min(n);
                        let mut acc: T = zero();
                        for start in (from..to).step_by(CHUNK) {
                            let len: usize = (to - start).min(CHUNK);
                            for (j, series) in self.columns.iter().enumerate() {
                                let values: &[f64] = &column(series)[start..start + len];
                                let to: &mut [f64] = &mut x.col_as_slice_mut(j)[..len];
                                for (to, value) in to.iter_mut().zip(values) {
                                    *to = value - offset[j];
                                }
                            }
                            f(&mut acc, start, x.as_ref().subrows(0, len));
                        }
                        acc
                    },
                )
                .collect()
        })
    }
}

/// Column `j` of a chunk that [`DesignMatrix::fold`] gives, as a slice.
pub(crate) fn col(x: MatRef<'_, f64>, j: usize) -> &[f64] {
    x.col(j)
        .try_as_col_major()
        .expect("fold gives column-major chunks")
        .as_slice()
}

/// `intercept[k] + Σ coef[k][j] × feature j` for each row and each output k, in one pass per
/// output: each chunk of rows adds `coef × column` into a small buffer that stays in the CPU
/// cache, so each column is read once and no temporary column is made. The chunks run on the
/// Polars pool. Nulls, NaN and text are errors, as in scikit-learn.
pub(crate) fn scores(
    dataset: &Dataset,
    names: &[String],
    coef: &[&[f64]],
    intercept: &[f64],
) -> Result<Vec<Vec<f64>>, ModelError> {
    let columns: Vec<Series> = THREAD_POOL.install(|| {
        names
            .par_iter()
            .map(|name| numeric(dataset, name))
            .collect::<Result<_, _>>()
    })?;
    let out: Vec<Vec<f64>> = coef
        .iter()
        .zip(intercept)
        .map(|(coef, &intercept)| {
            let mut out: Vec<f64> = vec![intercept; dataset.height()];
            // Each chunk is its own task and writes its own rows, so the order of the threads
            // does not change any sum.
            THREAD_POOL.install(|| {
                out.par_chunks_mut(CHUNK).enumerate().for_each(|(k, sums)| {
                    let start: usize = k * CHUNK;
                    let end: usize = start + sums.len();
                    for (series, w) in columns.iter().zip(*coef) {
                        for (sum, x) in sums.iter_mut().zip(&column(series)[start..end]) {
                            *sum += w * x;
                        }
                    }
                })
            });
            out
        })
        .collect();
    // A NaN or an infinity in a feature makes its row's score NaN or infinite, whatever the
    // coefficient (0 × inf is NaN). So the sum of the scores checks every feature without a
    // second read of every column; only a sum that is not finite looks for the column.
    if out.first().is_some_and(|scores| !sum(scores).is_finite()) {
        for (name, series) in names.iter().zip(&columns) {
            finite(name, series)?;
        }
    }
    Ok(out)
}

/// [`prepared`] for every name, one column per task on the Polars pool.
fn prepared_all(
    dataset: &Dataset,
    names: &[String],
) -> Result<Vec<(Series, f64)>, ModelError> {
    THREAD_POOL.install(|| {
        names
            .par_iter()
            .map(|name| prepared(dataset, name))
            .collect()
    })
}

/// The values of a column that [`prepared`] made: `Float64`, one chunk, no nulls.
pub(crate) fn column(series: &Series) -> &[f64] {
    series
        .f64()
        .and_then(|values| values.cont_slice())
        .expect("prepared() makes Float64 columns in one chunk with no nulls")
}

/// The sum of `values` in 8 running sums, so that the additions do not wait for each other.
/// On the 31 columns of S6E1, one thread: 7.6 ms, against 20.5 ms for `iter().sum()` and
/// 13.8 ms for `iter().all(is_finite)`.
fn sum(values: &[f64]) -> f64 {
    let mut lanes: [f64; 8] = [0.0; 8];
    let (chunks, rest): (&[[f64; 8]], &[f64]) = values.as_chunks::<8>();
    for chunk in chunks {
        for (lane, value) in lanes.iter_mut().zip(chunk) {
            *lane += value;
        }
    }
    lanes.iter().sum::<f64>() + rest.iter().sum::<f64>()
}

/// `name` as `Float64` in one chunk: a number column with no nulls. A column that is already
/// that is Polars' own memory, not a copy.
fn numeric(dataset: &Dataset, name: &str) -> Result<Series, ModelError> {
    let source = dataset.column(name)?;
    let dtype: &DataType = source.dtype();
    if !(dtype.is_primitive_numeric() || dtype.is_bool()) {
        return Err(ModelError::NotNumeric {
            column: name.into(),
            dtype: dtype.to_string(),
        });
    }
    let nulls: usize = source.null_count();
    if nulls > 0 {
        return Err(ModelError::FeatureNulls {
            column: name.into(),
            nulls,
        });
    }
    Ok(source
        .as_materialized_series()
        .cast(&DataType::Float64)?
        .rechunk())
}

/// An error when a value of `series` (the column `name`) is NaN or infinite.
fn finite(name: &str, series: &Series) -> Result<(), ModelError> {
    if column(series).iter().all(|value| value.is_finite()) {
        return Ok(());
    }
    Err(ModelError::NotFinite {
        column: name.into(),
    })
}

/// `name` as `Float64` in one chunk, after the checks that every model needs, and its sum.
pub(crate) fn prepared(dataset: &Dataset, name: &str) -> Result<(Series, f64), ModelError> {
    let series: Series = numeric(dataset, name)?;
    // A sum is finite only when every value is, so one pass is the check and gives the mean.
    // A sum that is not finite can also be an overflow of finite values: only then look again.
    let total: f64 = sum(column(&series));
    if !total.is_finite() {
        finite(name, &series)?;
    }
    Ok((series, total))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_adds_every_value_and_shows_a_value_that_is_not_finite() {
        let values: Vec<f64> = (1..=19).map(f64::from).collect(); // two lanes of 8, a rest of 3
        assert_eq!(sum(&values), 190.0);
        assert_eq!(sum(&[]), 0.0);
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut with_bad: Vec<f64> = values.clone();
            with_bad[4] = bad;
            with_bad[12] = -bad; // inf - inf is NaN, never a finite sum
            assert!(!sum(&with_bad).is_finite());
        }
    }
}
