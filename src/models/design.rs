//! [`DesignMatrix`]: the features and the target of a `fit`, checked once, as `f64` columns.

use faer::{Accum, Mat, Par, linalg::matmul::matmul};
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
/// own memory, not a copy. Kernels read the columns in chunks of [`CHUNK`] rows, so a fit never
/// holds a second copy of the whole table. Every pass runs on the Polars thread pool.
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
        let (y, y_sum): (Series, f64) = prepared(dataset, target)?;
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

    /// XᵀX and Xᵀy, with `x_means` taken from the features and `y_mean` from the target first.
    /// Each task sums one block of rows on the Polars pool: chunk by chunk, it copies the rows,
    /// centered, into its own small column-major buffer, and faer adds their product (on one
    /// thread, which was faster than faer's own threads on this tall, narrow shape).
    pub fn gram(&self, x_means: &[f64], y_mean: f64) -> (Mat<f64>, Mat<f64>) {
        let (n, p): (usize, usize) = (self.rows(), self.names.len());
        let size: usize = CHUNK.min(n.max(1));
        let starts: Vec<usize> = (0..n).step_by(BLOCK).collect();
        let parts: Vec<(Mat<f64>, Mat<f64>)> = THREAD_POOL.install(|| {
            starts
                .par_iter()
                .map_init(
                    || (Mat::zeros(size, p), Mat::zeros(size, 1)),
                    |(x, y), &start| {
                        self.block(x, y, start, (start + BLOCK).min(n), x_means, y_mean)
                    },
                )
                .collect()
        });
        let (mut g, mut v): (Mat<f64>, Mat<f64>) = (Mat::zeros(p, p), Mat::zeros(p, 1));
        for (block_g, block_v) in parts {
            g += block_g;
            v += block_v;
        }
        (g, v)
    }

    /// XᵀX and Xᵀy of rows `from..to`, through the buffers `x` and `y` (`CHUNK` rows each).
    fn block(
        &self,
        x: &mut Mat<f64>,
        y: &mut Mat<f64>,
        from: usize,
        to: usize,
        x_means: &[f64],
        y_mean: f64,
    ) -> (Mat<f64>, Mat<f64>) {
        let p: usize = self.names.len();
        let (mut g, mut v): (Mat<f64>, Mat<f64>) = (Mat::zeros(p, p), Mat::zeros(p, 1));
        let target: &[f64] = column(&self.y);
        for start in (from..to).step_by(CHUNK) {
            let len: usize = (to - start).min(CHUNK);
            for (j, series) in self.columns.iter().enumerate() {
                for (i, value) in column(series)[start..start + len].iter().enumerate() {
                    x[(i, j)] = value - x_means[j];
                }
            }
            for (i, value) in target[start..start + len].iter().enumerate() {
                y[(i, 0)] = value - y_mean;
            }
            let part = x.as_ref().subrows(0, len);
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
        }
        (g, v)
    }
}

/// [`prepared`] for every name, one column per task on the Polars pool.
pub(crate) fn prepared_all(
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

/// `name` as `Float64` in one chunk, after the checks that every model needs, and its sum.
pub(crate) fn prepared(dataset: &Dataset, name: &str) -> Result<(Series, f64), ModelError> {
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
    let series: Series = source
        .as_materialized_series()
        .cast(&DataType::Float64)?
        .rechunk();
    // A sum is finite only when every value is, so one pass is the check and gives the mean.
    // A sum that is not finite can also be an overflow of finite values: only then look again.
    let total: f64 = sum(column(&series));
    if !total.is_finite() && !column(&series).iter().all(|value| value.is_finite()) {
        return Err(ModelError::NotFinite {
            column: name.into(),
        });
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
