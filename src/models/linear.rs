//! [`LinearRegression`] and [`Ridge`]: least squares from one pass over the data.
//!
//! The pass computes XᵀX and Xᵀy of the centered data, read in chunks of rows. The solve works
//! on that small p × p matrix through its eigendecomposition, which also gives scikit-learn's
//! answer when the features are collinear: one-hot columns of every category add up to 1, so
//! with an intercept XᵀX is singular, and the solve returns the minimum-norm coefficients, as
//! scikit-learn's `lstsq` does.

use faer::{Mat, Side};
use polars::prelude::{NamedFrom, Series};
use polars_core::runtime::THREAD_POOL;
use rayon::prelude::*;

use crate::{
    dataset::Dataset,
    error::OrFatal,
    models::{
        ModelError, Regressor,
        design::{CHUNK, DesignMatrix, column, prepared_all},
    },
};

/// Eigenvalues of XᵀX + αI below this share of the largest one count as zero. Forming XᵀX
/// rounds its entries by about the number of rows times f64's epsilon, so smaller eigenvalues
/// are noise.
// ponytail: the solve works on XᵀX, which squares the condition number. Features that are
// nearly collinear (condition above about 1e5 at 630,000 rows) lose those directions; a QR of
// X itself (faer) keeps them, add it when a dataset needs it.
fn cutoff(rows: usize, largest: f64) -> f64 {
    largest * rows.max(1) as f64 * f64::EPSILON
}

/// The minimum-norm solution w of (G + αI) w = v.
fn solve(mut g: Mat<f64>, v: &Mat<f64>, alpha: f64, rows: usize) -> Result<Vec<f64>, ModelError> {
    let p: usize = g.nrows();
    for i in 0..p {
        g[(i, i)] += alpha;
    }
    let eigen = g
        .self_adjoint_eigen(Side::Lower)
        .map_err(|_| ModelError::Solve)?;
    let (u, s) = (eigen.U(), eigen.S().column_vector());
    let small: f64 = cutoff(rows, s[p - 1].abs()); // eigenvalues come in increasing order
    let mut w: Vec<f64> = vec![0.0; p];
    for k in (0..p).filter(|&k| s[k] > small) {
        let projection: f64 = (0..p).map(|i| u[(i, k)] * v[(i, 0)]).sum::<f64>() / s[k];
        for (i, wi) in w.iter_mut().enumerate() {
            *wi += projection * u[(i, k)];
        }
    }
    Ok(w)
}

/// What a linear fit learns.
#[derive(Debug, Clone, Default)]
struct Learned {
    names: Vec<String>,
    coef: Vec<f64>,
    intercept: f64,
}

/// Least squares with an L2 penalty `alpha` (0 for `LinearRegression`).
fn fit(
    dataset: &Dataset,
    target: &str,
    fit_intercept: bool,
    alpha: f64,
) -> Result<Learned, ModelError> {
    let matrix: DesignMatrix = DesignMatrix::try_new(dataset, target)?;
    let (x_means, y_mean): (Vec<f64>, f64) = if fit_intercept {
        matrix.means()
    } else {
        (vec![0.0; matrix.names.len()], 0.0)
    };
    let (g, v) = matrix.gram(&x_means, y_mean);
    let coef: Vec<f64> = solve(g, &v, alpha, matrix.rows())?;
    let intercept: f64 = y_mean - x_means.iter().zip(&coef).map(|(m, w)| m * w).sum::<f64>();
    Ok(Learned {
        names: matrix.names,
        coef,
        intercept,
    })
}

/// `intercept + Σ coef × feature` for each row, in one pass: each chunk of rows adds
/// `coef × column` into a small buffer that stays in the CPU cache, so each column is read once
/// and no temporary column is made. The chunks run on the Polars pool. Nulls, NaN and text
/// are errors, as in scikit-learn.
fn predict(dataset: &Dataset, learned: &Learned, target: &str) -> Result<Series, ModelError> {
    let columns: Vec<(Series, f64)> = prepared_all(dataset, &learned.names)?;
    let mut out: Vec<f64> = vec![learned.intercept; dataset.height()];
    // Each chunk is its own task and writes its own rows, so the order of the threads does not
    // change any sum.
    THREAD_POOL.install(|| {
        out.par_chunks_mut(CHUNK).enumerate().for_each(|(k, sums)| {
            let start: usize = k * CHUNK;
            let end: usize = start + sums.len();
            for ((series, _), w) in columns.iter().zip(&learned.coef) {
                for (sum, x) in sums.iter_mut().zip(&column(series)[start..end]) {
                    *sum += w * x;
                }
            }
        })
    });
    Ok(Series::new(target.into(), out))
}

/// Ordinary least squares, like scikit-learn's `LinearRegression`.
///
/// `fit` makes one pass over the data for XᵀX and Xᵀy, then solves the small p × p system.
/// When features are collinear (for example one-hot columns of every category, next to the
/// intercept), it returns the minimum-norm coefficients, the same as scikit-learn.
///
/// ```
/// use polars::df;
/// use rusmlai::prelude::*;
/// let ds = Dataset::new(
///     df!("x" => [1.0, 2.0, 3.0, 4.0], "y" => [3.0, 5.0, 7.0, 9.0]).unwrap(),
///     "",
/// );
/// let mut model = LinearRegression::new();
/// model.fit(&ds, "y");
/// assert!((model.coef_[0] - 2.0).abs() < 1e-12 && (model.intercept_ - 1.0).abs() < 1e-12);
/// assert!((model.score(&ds, "y") - 1.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone)]
pub struct LinearRegression {
    fit_intercept: bool,
    /// One coefficient per feature, in the order of `feature_names_in_`, learned by `fit`.
    pub coef_: Vec<f64>,
    /// The constant term, learned by `fit` (0 with `fit_intercept(false)`).
    pub intercept_: f64,
    /// The feature columns `fit` saw.
    pub feature_names_in_: Vec<String>,
    target: Option<String>,
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            coef_: Vec::new(),
            intercept_: 0.0,
            feature_names_in_: Vec::new(),
            target: None,
        }
    }
}

impl LinearRegression {
    /// Least squares with an intercept.
    pub fn new() -> Self {
        Self::default()
    }

    /// `true` (the default) learns a constant term; `false` fits through the origin.
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    fn learned(&self) -> Learned {
        Learned {
            names: self.feature_names_in_.clone(),
            coef: self.coef_.clone(),
            intercept: self.intercept_,
        }
    }
}

impl Regressor for LinearRegression {
    fn try_fit(&mut self, dataset: &Dataset, target: &str) -> Result<(), ModelError> {
        let learned: Learned = fit(dataset, target, self.fit_intercept, 0.0)?;
        (self.feature_names_in_, self.coef_, self.intercept_) =
            (learned.names, learned.coef, learned.intercept);
        self.target = Some(target.into());
        Ok(())
    }

    fn try_predict(&self, dataset: &Dataset) -> Result<Series, ModelError> {
        let target: &String = self.target.as_ref().ok_or(ModelError::NotFitted {
            model: "LinearRegression",
        })?;
        predict(dataset, &self.learned(), target)
    }
}

/// Least squares with an L2 penalty `alpha` on the coefficients, like scikit-learn's `Ridge`.
///
/// The same pass as [`LinearRegression`]; the solve adds `alpha` to the diagonal of XᵀX. The
/// intercept is not penalized.
///
/// ```
/// use polars::df;
/// use rusmlai::prelude::*;
/// let ds = Dataset::new(
///     df!("x" => [1.0, 2.0, 3.0, 4.0], "y" => [3.0, 5.0, 7.0, 9.0]).unwrap(),
///     "",
/// );
/// let mut model = Ridge::new().alpha(5.0);
/// model.fit(&ds, "y");
/// assert!((model.coef_[0] - 1.0).abs() < 1e-12); // pulled toward 0 from 2
/// ```
#[derive(Debug, Clone)]
pub struct Ridge {
    alpha: f64,
    fit_intercept: bool,
    /// One coefficient per feature, in the order of `feature_names_in_`, learned by `fit`.
    pub coef_: Vec<f64>,
    /// The constant term, learned by `fit` (0 with `fit_intercept(false)`).
    pub intercept_: f64,
    /// The feature columns `fit` saw.
    pub feature_names_in_: Vec<String>,
    target: Option<String>,
}

impl Default for Ridge {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            fit_intercept: true,
            coef_: Vec::new(),
            intercept_: 0.0,
            feature_names_in_: Vec::new(),
            target: None,
        }
    }
}

impl Ridge {
    /// Ridge with `alpha = 1.0` and an intercept.
    pub fn new() -> Self {
        Self::default()
    }

    /// The penalty, 0 or more. 0 is plain least squares.
    #[track_caller]
    pub fn alpha(mut self, alpha: f64) -> Self {
        if !(alpha.is_finite() && alpha >= 0.0) {
            Err::<(), ModelError>(ModelError::InvalidSetting {
                setting: "alpha",
                value: alpha.to_string(),
                expected: "a finite number of 0 or more",
            })
            .or_fatal();
        }
        self.alpha = alpha;
        self
    }

    /// `true` (the default) learns a constant term; `false` fits through the origin.
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl Regressor for Ridge {
    fn try_fit(&mut self, dataset: &Dataset, target: &str) -> Result<(), ModelError> {
        let learned: Learned = fit(dataset, target, self.fit_intercept, self.alpha)?;
        (self.feature_names_in_, self.coef_, self.intercept_) =
            (learned.names, learned.coef, learned.intercept);
        self.target = Some(target.into());
        Ok(())
    }

    fn try_predict(&self, dataset: &Dataset) -> Result<Series, ModelError> {
        let target: &String = self
            .target
            .as_ref()
            .ok_or(ModelError::NotFitted { model: "Ridge" })?;
        let learned: Learned = Learned {
            names: self.feature_names_in_.clone(),
            coef: self.coef_.clone(),
            intercept: self.intercept_,
        };
        predict(dataset, &learned, target)
    }
}

#[cfg(test)]
mod tests {
    use polars::df;

    use super::*;

    /// 8 rows; `a` and `b` are one-hot columns of one category column, so a + b = 1.
    fn data() -> Dataset {
        Dataset::new(
            df!(
                "x1" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "x2" => [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 9.5],
                "a" => [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                "b" => [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                "y" => [3.1, 4.2, 7.9, 8.1, 12.2, 12.8, 15.9, 19.3],
            )
            .unwrap(),
            "",
        )
    }

    /// `coef_` and `intercept_` against scikit-learn 1.9.1 on the same data.
    fn assert_fit(coef: &[f64], intercept: f64, expected: &[f64], expected_intercept: f64) {
        for (got, want) in coef.iter().zip(expected) {
            assert!(
                (got - want).abs() < 1e-9,
                "coef {coef:?}, want {expected:?}"
            );
        }
        assert!(
            (intercept - expected_intercept).abs() < 1e-9,
            "intercept {intercept}"
        );
    }

    #[test]
    fn coefficients_match_scikit_learn() {
        let ds: Dataset = data();
        let x: Dataset = ds.drop(&["a", "b"]);
        let mut ols: LinearRegression = LinearRegression::new();
        ols.fit(&x, "y");
        assert_fit(
            &ols.coef_,
            ols.intercept_,
            &[1.4180206378986875, 0.7656285178236396],
            0.3718198874296412,
        );
        assert!((ols.score(&x, "y") - 0.9972840488307306).abs() < 1e-12);

        let mut ridge: Ridge = Ridge::new();
        ridge.fit(&x, "y");
        assert_fit(
            &ridge.coef_,
            ridge.intercept_,
            &[1.3018487997133628, 0.8421497671085648],
            0.5263346470798993,
        );

        let mut origin: LinearRegression = LinearRegression::new().fit_intercept(false);
        origin.fit(&x, "y");
        assert_fit(
            &origin.coef_,
            origin.intercept_,
            &[1.4998620478234215, 0.7505824647455558],
            0.0,
        );
    }

    #[test]
    fn collinear_one_hot_columns_get_the_minimum_norm_answer() {
        let ds: Dataset = data().drop(&["x2"]);
        let mut ols: LinearRegression = LinearRegression::new();
        ols.fit(&ds, "y");
        assert_eq!(ols.feature_names_in_, ["x1", "a", "b"]);
        assert_fit(
            &ols.coef_,
            ols.intercept_,
            &[2.3126666666666673, 0.29700000000000004, -0.2970000000000001],
            0.03049999999999642,
        );

        let mut ridge: Ridge = Ridge::new();
        ridge.fit(&ds, "y");
        assert_fit(
            &ridge.coef_,
            ridge.intercept_,
            &[2.24492385786802, 0.1969543147208121, -0.1969543147208121],
            0.3353426395939092,
        );
    }

    #[test]
    fn errors_name_the_problem() {
        let fit = |ds: Dataset| {
            LinearRegression::new()
                .try_fit(&ds, "y")
                .unwrap_err()
                .to_string()
        };
        let text: Dataset = Dataset::new(df!("s" => ["a", "b"], "y" => [1.0, 2.0]).unwrap(), "");
        assert!(fit(text).starts_with("feature 's' is str, not a number"));
        let nulls: Dataset = Dataset::new(
            df!("x" => [Some(1.0), None], "y" => [1.0, 2.0]).unwrap(),
            "",
        );
        assert!(fit(nulls).starts_with("feature 'x' has 1 nulls"));
        let nan: Dataset =
            Dataset::new(df!("x" => [1.0, f64::NAN], "y" => [1.0, 2.0]).unwrap(), "");
        assert!(fit(nan).starts_with("column 'x' has NaN"));
        let alone: Dataset = Dataset::new(df!("y" => [1.0, 2.0]).unwrap(), "");
        assert!(fit(alone).starts_with("the dataset has no feature columns"));

        let ds: Dataset = data();
        assert!(matches!(
            LinearRegression::new().try_predict(&ds),
            Err(ModelError::NotFitted { .. })
        ));
        let mut model: LinearRegression = LinearRegression::new();
        model.fit(&ds, "y");
        assert!(model.try_predict(&ds.drop(&["x1"])).is_err());
    }
}
