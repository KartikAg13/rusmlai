//! [`LinearRegression`]: least squares, Ridge, Lasso and ElasticNet as one model.
//!
//! One pass computes XᵀX and Xᵀy of the centered data, read in chunks of rows. Every solve then
//! works on that small p × p matrix:
//!
//! - no L1 part and no sign limit: the eigendecomposition of XᵀX + n·alpha·I. It also gives
//!   scikit-learn's answer when the features are collinear: one-hot columns of every category
//!   add up to 1, so with an intercept XᵀX is singular, and the solve returns the minimum-norm
//!   coefficients, as scikit-learn's `lstsq` does.
//! - an L1 part (`l1_ratio > 0`) or `positive`: coordinate descent on XᵀX ("covariance
//!   updates"), the same steps and the same stopping rule (duality gap) as scikit-learn's
//!   `enet_coordinate_descent_gram`. Each step costs O(p), not O(rows).

use faer::{Mat, Side};
use polars::prelude::{NamedFrom, Series};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    dataset::{Dataset, split::Rng},
    error::{OrFatal, warn},
    models::{
        Model, ModelError, Regressor,
        design::{DesignMatrix, scores},
    },
    preprocess::one_of,
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

/// What one solve gives, besides the intercept.
#[derive(Debug, Default)]
pub(crate) struct Solved {
    pub coef: Vec<f64>,
    n_iter: Option<usize>,
    dual_gap: Option<f64>,
    rank: Option<usize>,
    singular: Option<Vec<f64>>,
    losses: Option<Vec<f64>>,
}

/// The minimum-norm solution w of (G + l2·I) w = v. With `l2 = 0` it also gives the rank and
/// the singular values of the centered X, like scikit-learn's `rank_` and `singular_`.
/// `LogisticRegression` solves its Newton steps with it too.
pub(crate) fn solve(
    mut g: Mat<f64>,
    v: &Mat<f64>,
    l2: f64,
    rows: usize,
) -> Result<Solved, ModelError> {
    let p: usize = g.nrows();
    for i in 0..p {
        g[(i, i)] += l2;
    }
    let eigen = g
        .self_adjoint_eigen(Side::Lower)
        .map_err(|_| ModelError::Solve)?;
    let (u, s) = (eigen.U(), eigen.S().column_vector());
    let small: f64 = cutoff(rows, s[p - 1].abs()); // eigenvalues come in increasing order
    let mut coef: Vec<f64> = vec![0.0; p];
    let kept: Vec<usize> = (0..p).filter(|&k| s[k] > small).collect();
    for &k in &kept {
        let projection: f64 = (0..p).map(|i| u[(i, k)] * v[(i, 0)]).sum::<f64>() / s[k];
        for (i, wi) in coef.iter_mut().enumerate() {
            *wi += projection * u[(i, k)];
        }
    }
    let plain: bool = l2 == 0.0;
    Ok(Solved {
        coef,
        rank: plain.then_some(kept.len()),
        singular: plain.then(|| (0..p).rev().map(|k| s[k].max(0.0).sqrt()).collect()),
        ..Solved::default()
    })
}

/// Penalties on the n-times-larger scale that the Gram solves use: the ElasticNet objective
/// times 2n is ‖y − Xw‖² + 2·l1·‖w‖₁ + l2·‖w‖², with these `l1` and `l2`.
struct Penalty {
    l1: f64,
    l2: f64,
    positive: bool,
}

impl Penalty {
    /// ‖y − Xw‖² + 2·l1·‖w‖₁ + l2·‖w‖², from G·w: O(p).
    fn objective(&self, w: &[f64], gw: &[f64], v: &Mat<f64>, y_norm2: f64) -> f64 {
        let dot = |a: &[f64]| a.iter().zip(w).map(|(x, y)| x * y).sum::<f64>();
        let v_dot_w: f64 = (0..w.len()).map(|j| v[(j, 0)] * w[j]).sum();
        let l1_norm: f64 = w.iter().map(|x| x.abs()).sum();
        y_norm2 + dot(gw) - 2.0 * v_dot_w + 2.0 * self.l1 * l1_norm + self.l2 * dot(w)
    }

    /// How far `w` is from the best answer: scikit-learn's duality gap (its "formulation A")
    /// when there is an L1 part. Without one, only `positive` brings coordinate descent here,
    /// and the gap has no closed form under a sign limit, so it is the squared norm of the
    /// gradient, with the parts that push a zero coefficient below zero left out.
    fn gap(&self, w: &[f64], gw: &[f64], v: &Mat<f64>, y_norm2: f64) -> f64 {
        let grad = |j: usize| v[(j, 0)] - gw[j] - self.l2 * w[j];
        if self.l1 == 0.0 {
            return (0..w.len())
                .map(|j| {
                    let g: f64 = grad(j);
                    if self.positive && w[j] == 0.0 {
                        g.max(0.0)
                    } else {
                        g
                    }
                })
                .map(|g| g * g)
                .sum();
        }
        let dot = |a: &[f64]| a.iter().zip(w).map(|(x, y)| x * y).sum::<f64>();
        let v_dot_w: f64 = (0..w.len()).map(|j| v[(j, 0)] * w[j]).sum();
        let r_norm2: f64 = y_norm2 + dot(gw) - 2.0 * v_dot_w;
        let ry: f64 = y_norm2 - v_dot_w;
        let l2_part: f64 = self.l2 * dot(w);
        let dual_norm: f64 = (0..w.len())
            .map(|j| {
                if self.positive {
                    grad(j)
                } else {
                    grad(j).abs()
                }
            })
            .fold(f64::NEG_INFINITY, f64::max);
        let primal: f64 =
            0.5 * (r_norm2 + l2_part) + self.l1 * w.iter().map(|x| x.abs()).sum::<f64>();
        let scale: f64 = if dual_norm > self.l1 {
            self.l1 / dual_norm
        } else {
            1.0
        };
        primal - (-0.5 * scale * scale * (r_norm2 + l2_part) + scale * ry)
    }
}

/// Coordinate descent on G = XᵀX and v = Xᵀy, from `w`, as scikit-learn's
/// `enet_coordinate_descent_gram` (without its screening, which only skips work). Returns the
/// coefficients, the number of rounds, the duality gap and the objective after each round
/// (times 2n, the scale of the Gram solves).
fn coordinate_descent(
    g: &Mat<f64>,
    v: &Mat<f64>,
    y_norm2: f64,
    penalty: &Penalty,
    params: &Params,
    mut w: Vec<f64>,
) -> (Vec<f64>, usize, f64, Vec<f64>) {
    let p: usize = w.len();
    let mut gw: Vec<f64> = (0..p)
        .map(|i| (0..p).map(|k| g[(i, k)] * w[k]).sum())
        .collect();
    let tol: f64 = params.tol * y_norm2;
    let mut gap: f64 = penalty.gap(&w, &gw, v, y_norm2);
    if (0.0..=tol).contains(&gap) {
        return (w, 0, gap, Vec::new());
    }
    let mut losses: Vec<f64> = Vec::new();
    let mut rng: Rng = Rng::new(params.random_state);
    let random: bool = params.selection == "random";
    let mut rounds: usize = 0;
    let mut converged: bool = false;
    for round in 0..params.max_iter {
        rounds = round + 1;
        let (mut w_max, mut step_max): (f64, f64) = (0.0, 0.0);
        for f in 0..p {
            let j: usize = if random {
                (rng.next_u64() % p as u64) as usize
            } else {
                f
            };
            let gjj: f64 = g[(j, j)];
            if gjj == 0.0 {
                continue;
            }
            let old: f64 = w[j];
            let tmp: f64 = v[(j, 0)] - gw[j] + old * gjj;
            w[j] = if penalty.positive && tmp < 0.0 {
                0.0
            } else {
                tmp.signum() * (tmp.abs() - penalty.l1).max(0.0) / (gjj + penalty.l2)
            };
            if w[j] != old {
                let step: f64 = w[j] - old;
                for (k, gwk) in gw.iter_mut().enumerate() {
                    *gwk += step * g[(k, j)];
                }
            }
            step_max = step_max.max((w[j] - old).abs());
            w_max = w_max.max(w[j].abs());
        }
        losses.push(penalty.objective(&w, &gw, v, y_norm2));
        if w_max == 0.0 || step_max / w_max <= params.tol || round + 1 == params.max_iter {
            gap = penalty.gap(&w, &gw, v, y_norm2);
            if gap <= tol {
                converged = true;
                break;
            }
        }
    }
    if !converged {
        warn(format_args!(
            "LinearRegression did not converge in max_iter = {} rounds (duality gap {gap:.3e}, \
             tolerance {tol:.3e}); raise max_iter or tol, or scale the features",
            params.max_iter
        ));
    }
    (w, rounds, gap, losses)
}

/// The settings of a [`LinearRegression`], with scikit-learn's names.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Params {
    fit_intercept: bool,
    alpha: f64,
    l1_ratio: f64,
    positive: bool,
    max_iter: usize,
    tol: f64,
    selection: String,
    random_state: u64,
    warm_start: bool,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            alpha: 0.0,
            l1_ratio: 0.0,
            positive: false,
            max_iter: 1000,
            tol: 1e-4,
            selection: "cyclic".into(),
            random_state: 42,
            warm_start: false,
        }
    }
}

/// Least squares with an optional ElasticNet penalty: scikit-learn's `LinearRegression`,
/// `Ridge`, `Lasso` and `ElasticNet` in one model. It minimizes
///
/// ```text
/// 1 / (2 · rows) · ‖y − Xw‖² + alpha · l1_ratio · ‖w‖₁ + alpha · (1 − l1_ratio) / 2 · ‖w‖²
/// ```
///
/// which is scikit-learn's `ElasticNet` objective. `alpha` and `l1_ratio` are both 0 unless you
/// set them, so the default is plain least squares.
///
/// | rusmlai | scikit-learn |
/// |---|---|
/// | `LinearRegression::new()` | `LinearRegression()` |
/// | `.alpha(a)` | `Ridge(alpha=a * rows)`, the same as `ElasticNet(alpha=a, l1_ratio=0)` |
/// | `.alpha(a).l1_ratio(1.0)` | `Lasso(alpha=a)` |
/// | `.alpha(a).l1_ratio(r)` | `ElasticNet(alpha=a, l1_ratio=r)` |
/// | `.positive(true)` | `LinearRegression(positive=True)` |
///
/// `Ridge` scales its penalty by the number of rows and `ElasticNet` does not, so one `alpha`
/// can't match both: rusmlai takes ElasticNet's. The intercept is never penalized.
///
/// `fit` makes one pass over the data for XᵀX and Xᵀy. Without an L1 part and without
/// `positive`, it solves that p × p system exactly; when features are collinear (for example
/// one-hot columns of every category, next to the intercept), it returns the minimum-norm
/// coefficients, the same as scikit-learn. Otherwise it runs coordinate descent on it, with
/// scikit-learn's steps and stopping rule.
///
/// scikit-learn settings with no setter: `copy_X` (nothing is copied), `n_jobs` (the threads
/// come from the Polars pool, `POLARS_MAX_THREADS`), `precompute` (the Gram matrix is always
/// used), `solver` (the exact solve gives what every scikit-learn solver converges to).
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
///
/// let mut ridge = LinearRegression::new().alpha(5.0 / 4.0); // Ridge(alpha=5) on 4 rows
/// ridge.fit(&ds, "y");
/// assert!((ridge.coef_[0] - 1.0).abs() < 1e-12); // pulled toward 0 from 2
/// ```
///
/// Python:
///
/// ```python
/// import pandas as pd
/// from sklearn.linear_model import LinearRegression, Ridge
/// ds = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [3.0, 5.0, 7.0, 9.0]})
/// model = LinearRegression().fit(ds[["x"]], ds["y"])
/// assert abs(model.coef_[0] - 2.0) < 1e-12 and abs(model.intercept_ - 1.0) < 1e-12
/// assert abs(model.score(ds[["x"]], ds["y"]) - 1.0) < 1e-12
///
/// ridge = Ridge(alpha=5.0).fit(ds[["x"]], ds["y"])
/// assert abs(ridge.coef_[0] - 1.0) < 1e-12
/// ```
///
/// Lasso and ElasticNet:
///
/// ```
/// use polars::df;
/// use rusmlai::prelude::*;
/// let ds = Dataset::new(
///     df!("x" => [1.0, 2.0, 3.0, 4.0], "noise" => [0.3, -0.1, 0.2, -0.4],
///         "y" => [3.0, 5.0, 7.0, 9.0]).unwrap(),
///     "",
/// );
/// let mut lasso = LinearRegression::new().alpha(0.5).l1_ratio(1.0);
/// lasso.fit(&ds, "y");
/// assert_eq!(lasso.coef_[1], 0.0); // L1 sets the useless feature to exactly 0
/// assert!(lasso.n_iter_.unwrap() > 0);
/// ```
///
/// Python:
///
/// ```python
/// import pandas as pd
/// from sklearn.linear_model import Lasso
/// ds = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "noise": [0.3, -0.1, 0.2, -0.4],
///                    "y": [3.0, 5.0, 7.0, 9.0]})
/// lasso = Lasso(alpha=0.5).fit(ds[["x", "noise"]], ds["y"])
/// assert lasso.coef_[1] == 0.0
/// assert lasso.n_iter_ > 0
/// ```
#[derive(Debug, Clone)]
pub struct LinearRegression {
    params: Params,
    /// One coefficient per feature, in the order of `feature_names_in_`, learned by `fit`.
    pub coef_: Vec<f64>,
    /// The constant term, learned by `fit` (0 with `fit_intercept(false)`).
    pub intercept_: f64,
    /// The feature columns `fit` saw.
    pub feature_names_in_: Vec<String>,
    /// How many features `fit` saw.
    pub n_features_in_: usize,
    /// Rounds of coordinate descent; `None` when the exact solve ran.
    pub n_iter_: Option<usize>,
    /// The duality gap at the end of coordinate descent, on the scale of the objective above;
    /// `None` when the exact solve ran.
    pub dual_gap_: Option<f64>,
    /// The rank of the centered features; only for plain least squares (`alpha = 0`, not
    /// `positive`).
    pub rank_: Option<usize>,
    /// The singular values of the centered features, largest first; only when `rank_` is set.
    pub singular_: Option<Vec<f64>>,
    /// The objective above after each round of coordinate descent, or one value (the
    /// objective of the answer) when the exact solve ran. The journal draws it.
    pub loss_curve_: Option<Vec<f64>>,
    target: Option<String>,
    run: Option<String>,
}

/// The learned values, as the model cache stores them.
#[derive(Serialize, Deserialize)]
struct Learned {
    coef_: Vec<f64>,
    intercept_: f64,
    feature_names_in_: Vec<String>,
    n_iter_: Option<usize>,
    dual_gap_: Option<f64>,
    rank_: Option<usize>,
    singular_: Option<Vec<f64>>,
    #[serde(default)]
    loss_curve_: Option<Vec<f64>>,
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self {
            params: Params::default(),
            coef_: Vec::new(),
            intercept_: 0.0,
            feature_names_in_: Vec::new(),
            n_features_in_: 0,
            n_iter_: None,
            dual_gap_: None,
            rank_: None,
            singular_: None,
            loss_curve_: None,
            target: None,
            run: None,
        }
    }
}

/// Panics at the caller's line with [`ModelError::InvalidSetting`] unless `ok`.
#[track_caller]
fn check(ok: bool, setting: &'static str, value: impl ToString, expected: &'static str) {
    if !ok {
        Err::<(), ModelError>(ModelError::InvalidSetting {
            setting,
            value: value.to_string(),
            expected,
        })
        .or_fatal();
    }
}

impl LinearRegression {
    /// Plain least squares with an intercept.
    pub fn new() -> Self {
        Self::default()
    }

    /// `true` (the default) learns a constant term; `false` fits through the origin.
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.params.fit_intercept = fit_intercept;
        self
    }

    /// The strength of the penalty, 0 (the default, no penalty) or more, on ElasticNet's scale.
    #[track_caller]
    pub fn alpha(mut self, alpha: f64) -> Self {
        check(
            alpha.is_finite() && alpha >= 0.0,
            "alpha",
            alpha,
            "a finite number of 0 or more",
        );
        self.params.alpha = alpha;
        self
    }

    /// The L1 share of the penalty, from 0 (the default, all L2: Ridge) to 1 (all L1: Lasso).
    #[track_caller]
    pub fn l1_ratio(mut self, l1_ratio: f64) -> Self {
        check(
            (0.0..=1.0).contains(&l1_ratio),
            "l1_ratio",
            l1_ratio,
            "a number from 0 to 1",
        );
        self.params.l1_ratio = l1_ratio;
        self
    }

    /// `true` keeps every coefficient at 0 or more. Default `false`.
    pub fn positive(mut self, positive: bool) -> Self {
        self.params.positive = positive;
        self
    }

    /// The most rounds of coordinate descent. Default 1000.
    #[track_caller]
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        check(max_iter > 0, "max_iter", max_iter, "1 or more");
        self.params.max_iter = max_iter;
        self
    }

    /// Coordinate descent stops when the duality gap is below `tol` times ‖y − ȳ‖². Default 1e-4.
    #[track_caller]
    pub fn tol(mut self, tol: f64) -> Self {
        check(
            tol.is_finite() && tol >= 0.0,
            "tol",
            tol,
            "a finite number of 0 or more",
        );
        self.params.tol = tol;
        self
    }

    /// `"cyclic"` (the default) updates the coefficients in order; `"random"` picks one at
    /// random each step, seeded by `random_state`, which can converge faster when `tol` is
    /// above 1e-4.
    #[track_caller]
    pub fn selection(mut self, selection: &str) -> Self {
        self.params.selection = one_of("selection", selection, &["cyclic", "random"]).or_fatal();
        self
    }

    /// The seed of `selection("random")`. Default 42.
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.params.random_state = random_state;
        self
    }

    /// `true` starts coordinate descent from the coefficients of the last `fit` when it saw
    /// the same features. Default `false`.
    pub fn warm_start(mut self, warm_start: bool) -> Self {
        self.params.warm_start = warm_start;
        self
    }

    /// Coordinate descent is needed for an L1 part or a sign limit.
    fn descent(&self) -> bool {
        self.params.positive || (self.params.alpha > 0.0 && self.params.l1_ratio > 0.0)
    }

    fn solve(
        &self,
        dataset: &Dataset,
        target: &str,
    ) -> Result<(Vec<String>, Solved, f64), ModelError> {
        let p: &Params = &self.params;
        let matrix: DesignMatrix = DesignMatrix::try_new(dataset, target)?;
        let (x_means, y_mean): (Vec<f64>, f64) = if p.fit_intercept {
            matrix.means()
        } else {
            (vec![0.0; matrix.names.len()], 0.0)
        };
        let (g, v) = matrix.gram(&x_means, y_mean);
        let rows: f64 = matrix.rows() as f64;
        let penalty: Penalty = Penalty {
            l1: p.alpha * p.l1_ratio * rows,
            l2: p.alpha * (1.0 - p.l1_ratio) * rows,
            positive: p.positive,
        };
        let solved: Solved = if self.descent() {
            let start: Vec<f64> = if p.warm_start && self.feature_names_in_ == matrix.names {
                self.coef_.clone()
            } else {
                vec![0.0; matrix.names.len()]
            };
            let y_norm2: f64 = matrix.y_norm2(y_mean);
            let (coef, rounds, gap, losses) =
                coordinate_descent(&g, &v, y_norm2, &penalty, p, start);
            Solved {
                coef,
                n_iter: Some(rounds),
                dual_gap: Some(gap / rows),
                // On scikit-learn's scale: the objective in the doc above.
                losses: Some(losses.into_iter().map(|l| l / (2.0 * rows)).collect()),
                ..Solved::default()
            }
        } else {
            let mut solved: Solved = solve(g.clone(), &v, penalty.l2, matrix.rows())?;
            // One point, the loss of the answer: G·w is p × p work, ‖y − ȳ‖² one column.
            let w: &[f64] = &solved.coef;
            let gw: Vec<f64> = (0..w.len())
                .map(|i| w.iter().enumerate().map(|(k, wk)| g[(i, k)] * wk).sum())
                .collect();
            let loss: f64 = penalty.objective(w, &gw, &v, matrix.y_norm2(y_mean));
            solved.losses = Some(vec![loss / (2.0 * rows)]);
            solved
        };
        let intercept: f64 = y_mean
            - x_means
                .iter()
                .zip(&solved.coef)
                .map(|(m, w)| m * w)
                .sum::<f64>();
        Ok((matrix.names, solved, intercept))
    }
}

impl Model for LinearRegression {
    fn try_train(&mut self, dataset: &Dataset, target: &str) -> Result<(), ModelError> {
        let (names, solved, intercept) = self.solve(dataset, target)?;
        self.n_features_in_ = names.len();
        self.feature_names_in_ = names;
        self.coef_ = solved.coef;
        self.intercept_ = intercept;
        self.n_iter_ = solved.n_iter;
        self.dual_gap_ = solved.dual_gap;
        self.rank_ = solved.rank;
        self.singular_ = solved.singular;
        self.loss_curve_ = solved.losses;
        self.target = Some(target.into());
        Ok(())
    }

    fn params(&self) -> Value {
        serde_json::to_value(&self.params).unwrap_or_default()
    }

    fn learned(&self) -> Value {
        serde_json::to_value(Learned {
            coef_: self.coef_.clone(),
            intercept_: self.intercept_,
            feature_names_in_: self.feature_names_in_.clone(),
            n_iter_: self.n_iter_,
            dual_gap_: self.dual_gap_,
            rank_: self.rank_,
            singular_: self.singular_.clone(),
            loss_curve_: self.loss_curve_.clone(),
        })
        .unwrap_or_default()
    }

    fn try_set_params(&mut self, params: Value) -> Result<(), ModelError> {
        self.params = serde_json::from_value(params)?;
        Ok(())
    }

    fn try_load(&mut self, learned: Value, target: &str) -> Result<(), ModelError> {
        let l: Learned = serde_json::from_value(learned)?;
        self.n_features_in_ = l.feature_names_in_.len();
        (self.coef_, self.intercept_, self.feature_names_in_) =
            (l.coef_, l.intercept_, l.feature_names_in_);
        (self.n_iter_, self.dual_gap_, self.rank_, self.singular_) =
            (l.n_iter_, l.dual_gap_, l.rank_, l.singular_);
        self.loss_curve_ = l.loss_curve_;
        self.target = Some(target.into());
        Ok(())
    }

    fn run(&self) -> Option<&str> {
        self.run.as_deref()
    }

    fn loss_curve(&self) -> Option<Vec<f64>> {
        self.loss_curve_.clone()
    }

    fn set_run(&mut self, run: Option<String>) {
        self.run = run;
    }
}

impl Regressor for LinearRegression {
    fn try_predict(&self, dataset: &Dataset) -> Result<Series, ModelError> {
        let target: &String = self.target.as_ref().ok_or(ModelError::NotFitted {
            model: "LinearRegression",
        })?;
        let mut out: Vec<Vec<f64>> = scores(
            dataset,
            &self.feature_names_in_,
            &[&self.coef_],
            &[self.intercept_],
        )?;
        Ok(Series::new(target.into(), out.pop().unwrap_or_default()))
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

    /// `x1`, `x2` and `y` only.
    fn plain() -> Dataset {
        data().drop(&["a", "b"])
    }

    fn fitted(model: LinearRegression, ds: &Dataset) -> LinearRegression {
        let mut model: LinearRegression = model;
        model.fit(ds, "y");
        model
    }

    /// `coef_` and `intercept_` against scikit-learn 1.9.1 on the same data.
    fn assert_fit(model: &LinearRegression, expected: &[f64], expected_intercept: f64, eps: f64) {
        for (got, want) in model.coef_.iter().zip(expected) {
            assert!(
                (got - want).abs() < eps,
                "coef {:?}, want {expected:?}",
                model.coef_
            );
        }
        let intercept: f64 = model.intercept_;
        assert!(
            (intercept - expected_intercept).abs() < eps,
            "intercept {intercept}"
        );
    }

    #[test]
    fn least_squares_and_ridge_match_scikit_learn() {
        let x: Dataset = plain();
        let ols: LinearRegression = fitted(LinearRegression::new(), &x);
        assert_fit(
            &ols,
            &[1.4180206378986875, 0.7656285178236396],
            0.3718198874296412,
            1e-9,
        );
        assert!((ols.score(&x, "y") - 0.9972840488307306).abs() < 1e-12);
        assert_eq!(ols.rank_, Some(2));
        let singular: Vec<f64> = ols.singular_.clone().unwrap();
        assert!((singular[0] - 9.929242295586475).abs() < 1e-9);
        assert!((singular[1] - 1.8381777486240647).abs() < 1e-9);
        assert_eq!((ols.n_iter_, ols.dual_gap_), (None, None));

        // Ridge(alpha=1) on 8 rows.
        let ridge: LinearRegression = fitted(LinearRegression::new().alpha(1.0 / 8.0), &x);
        assert_fit(
            &ridge,
            &[1.3018487997133628, 0.8421497671085648],
            0.5263346470798993,
            1e-9,
        );
        assert_eq!(ridge.rank_, None);

        let origin: LinearRegression = fitted(LinearRegression::new().fit_intercept(false), &x);
        assert_fit(
            &origin,
            &[1.4998620478234215, 0.7505824647455558],
            0.0,
            1e-9,
        );
    }

    #[test]
    fn lasso_elastic_net_and_positive_match_scikit_learn() {
        let x: Dataset = plain();
        let tight = || LinearRegression::new().tol(1e-12);
        let lasso: LinearRegression = fitted(tight().alpha(0.5).l1_ratio(1.0), &x);
        assert_fit(
            &lasso,
            &[1.2592964352794886, 0.8226641650973867],
            0.811594746711128,
            1e-9,
        );
        // The default tol stops where scikit-learn stops, after the same 51 rounds.
        let early: LinearRegression = fitted(LinearRegression::new().alpha(0.5).l1_ratio(1.0), &x);
        assert_fit(
            &early,
            &[1.2600571154247946, 0.8220711596271534],
            0.8110255248827478,
            1e-9,
        );
        assert_eq!(early.n_iter_, Some(51));
        assert!((early.dual_gap_.unwrap() - 0.0006655559327843896).abs() < 1e-9);

        let enet: LinearRegression = fitted(tight().alpha(0.5).l1_ratio(0.3), &x);
        assert_fit(
            &enet,
            &[1.1494995829897572, 0.9123512017561103],
            0.8740617180948114,
            1e-9,
        );
        let origin: LinearRegression =
            fitted(tight().alpha(0.5).l1_ratio(1.0).fit_intercept(false), &x);
        assert_fit(
            &origin,
            &[1.4379368486012722, 0.7898221949346386],
            0.0,
            1e-9,
        );

        // `-x2` wants a negative coefficient; `positive` holds it at 0.
        let flipped: Dataset = Dataset::new(
            df!(
                "x1" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "neg" => [-2.0, -1.0, -4.0, -3.0, -6.0, -5.0, -8.0, -9.5],
                "y" => [3.1, 4.2, 7.9, 8.1, 12.2, 12.8, 15.9, 19.3],
            )
            .unwrap(),
            "",
        );
        let nnls: LinearRegression = fitted(tight().positive(true), &flipped);
        assert_fit(&nnls, &[2.270238095238096, 0.0], 0.22142857142856798, 1e-9);
        let positive_enet: LinearRegression =
            fitted(tight().alpha(0.5).l1_ratio(0.3).positive(true), &flipped);
        assert_fit(&positive_enet, &[2.1015625, 0.0], 0.98046875, 1e-9);
    }

    #[test]
    fn random_selection_and_warm_start_reach_the_same_answer() {
        let x: Dataset = plain();
        let random: LinearRegression = fitted(
            LinearRegression::new()
                .alpha(0.5)
                .l1_ratio(1.0)
                .tol(1e-12)
                .selection("random"),
            &x,
        );
        assert_fit(
            &random,
            &[1.2592964352794886, 0.8226641650973867],
            0.811594746711128,
            1e-8,
        );

        let mut warm: LinearRegression = LinearRegression::new()
            .alpha(0.5)
            .l1_ratio(1.0)
            .tol(1e-12)
            .warm_start(true);
        warm.fit(&x, "y");
        let cold_rounds: usize = warm.n_iter_.unwrap();
        warm.fit(&x, "y");
        assert!(
            warm.n_iter_.unwrap() < cold_rounds,
            "a warm start begins at the answer"
        );
    }

    #[test]
    fn collinear_one_hot_columns_get_the_minimum_norm_answer() {
        let ds: Dataset = data().drop(&["x2"]);
        let ols: LinearRegression = fitted(LinearRegression::new(), &ds);
        assert_eq!(ols.feature_names_in_, ["x1", "a", "b"]);
        assert_eq!((ols.n_features_in_, ols.rank_), (3, Some(2)));
        assert_fit(
            &ols,
            &[2.3126666666666673, 0.29700000000000004, -0.2970000000000001],
            0.03049999999999642,
            1e-9,
        );
        let ridge: LinearRegression = fitted(LinearRegression::new().alpha(1.0 / 8.0), &ds);
        assert_fit(
            &ridge,
            &[2.24492385786802, 0.1969543147208121, -0.1969543147208121],
            0.3353426395939092,
            1e-9,
        );
    }

    #[test]
    fn errors_name_the_problem() {
        let fit = |ds: Dataset| {
            LinearRegression::new()
                .try_train(&ds, "y")
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
        let model: LinearRegression = fitted(LinearRegression::new(), &ds);
        assert!(model.try_predict(&ds.drop(&["x1"])).is_err());

        // `predict` checks its features through its output: a 0 coefficient still shows them.
        let frame = |noise: f64| {
            Dataset::new(
                df!("x" => [1.0, 2.0, 3.0, 4.0], "noise" => [0.3, -0.1, 0.2, noise],
                    "y" => [3.0, 5.0, 7.0, 9.0])
                .unwrap(),
                "",
            )
        };
        let lasso: LinearRegression =
            fitted(LinearRegression::new().alpha(0.5).l1_ratio(1.0), &frame(-0.4));
        assert_eq!(lasso.coef_[1], 0.0);
        for bad in [f64::INFINITY, f64::NAN] {
            let error: String = lasso.try_predict(&frame(bad)).unwrap_err().to_string();
            assert!(error.starts_with("column 'noise' has NaN"), "{error}");
        }
    }

    #[test]
    #[should_panic(expected = "l1_ratio = 2 is not valid")]
    fn a_bad_setting_panics_at_the_setter() {
        let _ = LinearRegression::new().l1_ratio(2.0);
    }
}
