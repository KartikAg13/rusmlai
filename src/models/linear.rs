//! [`LinearRegression`], [`Ridge`], [`Lasso`] and [`ElasticNet`]: scikit-learn's four linear
//! regressors, each a type with scikit-learn's settings and defaults, over one solve.
//!
//! One pass computes XᵀX and Xᵀy of the centered data, read in chunks of rows. Every solve then
//! works on that small p × p matrix:
//!
//! - no L1 part and no sign limit: the eigendecomposition of XᵀX + alpha·I (Ridge's alpha). It also gives
//!   scikit-learn's answer when the features are collinear: one-hot columns of every category
//!   add up to 1, so with an intercept XᵀX is singular, and the solve returns the minimum-norm
//!   coefficients, as scikit-learn's `lstsq` does.
//! - an L1 part ([`Lasso`], [`ElasticNet`]) or `positive`: coordinate descent on XᵀX ("covariance
//!   updates"), the same steps and the same stopping rule (duality gap) as scikit-learn's
//!   `enet_coordinate_descent_gram`. Each step costs O(p), not O(rows).

use faer::{Mat, Side};
use polars::prelude::{NamedFrom, Series};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;

use crate::{
    dataset::{Dataset, split::Rng},
    error::{OrFatal, warn},
    metadata::models::CachedRun,
    models::{
        Model, ModelError, Regressor, bad_learned, check,
        design::{DesignMatrix, scores},
    },
    preprocess::one_of,
};
use crate::models::{at_least_zero};

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
    params: &Engine,
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
            "{} did not converge in max_iter = {} rounds (duality gap {gap:.3e}, tolerance \
             {tol:.3e}); raise max_iter or tol, or scale the features",
            params.model, params.max_iter
        ));
    }
    (w, rounds, gap, losses)
}

/// What one solve runs with: every linear model's settings, the penalty on scikit-learn's
/// ElasticNet scale.
#[derive(Debug, Clone)]
struct Engine {
    /// The model's name, for the convergence warning.
    model: &'static str,
    fit_intercept: bool,
    /// The penalty's strength. With `ridge_scale` it is Ridge's (‖y − Xw‖² + α‖w‖²), divided
    /// by the rows before the solve; otherwise ElasticNet's.
    alpha: f64,
    ridge_scale: bool,
    l1_ratio: f64,
    positive: bool,
    max_iter: usize,
    tol: f64,
    selection: String,
    random_state: u64,
    warm_start: bool,
}

impl Engine {
    /// No penalty, an intercept, coordinate descent run to its end: what `LinearRegression`
    /// solves (it only needs coordinate descent for `positive`, where scikit-learn's `nnls`
    /// is exact).
    fn exact(model: &'static str, fit_intercept: bool, positive: bool) -> Self {
        Self {
            model,
            fit_intercept,
            alpha: 0.0,
            ridge_scale: false,
            l1_ratio: 0.0,
            positive,
            max_iter: 100_000,
            tol: 1e-12,
            selection: "cyclic".into(),
            random_state: 42,
            warm_start: false,
        }
    }

    /// Coordinate descent is needed for an L1 part or a sign limit.
    fn descent(&self) -> bool {
        self.positive || (self.alpha > 0.0 && self.l1_ratio > 0.0)
    }
}

/// The settings of one linear model under scikit-learn's names: what they make of the solve.
trait Settings: Clone + std::fmt::Debug + PartialEq + Serialize + DeserializeOwned {
    fn engine(&self) -> Engine;
}

/// `LinearRegression`'s settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct OlsParams {
    fit_intercept: bool,
    positive: bool,
}

impl Settings for OlsParams {
    fn engine(&self) -> Engine {
        Engine::exact("LinearRegression", self.fit_intercept, self.positive)
    }
}

/// `Ridge`'s settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RidgeParams {
    alpha: f64,
    fit_intercept: bool,
    positive: bool,
    /// `None`: until converged, at most 1,000 rounds (only `positive` iterates).
    max_iter: Option<usize>,
    tol: f64,
}

impl Settings for RidgeParams {
    fn engine(&self) -> Engine {
        Engine {
            alpha: self.alpha,
            ridge_scale: true,
            max_iter: self.max_iter.unwrap_or(1000),
            tol: self.tol,
            ..Engine::exact("Ridge", self.fit_intercept, self.positive)
        }
    }
}

/// The settings of `Lasso` (`l1_ratio` always 1) and `ElasticNet`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DescentParams {
    alpha: f64,
    /// `ElasticNet`'s only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    l1_ratio: Option<f64>,
    fit_intercept: bool,
    max_iter: usize,
    tol: f64,
    warm_start: bool,
    positive: bool,
    random_state: u64,
    selection: String,
}

impl DescentParams {
    fn new(l1_ratio: Option<f64>) -> Self {
        Self {
            alpha: 1.0,
            l1_ratio,
            fit_intercept: true,
            max_iter: 1000,
            tol: 1e-4,
            warm_start: false,
            positive: false,
            random_state: 42,
            selection: "cyclic".into(),
        }
    }
}

impl Settings for DescentParams {
    fn engine(&self) -> Engine {
        Engine {
            model: if self.l1_ratio.is_some() { "ElasticNet" } else { "Lasso" },
            fit_intercept: self.fit_intercept,
            alpha: self.alpha,
            ridge_scale: false,
            l1_ratio: self.l1_ratio.unwrap_or(1.0),
            positive: self.positive,
            max_iter: self.max_iter,
            tol: self.tol,
            selection: self.selection.clone(),
            random_state: self.random_state,
            warm_start: self.warm_start,
        }
    }
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

/// Fits `e` on `dataset`: the feature names, the solve and the intercept. `start` and
/// `seen` are the coefficients and features of the last fit, for a warm start.
fn fit_linear(
    e: &Engine,
    dataset: &Dataset,
    target: &str,
    (start, seen): (&[f64], &[String]),
) -> Result<(Vec<String>, Solved, f64), ModelError> {
    let matrix: DesignMatrix = DesignMatrix::try_new(dataset, target)?;
    let (x_means, y_mean): (Vec<f64>, f64) = if e.fit_intercept {
        matrix.means()
    } else {
        (vec![0.0; matrix.names.len()], 0.0)
    };
    let (g, v) = matrix.gram(&x_means, y_mean);
    let rows: f64 = matrix.rows() as f64;
    // Ridge's alpha is ElasticNet's times the rows.
    let alpha: f64 = if e.ridge_scale { e.alpha / rows } else { e.alpha };
    let penalty: Penalty = Penalty {
        l1: alpha * e.l1_ratio * rows,
        l2: alpha * (1.0 - e.l1_ratio) * rows,
        positive: e.positive,
    };
    let solved: Solved = if e.descent() {
        let start: Vec<f64> = if e.warm_start && seen == matrix.names {
            start.to_vec()
        } else {
            vec![0.0; matrix.names.len()]
        };
        let y_norm2: f64 = matrix.y_norm2(y_mean);
        let (coef, rounds, gap, losses) = coordinate_descent(&g, &v, y_norm2, &penalty, e, start);
        Solved {
            coef,
            n_iter: Some(rounds),
            dual_gap: Some(gap / rows),
            // On scikit-learn's ElasticNet scale: 1 / (2 · rows) of the Gram scale.
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
    let intercept: f64 = y_mean - x_means.iter().zip(&solved.coef).map(|(m, w)| m * w).sum::<f64>();
    Ok((matrix.names, solved, intercept))
}

/// The struct, `Model` impl and `Regressor` impl of a linear model whose settings are
/// `$params` (a [`Settings`]): each scikit-learn class is a type of its own over one solve.
macro_rules! linear_model {
    ($(#[$meta:meta])* pub struct $name:ident($params:ty = $default:expr);) => {
        $(#[$meta])*
        #[derive(Debug, Clone)]
        pub struct $name {
            params: $params,
            /// One coefficient per feature, in the order of `feature_names_in_`, learned by
            /// `fit`.
            pub coef_: Vec<f64>,
            /// The constant term, learned by `fit` (0 with `fit_intercept(false)`).
            pub intercept_: f64,
            /// The feature columns `fit` saw.
            pub feature_names_in_: Vec<String>,
            /// How many features `fit` saw.
            pub n_features_in_: usize,
            /// Rounds of coordinate descent; `None` when the exact solve ran.
            pub n_iter_: Option<usize>,
            /// The duality gap at the end of coordinate descent, on ElasticNet's scale;
            /// `None` when the exact solve ran.
            pub dual_gap_: Option<f64>,
            /// The rank of the centered features; only for a solve with no penalty.
            pub rank_: Option<usize>,
            /// The singular values of the centered features, largest first; only when
            /// `rank_` is set.
            pub singular_: Option<Vec<f64>>,
            /// ElasticNet's objective after each round of coordinate descent, or one value
            /// (that of the answer) when the exact solve ran. The journal draws it.
            pub loss_curve_: Option<Vec<f64>>,
            target: Option<String>,
            run: Option<CachedRun>,
        }

        impl Default for $name {
            fn default() -> Self {
                Self {
                    params: $default,
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

        impl Model for $name {
            fn target(&self) -> Option<&str> {
                self.target.as_deref()
            }

            fn try_train(&mut self, dataset: &Dataset, target: &str) -> Result<(), ModelError> {
                let last = (self.coef_.as_slice(), self.feature_names_in_.as_slice());
                let (names, solved, intercept) = fit_linear(&self.params.engine(), dataset, target, last)?;
                self.n_features_in_ = names.len();
                self.feature_names_in_ = names;
                (self.coef_, self.intercept_) = (solved.coef, intercept);
                (self.n_iter_, self.dual_gap_) = (solved.n_iter, solved.dual_gap);
                (self.rank_, self.singular_) = (solved.rank, solved.singular);
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

            fn try_evaluate(&self, dataset: &Dataset, target: &str, scoring: &str) -> Result<f64, ModelError> {
                crate::models::evaluate_regressor(self, dataset, target, scoring)
            }

            fn try_predict_method(&self, dataset: &Dataset, method: &str) -> Result<Dataset, ModelError> {
                crate::models::predict_regressor(self, dataset, method)
            }

            fn try_load(&mut self, learned: Value, _: &[u8], target: &str) -> Result<(), ModelError> {
                let l: Learned = serde_json::from_value(learned)?;
                if l.coef_.len() != l.feature_names_in_.len() {
                    return Err(bad_learned(format!(
                        "{} coefficients for {} features",
                        l.coef_.len(),
                        l.feature_names_in_.len()
                    )));
                }
                self.n_features_in_ = l.feature_names_in_.len();
                (self.coef_, self.intercept_, self.feature_names_in_) = (l.coef_, l.intercept_, l.feature_names_in_);
                (self.n_iter_, self.dual_gap_, self.rank_, self.singular_) = (l.n_iter_, l.dual_gap_, l.rank_, l.singular_);
                self.loss_curve_ = l.loss_curve_;
                self.target = Some(target.into());
                Ok(())
            }

            fn run(&self) -> Option<&CachedRun> {
                self.run.as_ref()
            }

            fn loss_curve(&self) -> Option<Vec<f64>> {
                self.loss_curve_.clone()
            }

            fn set_run(&mut self, run: Option<CachedRun>) {
                self.run = run;
            }
        }

        impl Regressor for $name {
            fn try_predict(&self, dataset: &Dataset) -> Result<Series, ModelError> {
                let target: &String = self.target.as_ref().ok_or(ModelError::NotFitted { model: stringify!($name) })?;
                let mut out: Vec<Vec<f64>> = scores(dataset, &self.feature_names_in_, &[&self.coef_], &[self.intercept_])?;
                Ok(Series::new(target.into(), out.pop().unwrap_or_default()))
            }
        }

        impl $name {
            /// `true` (the default) learns a constant term; `false` fits through the origin.
            /// The intercept is never penalized.
            pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
                self.params.fit_intercept = fit_intercept;
                self
            }

            /// `true` keeps every coefficient at 0 or more. Default `false`.
            pub fn positive(mut self, positive: bool) -> Self {
                self.params.positive = positive;
                self
            }
        }
    };
}

/// The setters every penalized model shares.
macro_rules! penalty_settings {
    ($model:ident, $alpha_doc:literal) => {
        impl $model {
            #[doc = $alpha_doc]
            #[track_caller]
            pub fn alpha(mut self, alpha: f64) -> Self {
                at_least_zero("alpha", alpha);
                self.params.alpha = alpha;
                self
            }

            /// Coordinate descent stops when the duality gap is below `tol` times ‖y − ȳ‖².
            /// Default 1e-4.
            #[track_caller]
            pub fn tol(mut self, tol: f64) -> Self {
                at_least_zero("tol", tol);
                self.params.tol = tol;
                self
            }
        }
    };
}

/// The setters of coordinate descent: `Lasso`'s and `ElasticNet`'s.
macro_rules! descent_settings {
    ($model:ident) => {
        impl $model {
            /// The most rounds of coordinate descent. Default 1000.
            #[track_caller]
            pub fn max_iter(mut self, max_iter: usize) -> Self {
                check(max_iter > 0, "max_iter", max_iter, "1 or more");
                self.params.max_iter = max_iter;
                self
            }

            /// `"cyclic"` (the default) updates the coefficients in order; `"random"` picks
            /// one at random each step, seeded by `random_state`, which can converge faster
            /// when `tol` is above 1e-4.
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

            /// `true` starts coordinate descent from the coefficients of the last `fit` when
            /// it saw the same features. Default `false`.
            pub fn warm_start(mut self, warm_start: bool) -> Self {
                self.params.warm_start = warm_start;
                self
            }
        }
    };
}

linear_model! {
    /// Ordinary least squares: predicts a number as a weighted sum of the features plus a
    /// constant, with the weights that make the squared error on the training rows smallest.
    /// scikit-learn's `LinearRegression`.
    ///
    /// `fit` makes one pass over the data for XᵀX and Xᵀy and solves that p × p system
    /// exactly. When features are collinear (for example one-hot columns of every category,
    /// next to the intercept), it returns the minimum-norm coefficients, the same as
    /// scikit-learn's `lstsq`; `rank_` and `singular_` tell. With `positive(true)` it runs
    /// coordinate descent to the exact answer, as scikit-learn's `nnls`.
    ///
    /// A penalty is another model: [`Ridge`] (L2), [`Lasso`] (L1) and [`ElasticNet`] (both).
    ///
    /// scikit-learn settings with no setter: `copy_X` (nothing is copied), `n_jobs` (the
    /// threads come from the Polars pool, `POLARS_MAX_THREADS`), `tol` (it is used only for
    /// sparse data, which rusmlai does not take).
    ///
    /// ```
    /// use rusmlai::prelude::*;
    /// let ds = Dataset::new(
    ///     df!("x" => [1.0, 2.0, 3.0, 4.0], "y" => [3.0, 5.0, 7.0, 9.0]).unwrap(),
    ///     "",
    /// );
    /// let mut model = LinearRegression::new();
    /// model.fit(&ds, "y");
    /// assert!((model.coef_[0] - 2.0).abs() < 1e-12 && (model.intercept_ - 1.0).abs() < 1e-12);
    /// assert!((model.score(&ds, "y") - 1.0).abs() < 1e-12);
    /// assert_eq!(model.rank_, Some(1));
    ///
    /// // `down` wants a negative weight; `positive` holds it at 0.
    /// let ds = Dataset::new(
    ///     df!("x" => [1.0, 2.0, 3.0, 4.0], "down" => [0.5, 2.5, 1.0, 3.0],
    ///         "y" => [3.1, 4.9, 7.2, 8.8]).unwrap(),
    ///     "",
    /// );
    /// let mut free = LinearRegression::new();
    /// free.fit(&ds, "y");
    /// assert!(free.coef_[1] < 0.0);
    /// let mut positive = LinearRegression::new().positive(true);
    /// positive.fit(&ds, "y");
    /// assert_eq!(positive.coef_[1], 0.0);
    /// assert!((positive.coef_[0] - 1.94).abs() < 1e-9);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// from sklearn.linear_model import LinearRegression
    /// ds = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [3.0, 5.0, 7.0, 9.0]})
    /// model = LinearRegression().fit(ds[["x"]], ds["y"])
    /// assert abs(model.coef_[0] - 2.0) < 1e-12 and abs(model.intercept_ - 1.0) < 1e-12
    /// assert abs(model.score(ds[["x"]], ds["y"]) - 1.0) < 1e-12
    /// assert model.rank_ == 1
    ///
    /// ds = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "down": [0.5, 2.5, 1.0, 3.0], "y": [3.1, 4.9, 7.2, 8.8]})
    /// free = LinearRegression().fit(ds[["x", "down"]], ds["y"])
    /// assert free.coef_[1] < 0.0
    /// positive = LinearRegression(positive=True).fit(ds[["x", "down"]], ds["y"])
    /// assert positive.coef_[1] == 0.0
    /// assert abs(positive.coef_[0] - 1.94) < 1e-9
    /// ```
    pub struct LinearRegression(OlsParams = OlsParams { fit_intercept: true, positive: false });
}

impl LinearRegression {
    /// Whether the model fits an intercept (for `summary`).
    pub(crate) fn fits_intercept(&self) -> bool {
        self.params.fit_intercept
    }

    /// Least squares with an intercept.
    pub fn new() -> Self {
        Self::default()
    }
}

linear_model! {
    /// Least squares with an L2 penalty, scikit-learn's `Ridge`: it minimizes
    ///
    /// ```text
    /// ‖y − Xw‖² + alpha · ‖w‖²
    /// ```
    ///
    /// The penalty pulls every coefficient toward 0, and more so for features that say little;
    /// none becomes exactly 0 ([`Lasso`] does that). `alpha` is 1 by default, as in
    /// scikit-learn, on Ridge's own scale: the squared error is summed over the rows, so the
    /// same `alpha` weighs less on more rows. The intercept is never penalized.
    ///
    /// `fit` makes one pass for XᵀX and Xᵀy and solves XᵀX + alpha·I exactly, which is what
    /// every scikit-learn solver converges to. With `positive(true)` it runs coordinate
    /// descent, where `tol` and `max_iter` apply (scikit-learn runs L-BFGS there).
    ///
    /// scikit-learn settings with no setter: `copy_X`, `solver` (the exact solve gives what
    /// each solver converges to), `random_state` (only the `sag` and `saga` solvers draw).
    ///
    /// ```
    /// use rusmlai::prelude::*;
    /// let ds = Dataset::new(
    ///     df!("x" => [1.0, 2.0, 3.0, 4.0], "y" => [3.0, 5.0, 7.0, 9.0]).unwrap(),
    ///     "",
    /// );
    /// let mut model = Ridge::new(); // alpha = 1
    /// model.fit(&ds, "y");
    /// assert!((model.coef_[0] - 2.0 * 5.0 / 6.0).abs() < 1e-12);
    ///
    /// let mut strong = Ridge::new().alpha(5.0);
    /// strong.fit(&ds, "y");
    /// assert!((strong.coef_[0] - 1.0).abs() < 1e-12); // pulled halfway to 0 from 2
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// from sklearn.linear_model import Ridge
    /// ds = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [3.0, 5.0, 7.0, 9.0]})
    /// model = Ridge().fit(ds[["x"]], ds["y"])
    /// assert abs(model.coef_[0] - 2.0 * 5.0 / 6.0) < 1e-12
    ///
    /// strong = Ridge(alpha=5.0).fit(ds[["x"]], ds["y"])
    /// assert abs(strong.coef_[0] - 1.0) < 1e-12
    /// ```
    pub struct Ridge(RidgeParams = RidgeParams { alpha: 1.0, fit_intercept: true, positive: false, max_iter: None, tol: 1e-4 });
}

penalty_settings!(Ridge, "The strength of the penalty, 0 or more. Default 1.0. On Ridge's scale: the squared error is summed, not averaged, over the rows.");

impl Ridge {
    /// Least squares with an L2 penalty of 1.
    pub fn new() -> Self {
        Self::default()
    }

    /// The most rounds of coordinate descent with `positive(true)`. Default: until
    /// converged (at most 1,000).
    #[track_caller]
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        check(max_iter > 0, "max_iter", max_iter, "1 or more");
        self.params.max_iter = Some(max_iter);
        self
    }
}

linear_model! {
    /// Least squares with an L1 penalty, scikit-learn's `Lasso`: it minimizes
    ///
    /// ```text
    /// 1 / (2 · rows) · ‖y − Xw‖² + alpha · ‖w‖₁
    /// ```
    ///
    /// The L1 penalty sets the coefficients of features that say too little to exactly 0, so
    /// it also picks features. `alpha` is 1 by default, as in scikit-learn; the squared error
    /// is averaged over the rows. The intercept is never penalized.
    ///
    /// `fit` makes one pass for XᵀX and Xᵀy, then runs coordinate descent on that p × p
    /// matrix with scikit-learn's steps and stopping rule (the duality gap below
    /// `tol · ‖y − ȳ‖²`): the same coefficients, `n_iter_` and `dual_gap_`. Each step costs
    /// O(features), not O(rows).
    ///
    /// scikit-learn settings with no setter: `copy_X`, `precompute` (the Gram matrix is always
    /// used).
    ///
    /// ```
    /// use rusmlai::prelude::*;
    /// let ds = Dataset::new(
    ///     df!("x" => [1.0, 2.0, 3.0, 4.0], "noise" => [0.3, -0.1, 0.2, -0.4],
    ///         "y" => [3.0, 5.0, 7.0, 9.0]).unwrap(),
    ///     "",
    /// );
    /// let mut lasso = Lasso::new().alpha(0.5);
    /// lasso.fit(&ds, "y");
    /// assert_eq!(lasso.coef_[1], 0.0); // L1 sets the useless feature to exactly 0
    /// assert!(lasso.n_iter_.unwrap() > 0);
    ///
    /// // alpha = 1 (the default) on 4 rows takes the slope from 2 to 1.2.
    /// let mut model = Lasso::new();
    /// model.fit(&ds, "y");
    /// assert!((model.coef_[0] - 1.2).abs() < 1e-3);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// from sklearn.linear_model import Lasso
    /// ds = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "noise": [0.3, -0.1, 0.2, -0.4], "y": [3.0, 5.0, 7.0, 9.0]})
    /// x, y = ds[["x", "noise"]], ds["y"]
    /// lasso = Lasso(alpha=0.5).fit(x, y)
    /// assert lasso.coef_[1] == 0.0
    /// assert lasso.n_iter_ > 0
    ///
    /// model = Lasso().fit(x, y)
    /// assert abs(model.coef_[0] - 1.2) < 1e-3
    /// ```
    pub struct Lasso(DescentParams = DescentParams::new(None));
}

penalty_settings!(Lasso, "The strength of the L1 penalty, 0 or more. Default 1.0. With 0 the model is least squares, which `LinearRegression` solves better.");
descent_settings!(Lasso);

impl Lasso {
    /// Least squares with an L1 penalty of 1.
    pub fn new() -> Self {
        Self::default()
    }
}

linear_model! {
    /// Least squares with both an L1 and an L2 penalty, scikit-learn's `ElasticNet`: it
    /// minimizes
    ///
    /// ```text
    /// 1 / (2 · rows) · ‖y − Xw‖² + alpha · l1_ratio · ‖w‖₁ + alpha · (1 − l1_ratio) / 2 · ‖w‖²
    /// ```
    ///
    /// The L1 part sets some coefficients to 0, like [`Lasso`]; the L2 part shares the weight
    /// among correlated features, like [`Ridge`]. `alpha` is 1 and `l1_ratio` 0.5 by default,
    /// as in scikit-learn. `l1_ratio = 1` is [`Lasso`]. The intercept is never penalized.
    ///
    /// `fit` runs coordinate descent on XᵀX, as [`Lasso`] does, with scikit-learn's steps and
    /// stopping rule.
    ///
    /// scikit-learn settings with no setter: `copy_X`, `precompute` (the Gram matrix is always
    /// used).
    ///
    /// ```
    /// use rusmlai::prelude::*;
    /// let ds = Dataset::new(
    ///     df!("x" => [1.0, 2.0, 3.0, 4.0], "noise" => [0.3, -0.1, 0.2, -0.4],
    ///         "y" => [3.0, 5.0, 7.0, 9.0]).unwrap(),
    ///     "",
    /// );
    /// let mut model = ElasticNet::new(); // alpha = 1, l1_ratio = 0.5
    /// model.fit(&ds, "y");
    /// assert_eq!(model.coef_[1], 0.0);
    /// assert!(model.coef_[0] > 0.0 && model.coef_[0] < 2.0);
    ///
    /// // Mostly L2: the noise keeps a small weight.
    /// let mut ridge_like = ElasticNet::new().alpha(0.1).l1_ratio(0.1);
    /// ridge_like.fit(&ds, "y");
    /// assert!(ridge_like.coef_[1] != 0.0);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// from sklearn.linear_model import ElasticNet
    /// ds = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "noise": [0.3, -0.1, 0.2, -0.4], "y": [3.0, 5.0, 7.0, 9.0]})
    /// x, y = ds[["x", "noise"]], ds["y"]
    /// model = ElasticNet().fit(x, y)
    /// assert model.coef_[1] == 0.0
    /// assert 0.0 < model.coef_[0] < 2.0
    ///
    /// ridge_like = ElasticNet(alpha=0.1, l1_ratio=0.1).fit(x, y)
    /// assert ridge_like.coef_[1] != 0.0
    /// ```
    pub struct ElasticNet(DescentParams = DescentParams::new(Some(0.5)));
}

penalty_settings!(ElasticNet, "The strength of the penalty, 0 or more. Default 1.0.");
descent_settings!(ElasticNet);

impl ElasticNet {
    /// Least squares with a penalty of 1, half L1 and half L2.
    pub fn new() -> Self {
        Self::default()
    }

    /// The L1 share of the penalty, from 0 (all L2) to 1 (all L1: [`Lasso`]). Default 0.5.
    #[track_caller]
    pub fn l1_ratio(mut self, l1_ratio: f64) -> Self {
        check((0.0..=1.0).contains(&l1_ratio), "l1_ratio", l1_ratio, "a number from 0 to 1");
        self.params.l1_ratio = Some(l1_ratio);
        self
    }
}

/// The mean squared leave-one-out error of Ridge at each alpha, scikit-learn's `_RidgeGCV`:
/// with XᵀX = U Λ Uᵀ of the centered features, row i's hat value is
/// 1/n + Σₖ zᵢₖ² / (λₖ + α) and its leave-one-out residual (yᵢ − ŷᵢ) / (1 − hᵢᵢ), where
/// zᵢ = Uᵀ xᵢ. One pass over the rows gives every alpha.
fn gcv_errors(dataset: &Dataset, target: &str, alphas: &[f64], fit_intercept: bool) -> Result<Vec<f64>, ModelError> {
    let matrix: DesignMatrix = DesignMatrix::try_new(dataset, target)?;
    let p: usize = matrix.names.len();
    let (x_means, y_mean): (Vec<f64>, f64) = if fit_intercept {
        matrix.means()
    } else {
        (vec![0.0; p], 0.0)
    };
    let (g, v) = matrix.gram(&x_means, y_mean);
    let eigen = g.self_adjoint_eigen(Side::Lower).map_err(|_| ModelError::Solve)?;
    let u: Mat<f64> = eigen.U().to_owned();
    let lambda: Vec<f64> = (0..p).map(|k| eigen.S().column_vector()[k].max(0.0)).collect();
    let uv: Vec<f64> = (0..p).map(|k| (0..p).map(|i| u[(i, k)] * v[(i, 0)]).sum()).collect();
    let n: usize = matrix.rows();
    let base: f64 = if fit_intercept { 1.0 / n as f64 } else { 0.0 };
    let y: &[f64] = matrix.y();
    // Per alpha: 1 / (λₖ + α) and Uᵀv / (λₖ + α).
    let inverse: Vec<Vec<f64>> = alphas.iter().map(|a| lambda.iter().map(|l| 1.0 / (l + a)).collect()).collect();
    let size: usize = crate::models::design::CHUNK.min(n.max(1));
    let parts = matrix.fold(
        &x_means,
        false,
        || (vec![0.0; alphas.len()], Mat::<f64>::zeros(size, p)),
        |(errors, z), start, part| {
            let len: usize = part.nrows();
            let mut z = z.as_mut().subrows_mut(0, len);
            faer::linalg::matmul::matmul(z.as_mut(), faer::Accum::Replace, part, u.as_ref(), 1.0, faer::Par::Seq);
            for i in 0..len {
                let yc: f64 = y[start + i] - y_mean;
                for (error, inv) in errors.iter_mut().zip(&inverse) {
                    let (mut h, mut fit): (f64, f64) = (base, 0.0);
                    for k in 0..p {
                        let zk: f64 = z[(i, k)];
                        h += zk * zk * inv[k];
                        fit += zk * uv[k] * inv[k];
                    }
                    let e: f64 = (yc - fit) / (1.0 - h);
                    *error += e * e;
                }
            }
        },
    );
    let mut total: Vec<f64> = vec![0.0; alphas.len()];
    for (errors, _) in parts {
        total.iter_mut().zip(errors).for_each(|(t, e)| *t += e);
    }
    Ok(total.into_iter().map(|t| t / n as f64).collect())
}

/// `RidgeCV`'s settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RidgeCvParams {
    alphas: Vec<f64>,
    fit_intercept: bool,
}

/// [`Ridge`] that picks its `alpha` from a list by the leave-one-out error, scikit-learn's
/// `RidgeCV` with its default `cv=None` (generalized cross-validation). Each alpha is scored by
/// the mean squared error of predicting every row from a fit on all the others, which a
/// closed form gives without refitting; the best one is `alpha_`, and the model is then
/// `Ridge::new().alpha(alpha_)` on all the rows. It is the default final model of
/// [`StackingRegressor`](crate::models::StackingRegressor).
///
/// `fit` costs one pass for XᵀX, one eigendecomposition of it, and one more pass that scores
/// every alpha at once. Ties go to the first alpha, as in scikit-learn.
///
/// scikit-learn settings with no setter yet: `scoring` (the squared error only), `cv` (a fold
/// count or splitter: use [`GridSearchCV`](crate::models::GridSearchCV) over `Ridge`),
/// `gcv_mode` (one exact way here), `store_cv_results`, `alpha_per_target`.
///
/// ```
/// use rusmlai::prelude::*;
/// let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
/// let y: Vec<f64> = x.iter().map(|x| 3.0 * x + ((x * 7.0) % 5.0)).collect();
/// let ds = Dataset::new(df!("x" => x, "y" => y).unwrap(), "");
/// let mut model = RidgeCV::new(); // alphas 0.1, 1 and 10
/// model.fit(&ds, "y");
/// assert_eq!(model.alpha_, 0.1);
/// assert!((model.best_score_ + 2.437_092_965_815_825_4).abs() < 1e-9);
/// assert!((model.coef_()[0] - 3.029_619_606_074_274).abs() < 1e-9);
/// ```
///
/// Python:
///
/// ```python
/// import numpy as np
/// from sklearn.linear_model import RidgeCV
/// x = np.arange(20.0).reshape(-1, 1)
/// y = 3 * x[:, 0] + (x[:, 0] * 7) % 5
/// model = RidgeCV().fit(x, y)
/// assert model.alpha_ == 0.1
/// assert abs(model.best_score_ + 2.4370929658158254) < 1e-9
/// assert abs(model.coef_[0] - 3.029619606074274) < 1e-9
/// ```
#[derive(Debug, Clone)]
pub struct RidgeCV {
    params: RidgeCvParams,
    /// The alpha that scored best, learned by `fit`.
    pub alpha_: f64,
    /// Minus the mean squared leave-one-out error at `alpha_` (higher is better), learned by
    /// `fit`.
    pub best_score_: f64,
    ridge: Ridge,
}

impl Default for RidgeCV {
    fn default() -> Self {
        Self::new()
    }
}

/// What `RidgeCV` learns besides its ridge.
#[derive(Serialize, Deserialize)]
struct RidgeCvLearned {
    alpha_: f64,
    best_score_: f64,
    ridge: Value,
}

impl RidgeCV {
    /// The alphas 0.1, 1 and 10, with an intercept.
    pub fn new() -> Self {
        Self {
            params: RidgeCvParams { alphas: vec![0.1, 1.0, 10.0], fit_intercept: true },
            alpha_: f64::NAN,
            best_score_: f64::NAN,
            ridge: Ridge::new(),
        }
    }

    /// The alphas to try, each above 0. Default `[0.1, 1.0, 10.0]`.
    #[track_caller]
    pub fn alphas(mut self, alphas: impl IntoIterator<Item = f64>) -> Self {
        let alphas: Vec<f64> = alphas.into_iter().collect();
        let ok: bool = !alphas.is_empty() && alphas.iter().all(|a| a.is_finite() && *a > 0.0);
        check(ok, "alphas", format!("{alphas:?}"), "one or more finite numbers above 0");
        self.params.alphas = alphas;
        self
    }

    /// `true` (the default) learns a constant term, never penalized.
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.params.fit_intercept = fit_intercept;
        self
    }

    /// The coefficients of the ridge at `alpha_`, in the order of `feature_names_in_()`.
    pub fn coef_(&self) -> &[f64] {
        &self.ridge.coef_
    }

    /// The constant term of the ridge at `alpha_`.
    pub fn intercept_(&self) -> f64 {
        self.ridge.intercept_
    }

    /// The feature columns `fit` saw.
    pub fn feature_names_in_(&self) -> &[String] {
        &self.ridge.feature_names_in_
    }
}

impl Model for RidgeCV {
    fn target(&self) -> Option<&str> {
        self.ridge.target()
    }

    fn try_train(&mut self, dataset: &Dataset, target: &str) -> Result<(), ModelError> {
        let errors: Vec<f64> = gcv_errors(dataset, target, &self.params.alphas, self.params.fit_intercept)?;
        // The first of the lowest, as scikit-learn keeps a score only when it is higher.
        let best: usize = (1..errors.len()).fold(0, |best, i| if errors[i] < errors[best] { i } else { best });
        (self.alpha_, self.best_score_) = (self.params.alphas[best], -errors[best]);
        self.ridge = Ridge::new().alpha(self.alpha_).fit_intercept(self.params.fit_intercept);
        self.ridge.try_train(dataset, target)
    }

    fn params(&self) -> Value {
        serde_json::to_value(&self.params).unwrap_or_default()
    }

    fn learned(&self) -> Value {
        serde_json::to_value(RidgeCvLearned {
            alpha_: self.alpha_,
            best_score_: self.best_score_,
            ridge: self.ridge.learned(),
        })
        .unwrap_or_default()
    }

    fn try_load(&mut self, learned: Value, weights: &[u8], target: &str) -> Result<(), ModelError> {
        let l: RidgeCvLearned = serde_json::from_value(learned)?;
        let mut ridge: Ridge = Ridge::new().fit_intercept(self.params.fit_intercept);
        ridge.params.alpha = l.alpha_;
        ridge.try_load(l.ridge, weights, target)?;
        (self.alpha_, self.best_score_, self.ridge) = (l.alpha_, l.best_score_, ridge);
        Ok(())
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

    fn run(&self) -> Option<&CachedRun> {
        self.ridge.run()
    }

    fn set_run(&mut self, run: Option<CachedRun>) {
        self.ridge.set_run(run);
    }

    fn loss_curve(&self) -> Option<Vec<f64>> {
        self.ridge.loss_curve()
    }
}

impl Regressor for RidgeCV {
    fn try_predict(&self, dataset: &Dataset) -> Result<Series, ModelError> {
        if self.ridge.target.is_none() {
            return Err(ModelError::NotFitted { model: "RidgeCV" });
        }
        self.ridge.try_predict(dataset)
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

    fn fitted<M: Model>(mut model: M, ds: &Dataset) -> M {
        model.fit(ds, "y");
        model
    }

    /// `coef` and `intercept` against scikit-learn 1.9.1 on the same data.
    fn assert_fit(coef: &[f64], intercept: f64, expected: &[f64], expected_intercept: f64, eps: f64) {
        for (got, want) in coef.iter().zip(expected) {
            assert!((got - want).abs() < eps, "coef {coef:?}, want {expected:?}");
        }
        assert!((intercept - expected_intercept).abs() < eps, "intercept {intercept}");
    }

    #[test]
    fn least_squares_and_ridge_match_scikit_learn() {
        let x: Dataset = plain();
        let ols: LinearRegression = fitted(LinearRegression::new(), &x);
        assert_fit(&ols.coef_, ols.intercept_, &[1.4180206378986875, 0.7656285178236396], 0.3718198874296412, 1e-9);
        assert!((ols.score(&x, "y") - 0.9972840488307306).abs() < 1e-12);
        assert_eq!(ols.rank_, Some(2));
        let singular: Vec<f64> = ols.singular_.clone().unwrap();
        assert!((singular[0] - 9.929242295586475).abs() < 1e-9);
        assert!((singular[1] - 1.8381777486240647).abs() < 1e-9);
        assert_eq!((ols.n_iter_, ols.dual_gap_), (None, None));

        // Ridge() has alpha = 1, on Ridge's own scale.
        let ridge: Ridge = fitted(Ridge::new(), &x);
        assert_fit(&ridge.coef_, ridge.intercept_, &[1.3018487997133628, 0.8421497671085648], 0.5263346470798993, 1e-9);
        assert_eq!(ridge.rank_, None);
        // Ridge(positive=True) runs L-BFGS in scikit-learn, coordinate descent here.
        let positive: Ridge = fitted(Ridge::new().positive(true).tol(1e-10), &x);
        assert_fit(&positive.coef_, positive.intercept_, &[1.3018491888733943, 0.8421500546440217], 0.5263315120953713, 1e-5);

        let origin: LinearRegression = fitted(LinearRegression::new().fit_intercept(false), &x);
        assert_fit(&origin.coef_, origin.intercept_, &[1.4998620478234215, 0.7505824647455558], 0.0, 1e-9);
    }

    #[test]
    fn lasso_elastic_net_and_positive_match_scikit_learn() {
        let x: Dataset = plain();
        // The defaults: alpha = 1, and l1_ratio = 0.5 for ElasticNet, with the same rounds.
        let lasso: Lasso = fitted(Lasso::new(), &x);
        assert_fit(&lasso.coef_, lasso.intercept_, &[1.1012780672523725, 0.8791495629965872], 1.250841425443248, 1e-9);
        assert_eq!(lasso.n_iter_, Some(52));
        assert!((lasso.dual_gap_.unwrap() - 0.0005397494157759475).abs() < 1e-9);
        let enet: ElasticNet = fitted(ElasticNet::new(), &x);
        assert_fit(&enet.coef_, enet.intercept_, &[1.037089606366662, 0.9336169755131772], 1.2775650766928557, 1e-9);
        assert_eq!(enet.n_iter_, Some(29));
        let tight: ElasticNet = fitted(ElasticNet::new().tol(1e-12), &x);
        assert_fit(&tight.coef_, tight.intercept_, &[1.0368601387076686, 0.9337846763523829], 1.2777906208696486, 1e-9);

        let small: Lasso = fitted(Lasso::new().alpha(0.5).tol(1e-12), &x);
        assert_fit(&small.coef_, small.intercept_, &[1.2592964352794886, 0.8226641650973867], 0.811594746711128, 1e-9);
        // The default tol stops where scikit-learn stops, after the same 51 rounds.
        let early: Lasso = fitted(Lasso::new().alpha(0.5), &x);
        assert_fit(&early.coef_, early.intercept_, &[1.2600571154247946, 0.8220711596271534], 0.8110255248827478, 1e-9);
        assert_eq!(early.n_iter_, Some(51));
        assert!((early.dual_gap_.unwrap() - 0.0006655559327843896).abs() < 1e-9);

        let mixed: ElasticNet = fitted(ElasticNet::new().alpha(0.5).l1_ratio(0.3).tol(1e-12), &x);
        assert_fit(&mixed.coef_, mixed.intercept_, &[1.1494995829897572, 0.9123512017561103], 0.8740617180948114, 1e-9);
        let origin: Lasso = fitted(Lasso::new().alpha(0.5).tol(1e-12).fit_intercept(false), &x);
        assert_fit(&origin.coef_, origin.intercept_, &[1.4379368486012722, 0.7898221949346386], 0.0, 1e-9);

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
        let nnls: LinearRegression = fitted(LinearRegression::new().positive(true), &flipped);
        assert_fit(&nnls.coef_, nnls.intercept_, &[2.270238095238096, 0.0], 0.22142857142856798, 1e-9);
        let positive_enet: ElasticNet = fitted(ElasticNet::new().alpha(0.5).l1_ratio(0.3).tol(1e-12).positive(true), &flipped);
        assert_fit(&positive_enet.coef_, positive_enet.intercept_, &[2.1015625, 0.0], 0.98046875, 1e-9);
    }

    #[test]
    fn random_selection_and_warm_start_reach_the_same_answer() {
        let x: Dataset = plain();
        let random: Lasso = fitted(Lasso::new().alpha(0.5).tol(1e-12).selection("random"), &x);
        assert_fit(&random.coef_, random.intercept_, &[1.2592964352794886, 0.8226641650973867], 0.811594746711128, 1e-8);

        let mut warm: Lasso = Lasso::new().alpha(0.5).tol(1e-12).warm_start(true);
        warm.fit(&x, "y");
        let cold_rounds: usize = warm.n_iter_.unwrap();
        warm.fit(&x, "y");
        assert!(warm.n_iter_.unwrap() < cold_rounds, "a warm start begins at the answer");
    }

    #[test]
    fn collinear_one_hot_columns_get_the_minimum_norm_answer() {
        let ds: Dataset = data().drop(&["x2"]);
        let ols: LinearRegression = fitted(LinearRegression::new(), &ds);
        assert_eq!(ols.feature_names_in_, ["x1", "a", "b"]);
        assert_eq!((ols.n_features_in_, ols.rank_), (3, Some(2)));
        assert_fit(&ols.coef_, ols.intercept_, &[2.3126666666666673, 0.29700000000000004, -0.2970000000000001], 0.03049999999999642, 1e-9);
        let ridge: Ridge = fitted(Ridge::new(), &ds);
        assert_fit(&ridge.coef_, ridge.intercept_, &[2.24492385786802, 0.1969543147208121, -0.1969543147208121], 0.3353426395939092, 1e-9);
    }

    #[test]
    fn errors_name_the_problem() {
        let fit = |ds: Dataset| LinearRegression::new().try_train(&ds, "y").unwrap_err().to_string();
        let text: Dataset = Dataset::new(df!("s" => ["a", "b"], "y" => [1.0, 2.0]).unwrap(), "");
        assert!(fit(text).starts_with("feature 's' is str, not a number"));
        let nulls: Dataset = Dataset::new(df!("x" => [Some(1.0), None], "y" => [1.0, 2.0]).unwrap(), "");
        assert!(fit(nulls).starts_with("feature 'x' has 1 null:"));
        let nan: Dataset = Dataset::new(df!("x" => [1.0, f64::NAN], "y" => [1.0, 2.0]).unwrap(), "");
        assert!(fit(nan).starts_with("column 'x' has NaN"));
        let alone: Dataset = Dataset::new(df!("y" => [1.0, 2.0]).unwrap(), "");
        assert!(fit(alone).starts_with("the dataset has no feature columns"));

        let ds: Dataset = data();
        assert!(matches!(LinearRegression::new().try_predict(&ds), Err(ModelError::NotFitted { model: "LinearRegression" })));
        assert!(matches!(Ridge::new().try_predict(&ds), Err(ModelError::NotFitted { model: "Ridge" })));
        let model: LinearRegression = fitted(LinearRegression::new(), &ds);
        assert!(model.try_predict(&ds.drop(&["x1"])).is_err());

        // `predict` checks its features through its output: a 0 coefficient still shows them.
        let frame = |noise: f64| {
            Dataset::new(
                df!("x" => [1.0, 2.0, 3.0, 4.0], "noise" => [0.3, -0.1, 0.2, noise], "y" => [3.0, 5.0, 7.0, 9.0]).unwrap(),
                "",
            )
        };
        let lasso: Lasso = fitted(Lasso::new().alpha(0.5), &frame(-0.4));
        assert_eq!(lasso.coef_[1], 0.0);
        for bad in [f64::INFINITY, f64::NAN] {
            let error: String = lasso.try_predict(&frame(bad)).unwrap_err().to_string();
            assert!(error.starts_with("column 'noise' has NaN"), "{error}");
        }
    }

    #[test]
    #[should_panic(expected = "l1_ratio = 2 is not valid")]
    fn a_bad_setting_panics_at_the_setter() {
        let _ = ElasticNet::new().l1_ratio(2.0);
    }

    /// Each model takes its own settings by name only, as scikit-learn's `set_params` does:
    /// a grid search can't give Ridge an `l1_ratio`.
    #[test]
    fn each_model_takes_only_its_own_settings() {
        let mut ridge: Ridge = Ridge::new();
        assert_eq!(ridge.params(), serde_json::json!({"alpha": 1.0, "fit_intercept": true, "positive": false, "max_iter": null, "tol": 1e-4}));
        let mut with_l1 = ridge.params();
        with_l1["l1_ratio"] = serde_json::json!(0.5);
        assert!(ridge.try_set_params(with_l1).is_err());
        assert_eq!(LinearRegression::new().params(), serde_json::json!({"fit_intercept": true, "positive": false}));
        assert!(Lasso::new().params().get("l1_ratio").is_none());
        assert_eq!(ElasticNet::new().params()["l1_ratio"], 0.5);
        assert_eq!((Lasso::new().name(), ElasticNet::new().name()), ("Lasso", "ElasticNet"));
    }

    #[test]
    fn a_cached_record_of_the_wrong_shape_is_refused_and_changes_nothing() {
        let mut model: LinearRegression = LinearRegression::new();
        let learned: Value = serde_json::json!({
            "coef_": [1.0, 2.0, 3.0],
            "intercept_": 0.0,
            "feature_names_in_": ["a", "b"],
            "n_iter_": null, "dual_gap_": null, "rank_": null, "singular_": null,
        });
        let err: String = model.try_load(learned, &[], "y").unwrap_err().to_string();
        assert!(err.contains("3 coefficients for 2 features"), "{err}");
        assert!(model.coef_.is_empty());
    }
    /// scikit-learn 1.9 on `tree::tests::table()`
    /// (analysis/2026-11-18/new-features/folds-parity.py).
    #[test]
    fn ridge_cv_picks_scikit_learn_s_alpha() {
        let ds: Dataset = crate::models::tree::tests::table().drop(&["class"]);
        let cases = [
            (true, -0.8947672190197857, [0.25068602476592156, 0.7436144398584034, -1.24878125821263], 1.511322900266924),
            (false, -1.1562229233804586, [0.26063238312672676, 0.8008715765243966, -1.1208461938537542], 0.0),
        ];
        for (intercept, score, coef, b) in cases {
            let mut model = RidgeCV::new().alphas([0.01, 1.0, 100.0, 1e4, 1e6]).fit_intercept(intercept);
            model.try_train(&ds, "y").unwrap();
            assert_eq!(model.alpha_, 1.0);
            assert!((model.best_score_ - score).abs() < 1e-10, "{}", model.best_score_);
            assert!(model.coef_().iter().zip(coef).all(|(g, w)| (g - w).abs() < 1e-10), "{:?}", model.coef_());
            assert!((model.intercept_() - b).abs() < 1e-9);
        }
    }

}
