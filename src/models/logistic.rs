//! [`LogisticRegression`]: two classes (sigmoid) or more (softmax), with an L2 penalty, fitted by
//! Newton's method on the weighted Gram matrix; [`LogisticRegressionCV`] picks its `C` by
//! cross-validation.
//!
//! Each Newton step costs one pass over the rows ([`DesignMatrix::fold`]): chunk by chunk, the
//! probabilities, the log loss, the gradient Xᵀ(p − y) and the Hessian Xᵀ·diag(h)·X. The step
//! then solves that small system. A fit takes a handful of passes. The line search and the
//! stopping rule are scikit-learn's `solver="newton-cholesky"`, so both take the same steps.

use faer::{Accum, Mat, MatRef, Par, linalg::matmul::matmul};
use polars::prelude::{Column, DataFrame, DataType, IdxCa, IdxSize, NamedFrom, Series};
use polars_core::runtime::THREAD_POOL;
use rayon::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use crate::{
    dataset::Dataset,
    error::{OrFatal, warn},
    metadata::models::CachedRun,
    models::{
        Classifier, Model, ModelError, bad_learned, check, class_labels,
        design::{CHUNK, DesignMatrix, col, scores},
        folds::Cv,
        label_target,
        linear::solve,
    },
};
use crate::models::at_least_zero;

/// `C` is a number or `"inf"`: JSON has no infinity, and `C = inf` means no penalty.
mod c_value {
    use super::*;

    pub fn serialize<S: Serializer>(c: &f64, s: S) -> Result<S::Ok, S::Error> {
        if c.is_finite() {
            s.serialize_f64(*c)
        } else {
            s.serialize_str("inf")
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum C {
            Number(f64),
            Text(serde::de::IgnoredAny),
        }
        Ok(match C::deserialize(d)? {
            C::Number(c) => c,
            C::Text(_) => f64::INFINITY,
        })
    }
}

/// The settings of a [`LogisticRegression`], with scikit-learn's names.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Params {
    #[serde(rename = "C", with = "c_value")]
    c: f64,
    fit_intercept: bool,
    max_iter: usize,
    tol: f64,
    warm_start: bool,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            c: 1.0,
            fit_intercept: true,
            max_iter: 100,
            tol: 1e-4,
            warm_start: false,
        }
    }
}

/// The learned values, as the model cache stores them.
#[derive(Serialize, Deserialize)]
struct Learned {
    classes_: Vec<String>,
    coef_: Vec<Vec<f64>>,
    intercept_: Vec<f64>,
    feature_names_in_: Vec<String>,
    n_iter_: usize,
    loss_curve_: Vec<f64>,
    /// The target's type, so `predict` gives labels of that type.
    dtype: DataType,
}

/// Predicts a class (yes / no, or one of several) and how likely each class is, from a weighted
/// sum of the features.
///
/// Logistic regression, like scikit-learn's `LogisticRegression`: the probability of each class
/// is the sigmoid (two classes) or the softmax (more) of a linear function of the features. It
/// minimizes the mean log loss of the rows plus `1 / (2 · C · rows) · ‖w‖²`, which is
/// scikit-learn's objective; the intercept is never penalized.
///
/// The target can be text, integers or booleans: `classes_` are its values, sorted, and
/// `predict` gives them back in the target's type. With two classes, `coef_` has one row, for
/// `classes_[1]`; with more, one row per class, like scikit-learn.
///
/// `fit` runs Newton's method with scikit-learn's line search and stopping rule
/// (`solver="newton-cholesky"`): with the same `tol` it takes the same steps and gives the same
/// coefficients. scikit-learn's default solver, L-BFGS, stops at a slightly different point of
/// the same optimum: about 1e-4 apart with the default `tol`.
///
/// scikit-learn settings with no setter: `solver` (always Newton), `penalty` (deprecated in
/// scikit-learn 1.8: `C = inf` is no penalty), `dual` and `intercept_scaling` (liblinear only),
/// `random_state` (only the sag, saga and liblinear solvers shuffle), `n_jobs` (the threads come
/// from the Polars pool, `POLARS_MAX_THREADS`), `verbose`. Not yet: `l1_ratio` (an L1 part needs
/// another solver) and `class_weight` (backlog).
///
/// ```
/// use rusmlai::prelude::*;
/// let ds = Dataset::new(
///     df!(
///         "hours" => [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
///         "passed" => ["no", "no", "no", "yes", "no", "yes", "no", "yes", "yes", "yes"],
///     )
///     .unwrap(),
///     "",
/// );
/// let mut model = LogisticRegression::new();
/// model.fit(&ds, "passed");
/// assert_eq!(model.classes_, ["no", "yes"]);
/// assert!((model.coef_[0][0] - 0.917_523_422_869_693_8).abs() < 1e-9);
/// assert!((model.intercept_[0] + 2.523_189_412_891_658).abs() < 1e-9);
/// assert_eq!(model.n_iter_, 3);
/// assert_eq!(model.score(&ds, "passed"), 0.8);
///
/// let proba = model.predict_proba(&ds); // one column per class
/// assert_eq!(proba.columns(), ["no", "yes"]);
/// ```
///
/// Python:
///
/// ```python
/// import pandas as pd
/// from sklearn.linear_model import LogisticRegression
/// ds = pd.DataFrame({
///     "hours": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
///     "passed": ["no", "no", "no", "yes", "no", "yes", "no", "yes", "yes", "yes"],
/// })
/// model = LogisticRegression(solver="newton-cholesky").fit(ds[["hours"]], ds["passed"])
/// assert model.classes_.tolist() == ["no", "yes"]
/// assert abs(model.coef_[0][0] - 0.9175234228696938) < 1e-9
/// assert abs(model.intercept_[0] + 2.523189412891658) < 1e-9
/// assert model.n_iter_[0] == 3
/// assert model.score(ds[["hours"]], ds["passed"]) == 0.8
///
/// proba = model.predict_proba(ds[["hours"]])
/// assert proba.shape == (10, 2)
/// ```
#[derive(Debug, Clone, Default)]
pub struct LogisticRegression {
    params: Params,
    /// The classes `fit` saw, sorted: the order of `predict_proba`'s columns.
    pub classes_: Vec<String>,
    /// The coefficients, in the order of `feature_names_in_`: one row for `classes_[1]` with
    /// two classes, one row per class with more.
    pub coef_: Vec<Vec<f64>>,
    /// The constant term of each row of `coef_` (0 with `fit_intercept(false)`).
    pub intercept_: Vec<f64>,
    /// The feature columns `fit` saw.
    pub feature_names_in_: Vec<String>,
    /// How many features `fit` saw.
    pub n_features_in_: usize,
    /// The Newton steps `fit` took.
    pub n_iter_: usize,
    /// The objective after each Newton step: the mean log loss of the training rows plus the
    /// penalty. The journal draws it.
    pub loss_curve_: Vec<f64>,
    dtype: Option<DataType>,
    target: Option<String>,
    run: Option<CachedRun>,
}

impl LogisticRegression {
    /// L2 penalty with `C = 1`, an intercept, at most 100 Newton steps.
    pub fn new() -> Self {
        Self::default()
    }

    /// The inverse strength of the L2 penalty: smaller is stronger. Above 0; `f64::INFINITY`
    /// is no penalty. Default 1.
    #[track_caller]
    #[allow(non_snake_case)]
    pub fn C(mut self, c: f64) -> Self {
        check(
            c > 0.0,
            "C",
            c,
            "a number above 0, or f64::INFINITY for no penalty",
        );
        self.params.c = c;
        self
    }

    /// `true` (the default) learns a constant term; `false` has none.
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.params.fit_intercept = fit_intercept;
        self
    }

    /// The most Newton steps. Default 100.
    #[track_caller]
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        check(max_iter > 0, "max_iter", max_iter, "1 or more");
        self.params.max_iter = max_iter;
        self
    }

    /// Newton stops when the largest gradient entry and half the Newton decrement are both at
    /// most `tol`. Default 1e-4.
    #[track_caller]
    pub fn tol(mut self, tol: f64) -> Self {
        at_least_zero("tol", tol);
        self.params.tol = tol;
        self
    }

    /// `true` starts Newton from the coefficients of the last `fit` when it saw the same
    /// features and classes. Default `false`.
    pub fn warm_start(mut self, warm_start: bool) -> Self {
        self.params.warm_start = warm_start;
        self
    }

    /// The linear scores of each row: one column per row of `coef_`.
    fn decision(&self, dataset: &Dataset) -> Result<Vec<Vec<f64>>, ModelError> {
        if self.target.is_none() {
            return Err(ModelError::NotFitted {
                model: "LogisticRegression",
            });
        }
        let coef: Vec<&[f64]> = self.coef_.iter().map(Vec::as_slice).collect();
        scores(dataset, &self.feature_names_in_, &coef, &self.intercept_)
    }
}

/// The probability of class 1 at score `z`, and ln(1 + eᶻ), from one exponential and without
/// overflow.
pub(crate) fn logistic(z: f64) -> (f64, f64) {
    let e: f64 = (-z.abs()).exp();
    let p: f64 = if z >= 0.0 {
        1.0 / (1.0 + e)
    } else {
        e / (1.0 + e)
    };
    (p, z.max(0.0) + e.ln_1p())
}

/// The softmax of `z` into `out`, and ln Σ eᶻ.
pub(crate) fn softmax(z: &[f64], out: &mut [f64]) -> f64 {
    let top: f64 = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sum: f64 = 0.0;
    for (o, zk) in out.iter_mut().zip(z) {
        *o = (zk - top).exp();
        sum += *o;
    }
    for o in out.iter_mut() {
        *o /= sum;
    }
    top + sum.ln()
}

/// The objective, its gradient and its Hessian at one point.
struct Point {
    loss: f64,
    grad: Vec<f64>,
    hess: Mat<f64>,
}

/// One task's sums, and its buffers for one chunk of rows.
struct Sums {
    loss: f64,
    grad: Mat<f64>,
    hess: Mat<f64>,
    z: Mat<f64>,
    prob: Mat<f64>,
    resid: Mat<f64>,
    scaled: Mat<f64>,
    weight: Vec<f64>,
}

/// The shape of the problem: `outputs` score columns (1 for two classes, else one per class) of
/// `width` parameters each (the features, then the intercept when `ones`). `theta[k · width +
/// j]` is parameter `j` of output `k`.
struct Shape {
    outputs: usize,
    width: usize,
    ones: bool,
    /// The L2 strength on scikit-learn's scale, `1 / (C · rows)`.
    l2: f64,
}

impl Shape {
    /// Whether parameter `j` of an output is penalized: every one but the intercept.
    fn penalized(&self, j: usize) -> bool {
        !(self.ones && j + 1 == self.width)
    }

    /// The objective, gradient and (with `hessian`) Hessian at `theta`, in one pass over the
    /// rows of `matrix`. Without the Hessian, the loss and gradient are the same to the bit.
    fn evaluate(&self, matrix: &DesignMatrix, theta: &[f64], hessian: bool) -> Point {
        let (m, d, n): (usize, usize, usize) = (self.outputs, self.width, matrix.rows());
        let md: usize = if hessian { m * d } else { 0 };
        let w: MatRef<'_, f64> = MatRef::from_column_major_slice(theta, d, m);
        let y: &[f64] = matrix.y();
        let size: usize = CHUNK.min(n.max(1));
        let offset: Vec<f64> = vec![0.0; matrix.names.len()];
        let zero = || Sums {
            loss: 0.0,
            grad: Mat::zeros(d, m),
            hess: Mat::zeros(md, md),
            z: Mat::zeros(size, m),
            prob: Mat::zeros(size, m),
            resid: Mat::zeros(size, m),
            scaled: Mat::zeros(size, d),
            weight: vec![0.0; size],
        };
        let parts: Vec<Sums> = matrix.fold(&offset, self.ones, zero, |s, start, x| {
            let len: usize = x.nrows();
            matmul(
                s.z.as_mut().subrows_mut(0, len),
                Accum::Replace,
                x,
                w,
                1.0,
                Par::Seq,
            );
            let classes: &[f64] = &y[start..start + len];
            if m == 1 {
                let z: &[f64] = &s.z.col_as_slice(0)[..len];
                let prob: &mut [f64] = s.prob.col_as_slice_mut(0);
                let resid: &mut [f64] = s.resid.col_as_slice_mut(0);
                let mut loss: f64 = 0.0;
                for i in 0..len {
                    let hit: f64 = f64::from(classes[i] == 1.0);
                    let (p, softplus): (f64, f64) = logistic(z[i]);
                    loss += softplus - hit * z[i];
                    prob[i] = p;
                    resid[i] = p - hit;
                }
                s.loss += loss;
            } else {
                let (mut z_row, mut p_row): (Vec<f64>, Vec<f64>) = (vec![0.0; m], vec![0.0; m]);
                for (i, &class) in classes.iter().enumerate() {
                    let class: usize = class as usize;
                    for (k, zk) in z_row.iter_mut().enumerate() {
                        *zk = s.z[(i, k)];
                    }
                    s.loss += softmax(&z_row, &mut p_row) - z_row[class];
                    for (k, pk) in p_row.iter().enumerate() {
                        s.prob[(i, k)] = *pk;
                        s.resid[(i, k)] = pk - f64::from(class == k);
                    }
                }
            }
            let resid = s.resid.as_ref().subrows(0, len);
            matmul(
                s.grad.as_mut(),
                Accum::Add,
                x.transpose(),
                resid,
                1.0,
                Par::Seq,
            );
            if !hessian {
                return;
            }
            // Block (k, l) of the Hessian is Xᵀ·diag(p_k (δ_kl − p_l))·X; with two classes,
            // the one block is Xᵀ·diag(p (1 − p))·X. The weights are made once per row.
            for k in 0..m {
                for l in k..m {
                    let (pk, pl): (&[f64], &[f64]) =
                        (s.prob.col_as_slice(k), s.prob.col_as_slice(l));
                    let same: f64 = f64::from(k == l);
                    for ((h, a), b) in s.weight[..len].iter_mut().zip(pk).zip(pl) {
                        *h = a * (same - b);
                    }
                    for j in 0..d {
                        let to: &mut [f64] = &mut s.scaled.col_as_slice_mut(j)[..len];
                        for ((to, h), v) in to.iter_mut().zip(&s.weight).zip(col(x, j)) {
                            *to = h * v;
                        }
                    }
                    matmul(
                        s.hess.as_mut().submatrix_mut(k * d, l * d, d, d),
                        Accum::Add,
                        x.transpose(),
                        s.scaled.as_ref().subrows(0, len),
                        1.0,
                        Par::Seq,
                    );
                }
            }
        });
        let (mut loss, mut grad, mut hess): (f64, Mat<f64>, Mat<f64>) =
            (0.0, Mat::zeros(d, m), Mat::zeros(md, md));
        for part in parts {
            loss += part.loss;
            grad += part.grad;
            hess += part.hess;
        }
        for k in 0..if hessian { m } else { 0 } {
            for l in k + 1..m {
                for a in 0..d {
                    for b in 0..d {
                        hess[(l * d + b, k * d + a)] = hess[(k * d + a, l * d + b)];
                    }
                }
            }
        }
        let rows: f64 = n.max(1) as f64;
        let mut point: Point = Point {
            loss: loss / rows,
            grad: (0..m * d).map(|t| grad[(t % d, t / d)] / rows).collect(),
            hess: Mat::from_fn(md, md, |a, b| hess[(a, b)] / rows),
        };
        for (t, &value) in theta.iter().enumerate() {
            if self.penalized(t % d) {
                point.loss += 0.5 * self.l2 * value * value;
                point.grad[t] += self.l2 * value;
                if hessian {
                    point.hess[(t, t)] += self.l2;
                }
            }
        }
        point
    }
}

/// Newton's method from `theta`, as scikit-learn's `NewtonCholeskySolver`: the step solves
/// H·s = −g (the minimum-norm solution, so the redundant direction of the softmax intercepts
/// costs nothing), then a backtracking line search with Armijo's rule. It stops when
/// max |g| ≤ tol and ½·sᵀHs ≤ tol. Returns the parameters, the steps taken and the objective
/// after each step.
fn newton(
    matrix: &DesignMatrix,
    shape: &Shape,
    params: &Params,
    mut theta: Vec<f64>,
) -> Result<(Vec<f64>, usize, Vec<f64>), ModelError> {
    const SIGMA: f64 = 1.0 / 2048.0;
    let rows: usize = matrix.rows();
    let mut at: Point = shape.evaluate(matrix, &theta, true);
    let mut losses: Vec<f64> = Vec::new();
    let mut steps: usize = 0;
    for step in 1..=params.max_iter {
        steps = step;
        let minus_g: Mat<f64> = Mat::from_fn(theta.len(), 1, |t, _| -at.grad[t]);
        let s: Vec<f64> = solve(at.hess.clone(), &minus_g, 0.0, rows)?.coef;
        let slope: f64 = at.grad.iter().zip(&s).map(|(g, s)| g * s).sum();
        let hs: Vec<f64> = (0..s.len())
            .map(|a| (0..s.len()).map(|b| at.hess[(a, b)] * s[b]).sum())
            .collect();
        let decrement: f64 = 0.5 * s.iter().zip(&hs).map(|(a, b)| a * b).sum::<f64>();
        let old_l1: f64 = at.grad.iter().map(|g| g.abs()).sum();
        // A small decrement means the next point is likely the answer, and its Hessian is then
        // never used: leave it out, and make it in one more pass only if Newton goes on.
        let last: bool = decrement <= params.tol;
        let mut t: f64 = 1.0;
        let mut accepted: Option<(Vec<f64>, Point)> = None;
        for _ in 0..21 {
            let trial: Vec<f64> = theta.iter().zip(&s).map(|(a, b)| a + t * b).collect();
            let next: Point = shape.evaluate(matrix, &trial, !last);
            let improvement: f64 = next.loss - at.loss;
            // Armijo, or a change below rounding that still shrinks the gradient.
            let tiny: bool = improvement.abs() <= (at.loss * 16.0 * f64::EPSILON).abs()
                && next.grad.iter().map(|g| g.abs()).sum::<f64>() < old_l1;
            if improvement <= t * SIGMA * slope || tiny {
                accepted = Some((trial, next));
                break;
            }
            t *= 0.5;
        }
        let Some((next_theta, next)) = accepted else {
            warn(format_args!(
                "LogisticRegression: the line search of Newton step {step} found no lower \
                 loss; stopping there. Scale the features, or lower C"
            ));
            return Ok((theta, step, losses));
        };
        (theta, at) = (next_theta, next);
        losses.push(at.loss);
        let largest: f64 = at.grad.iter().fold(0.0, |a, g| a.max(g.abs()));
        if largest <= params.tol && last {
            return Ok((theta, steps, losses));
        }
        if last {
            at = shape.evaluate(matrix, &theta, true);
        }
    }
    let largest: f64 = at.grad.iter().fold(0.0, |a, g| a.max(g.abs()));
    warn(format_args!(
        "LogisticRegression did not converge in max_iter = {} Newton steps (largest gradient \
         {largest:.3e}, tolerance {:.3e}); raise max_iter, or scale the features",
        params.max_iter, params.tol
    ));
    Ok((theta, steps, losses))
}

impl Model for LogisticRegression {
    fn target(&self) -> Option<&str> {
        self.target.as_deref()
    }

    fn try_train(&mut self, dataset: &Dataset, target: &str) -> Result<(), ModelError> {
        let (classes, codes): (Vec<String>, Series) = label_target(dataset, target)?;
        let matrix: DesignMatrix = DesignMatrix::try_with_target(dataset, target, codes, 0.0)?;
        let p: &Params = &self.params;
        let features: usize = matrix.names.len();
        let shape: Shape = Shape {
            outputs: if classes.len() == 2 { 1 } else { classes.len() },
            width: features + usize::from(p.fit_intercept),
            ones: p.fit_intercept,
            l2: 1.0 / (p.c * matrix.rows() as f64),
        };
        let (m, d): (usize, usize) = (shape.outputs, shape.width);
        let warm: bool = p.warm_start
            && self.feature_names_in_ == matrix.names
            && self.classes_ == classes
            && self.coef_.len() == m;
        let start: Vec<f64> = (0..m * d)
            .map(|t| match (warm, t % d < features) {
                (false, _) => 0.0,
                (true, true) => self.coef_[t / d][t % d],
                (true, false) => self.intercept_[t / d],
            })
            .collect();
        let (mut theta, steps, losses) = newton(&matrix, &shape, p, start)?;
        // scikit-learn's symmetric form of the softmax: the intercepts (and with no penalty,
        // every parameter) sum to 0 over the classes. The probabilities don't change.
        if m > 1 {
            let centered: Vec<usize> = if p.c.is_infinite() {
                (0..d).collect()
            } else if p.fit_intercept {
                vec![d - 1]
            } else {
                Vec::new()
            };
            for j in centered {
                let mean: f64 = (0..m).map(|k| theta[k * d + j]).sum::<f64>() / m as f64;
                for k in 0..m {
                    theta[k * d + j] -= mean;
                }
            }
        }
        self.coef_ = (0..m)
            .map(|k| theta[k * d..k * d + features].to_vec())
            .collect();
        self.intercept_ = (0..m)
            .map(|k| {
                if p.fit_intercept {
                    theta[k * d + features]
                } else {
                    0.0
                }
            })
            .collect();
        self.dtype = Some(dataset.column(target)?.dtype().clone());
        self.n_features_in_ = features;
        self.feature_names_in_ = matrix.names;
        self.classes_ = classes;
        self.n_iter_ = steps;
        self.loss_curve_ = losses;
        self.target = Some(target.into());
        Ok(())
    }

    fn params(&self) -> Value {
        serde_json::to_value(&self.params).unwrap_or_default()
    }

    fn learned(&self) -> Value {
        serde_json::to_value(Learned {
            classes_: self.classes_.clone(),
            coef_: self.coef_.clone(),
            intercept_: self.intercept_.clone(),
            feature_names_in_: self.feature_names_in_.clone(),
            n_iter_: self.n_iter_,
            loss_curve_: self.loss_curve_.clone(),
            dtype: self.dtype.clone().unwrap_or(DataType::String),
        })
        .unwrap_or_default()
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

    fn try_load(&mut self, learned: Value, _: &[u8], target: &str) -> Result<(), ModelError> {
        let l: Learned = serde_json::from_value(learned)?;
        let outputs: usize = if l.classes_.len() == 2 {
            1
        } else {
            l.classes_.len()
        };
        let features: usize = l.feature_names_in_.len();
        if l.classes_.len() < 2
            || l.coef_.len() != outputs
            || l.intercept_.len() != outputs
            || l.coef_.iter().any(|row| row.len() != features)
        {
            return Err(bad_learned(format!(
                "{} classes, {} coefficient rows and {} intercepts for {features} features",
                l.classes_.len(),
                l.coef_.len(),
                l.intercept_.len()
            )));
        }
        self.n_features_in_ = features;
        (self.classes_, self.coef_, self.intercept_) = (l.classes_, l.coef_, l.intercept_);
        (self.feature_names_in_, self.n_iter_) = (l.feature_names_in_, l.n_iter_);
        (self.loss_curve_, self.dtype) = (l.loss_curve_, Some(l.dtype));
        self.target = Some(target.into());
        Ok(())
    }

    fn run(&self) -> Option<&CachedRun> {
        self.run.as_ref()
    }

    fn set_run(&mut self, run: Option<CachedRun>) {
        self.run = run;
    }

    fn loss_curve(&self) -> Option<Vec<f64>> {
        (!self.loss_curve_.is_empty()).then(|| self.loss_curve_.clone())
    }
}

impl Classifier for LogisticRegression {
    /// `classes_[1]` where the score is above 0 with two classes, else the class with the
    /// highest score (the first one on a tie), like scikit-learn.
    fn try_predict(&self, dataset: &Dataset) -> Result<Series, ModelError> {
        let z: Vec<Vec<f64>> = self.decision(dataset)?;
        let picks: Vec<IdxSize> = THREAD_POOL.install(|| {
            (0..dataset.height())
                .into_par_iter()
                .map(|i| match z.as_slice() {
                    [only] => IdxSize::from(only[i] > 0.0),
                    _ => {
                        (1..z.len()).fold(0, |best, k| if z[k][i] > z[best][i] { k } else { best })
                            as IdxSize
                    }
                })
                .collect()
        });
        let (target, dtype) = (self.target.as_deref(), self.dtype.as_ref());
        let labels: Series = class_labels(
            target.unwrap_or_default(),
            &self.classes_,
            dtype.unwrap_or(&DataType::String),
        )?;
        Ok(labels.take(&IdxCa::from_vec("".into(), picks))?)
    }

    fn try_predict_proba(&self, dataset: &Dataset) -> Result<Dataset, ModelError> {
        let z: Vec<Vec<f64>> = self.decision(dataset)?;
        let n: usize = dataset.height();
        // On the Polars pool: the sigmoid or softmax of each row is its own work.
        let columns: Vec<Vec<f64>> = THREAD_POOL.install(|| match z.as_slice() {
            [only] => {
                let p: Vec<f64> = only.par_iter().map(|zi| logistic(*zi).0).collect();
                vec![p.par_iter().map(|pi| 1.0 - pi).collect(), p]
            }
            _ => {
                let k: usize = z.len();
                let mut rows: Vec<f64> = vec![0.0; n * k];
                rows.par_chunks_mut(k * CHUNK)
                    .enumerate()
                    .for_each(|(c, block)| {
                        let mut z_row: Vec<f64> = vec![0.0; k];
                        for (r, out) in block.chunks_mut(k).enumerate() {
                            for (zk, column) in z_row.iter_mut().zip(&z) {
                                *zk = column[c * CHUNK + r];
                            }
                            softmax(&z_row, out);
                        }
                    });
                (0..k)
                    .map(|c| rows.iter().skip(c).step_by(k).copied().collect())
                    .collect()
            }
        });
        let columns: Vec<Column> = self
            .classes_
            .iter()
            .zip(columns)
            .map(|(class, p)| Series::new(class.into(), p).into())
            .collect();
        Ok(dataset.reshaped(DataFrame::new_infer_height(columns)?, |_| false))
    }
}

/// The values of `C` that [`LogisticRegressionCV`] tries: a count (that many on a log scale
/// from 1e-4 to 1e4, numpy's `logspace(-4, 4, n)`) or the values themselves.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Cs {
    /// That many values, from 1e-4 to 1e4.
    Count(usize),
    /// These values.
    Values(Vec<f64>),
}

impl From<usize> for Cs {
    fn from(n: usize) -> Self {
        Cs::Count(n)
    }
}

/// So that a literal such as `10` is a `Cs`: Rust reads an untyped integer as `i32`.
impl From<i32> for Cs {
    fn from(n: i32) -> Self {
        Cs::Count(usize::try_from(n).unwrap_or(0))
    }
}

impl From<Vec<f64>> for Cs {
    fn from(values: Vec<f64>) -> Self {
        Cs::Values(values)
    }
}

impl<const N: usize> From<[f64; N]> for Cs {
    fn from(values: [f64; N]) -> Self {
        Cs::Values(values.to_vec())
    }
}

impl Cs {
    /// The values, as numpy's `logspace(-4, 4, n)` gives a count.
    fn values(&self) -> Vec<f64> {
        match self {
            Cs::Values(values) => values.clone(),
            Cs::Count(1) => vec![1e-4],
            Cs::Count(n) => {
                let step: f64 = 8.0 / (*n - 1) as f64;
                (0..*n).map(|i| 10f64.powf(if i + 1 == *n { 4.0 } else { i as f64 * step - 4.0 })).collect()
            }
        }
    }
}

/// `LogisticRegressionCV`'s settings, with scikit-learn's names.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CvParams {
    #[serde(rename = "Cs")]
    cs: Cs,
    cv: Cv,
    scoring: String,
    fit_intercept: bool,
    max_iter: usize,
    tol: f64,
    refit: bool,
}

/// What `LogisticRegressionCV` learns besides its final model.
#[derive(Serialize, Deserialize)]
#[allow(non_snake_case)]
struct CvLearned {
    Cs_: Vec<f64>,
    C_: f64,
    scores_: Vec<Vec<f64>>,
    n_iter_: Vec<Vec<usize>>,
    model: Value,
}

/// [`LogisticRegression`] that picks its `C` by cross-validation: scikit-learn's
/// `LogisticRegressionCV`. Each fold fits every `C` in `Cs_`, smallest first, each fit
/// starting from the last one's coefficients (scikit-learn's path), and scores it on the
/// fold's held-out rows with `scoring`. The `C` with the best mean score is `C_` (the first on
/// a tie); with `refit` the model then fits every row at `C_`, starting from the mean of the
/// folds' coefficients there. The folds are 5 stratified, unshuffled ones by default, as
/// scikit-learn's `cv=5` ([`Cv`]).
///
/// Every fit is Newton's method with the steps of scikit-learn's `solver="newton-cholesky"`;
/// the folds run as tasks on the Polars pool.
///
/// scikit-learn settings with no setter: `solver` (always Newton), `penalty` (deprecated),
/// `dual` and `intercept_scaling` (liblinear only), `random_state` (only the sag, saga and
/// liblinear solvers shuffle), `n_jobs`, `verbose`, `use_legacy_attributes` (`C_` is one
/// number and `scores_` one grid, its new attributes). Not yet: `l1_ratios` (an L1 part needs
/// another solver) and `class_weight`.
///
/// ```
/// use rusmlai::prelude::*;
/// let n: usize = 60;
/// let x: Vec<f64> = (0..n).map(|i| (i * 37 % 101) as f64 / 10.0).collect();
/// let z: Vec<f64> = (0..n).map(|i| (i * 53 % 17) as f64 / 3.0).collect();
/// let noise = |i: usize| ((i * 7) % 11) as f64 / 2.0 - 2.5;
/// let y: Vec<&str> = (0..n).map(|i| if 0.4 * x[i] - 0.5 * z[i] + noise(i) > 0.5 { "yes" } else { "no" }).collect();
/// let ds = Dataset::new(df!("x" => x, "z" => z, "y" => y).unwrap(), "");
/// let mut model = LogisticRegressionCV::new().scoring("neg_log_loss").tol(1e-10);
/// model.fit(&ds, "y"); // 10 values of C, 5 folds
/// assert_eq!(model.Cs_.len(), 10);
/// assert!((model.C_ - 0.359_381_366_380_462_6).abs() < 1e-15);
/// assert!((model.coef_()[0][1] + 0.620_407_574_466_324_7).abs() < 1e-9);
/// assert!((model.intercept_()[0] - 1.915_184_372_416_104_2).abs() < 1e-9);
/// assert_eq!((model.scores_.len(), model.scores_[0].len()), (5, 10));
///
/// // The C of each fold's best score, averaged, with no refit:
/// let mut averaged = LogisticRegressionCV::new().scoring("neg_log_loss").refit(false).tol(1e-10);
/// averaged.fit(&ds, "y");
/// assert!((averaged.C_ - 6_000.073_075_241_778).abs() < 1e-9);
/// ```
///
/// Python:
///
/// ```python
/// import numpy as np
/// import pandas as pd
/// from sklearn.linear_model import LogisticRegressionCV
/// n = 60
/// x = np.array([(i * 37 % 101) / 10 for i in range(n)])
/// z = np.array([(i * 53 % 17) / 3 for i in range(n)])
/// noise = np.array([((i * 7) % 11) / 2 - 2.5 for i in range(n)])
/// y = np.where(0.4 * x - 0.5 * z + noise > 0.5, "yes", "no")
/// X = pd.DataFrame({"x": x, "z": z})
/// settings = dict(cv=5, scoring="neg_log_loss", solver="newton-cholesky", tol=1e-10,
///                 l1_ratios=(0.0,), use_legacy_attributes=False)
/// model = LogisticRegressionCV(**settings).fit(X, y)
/// assert len(model.Cs_) == 10
/// assert abs(model.C_ - 0.3593813663804626) < 1e-15
/// assert abs(model.coef_[0][1] + 0.6204075744663247) < 1e-9
/// assert abs(model.intercept_[0] - 1.9151843724161042) < 1e-9
/// assert model.scores_.shape == (5, 1, 10)
///
/// averaged = LogisticRegressionCV(**settings, refit=False).fit(X, y)
/// assert abs(averaged.C_ - 6000.073075241778) < 1e-9
/// ```
#[derive(Debug, Clone)]
#[allow(non_snake_case)]
pub struct LogisticRegressionCV {
    params: CvParams,
    /// The values of `C` tried, learned by `fit`.
    pub Cs_: Vec<f64>,
    /// The `C` picked: the best mean score, or with `refit(false)` the mean of each fold's best.
    pub C_: f64,
    /// The score of each fold (rows) at each `C` (columns).
    pub scores_: Vec<Vec<f64>>,
    /// The Newton steps of each fold at each `C`.
    pub n_iter_: Vec<Vec<usize>>,
    model: LogisticRegression,
}

impl Default for LogisticRegressionCV {
    fn default() -> Self {
        Self::new()
    }
}

impl LogisticRegressionCV {
    /// 10 values of `C`, 5 folds, scored by accuracy, with a refit: scikit-learn's defaults.
    pub fn new() -> Self {
        Self {
            params: CvParams {
                cs: Cs::Count(10),
                cv: Cv::Folds(5),
                scoring: String::new(),
                fit_intercept: true,
                max_iter: 100,
                tol: 1e-4,
                refit: true,
            },
            Cs_: Vec::new(),
            C_: f64::NAN,
            scores_: Vec::new(),
            n_iter_: Vec::new(),
            model: LogisticRegression::new(),
        }
    }

    /// The values of `C` to try ([`Cs`]): a count such as `10` (from 1e-4 to 1e4 on a log
    /// scale) or values such as `[0.1, 1.0, 10.0]`, each above 0. Default 10.
    #[track_caller]
    #[allow(non_snake_case)]
    pub fn Cs(mut self, cs: impl Into<Cs>) -> Self {
        let cs: Cs = cs.into();
        let ok: bool = match &cs {
            Cs::Count(n) => *n > 0,
            Cs::Values(values) => !values.is_empty() && values.iter().all(|c| c.is_finite() && *c > 0.0),
        };
        check(ok, "Cs", format!("{cs:?}"), "a count of 1 or more, or finite values above 0");
        self.params.cs = cs;
        self
    }

    /// The folds ([`Cv`]): a count such as `5` (stratified, not shuffled) or a splitter such
    /// as `StratifiedKFold::new().shuffle(true)`. Default 5.
    #[track_caller]
    pub fn cv(mut self, cv: impl Into<Cv>) -> Self {
        let cv: Cv = cv.into();
        cv.check().or_fatal();
        self.params.cv = cv;
        self
    }

    /// The scikit-learn scorer that ranks the values of `C` on the held-out rows, such as
    /// `"roc_auc"` or `"neg_log_loss"` (the classifier scorers of [`Model::try_evaluate`]).
    /// Default `""`: the accuracy, scikit-learn 1.9's default.
    #[track_caller]
    pub fn scoring(mut self, scoring: &str) -> Self {
        crate::models::search::check_scoring(scoring).or_fatal();
        self.params.scoring = scoring.into();
        self
    }

    /// `true` (the default) learns a constant term; `false` has none.
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.params.fit_intercept = fit_intercept;
        self
    }

    /// The most Newton steps of each fit. Default 100.
    #[track_caller]
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        check(max_iter > 0, "max_iter", max_iter, "1 or more");
        self.params.max_iter = max_iter;
        self
    }

    /// Newton's stopping tolerance in each fit ([`LogisticRegression::tol`]). Default 1e-4.
    #[track_caller]
    pub fn tol(mut self, tol: f64) -> Self {
        at_least_zero("tol", tol);
        self.params.tol = tol;
        self
    }

    /// `true` (the default) fits every row at the best `C`; `false` averages the coefficients
    /// of each fold at its own best `C`, and `C_` is the mean of those.
    pub fn refit(mut self, refit: bool) -> Self {
        self.params.refit = refit;
        self
    }

    /// The classes `fit` saw, sorted.
    pub fn classes_(&self) -> &[String] {
        &self.model.classes_
    }

    /// The coefficients of the final model ([`LogisticRegression::coef_`]).
    pub fn coef_(&self) -> &[Vec<f64>] {
        &self.model.coef_
    }

    /// The constant terms of the final model.
    pub fn intercept_(&self) -> &[f64] {
        &self.model.intercept_
    }

    /// The feature columns `fit` saw.
    pub fn feature_names_in_(&self) -> &[String] {
        &self.model.feature_names_in_
    }
}

/// One fold's fit at one `C`: its held-out score, Newton steps, coefficients and intercepts.
type PathPoint = (f64, usize, Vec<Vec<f64>>, Vec<f64>);

/// The index of the first highest value, as numpy's `argmax`.
fn first_best(values: &[f64]) -> usize {
    (1..values.len()).fold(0, |best, i| if values[i] > values[best] { i } else { best })
}

impl Model for LogisticRegressionCV {
    fn target(&self) -> Option<&str> {
        self.model.target()
    }

    fn try_train(&mut self, dataset: &Dataset, target: &str) -> Result<(), ModelError> {
        let p: &CvParams = &self.params;
        let cs: Vec<f64> = p.cs.values();
        let folds: Vec<(Dataset, Dataset)> = p.cv.datasets(dataset, target, true)?;
        let (classes, _): (Vec<String>, Series) = label_target(dataset, target)?;
        let base: LogisticRegression = LogisticRegression {
            params: Params { c: cs[0], fit_intercept: p.fit_intercept, max_iter: p.max_iter, tol: p.tol, warm_start: true },
            ..LogisticRegression::default()
        };
        let path = |(train, test): &(Dataset, Dataset)| -> Result<(Vec<PathPoint>, LogisticRegression), ModelError> {
            let mut model: LogisticRegression = base.clone();
            let mut points: Vec<PathPoint> = Vec::with_capacity(cs.len());
            for &c in &cs {
                model.params.c = c;
                model.try_train(train, target)?;
                if model.classes_ != classes {
                    return Err(ModelError::InvalidSetting {
                        setting: "cv",
                        value: format!("a fold whose training rows have the classes {:?} of {classes:?}", model.classes_),
                        expected: "folds whose training rows have every class: fewer folds, or StratifiedKFold",
                    });
                }
                let score: f64 = model.try_evaluate(test, target, &p.scoring)?;
                points.push((score, model.n_iter_, model.coef_.clone(), model.intercept_.clone()));
            }
            Ok((points, model))
        };
        let paths: Vec<(Vec<PathPoint>, LogisticRegression)> =
            THREAD_POOL.install(|| folds.par_iter().map(path).collect::<Result<_, ModelError>>())?;
        let scores: Vec<Vec<f64>> = paths.iter().map(|(points, _)| points.iter().map(|x| x.0).collect()).collect();
        // The mean over the folds of the coefficients at `picks[fold]`.
        let mean_at = |picks: &[usize]| -> (Vec<Vec<f64>>, Vec<f64>) {
            let share: f64 = 1.0 / paths.len() as f64;
            let (_, _, coef, intercept) = &paths[0].0[picks[0]];
            let mut coef: Vec<Vec<f64>> = coef.iter().map(|row| vec![0.0; row.len()]).collect();
            let mut intercept: Vec<f64> = vec![0.0; intercept.len()];
            for ((points, _), &pick) in paths.iter().zip(picks) {
                let (_, _, c, b) = &points[pick];
                coef.iter_mut().flatten().zip(c.iter().flatten()).for_each(|(a, v)| *a += v * share);
                intercept.iter_mut().zip(b).for_each(|(a, v)| *a += v * share);
            }
            (coef, intercept)
        };
        let mut model: LogisticRegression = paths[0].1.clone();
        if p.refit {
            let sums: Vec<f64> = (0..cs.len()).map(|c| scores.iter().map(|fold| fold[c]).sum()).collect();
            let best: usize = first_best(&sums);
            (model.coef_, model.intercept_) = mean_at(&vec![best; paths.len()]);
            model.params.c = cs[best];
            model.try_train(dataset, target)?;
            self.C_ = cs[best];
        } else {
            let picks: Vec<usize> = scores.iter().map(|fold| first_best(fold)).collect();
            (model.coef_, model.intercept_) = mean_at(&picks);
            self.C_ = picks.iter().map(|&i| cs[i]).sum::<f64>() / picks.len() as f64;
            model.params.c = self.C_;
            model.loss_curve_.clear();
        }
        model.params.warm_start = false;
        self.n_iter_ = paths.iter().map(|(points, _)| points.iter().map(|x| x.1).collect()).collect();
        (self.Cs_, self.scores_, self.model) = (cs, scores, model);
        Ok(())
    }

    fn params(&self) -> Value {
        serde_json::to_value(&self.params).unwrap_or_default()
    }

    fn learned(&self) -> Value {
        serde_json::to_value(CvLearned {
            Cs_: self.Cs_.clone(),
            C_: self.C_,
            scores_: self.scores_.clone(),
            n_iter_: self.n_iter_.clone(),
            model: self.model.learned(),
        })
        .unwrap_or_default()
    }

    fn try_load(&mut self, learned: Value, weights: &[u8], target: &str) -> Result<(), ModelError> {
        let l: CvLearned = serde_json::from_value(learned)?;
        if l.scores_.iter().any(|row| row.len() != l.Cs_.len()) || l.n_iter_.len() != l.scores_.len() {
            return Err(bad_learned(format!("{} values of C and a score grid of {} folds", l.Cs_.len(), l.scores_.len())));
        }
        let mut model: LogisticRegression = LogisticRegression::new().fit_intercept(self.params.fit_intercept);
        model.params.c = l.C_;
        model.try_load(l.model, weights, target)?;
        (self.Cs_, self.C_, self.scores_, self.n_iter_, self.model) = (l.Cs_, l.C_, l.scores_, l.n_iter_, model);
        Ok(())
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

    fn run(&self) -> Option<&CachedRun> {
        self.model.run()
    }

    fn set_run(&mut self, run: Option<CachedRun>) {
        self.model.set_run(run);
    }

    fn loss_curve(&self) -> Option<Vec<f64>> {
        self.model.loss_curve()
    }
}

impl Classifier for LogisticRegressionCV {
    fn try_predict(&self, dataset: &Dataset) -> Result<Series, ModelError> {
        if self.model.target.is_none() {
            return Err(ModelError::NotFitted { model: "LogisticRegressionCV" });
        }
        self.model.try_predict(dataset)
    }

    fn try_predict_proba(&self, dataset: &Dataset) -> Result<Dataset, ModelError> {
        if self.model.target.is_none() {
            return Err(ModelError::NotFitted { model: "LogisticRegressionCV" });
        }
        self.model.try_predict_proba(dataset)
    }
}

#[cfg(test)]
mod tests {
    use polars::df;

    use super::*;

    /// 10 rows, two features, two classes that overlap.
    fn binary() -> Dataset {
        Dataset::new(
            df!(
                "x1" => [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                "x2" => [1.0, 0.0, 2.0, 1.0, 3.0, 1.5, 0.5, 2.5, 2.0, 3.5],
                "y" => ["no", "no", "no", "yes", "no", "yes", "no", "yes", "yes", "yes"],
            )
            .unwrap(),
            "",
        )
    }

    /// 12 rows, two features, three classes coded 0, 1, 2.
    fn three() -> Dataset {
        Dataset::new(
            df!(
                "x1" => [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
                "x2" => [1.0, 2.0, 0.5, 1.5, 3.0, 2.5, 1.0, 0.0, 2.0, 3.5, 1.5, 3.0],
                "y" => [0i64, 0, 1, 0, 1, 2, 1, 1, 2, 2, 1, 2],
            )
            .unwrap(),
            "",
        )
    }

    fn fitted(model: LogisticRegression, ds: &Dataset) -> LogisticRegression {
        let mut model: LogisticRegression = model;
        model.fit(ds, "y");
        model
    }

    /// `coef_` and `intercept_` against scikit-learn 1.9.1 (`solver="newton-cholesky"`).
    fn assert_fit(model: &LogisticRegression, coef: &[&[f64]], intercept: &[f64], eps: f64) {
        for (got, want) in model
            .coef_
            .iter()
            .flatten()
            .zip(coef.iter().copied().flatten())
        {
            assert!(
                (got - want).abs() < eps,
                "coef {:?}, want {coef:?}",
                model.coef_
            );
        }
        for (got, want) in model.intercept_.iter().zip(intercept) {
            assert!((got - want).abs() < eps, "intercept {:?}", model.intercept_);
        }
        assert_eq!(model.coef_.len(), coef.len());
    }

    #[test]
    fn two_classes_match_scikit_learn() {
        let ds: Dataset = binary();
        let tight = || LogisticRegression::new().tol(1e-12).max_iter(1000);
        let model: LogisticRegression = fitted(tight(), &ds);
        assert_fit(
            &model,
            &[&[0.8771587640526366, 0.1614289145036286]],
            &[-2.683140607726582],
            1e-9,
        );
        let proba: Dataset = model.predict_proba(&ds);
        let no: f64 = proba
            .dataframe
            .column("no")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();
        assert!((no - 0.8892528864763347).abs() < 1e-9);
        assert_eq!(model.score(&ds, "y"), 0.8);
        let y: &Series = ds.dataframe.column("y").unwrap().as_materialized_series();
        let loss: f64 = crate::metrics::log_loss(y, &proba);
        assert!((loss - 0.4507631936917095).abs() < 1e-9);

        // The default tol stops where scikit-learn stops, after the same 3 steps.
        let early: LogisticRegression = fitted(LogisticRegression::new(), &ds);
        assert_eq!(early.n_iter_, 3);
        assert_fit(
            &early,
            &[&[0.8770853162535097, 0.16138187230746498]],
            &[-2.682862363904318],
            1e-9,
        );
        assert_eq!(early.loss_curve_.len(), 3);

        let strong: LogisticRegression = fitted(tight().C(0.1), &ds);
        assert_fit(
            &strong,
            &[&[0.30221573839726107, 0.10676944485008451]],
            &[-1.0119780650633414],
            1e-9,
        );
        let free: LogisticRegression = fitted(tight().C(f64::INFINITY), &ds);
        assert_fit(
            &free,
            &[&[1.3300517447958298, 0.0896121643514498]],
            &[-3.80852853619352],
            1e-8,
        );
        let origin: LogisticRegression = fitted(tight().fit_intercept(false), &ds);
        assert_fit(
            &origin,
            &[&[0.3145101434623837, -0.205807673789368]],
            &[0.0],
            1e-9,
        );
    }

    #[test]
    fn three_classes_match_scikit_learn() {
        let ds: Dataset = three();
        let tight = || LogisticRegression::new().tol(1e-12).max_iter(1000);
        let model: LogisticRegression = fitted(tight(), &ds);
        assert_eq!(model.classes_, ["0", "1", "2"]);
        assert_fit(
            &model,
            &[
                &[-0.938977747001148, -0.05693941587039725],
                &[0.2823420587009694, -0.6274343138144525],
                &[0.656635688300178, 0.6843737296848498],
            ],
            &[1.9002623901108433, 1.0933248481954334, -2.993587238306277],
            1e-9,
        );
        let predicted: Series = model.predict(&ds);
        assert_eq!(
            predicted.dtype(),
            &DataType::Int64,
            "labels keep the target's type"
        );
        let got: Vec<i64> = predicted.i64().unwrap().into_no_null_iter().collect();
        assert_eq!(got, [0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 1, 2]);
        assert!((model.score(&ds, "y") - 0.8333333333333334).abs() < 1e-15);

        assert_eq!(fitted(LogisticRegression::new(), &ds).n_iter_, 5);
        let free: LogisticRegression = fitted(tight().C(f64::INFINITY), &ds);
        assert_fit(
            &free,
            &[
                &[-2.8040707881839024, 0.24112683612333474],
                &[0.9346877318026039, -1.4656063974213367],
                &[1.869383056381298, 1.2244795612980017],
            ],
            &[4.659820492587766, 2.1834378853480167, -6.843258377935783],
            1e-6,
        );
        let origin: LogisticRegression = fitted(tight().fit_intercept(false), &ds);
        assert_fit(
            &origin,
            &[
                &[-0.7046777118450727, 0.5241761109561596],
                &[0.4805513524016924, -0.5560786612618362],
                &[0.22412635944337994, 0.03190255030567704],
            ],
            &[0.0, 0.0, 0.0],
            1e-9,
        );
    }

    #[test]
    fn boolean_labels_and_warm_start() {
        let ds: Dataset = Dataset::new(
            df!(
                "x" => [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                "y" => [false, false, true, false, true, true],
            )
            .unwrap(),
            "",
        );
        let mut model: LogisticRegression = LogisticRegression::new().warm_start(true);
        model.fit(&ds, "y");
        assert_eq!(model.classes_, ["false", "true"]);
        assert_eq!(model.predict(&ds).dtype(), &DataType::Boolean);
        let cold: usize = model.n_iter_;
        model.fit(&ds, "y");
        assert!(model.n_iter_ < cold, "a warm start begins at the answer");
    }

    #[test]
    fn a_label_unseen_in_fit_still_gets_an_accuracy() {
        let model: LogisticRegression = fitted(LogisticRegression::new(), &binary());
        let unseen: Dataset = Dataset::new(
            df!("x1" => [0.5, 5.0], "x2" => [1.0, 3.5], "y" => ["maybe", "yes"]).unwrap(),
            "",
        );
        // scikit-learn's score is the accuracy; only the journal's log loss is left out.
        assert_eq!(model.try_score(&unseen, "y").unwrap(), 0.5);
    }

    #[test]
    fn errors_name_the_problem() {
        let fit = |ds: Dataset| {
            LogisticRegression::new()
                .try_train(&ds, "y")
                .unwrap_err()
                .to_string()
        };
        let one: Dataset = Dataset::new(df!("x" => [1.0, 2.0], "y" => ["a", "a"]).unwrap(), "");
        assert!(fit(one).starts_with("target 'y' has one class only (a)"));
        let nulls: Dataset = Dataset::new(
            df!("x" => [1.0, 2.0], "y" => [Some("a"), None]).unwrap(),
            "",
        );
        assert!(fit(nulls).starts_with("target 'y' has 1 null;"));
        let text: Dataset = Dataset::new(df!("s" => ["u", "v"], "y" => ["a", "b"]).unwrap(), "");
        assert!(fit(text).starts_with("feature 's' is str, not a number"));
        // scikit-learn: "Unknown label type: continuous"; whole numbers are classes.
        let scores: Dataset = Dataset::new(
            df!("x" => [1.0, 2.0, 3.0], "y" => [1.0, 0.5, 1.5]).unwrap(),
            "",
        );
        assert!(fit(scores).starts_with("target 'y' has continuous values (like 0.5)"));
        let codes: Dataset = Dataset::new(
            df!("x" => [1.0, 2.0, 3.0, 4.0], "y" => [0.0, 0.0, 1.0, 1.0]).unwrap(),
            "",
        );
        LogisticRegression::new().try_train(&codes, "y").unwrap();
        assert!(matches!(
            LogisticRegression::new().try_predict(&binary()),
            Err(ModelError::NotFitted { .. })
        ));
    }

    #[test]
    #[should_panic(expected = "C = 0 is not valid")]
    fn a_bad_setting_panics_at_the_setter() {
        let _ = LogisticRegression::new().C(0.0);
    }

    #[test]
    fn a_cached_record_of_the_wrong_shape_is_refused_and_changes_nothing() {
        let mut model: LogisticRegression = LogisticRegression::new();
        let learned: Value = serde_json::json!({
            "classes_": ["no", "yes"],
            "coef_": [[1.0], [2.0]],
            "intercept_": [0.0, 0.0],
            "feature_names_in_": ["x"],
            "n_iter_": 1,
            "loss_curve_": [],
            "dtype": serde_json::to_value(DataType::String).unwrap(),
        });
        let err: String = model.try_load(learned, &[], "y").unwrap_err().to_string();
        assert!(err.contains("2 classes, 2 coefficient rows"), "{err}");
        assert!(model.coef_.is_empty());
    }

    /// The 60 rows of `LogisticRegressionCV`'s doc example, with two classes or three.
    fn sixty(three: bool) -> Dataset {
        let x: Vec<f64> = (0..60).map(|i| (i * 37 % 101) as f64 / 10.0).collect();
        let z: Vec<f64> = (0..60).map(|i| (i * 53 % 17) as f64 / 3.0).collect();
        let s: Vec<f64> = (0..60).map(|i| 0.4 * x[i] - 0.5 * z[i] + ((i * 7) % 11) as f64 / 2.0 - 2.5).collect();
        let y: Vec<&str> = match three {
            false => s.iter().map(|s| if *s > 0.5 { "yes" } else { "no" }).collect(),
            true => s.iter().map(|s| if *s < -1.0 { "a" } else if *s < 1.5 { "b" } else { "c" }).collect(),
        };
        Dataset::new(df!("x" => x, "z" => z, "y" => y).unwrap(), "")
    }

    /// scikit-learn 1.9.1's `LogisticRegressionCV(solver="newton-cholesky", tol=1e-10)` on the
    /// same rows: the score grid, the first C of a tie, `refit=False`, and three classes.
    #[test]
    fn logistic_regression_cv_matches_scikit_learn() {
        let ds: Dataset = sixty(false);
        let cv = || LogisticRegressionCV::new().tol(1e-10);
        let mut model: LogisticRegressionCV = cv().scoring("neg_log_loss");
        model.fit(&ds, "y");
        let near = |a: f64, b: f64| assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        near(model.scores_[0][0], -0.636_543_424_175_189_8);
        near(model.scores_[0][2], -0.633_193_550_667_578_4);
        near(model.scores_[4][9], -0.786_788_615_023_289_6);
        let mut averaged: LogisticRegressionCV = cv().scoring("neg_log_loss").refit(false);
        averaged.fit(&ds, "y");
        near(averaged.coef_()[0][0], 0.111_157_330_769_604_85);
        near(averaged.intercept_()[0], 1.698_181_321_156_544_9);
        // Accuracy ties from the sixth C on; the first of them wins.
        let mut accuracy: LogisticRegressionCV = cv();
        accuracy.fit(&ds, "y");
        assert_eq!(accuracy.C_, 2.782_559_402_207_126);
        let mut auc: LogisticRegressionCV = cv().scoring("roc_auc");
        auc.fit(&ds, "y");
        assert_eq!(auc.C_, 0.046_415_888_336_127_774);
        near(auc.scores_.iter().map(|fold| fold[3]).sum::<f64>() / 5.0, 0.731_25);

        let three: Dataset = sixty(true);
        let mut model: LogisticRegressionCV = cv().Cs([0.1, 1.0, 10.0]).cv(3).scoring("neg_log_loss");
        model.fit(&three, "y");
        assert_eq!((model.C_, model.classes_()), (1.0, &["a".to_string(), "b".into(), "c".into()][..]));
        near(model.coef_()[1][1], 1.194_294_609_843_418_8);
        near(model.intercept_()[0], 3.540_537_000_296_443_4);
        assert_eq!(model.predict_proba(&three).columns(), ["a", "b", "c"]);
    }

    #[test]
    #[should_panic(expected = "Cs = Count(0) is not valid")]
    fn logistic_regression_cv_refuses_no_c() {
        let _ = LogisticRegressionCV::new().Cs(0);
    }
}
