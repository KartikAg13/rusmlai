//! [`LogisticRegression`]: two classes (sigmoid) or more (softmax), with an L2 penalty, fitted by
//! Newton's method on the weighted Gram matrix.
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
    models::{
        Classifier, Model, ModelError, class_labels,
        design::{CHUNK, DesignMatrix, col, scores},
        label_target,
        linear::solve,
    },
};

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
/// use polars::df;
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
    run: Option<String>,
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
        check(
            tol.is_finite() && tol >= 0.0,
            "tol",
            tol,
            "a finite number of 0 or more",
        );
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
fn logistic(z: f64) -> (f64, f64) {
    let e: f64 = (-z.abs()).exp();
    let p: f64 = if z >= 0.0 {
        1.0 / (1.0 + e)
    } else {
        e / (1.0 + e)
    };
    (p, z.max(0.0) + e.ln_1p())
}

/// The softmax of `z` into `out`, and ln Σ eᶻ.
fn softmax(z: &[f64], out: &mut [f64]) -> f64 {
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

    fn try_load(&mut self, learned: Value, target: &str) -> Result<(), ModelError> {
        let l: Learned = serde_json::from_value(learned)?;
        self.n_features_in_ = l.feature_names_in_.len();
        (self.classes_, self.coef_, self.intercept_) = (l.classes_, l.coef_, l.intercept_);
        (self.feature_names_in_, self.n_iter_) = (l.feature_names_in_, l.n_iter_);
        (self.loss_curve_, self.dtype) = (l.loss_curve_, Some(l.dtype));
        self.target = Some(target.into());
        Ok(())
    }

    fn run(&self) -> Option<&str> {
        self.run.as_deref()
    }

    fn set_run(&mut self, run: Option<String>) {
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
        Ok(dataset.derived(DataFrame::new_infer_height(columns)?))
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
        assert!(fit(nulls).starts_with("target 'y' has 1 nulls"));
        let text: Dataset = Dataset::new(df!("s" => ["u", "v"], "y" => ["a", "b"]).unwrap(), "");
        assert!(fit(text).starts_with("feature 's' is str, not a number"));
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
}
