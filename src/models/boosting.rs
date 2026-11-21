//! [`GradientBoostingClassifier`] and [`GradientBoostingRegressor`]: histogram gradient
//! boosting, like scikit-learn's `HistGradientBoosting*` and LightGBM.
//!
//! The model is a sum of small trees. Each round grows one tree (one per class for more than
//! two classes) on the gradients of the loss at the current predictions: the tree's leaves
//! hold the Newton step that lowers the loss most for their rows, times the learning rate.
//!
//! The features are put into buckets once (`models/hist.rs`), and every tree reads the same
//! buckets. A tree grows leaf by leaf, up to `max_leaf_nodes`. After a tree, each leaf adds
//! its value to the scores of its rows, which the tree already holds leaf by leaf, and one
//! pass over the rows makes the next gradients.

use std::io::{self, Write};

use polars::prelude::{DataType, NamedFrom, Series};
use polars_core::runtime::THREAD_POOL;
use rayon::prelude::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;

use crate::{
    dataset::{Dataset, split::Rng},
    metadata::models::CachedRun,
    models::{
        Classifier, Model, ModelError, Regressor, bad_learned, check,
        design::{column, prepared},
        hist::{Bins, Categories, Growth, Grown, Rules, Step, Target, draw_features, feature, grow, is_text},
        label_target,
        logistic::{logistic, softmax},
        numeric_target,
        tree::{ClassWeight, Tree, pick_classes, read_trees, share_columns, sum_leaves_on},
    },
};

/// Rows in one task of the pass that makes the gradients. The blocks are fixed and their
/// losses are added in block order, so the result is the same on any number of threads.
const BLOCK: usize = 16_384;

/// What boosting trains with, whatever library a model is named after: each library's
/// settings become a plan ([`Library::plan`]).
#[derive(Debug, Clone)]
pub(crate) struct Plan {
    /// The most rounds.
    pub rounds: usize,
    pub step: Step,
    pub max_depth: Option<usize>,
    /// `Some`: each tree grows leaf by leaf up to this many leaves.
    pub max_leaf_nodes: Option<usize>,
    pub min_samples_leaf: usize,
    /// scikit-learn's share of the features each node draws, rounded up.
    pub max_features: f64,
    /// The share of the features each tree draws, of those each depth, and of those each
    /// node (XGBoost's `colsample_by*`).
    pub by_tree: f64,
    pub by_level: f64,
    pub by_node: f64,
    /// The share of the training rows a draw takes, and for how many rounds a draw lasts
    /// (LightGBM's `bagging_freq`; 0: no draw).
    pub subsample: f64,
    pub subsample_freq: usize,
    pub rules: Rules,
    pub seed: u64,
    /// Early stopping, decided: whether, the share of rows held out, the rounds to wait and
    /// the least change that counts.
    pub early_stopping: bool,
    pub validation_fraction: f64,
    pub n_iter_no_change: usize,
    pub tol: f64,
    /// Where every score starts, instead of the constant with the lowest loss: XGBoost's
    /// `base_score` through the link, 0 for LightGBM's `boost_from_average=False`.
    pub base_score: Option<f64>,
    /// `Some`: CatBoost's symmetric trees instead of the engine's (`models/catboost.rs`).
    pub symmetric: Option<crate::models::catboost::Symmetric>,
}

impl Plan {
    /// Whether the scores of the last `n_iter_no_change` rounds are all no better than the
    /// one before them plus `tol`: scikit-learn's rule. A score is minus a loss.
    fn should_stop(&self, scores: &[f64]) -> bool {
        let back: usize = self.n_iter_no_change + 1;
        if scores.len() < back {
            return false;
        }
        let reference: f64 = scores[scores.len() - back] + self.tol;
        !scores[scores.len() - back + 1..]
            .iter()
            .any(|score| *score > reference)
    }
}

/// One library's settings of a boosted model, under that library's names and defaults: what
/// they make of the features, the target and the rounds. scikit-learn's are [`Params`];
/// `models/xgboost.rs` and `models/lightgbm.rs` hold the others.
pub(crate) trait Library: Clone + std::fmt::Debug + PartialEq + Serialize + DeserializeOwned + Send + Sync {
    /// The rounds and the trees, for `rows` rows.
    fn plan(&self, rows: usize) -> Plan;
    /// A classifier's plan, for `rows` rows of `classes` classes.
    fn classifier_plan(&self, rows: usize, _classes: usize) -> Plan {
        self.plan(rows)
    }
    /// The most buckets per feature.
    fn max_bins(&self) -> usize;
    /// The fewest rows of a bucket of neighbouring values (LightGBM's `min_data_in_bin`).
    fn min_data_in_bin(&self) -> usize {
        1
    }
    /// The features split as categories.
    fn categorical(&self, dataset: &Dataset, target: &str) -> Result<Vec<String>, ModelError>;
    /// A regressor's loss on target `y`, or why the settings do not fit it.
    fn regression<'a>(&self, y: &'a [f64]) -> Result<Loss<'a>, ModelError>;
    /// A classifier's loss on classes `y`, `classes` of them.
    fn classification<'a>(&self, y: &'a [u32], classes: usize) -> Result<Loss<'a>, ModelError>;
    /// A classifier's class weights.
    fn class_weight(&self) -> Option<&ClassWeight> {
        None
    }
    /// Whether a regressor's prediction is exp of its score (a log link).
    fn log_link(&self) -> bool;
}

/// The settings of a boosted model, with the names of scikit-learn's `HistGradientBoosting*`,
/// and for the settings it lacks, LightGBM's (`subsample`, `min_split_gain`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Params {
    loss: String,
    /// The quantile that the `"quantile"` loss predicts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    quantile: Option<f64>,
    learning_rate: f64,
    max_iter: usize,
    max_leaf_nodes: Option<usize>,
    max_depth: Option<usize>,
    min_samples_leaf: usize,
    l2_regularization: f64,
    #[serde(default)]
    l1_regularization: f64,
    #[serde(default)]
    min_split_gain: f64,
    #[serde(default = "one")]
    max_features: f64,
    #[serde(default = "one")]
    subsample: f64,
    max_bins: usize,
    /// `None` is scikit-learn's `"from_dtype"`: the text columns.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    categorical_features: Option<Vec<String>>,
    /// A classifier's only, and written only when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    class_weight: Option<ClassWeight>,
    /// `None` is scikit-learn's `"auto"`: on above 10,000 rows.
    early_stopping: Option<bool>,
    validation_fraction: f64,
    n_iter_no_change: usize,
    tol: f64,
    random_state: u64,
}

fn one() -> f64 {
    1.0
}

/// The text columns of `dataset` but `target`, or the columns `names`, which must exist.
pub(crate) fn categorical_columns(
    dataset: &Dataset,
    target: &str,
    names: Option<&[String]>,
) -> Result<Vec<String>, ModelError> {
    match names {
        Some(names) => {
            for name in names {
                dataset.column(name)?;
            }
            Ok(names.to_vec())
        }
        None => Ok(dataset
            .columns()
            .into_iter()
            .filter(|name| name != target)
            .filter(|name| dataset.column(name).is_ok_and(|c| is_text(c.dtype())))
            .collect()),
    }
}

impl Params {
    fn new(loss: &str) -> Self {
        Self {
            loss: loss.into(),
            quantile: None,
            learning_rate: 0.1,
            max_iter: 100,
            max_leaf_nodes: Some(31),
            max_depth: None,
            min_samples_leaf: 20,
            l2_regularization: 0.0,
            l1_regularization: 0.0,
            min_split_gain: 0.0,
            max_features: 1.0,
            subsample: 1.0,
            max_bins: 255,
            categorical_features: None,
            class_weight: None,
            early_stopping: None,
            validation_fraction: 0.1,
            n_iter_no_change: 10,
            tol: 1e-7,
            random_state: 42,
        }
    }
}

impl Library for Params {
    fn plan(&self, rows: usize) -> Plan {
        Plan {
            rounds: self.max_iter,
            step: Step {
                l1: self.l1_regularization,
                min_gain: self.min_split_gain,
                ..Step::new(self.l2_regularization, self.learning_rate)
            },
            max_depth: self.max_depth,
            max_leaf_nodes: self.max_leaf_nodes,
            min_samples_leaf: self.min_samples_leaf,
            max_features: self.max_features,
            by_tree: 1.0,
            by_level: 1.0,
            by_node: 1.0,
            subsample: self.subsample,
            subsample_freq: 1,
            rules: Rules::Sklearn,
            seed: self.random_state,
            early_stopping: self.early_stopping.unwrap_or(rows > 10_000),
            validation_fraction: self.validation_fraction,
            n_iter_no_change: self.n_iter_no_change,
            tol: self.tol,
            base_score: None,
            symmetric: None,
        }
    }

    fn max_bins(&self) -> usize {
        self.max_bins
    }

    /// Those named in `categorical_features`, or by default (scikit-learn's `"from_dtype"`)
    /// every text column but the target.
    fn categorical(&self, dataset: &Dataset, target: &str) -> Result<Vec<String>, ModelError> {
        categorical_columns(dataset, target, self.categorical_features.as_deref())
    }

    fn regression<'a>(&self, y: &'a [f64]) -> Result<Loss<'a>, ModelError> {
        let refuse = |expected: &'static str| ModelError::InvalidSetting {
            setting: "loss",
            value: self.loss.clone(),
            expected,
        };
        Ok(match self.loss.as_str() {
            "squared_error" => Loss::Squared(y),
            "absolute_error" => Loss::Absolute(y),
            "quantile" => Loss::Quantile(y, self.quantile.ok_or(ModelError::MissingSetting {
                setting: "quantile",
                reason: "loss \"quantile\" predicts a quantile of the target, and quantile(q) names it",
            })?),
            "poisson" if y.iter().all(|v| *v >= 0.0) && y.iter().sum::<f64>() > 0.0 => Loss::Poisson(y, 0.0),
            "poisson" => return Err(refuse("\"poisson\" only with a target of 0 or more that is not all 0")),
            "gamma" if y.iter().all(|v| *v > 0.0) => Loss::Gamma(y),
            "gamma" => return Err(refuse("\"gamma\" only with a target above 0 in every row")),
            _ => return Err(refuse("\"squared_error\", \"absolute_error\", \"quantile\", \"poisson\" or \"gamma\"")),
        })
    }

    fn classification<'a>(&self, y: &'a [u32], classes: usize) -> Result<Loss<'a>, ModelError> {
        Ok(match classes {
            2 => Loss::Binary(y, [1.0, 1.0]),
            k => Loss::Classes(y, k, 1.0),
        })
    }

    fn class_weight(&self) -> Option<&ClassWeight> {
        self.class_weight.as_ref()
    }

    fn log_link(&self) -> bool {
        matches!(self.loss.as_str(), "poisson" | "gamma")
    }
}

/// The loss a model boosts, with the target it measures against.
pub(crate) enum Loss<'a> {
    /// ½ (score − y)².
    Squared(&'a [f64]),
    /// |score − y|; each leaf then takes the median of its rows' residuals.
    Absolute(&'a [f64]),
    /// The pinball loss of quantile `q`; each leaf then takes that quantile of its residuals.
    Quantile(&'a [f64], f64),
    /// Half the Poisson deviance with a log link: the score is the log of the prediction. The
    /// hessian is exp(score + the second number): XGBoost and LightGBM add 0.7
    /// (`max_delta_step`, `poisson_max_delta_step`) so that a leaf moves less.
    Poisson(&'a [f64], f64),
    /// Half the gamma deviance with a log link.
    Gamma(&'a [f64]),
    /// The Tweedie deviance of variance power ρ (1 to 2), with a log link.
    Tweedie(&'a [f64], f64),
    /// LightGBM's Huber loss: the gradient is the residual, cut at ±α; the hessian is 1.
    Huber(&'a [f64], f64),
    /// XGBoost's pseudo-Huber loss of slope δ: δ² (√(1 + (r/δ)²) − 1).
    PseudoHuber(&'a [f64], f64),
    /// LightGBM's fair loss of `c`: c |r| − c² ln(1 + |r| / c).
    Fair(&'a [f64], f64),
    /// LightGBM's mean absolute percentage error: |r| / max(1, |y|); each leaf then takes the
    /// median of its residuals, weighted the same.
    Mape(&'a [f64]),
    /// The log loss of two classes: one score per row, the log-odds of class 1. The weights
    /// of class 0 and class 1 scale each row's gradient and hessian but not the start
    /// (XGBoost's `scale_pos_weight`, LightGBM's `is_unbalance`).
    Binary(&'a [u32], [f64; 2]),
    /// The log loss of this many classes: one score per class and row. The hessian is
    /// p (1 − p) times the third number: 1 in scikit-learn, 2 in XGBoost, k / (k − 1) in
    /// LightGBM.
    Classes(&'a [u32], usize, f64),
}

impl Loss<'_> {
    /// The scores per row, and so the trees per round.
    pub(crate) fn outputs(&self) -> usize {
        match self {
            Loss::Classes(_, classes, _) => *classes,
            _ => 1,
        }
    }

    /// The class of each row, for a split of the rows that keeps the share of each class.
    fn classes(&self) -> Option<(&[u32], usize)> {
        match self {
            Loss::Binary(y, _) => Some((y, 2)),
            Loss::Classes(y, classes, _) => Some((y, *classes)),
            _ => None,
        }
    }

    /// The target of a regression loss.
    fn regression_target(&self) -> Option<&[f64]> {
        match self {
            Loss::Squared(y)
            | Loss::Absolute(y)
            | Loss::Quantile(y, _)
            | Loss::Poisson(y, _)
            | Loss::Gamma(y)
            | Loss::Tweedie(y, _)
            | Loss::Huber(y, _)
            | Loss::PseudoHuber(y, _)
            | Loss::Fair(y, _)
            | Loss::Mape(y) => Some(y),
            Loss::Binary(..) | Loss::Classes(..) => None,
        }
    }

    /// Whether every hessian is 1 (without row weights): a bucket then needs no Σh.
    fn constant_hessian(&self) -> bool {
        matches!(self, Loss::Squared(_) | Loss::Absolute(_) | Loss::Quantile(..) | Loss::Huber(..) | Loss::Mape(_))
    }

    /// For a loss that is not smooth, the quantile of its rows' residuals that a leaf takes
    /// after its tree grows, as in scikit-learn (`_update_leaves_values`): the median for
    /// the absolute error. `None` keeps the Newton step.
    fn leaf_quantile(&self) -> Option<f64> {
        match self {
            Loss::Absolute(_) | Loss::Mape(_) => Some(0.5),
            Loss::Quantile(_, q) => Some(*q),
            _ => None,
        }
    }

    /// The weight of a row's residual in a leaf's quantile: 1 / max(1, |y|) for the MAPE.
    fn label_weight(&self, y: f64) -> f64 {
        match self {
            Loss::Mape(_) => 1.0 / y.abs().max(1.0),
            _ => 1.0,
        }
    }

    /// The scores that every row starts from: the constant with the lowest loss on `rows`,
    /// like scikit-learn's `fit_intercept_only`. The (weighted) mean, the median or a
    /// quantile, the log of the mean, the log-odds of class 1, or the logs of the class
    /// shares around their mean.
    pub(crate) fn baseline(&self, rows: &[u32], weights: Option<&[f64]>, rules: Rules) -> Vec<f64> {
        let weight = |row: u32| weights.map_or(1.0, |w| w[row as usize]);
        let total: f64 = rows.iter().map(|row| weight(*row)).sum();
        let mean = |y: &[f64]| rows.iter().map(|r| weight(*r) * y[*r as usize]).sum::<f64>() / total;
        let share = |y: &[u32], class: u32| {
            let count: f64 = rows.iter().filter(|r| y[**r as usize] == class).map(|r| weight(*r)).sum();
            (count / total).clamp(f64::EPSILON, 1.0 - f64::EPSILON)
        };
        let quantile_of = |y: &[f64], q: f64| {
            let mut values: Vec<f64> = rows.iter().map(|r| y[*r as usize]).collect();
            let w: Option<Vec<f64>> = match (weights, self) {
                (_, Loss::Mape(_)) => Some(rows.iter().map(|r| weight(*r) * self.label_weight(y[*r as usize])).collect()),
                (Some(w), _) => Some(rows.iter().map(|r| w[*r as usize]).collect()),
                (None, _) => None,
            };
            quantile(&mut values, w.as_deref(), q, rules, matches!(self, Loss::Absolute(_)))
        };
        // XGBoost's `FitStump`: one Newton step from 0, for the pseudo-Huber loss and for two
        // classes of unequal weight.
        let stump: bool = rules == Rules::Xgboost
            && (matches!(self, Loss::PseudoHuber(..)) || matches!(self, Loss::Binary(_, labels) if labels[1] != 1.0));
        if stump {
            let (mut g, mut h): (f64, f64) = (0.0, 0.0);
            for row in rows {
                let w: f64 = weight(*row);
                let (_, gradient, hessian): (f64, f64, f64) = match self {
                    Loss::Binary(y, labels) => {
                        let label: f64 = labels[y[*row as usize] as usize];
                        (0.0, label * (0.5 - f64::from(y[*row as usize])), label * 0.25)
                    }
                    _ => self.point(self.regression_target().expect("regression")[*row as usize], 0.0, rules),
                };
                (g, h) = (g + w * gradient, h + w * hessian);
            }
            return vec![-g / h];
        }
        match self {
            Loss::Squared(y) | Loss::Huber(y, _) | Loss::PseudoHuber(y, _) | Loss::Fair(y, _) => vec![mean(y)],
            Loss::Absolute(y) | Loss::Mape(y) => vec![quantile_of(y, 0.5)],
            Loss::Quantile(y, q) => vec![quantile_of(y, *q)],
            // scikit-learn keeps the mean at least 10 ε before the log.
            Loss::Poisson(y, _) | Loss::Gamma(y) | Loss::Tweedie(y, _) => vec![mean(y).max(10.0 * f64::EPSILON).ln()],
            Loss::Binary(y, _) => {
                let p: f64 = share(y, 1);
                vec![(p / (1.0 - p)).ln()]
            }
            Loss::Classes(y, classes, _) => {
                let logs: Vec<f64> = (0..*classes as u32).map(|c| share(y, c).ln()).collect();
                let mean: f64 = logs.iter().sum::<f64>() / logs.len() as f64;
                logs.iter().map(|log| log - mean).collect()
            }
        }
    }

    /// One regression row: (loss, gradient, hessian) at score `s` for target `y`, as
    /// scikit-learn's `_loss`, XGBoost's and LightGBM's objectives compute them. The sign of
    /// a residual of 0 is −1 in scikit-learn and 0 in the other two.
    fn point(&self, y: f64, s: f64, rules: Rules) -> (f64, f64, f64) {
        let sign = |r: f64| match rules {
            Rules::Sklearn => if r > 0.0 { 1.0 } else { -1.0 },
            _ => f64::from(i8::from(r > 0.0) - i8::from(r < 0.0)),
        };
        match self {
            Loss::Squared(_) => (0.5 * (s - y) * (s - y), s - y, 1.0),
            Loss::Absolute(_) => ((s - y).abs(), sign(s - y), 1.0),
            // XGBoost and LightGBM take a residual of 0 as above the quantile, scikit-learn as
            // below.
            Loss::Quantile(_, q) => match if rules == Rules::Sklearn { y >= s } else { y > s } {
                true => (q * (y - s), -q, 1.0),
                false => ((1.0 - q) * (s - y), 1.0 - q, 1.0),
            },
            Loss::Poisson(_, extra) => {
                let e: f64 = s.exp();
                (e - y * s, e - y, (s + extra).exp())
            }
            Loss::Gamma(_) => {
                let e: f64 = (-s).exp();
                (s + y * e, 1.0 - y * e, y * e)
            }
            Loss::Tweedie(_, rho) => {
                let (a, b): (f64, f64) = (((1.0 - rho) * s).exp(), ((2.0 - rho) * s).exp());
                let loss: f64 = -y * a / (1.0 - rho) + b / (2.0 - rho);
                (loss, -y * a + b, -y * (1.0 - rho) * a + (2.0 - rho) * b)
            }
            Loss::Huber(_, alpha) => {
                let r: f64 = s - y;
                let loss: f64 = if r.abs() <= *alpha { 0.5 * r * r } else { alpha * (r.abs() - 0.5 * alpha) };
                (loss, r.clamp(-alpha, *alpha), 1.0)
            }
            Loss::PseudoHuber(_, slope) => {
                let r: f64 = s - y;
                let scale: f64 = 1.0 + (r / slope) * (r / slope);
                let root: f64 = scale.sqrt();
                (slope * slope * (root - 1.0), r / root, 1.0 / (scale * root))
            }
            Loss::Fair(_, c) => {
                let r: f64 = s - y;
                let loss: f64 = c * r.abs() - c * c * (r.abs() / c).ln_1p();
                (loss, c * r / (r.abs() + c), c * c / ((r.abs() + c) * (r.abs() + c)))
            }
            Loss::Mape(_) => {
                let w: f64 = self.label_weight(y);
                (w * (s - y).abs(), w * sign(s - y), 1.0)
            }
            Loss::Binary(..) | Loss::Classes(..) => unreachable!("a classifier's rows are in gradients"),
        }
    }

    /// One pass over the rows: the gradient and the hessian of the loss at each row's
    /// `scores`, into `g` and `h` (one list per score of a row; `h` stays empty when every
    /// hessian is 1), each times the row's weight. With `measure`, also the weighted loss of
    /// the rows that train and of the rows `held` out. `shares` is room for class shares.
    #[allow(clippy::too_many_arguments)]
    fn gradients(
        &self,
        scores: &[f64],
        held: &[bool],
        (g, h): (&mut [Vec<f32>], &mut [Vec<f32>]),
        shares: &mut Vec<f64>,
        measure: bool,
        weights: Option<&[f64]>,
        rules: Rules,
    ) -> (f64, f64) {
        let weight = |row: usize| weights.map_or(1.0, |w| w[row]);
        // The loss of one row goes to the sum of its side.
        let add = |sums: &mut (f64, f64), row: usize, loss: f64| {
            if held[row] {
                sums.1 += loss;
            } else {
                sums.0 += loss;
            }
        };
        let blocks: Vec<(f64, f64)> = THREAD_POOL.install(|| match self {
            Loss::Binary(y, labels) => g[0]
                .par_chunks_mut(BLOCK)
                .zip(h[0].par_chunks_mut(BLOCK))
                .enumerate()
                .map(|(b, (g, h))| {
                    let mut sums: (f64, f64) = (0.0, 0.0);
                    for (row, (g, h)) in (b * BLOCK..).zip(g.iter_mut().zip(h)) {
                        let (p, softplus): (f64, f64) = logistic(scores[row]);
                        let hit: f64 = f64::from(y[row]);
                        let w: f64 = weight(row) * labels[y[row] as usize];
                        *g = (w * (p - hit)) as f32;
                        *h = (w * floored(p * (1.0 - p), rules)) as f32;
                        if measure {
                            add(&mut sums, row, w * (softplus - hit * scores[row]));
                        }
                    }
                    sums
                })
                .collect(),
            Loss::Classes(y, k, _) => {
                shares.resize(scores.len(), 0.0);
                shares
                    .par_chunks_mut(BLOCK * k)
                    .enumerate()
                    .map(|(b, shares)| {
                        let mut sums: (f64, f64) = (0.0, 0.0);
                        for (row, shares) in (b * BLOCK..).zip(shares.chunks_mut(*k)) {
                            let z: &[f64] = &scores[row * k..][..*k];
                            let log_sum: f64 = softmax(z, shares);
                            if measure {
                                add(&mut sums, row, weight(row) * (log_sum - z[y[row] as usize]));
                            }
                        }
                        sums
                    })
                    .collect()
            }
            _ => {
                let y: &[f64] = self.regression_target().expect("a regression loss");
                let block = |b: usize, g: &mut [f32], mut h: Option<&mut [f32]>| {
                    let mut sums: (f64, f64) = (0.0, 0.0);
                    for (i, g) in g.iter_mut().enumerate() {
                        let row: usize = b * BLOCK + i;
                        let (loss, gradient, hessian): (f64, f64, f64) = self.point(y[row], scores[row], rules);
                        let w: f64 = weight(row);
                        *g = (w * gradient) as f32;
                        if let Some(h) = h.as_deref_mut() {
                            h[i] = (w * hessian) as f32;
                        }
                        if measure {
                            add(&mut sums, row, w * loss);
                        }
                    }
                    sums
                };
                match h[0].is_empty() {
                    true => g[0].par_chunks_mut(BLOCK).enumerate().map(|(b, g)| block(b, g, None)).collect(),
                    false => (g[0].par_chunks_mut(BLOCK))
                        .zip(h[0].par_chunks_mut(BLOCK))
                        .enumerate()
                        .map(|(b, (g, h))| block(b, g, Some(h)))
                        .collect(),
                }
            }
        });
        if let Loss::Classes(y, k, factor) = self {
            let shares: &[f64] = shares;
            THREAD_POOL.install(|| {
                g.par_iter_mut().zip(h.par_iter_mut()).enumerate().for_each(|(class, (g, h))| {
                    for (row, (g, h)) in g.iter_mut().zip(h.iter_mut()).enumerate() {
                        let (p, w): (f64, f64) = (shares[row * k + class], weight(row));
                        *g = (w * (p - f64::from(y[row] as usize == class))) as f32;
                        *h = (w * floored(factor * p * (1.0 - p), rules)) as f32;
                    }
                })
            });
        }
        blocks
            .into_iter()
            .fold((0.0, 0.0), |sum, block| (sum.0 + block.0, sum.1 + block.1))
    }
}

/// A class hessian, at least 1e-16 in XGBoost (`fmaxf(h, eps)`).
fn floored(h: f64, rules: Rules) -> f64 {
    if rules == Rules::Xgboost { h.max(1e-16) } else { h }
}

/// Quantile `q` of `values` (reordered), with `weights` or not, as each library takes it.
/// scikit-learn's `fit_intercept_only`: without weights numpy's `median` (`median`) or its
/// linear `percentile`; with weights `_weighted_percentile`, the first value whose running
/// weight reaches `q` of the total. XGBoost's `common::Quantile` and `WeightedQuantile`.
/// LightGBM's `PercentileFun` and `WeightedPercentileFun`.
fn quantile(values: &mut [f64], weights: Option<&[f64]>, q: f64, rules: Rules, median: bool) -> f64 {
    let n: usize = values.len();
    if n == 1 {
        return values[0];
    }
    if let Some(weights) = weights {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|a, b| values[*a].total_cmp(&values[*b]));
        let cdf: Vec<f64> = order
            .iter()
            .scan(0.0, |sum, i| {
                *sum += weights[*i];
                Some(*sum)
            })
            .collect();
        let value = |at: usize| values[order[at]];
        return match rules {
            Rules::Sklearn => {
                let mut rank: f64 = (100.0 * q) / 100.0 * cdf[n - 1];
                if rank == 0.0 {
                    rank = f64::from_bits(1); // numpy's nextafter(0, 1): leading zero weights skip
                }
                value(cdf.partition_point(|c| *c < rank).min(n - 1))
            }
            Rules::Xgboost => value(cdf.partition_point(|c| *c < q * cdf[n - 1]).min(n - 1)),
            Rules::Lightgbm => {
                let threshold: f64 = cdf[n - 1] * q;
                let at: usize = cdf.partition_point(|c| *c <= threshold).min(n - 1);
                if at == 0 || at == n - 1 {
                    return value(at);
                }
                let (v1, v2): (f64, f64) = (value(at - 1), value(at));
                match cdf[at + 1] - cdf[at] >= 1.0 {
                    true => (threshold - cdf[at]) / (cdf[at + 1] - cdf[at]) * (v2 - v1) + v1,
                    false => v1,
                }
            }
        };
    }
    values.sort_unstable_by(f64::total_cmp);
    match rules {
        Rules::Sklearn if median => match n % 2 {
            1 => values[n / 2],
            _ => (values[n / 2 - 1] + values[n / 2]) / 2.0,
        },
        Rules::Sklearn => {
            // numpy's "linear" method, step for step.
            let q: f64 = (100.0 * q) / 100.0;
            let at: f64 = n as f64 * q + (1.0 - q) - 1.0;
            let below: f64 = at.floor().clamp(0.0, (n - 1) as f64);
            let (a, b): (f64, f64) = (values[below as usize], values[(below as usize + 1).min(n - 1)]);
            let t: f64 = at - below;
            let diff: f64 = b - a;
            if t >= 0.5 { b - diff * (1.0 - t) } else { a + diff * t }
        }
        Rules::Xgboost => {
            let n: f64 = n as f64;
            if q <= 1.0 / (n + 1.0) {
                return values[0];
            }
            if q >= n / (n + 1.0) {
                return values[values.len() - 1];
            }
            let x: f64 = q * (n + 1.0);
            let k: f64 = x.floor() - 1.0;
            let d: f64 = (x - 1.0) - k;
            let (v0, v1): (f64, f64) = (values[k as usize], values[k as usize + 1]);
            v0 + d * (v1 - v0)
        }
        Rules::Lightgbm => {
            // In falling order, (n − 1)(1 − q) places down: numpy's linear percentile.
            let at: f64 = (n - 1) as f64 * (1.0 - q);
            let pos: usize = at as usize;
            if pos >= n - 1 {
                return values[0];
            }
            let (v1, v2): (f64, f64) = (values[n - 1 - pos], values[n - 2 - pos]);
            v1 - (v1 - v2) * (at - pos as f64)
        }
    }
}

/// What boosting learned.
pub(crate) struct Boosted {
    /// One tree per round, or one per class and round, round after round.
    pub(crate) trees: Vec<Tree>,
    pub(crate) baseline: Vec<f64>,
    /// Minus the loss before the first round and after each round, on the training rows and
    /// on the rows held out; empty without early stopping, like scikit-learn's.
    pub(crate) train_score: Vec<f64>,
    pub(crate) validation_score: Vec<f64>,
    pub(crate) n_iter: usize,
    pub(crate) importances: Vec<f64>,
}

/// The rows held out of training to stop early: `validation_fraction` of the rows, or of the
/// rows of each class, drawn from the seed.
fn hold_out(loss: &Loss, rows: usize, p: &Plan) -> Vec<bool> {
    let mut rng: Rng = Rng::new(p.seed);
    let mut held: Vec<bool> = vec![false; rows];
    let mut draw = |mut group: Vec<u32>| {
        rng.shuffle(&mut group);
        let count: usize = (p.validation_fraction * group.len() as f64).round() as usize;
        group[..count].iter().for_each(|row| held[*row as usize] = true);
    };
    match loss.classes() {
        None => draw((0..rows as u32).collect()),
        Some((y, classes)) => (0..classes as u32)
            .for_each(|class| draw((0..rows as u32).filter(|r| y[*r as usize] == class).collect())),
    }
    held
}

/// A seed from `seed` and the place of a tree among the trees.
pub(crate) fn mixed(seed: u64, place: u64) -> u64 {
    Rng::new(seed ^ place.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15)).next_u64()
}

/// Round `round`'s rows: each of `train` with chance `share`, in blocks of [`BLOCK`] rows
/// with seeds of their own, so the draw is the same on any number of threads. Gives the
/// rows drawn and the rest, both in rising order.
fn subsample(train: &[u32], share: f64, seed: u64) -> (Vec<u32>, Vec<u32>) {
    let cut: u64 = (share * u64::MAX as f64) as u64;
    let blocks: Vec<(Vec<u32>, Vec<u32>)> = THREAD_POOL.install(|| {
        train
            .par_chunks(BLOCK)
            .enumerate()
            .map(|(b, rows)| {
                let mut rng: Rng = Rng::new(mixed(seed, b as u64));
                rows.iter().partition(|_| rng.next_u64() < cut)
            })
            .collect()
    });
    let (mut drawn, mut rest): (Vec<u32>, Vec<u32>) = (Vec::new(), Vec::new());
    for (d, r) in blocks {
        drawn.extend(d);
        rest.extend(r);
    }
    (drawn, rest)
}

/// Every feature as a tree reads it at predict time, for `rows` (`None`: every row):
/// numbers with NaN for a missing value, or a categorical feature's category codes.
fn walk_columns(dataset: &Dataset, bins: &Bins, rows: Option<&[u32]>) -> Result<Vec<Vec<f64>>, ModelError> {
    let pick = |all: Vec<f64>| match rows {
        Some(rows) => rows.iter().map(|row| all[*row as usize]).collect(),
        None => all,
    };
    THREAD_POOL.install(|| {
        (0..bins.names.len())
            .into_par_iter()
            .map(|f| match bins.categories(f) {
                Some(_) => Ok(pick(bins.codes(f))),
                None => Ok(pick(column(&feature(dataset, &bins.names[f])?).to_vec())),
            })
            .collect()
    })
}

/// The leaf value in `tree` of each of `rows`, walked from `columns` (every row of the
/// data), in chunks on the Polars pool.
fn walked(tree: &Tree, columns: &[Vec<f64>], rows: &[u32]) -> Vec<f64> {
    THREAD_POOL.install(|| {
        rows.par_chunks(4096)
            .flat_map_iter(|chunk| {
                let local: Vec<Vec<f64>> = (columns.iter())
                    .map(|c| chunk.iter().map(|row| c[*row as usize]).collect())
                    .collect();
                let local: Vec<&[f64]> = local.iter().map(Vec::as_slice).collect();
                let mut leaves: Vec<u32> = vec![0; chunk.len()];
                tree.walk(&local, 0, &mut leaves);
                leaves.into_iter().map(|leaf| tree.value()[leaf as usize])
            })
            .collect()
    })
}

/// Boosts `p.rounds` rounds of trees on the buckets, or fewer when the loss on the rows
/// held out stops getting better. `weights` is a weight per row (`None`: each counts once).
fn boost(
    dataset: &Dataset,
    bins: &Bins,
    loss: &Loss,
    p: &Plan,
    weights: Option<&[f64]>,
) -> Result<Boosted, ModelError> {
    if let Some(symmetric) = &p.symmetric {
        return Ok(crate::models::catboost::boost_symmetric(bins, loss, p, symmetric, weights));
    }
    let (n, k, features): (usize, usize, usize) = (dataset.height(), loss.outputs(), bins.names.len());
    let mut held: Vec<bool> = match p.early_stopping {
        true => hold_out(loss, n, p),
        false => vec![false; n],
    };
    let held_rows: Vec<u32> = (0..n as u32).filter(|row| held[*row as usize]).collect();
    // Too few rows to hold one out: no early stopping.
    let early: bool = !held_rows.is_empty() && held_rows.len() < n;
    if !early {
        held.fill(false);
    }
    let train: Vec<u32> = (0..n as u32).filter(|row| !held[*row as usize]).collect();
    let sampled: bool = p.subsample < 1.0 && p.subsample_freq > 0;
    // Rows outside a tree's own walk it: the rows held out (their columns in line, made
    // once), and the rows a round did not draw.
    let columns: Vec<Vec<f64>> = match sampled {
        true => walk_columns(dataset, bins, None)?,
        false => Vec::new(),
    };
    let held_columns: Vec<Vec<f64>> = match early {
        true => walk_columns(dataset, bins, Some(&held_rows))?,
        false => Vec::new(),
    };
    let held_columns: Vec<&[f64]> = held_columns.iter().map(Vec::as_slice).collect();
    let mut held_leaves: Vec<u32> = vec![0; if early { held_rows.len() } else { 0 }];
    let weight_of = |rows: &[u32]| -> f64 { weights.map_or(rows.len() as f64, |w| rows.iter().map(|r| w[*r as usize]).sum()) };
    let (train_weight, held_weight): (f64, f64) = (weight_of(&train), weight_of(&held_rows));

    let baseline: Vec<f64> = match p.base_score {
        Some(score) => vec![score; k],
        None => loss.baseline(&train, weights, p.rules),
    };
    let mut scores: Vec<f64> = (0..n * k).map(|i| baseline[i % k]).collect();
    let mut g: Vec<Vec<f32>> = vec![vec![0.0; n]; k];
    let hessians: usize = if loss.constant_hessian() && weights.is_none() { 0 } else { n };
    let mut h: Vec<Vec<f32>> = vec![vec![0.0; hessians]; k];
    let mut shares: Vec<f64> = Vec::new();
    let growth: Growth = Growth {
        max_depth: p.max_depth,
        min_samples_split: 2,
        min_samples_leaf: p.min_samples_leaf,
        min_impurity_decrease: 0.0,
        // scikit-learn's count: the share of the features, rounded up, at least 1.
        max_features: ((p.max_features * features as f64).ceil() as usize).clamp(1, features),
        seed: p.seed,
        max_leaf_nodes: p.max_leaf_nodes,
        features: None,
        by_level: p.by_level,
        by_node: p.by_node,
        rules: p.rules,
        random_splits: false,
    };
    let mut boosted: Boosted = Boosted {
        trees: Vec::new(),
        baseline,
        train_score: Vec::new(),
        validation_score: Vec::new(),
        n_iter: 0,
        importances: vec![0.0; features],
    };
    let (mut drawn, mut rest): (Vec<u32>, Vec<u32>) = (train.clone(), Vec::new());
    loop {
        if !early && boosted.n_iter == p.rounds {
            break;
        }
        let (train_loss, held_loss): (f64, f64) =
            loss.gradients(&scores, &held, (&mut g, &mut h), &mut shares, early, weights, p.rules);
        if early {
            boosted.train_score.push(-train_loss / train_weight);
            boosted.validation_score.push(-held_loss / held_weight);
            if boosted.n_iter == p.rounds || p.should_stop(&boosted.validation_score) {
                break;
            }
        }
        let round: u64 = boosted.n_iter as u64;
        // The rows this round's trees grow on: all, or a `subsample` share drawn anew every
        // `subsample_freq` rounds.
        if sampled && round.is_multiple_of(p.subsample_freq as u64) {
            (drawn, rest) = subsample(&train, p.subsample, mixed(p.seed, round));
        }
        for class in 0..k {
            let target: Target = Target::Gradients {
                g: std::mem::take(&mut g[class]),
                h: std::mem::take(&mut h[class]),
                step: Step { rules: p.rules, ..p.step },
            };
            // Each tree draws its features from a seed of its own.
            let seed: u64 = mixed(p.seed, round * k as u64 + class as u64);
            let all: Vec<u32> = (0..features as u32).collect();
            let growth: Growth = Growth {
                seed,
                features: (p.by_tree < 1.0).then(|| {
                    let mut rng: Rng = Rng::new(seed ^ 0x5DEE_CE66_D1CE_4E5B);
                    draw_features(&all, p.rules.count(features, p.by_tree), &mut rng)
                }),
                ..growth.clone()
            };
            let mut grown: Grown = grow(bins, &target, &growth, None, Some(drawn.clone()));
            if let Target::Gradients { g: gradients, h: hessians, .. } = target {
                (g[class], h[class]) = (gradients, hessians);
            }
            // The tree holds its rows leaf by leaf, each leaf's in rising order.
            let tree: &mut Tree = &mut grown.tree;
            let mut leaves: Vec<(usize, &[u32])> = Vec::new();
            let mut at: usize = 0;
            for node in 0..tree.node_count() {
                if tree.is_leaf(node) {
                    let rows: usize = tree.n_node_samples()[node] as usize;
                    leaves.push((node, &grown.rows[at..at + rows]));
                    at += rows;
                }
            }
            // A loss that is not smooth: each leaf takes a quantile of its rows' residuals.
            if let (Some(q), Some(y)) = (loss.leaf_quantile(), loss.regression_target()) {
                for (node, rows) in &leaves {
                    let mut residuals: Vec<f64> = rows.iter().map(|r| y[*r as usize] - scores[*r as usize]).collect();
                    let w: Option<Vec<f64>> = match (weights, loss) {
                        (_, Loss::Mape(_)) => Some(
                            (rows.iter())
                                .map(|r| weights.map_or(1.0, |w| w[*r as usize]) * loss.label_weight(y[*r as usize]))
                                .collect(),
                        ),
                        (Some(w), _) => Some(rows.iter().map(|r| w[*r as usize]).collect()),
                        (None, _) => None,
                    };
                    let median: bool = matches!(loss, Loss::Absolute(_));
                    tree.value_mut()[*node] = p.step.rate * quantile(&mut residuals, w.as_deref(), q, p.rules, median);
                }
            }
            // Each leaf adds its value to its rows, in blocks of rows, one task each.
            let values: Vec<(&[u32], f64)> = leaves.iter().map(|(node, rows)| (*rows, tree.value()[*node])).collect();
            THREAD_POOL.install(|| {
                scores.par_chunks_mut(BLOCK * k).enumerate().for_each(|(b, block)| {
                    let (lo, hi): (u32, u32) = ((b * BLOCK) as u32, ((b + 1) * BLOCK) as u32);
                    for (rows, value) in &values {
                        let from: usize = rows.partition_point(|row| *row < lo);
                        let to: usize = rows.partition_point(|row| *row < hi);
                        for row in &rows[from..to] {
                            block[(row - lo) as usize * k + class] += value;
                        }
                    }
                });
            });
            // The rows the tree did not grow on walk it.
            if !rest.is_empty() {
                for (row, value) in rest.iter().zip(walked(tree, &columns, &rest)) {
                    scores[*row as usize * k + class] += value;
                }
            }
            if early {
                tree.leaves(&held_columns, &mut held_leaves);
                for (row, leaf) in held_rows.iter().zip(&held_leaves) {
                    scores[*row as usize * k + class] += tree.value()[*leaf as usize];
                }
            }
            for (sum, share) in boosted.importances.iter_mut().zip(&grown.importances) {
                *sum += share;
            }
            boosted.trees.push(grown.tree);
        }
        boosted.n_iter += 1;
    }
    let sum: f64 = boosted.importances.iter().sum();
    if sum > 0.0 {
        boosted.importances.iter_mut().for_each(|share| *share /= sum);
    }
    Ok(boosted)
}

/// The scores of each row of `dataset`: the baseline plus what each tree's leaf adds, one
/// number per row, or one per class and row. A categorical feature is read as its codes.
pub(crate) fn scores_of(
    trees: &[Tree],
    baseline: &[f64],
    dataset: &Dataset,
    names: &[String],
    categories: &[Option<Categories>],
) -> Result<Vec<f64>, ModelError> {
    let k: usize = baseline.len();
    let series: Vec<Series> = THREAD_POOL.install(|| {
        (0..names.len())
            .into_par_iter()
            .map(|f| match categories.get(f).and_then(Option::as_ref) {
                Some(known) => Ok(Series::new(names[f].as_str().into(), known.codes(dataset, &names[f])?)),
                None => feature(dataset, &names[f]),
            })
            .collect::<Result<_, ModelError>>()
    })?;
    let columns: Vec<&[f64]> = series.iter().map(column).collect();
    let add = |t: usize, value: &[f64], sum: &mut [f64]| sum[t % k] += value[0];
    let mut scores: Vec<f64> = sum_leaves_on(trees, &columns, dataset.height(), k, add);
    for row in scores.chunks_mut(k) {
        row.iter_mut().zip(baseline).for_each(|(score, base)| *score += base);
    }
    Ok(scores)
}

/// The learned values that are small, as the model cache stores them in JSON. The trees are
/// bytes.
#[derive(Serialize, Deserialize)]
pub(crate) struct Learned {
    /// Empty for a regressor.
    pub(crate) classes_: Vec<String>,
    pub(crate) feature_names_in_: Vec<String>,
    pub(crate) feature_importances_: Vec<f64>,
    pub(crate) dtype: DataType,
    pub(crate) n_iter_: usize,
    pub(crate) train_score_: Vec<f64>,
    pub(crate) validation_score_: Vec<f64>,
    pub(crate) baseline_: Vec<f64>,
    /// The categories of each categorical feature, `None` for the others; empty in a record
    /// from before categorical features.
    #[serde(default)]
    pub(crate) categories_: Vec<Option<Categories>>,
}

impl Learned {
    /// The record with its trees, when its parts fit together: one tree per round and score.
    pub(crate) fn checked(learned: Value, weights: &[u8]) -> Result<(Self, Vec<Tree>), ModelError> {
        let l: Learned = serde_json::from_value(learned)?;
        let features: usize = l.feature_names_in_.len();
        let trees: Vec<Tree> = read_trees(weights, features, 1)?;
        let outputs: usize = l.baseline_.len();
        let classes: usize = l.classes_.len();
        let fits: bool = match classes {
            0 | 2 => outputs == 1,
            _ => outputs == classes,
        };
        let categories: bool = l.categories_.is_empty() || l.categories_.len() == features;
        if !fits || !categories || classes == 1 || trees.len() != l.n_iter_ * outputs || l.feature_importances_.len() != features {
            return Err(bad_learned(format!(
                "{} trees for {} rounds of {outputs} scores, {classes} classes, and {} \
                 importances for {features} features",
                trees.len(),
                l.n_iter_,
                l.feature_importances_.len()
            )));
        }
        Ok((l, trees))
    }
}

/// The buckets of `dataset`'s features for settings `p`, with each feature's categories
/// (`None` for a feature that is not categorical).
fn binned<L: Library>(dataset: &Dataset, target: &str, p: &L) -> Result<(Bins, Vec<Option<Categories>>), ModelError> {
    let categorical: Vec<String> = p.categorical(dataset, target)?;
    let bins: Bins = Bins::try_with(dataset, target, p.max_bins(), &categorical)?.merged(p.min_data_in_bin());
    let categories: Vec<Option<Categories>> = (0..bins.names.len()).map(|f| bins.categories(f)).collect();
    Ok((bins, categories))
}

/// A weight per row from `weights` (a caller's, such as a bag's), or none.
fn widened(weights: Option<&[f32]>) -> Option<Vec<f64>> {
    weights.map(|w| w.iter().map(|w| f64::from(*w)).collect())
}

/// The settings both boosted models share, as setters on `$model`.
/// What a fit of a boosted model learned, before it goes into the model's fields.
pub(crate) struct Fitted {
    pub(crate) boosted: Boosted,
    pub(crate) names: Vec<String>,
    pub(crate) categories: Vec<Option<Categories>>,
    /// A classifier's classes and the target's type.
    pub(crate) classes: Vec<String>,
    pub(crate) dtype: Option<DataType>,
}

/// Fits a regressor of settings `p` on `dataset`, each row with its weight in `weights`
/// (`None`: once).
pub(crate) fn fit_regressor<L: Library>(
    p: &L,
    dataset: &Dataset,
    target: &str,
    weights: Option<&[f32]>,
) -> Result<Fitted, ModelError> {
    numeric_target(dataset, target)?;
    let (y, _): (Series, f64) = prepared(dataset, target)?;
    let y: &[f64] = column(&y);
    let loss: Loss = p.regression(y)?;
    let (bins, categories): (Bins, Vec<Option<Categories>>) = binned(dataset, target, p)?;
    let weights: Option<Vec<f64>> = widened(weights);
    let boosted: Boosted = boost(dataset, &bins, &loss, &p.plan(dataset.height()), weights.as_deref())?;
    Ok(Fitted { boosted, names: bins.names, categories, classes: Vec::new(), dtype: None })
}

/// Fits a classifier of settings `p` on `dataset`, each row with its weight in `weights`
/// (`None`: once) times its class weight.
pub(crate) fn fit_classifier<L: Library>(
    p: &L,
    dataset: &Dataset,
    target: &str,
    weights: Option<&[f32]>,
) -> Result<Fitted, ModelError> {
    let (classes, codes): (Vec<String>, Series) = label_target(dataset, target)?;
    let y: Vec<u32> = column(&codes).iter().map(|code| *code as u32).collect();
    let (bins, categories): (Bins, Vec<Option<Categories>>) = binned(dataset, target, p)?;
    // A class weight times the caller's weight of the row, in f64 like scikit-learn's.
    let weights: Option<Vec<f64>> = match p.class_weight() {
        Some(class_weight) => {
            let of_classes: Vec<f64> = class_weight.exact(&classes, &y, None)?;
            let caller = |row: usize| weights.map_or(1.0, |w| f64::from(w[row]));
            Some(y.iter().enumerate().map(|(row, c)| of_classes[*c as usize] * caller(row)).collect())
        }
        None => widened(weights),
    };
    let loss: Loss = p.classification(&y, classes.len())?;
    let plan: Plan = p.classifier_plan(dataset.height(), classes.len());
    let boosted: Boosted = boost(dataset, &bins, &loss, &plan, weights.as_deref())?;
    let dtype: Option<DataType> = Some(dataset.column(target)?.dtype().clone());
    Ok(Fitted { boosted, names: bins.names, categories, classes, dtype })
}

/// The share of each class for each row of `scores` (one score per row for two classes, else
/// one per class), row after row.
pub(crate) fn class_shares(scores: &[f64], k: usize, outputs: usize) -> Vec<f64> {
    match outputs {
        // One score per row: the log-odds of the second class.
        1 => (scores.iter())
            .flat_map(|score| {
                let p: f64 = logistic(*score).0;
                [1.0 - p, p]
            })
            .collect(),
        _ => {
            let mut shares: Vec<f64> = vec![0.0; scores.len()];
            for (z, out) in scores.chunks(k).zip(shares.chunks_mut(k)) {
                softmax(z, out);
            }
            shares
        }
    }
}

/// The fields, `Model` impl and `Regressor` or `Classifier` impl of a boosted model whose
/// settings are `$params` (a [`Library`]), so that each library's model is a type of its
/// own with its own setters, over one engine.
macro_rules! boosted_model {
    (@fields $(#[$meta:meta])* $name:ident, $params:ty, $($extra:tt)*) => {
        $(#[$meta])*
        #[derive(Debug, Clone)]
        pub struct $name {
            params: $params,
            $($extra)*
            /// The feature columns `fit` saw.
            pub feature_names_in_: Vec<String>,
            /// How many features `fit` saw.
            pub n_features_in_: usize,
            /// The rounds `fit` took: all of them, or fewer with early stopping.
            pub n_iter_: usize,
            /// With early stopping: minus the loss of the training rows before the first
            /// round and after each one, like scikit-learn's.
            pub train_score_: Vec<f64>,
            /// The same for the rows held out.
            pub validation_score_: Vec<f64>,
            /// How much of the loss the splits on each feature took off, as shares that sum
            /// to 1.
            pub feature_importances_: Vec<f64>,
            /// The trees: one per round, or one per class and round, round after round.
            pub estimators_: Vec<Tree>,
            /// Whether each feature is categorical.
            pub is_categorical_: Vec<bool>,
            categories: Vec<Option<Categories>>,
            baseline: Vec<f64>,
            target: Option<String>,
            run: Option<CachedRun>,
        }
    };
    (@common $name:ident) => {
        impl $name {
            fn take(&mut self, fitted: Fitted, target: &str) {
                let Fitted { boosted, names, categories, .. } = fitted;
                self.n_features_in_ = names.len();
                self.feature_names_in_ = names;
                (self.estimators_, self.baseline) = (boosted.trees, boosted.baseline);
                (self.train_score_, self.validation_score_) = (boosted.train_score, boosted.validation_score);
                (self.n_iter_, self.feature_importances_) = (boosted.n_iter, boosted.importances);
                self.is_categorical_ = categories.iter().map(Option::is_some).collect();
                self.categories = categories;
                self.target = Some(target.into());
            }

            fn take_learned(&mut self, l: Learned, trees: Vec<Tree>, target: &str) {
                self.n_features_in_ = l.feature_names_in_.len();
                (self.feature_names_in_, self.feature_importances_) = (l.feature_names_in_, l.feature_importances_);
                (self.train_score_, self.validation_score_) = (l.train_score_, l.validation_score_);
                (self.estimators_, self.baseline, self.n_iter_) = (trees, l.baseline_, l.n_iter_);
                self.is_categorical_ = l.categories_.iter().map(Option::is_some).collect();
                self.categories = l.categories_;
                self.target = Some(target.into());
            }

            /// The loss before the first round and after each round; empty without early
            /// stopping. scikit-learn's `-train_score_`.
            fn curve(scores: &[f64]) -> Option<Vec<f64>> {
                (!scores.is_empty()).then(|| scores.iter().map(|score| -score).collect())
            }

            /// The scores of each row: the baseline plus what each tree's leaf adds.
            fn scores(&self, dataset: &Dataset) -> Result<Vec<f64>, ModelError> {
                if self.target.is_none() {
                    return Err(ModelError::NotFitted { model: stringify!($name) });
                }
                let names: &[String] = &self.feature_names_in_;
                scores_of(&self.estimators_, &self.baseline, dataset, names, &self.categories)
            }
        }
    };
    (@model $classifier:literal) => {
        fn try_train(&mut self, dataset: &Dataset, target: &str) -> Result<(), ModelError> {
            self.train(dataset, target, None)
        }

        fn try_train_weighted(
            &mut self,
            dataset: &Dataset,
            target: &str,
            weights: &[f32],
        ) -> Option<Result<(), ModelError>> {
            Some(self.train(dataset, target, Some(weights)))
        }

        fn params(&self) -> Value {
            serde_json::to_value(&self.params).unwrap_or_default()
        }

        fn learned(&self) -> Value {
            let (classes_, dtype): (Vec<String>, DataType) = self.labels();
            serde_json::to_value(Learned {
                classes_,
                dtype,
                feature_names_in_: self.feature_names_in_.clone(),
                feature_importances_: self.feature_importances_.clone(),
                n_iter_: self.n_iter_,
                train_score_: self.train_score_.clone(),
                validation_score_: self.validation_score_.clone(),
                baseline_: self.baseline.clone(),
                categories_: self.categories.clone(),
            })
            .unwrap_or_default()
        }

        fn write_weights(&self, out: &mut dyn Write) -> io::Result<()> {
            self.estimators_.iter().try_for_each(|tree| tree.write(out))
        }

        fn try_set_params(&mut self, params: Value) -> Result<(), ModelError> {
            self.params = serde_json::from_value(params)?;
            Ok(())
        }

        fn is_classifier(&self) -> bool {
            $classifier
        }

        fn run(&self) -> Option<&CachedRun> {
            self.run.as_ref()
        }

        fn set_run(&mut self, run: Option<CachedRun>) {
            self.run = run;
        }

        fn loss_curve(&self) -> Option<Vec<f64>> {
            Self::curve(&self.train_score_)
        }

        fn validation_curve(&self) -> Option<Vec<f64>> {
            Self::curve(&self.validation_score_)
        }
    };
    (regressor $(#[$meta:meta])* pub struct $name:ident($params:ty = $default:expr);) => {
        boosted_model!(@fields $(#[$meta])* $name, $params,);

        impl Default for $name {
            fn default() -> Self {
                Self {
                    params: $default,
                    feature_names_in_: Vec::new(),
                    n_features_in_: 0,
                    n_iter_: 0,
                    train_score_: Vec::new(),
                    validation_score_: Vec::new(),
                    feature_importances_: Vec::new(),
                    estimators_: Vec::new(),
                    is_categorical_: Vec::new(),
                    categories: Vec::new(),
                    baseline: Vec::new(),
                    target: None,
                    run: None,
                }
            }
        }

        boosted_model!(@common $name);

        impl $name {
            fn train(&mut self, dataset: &Dataset, target: &str, weights: Option<&[f32]>) -> Result<(), ModelError> {
                let fitted: Fitted = fit_regressor(&self.params, dataset, target, weights)?;
                self.take(fitted, target);
                Ok(())
            }

            /// No classes; the target is a number.
            fn labels(&self) -> (Vec<String>, DataType) {
                (Vec::new(), DataType::Float64)
            }
        }

        impl Model for $name {
            fn target(&self) -> Option<&str> {
                self.target.as_deref()
            }

            boosted_model!(@model false);

            fn try_evaluate(&self, dataset: &Dataset, target: &str, scoring: &str) -> Result<f64, ModelError> {
                crate::models::evaluate_regressor(self, dataset, target, scoring)
            }

            fn try_predict_method(&self, dataset: &Dataset, method: &str) -> Result<Dataset, ModelError> {
                crate::models::predict_regressor(self, dataset, method)
            }

            fn try_load(&mut self, learned: Value, weights: &[u8], target: &str) -> Result<(), ModelError> {
                let (l, trees): (Learned, Vec<Tree>) = Learned::checked(learned, weights)?;
                if !l.classes_.is_empty() {
                    return Err(bad_learned("a regressor with classes".into()));
                }
                self.take_learned(l, trees, target);
                Ok(())
            }
        }

        impl Regressor for $name {
            /// The baseline plus what each tree's leaf adds; exp of that with a log link.
            fn try_predict(&self, dataset: &Dataset) -> Result<Series, ModelError> {
                let mut predicted: Vec<f64> = self.scores(dataset)?;
                if self.params.log_link() {
                    predicted.iter_mut().for_each(|score| *score = score.exp());
                }
                Ok(Series::new(self.target.as_deref().unwrap_or_default().into(), predicted))
            }
        }
    };
    (classifier $(#[$meta:meta])* pub struct $name:ident($params:ty = $default:expr);) => {
        boosted_model!(@fields $(#[$meta])* $name, $params,
            /// The classes `fit` saw, sorted: the order of `predict_proba`'s columns.
            pub classes_: Vec<String>,
            dtype: Option<DataType>,
        );

        impl Default for $name {
            fn default() -> Self {
                Self {
                    params: $default,
                    classes_: Vec::new(),
                    dtype: None,
                    feature_names_in_: Vec::new(),
                    n_features_in_: 0,
                    n_iter_: 0,
                    train_score_: Vec::new(),
                    validation_score_: Vec::new(),
                    feature_importances_: Vec::new(),
                    estimators_: Vec::new(),
                    is_categorical_: Vec::new(),
                    categories: Vec::new(),
                    baseline: Vec::new(),
                    target: None,
                    run: None,
                }
            }
        }

        boosted_model!(@common $name);

        impl $name {
            fn train(&mut self, dataset: &Dataset, target: &str, weights: Option<&[f32]>) -> Result<(), ModelError> {
                let mut fitted: Fitted = fit_classifier(&self.params, dataset, target, weights)?;
                (self.classes_, self.dtype) = (std::mem::take(&mut fitted.classes), fitted.dtype.take());
                self.take(fitted, target);
                Ok(())
            }

            /// The share of each class for each row, row after row.
            fn shares(&self, dataset: &Dataset) -> Result<Vec<f64>, ModelError> {
                Ok(class_shares(&self.scores(dataset)?, self.classes_.len(), self.baseline.len()))
            }

            /// The classes and the target's type, as the cache stores them.
            fn labels(&self) -> (Vec<String>, DataType) {
                (self.classes_.clone(), self.dtype.clone().unwrap_or(DataType::String))
            }
        }

        impl Model for $name {
            fn target(&self) -> Option<&str> {
                self.target.as_deref()
            }

            boosted_model!(@model true);

            fn try_evaluate(&self, dataset: &Dataset, target: &str, scoring: &str) -> Result<f64, ModelError> {
                crate::models::evaluate_classifier(self, dataset, target, scoring)
            }

            fn try_predict_method(&self, dataset: &Dataset, method: &str) -> Result<Dataset, ModelError> {
                crate::models::predict_classifier(self, dataset, method)
            }

            fn try_load(&mut self, learned: Value, weights: &[u8], target: &str) -> Result<(), ModelError> {
                let (l, trees): (Learned, Vec<Tree>) = Learned::checked(learned, weights)?;
                if l.classes_.len() < 2 {
                    return Err(bad_learned("a classifier with no classes".into()));
                }
                (self.classes_, self.dtype) = (l.classes_.clone(), Some(l.dtype.clone()));
                self.take_learned(l, trees, target);
                Ok(())
            }
        }

        impl Classifier for $name {
            /// The class with the largest share; the first one on a tie.
            fn try_predict(&self, dataset: &Dataset) -> Result<Series, ModelError> {
                pick_classes(
                    &self.shares(dataset)?,
                    &self.classes_,
                    self.target.as_deref().unwrap_or_default(),
                    self.dtype.as_ref().unwrap_or(&DataType::String),
                )
            }

            /// The sigmoid (two classes) or the softmax (more) of the scores.
            fn try_predict_proba(&self, dataset: &Dataset) -> Result<Dataset, ModelError> {
                share_columns(dataset, &self.shares(dataset)?, &self.classes_, 1)
            }
        }
    };
}
pub(crate) use boosted_model;

macro_rules! boosting_settings {
    ($model:ident) => {
        impl $model {
            /// How much of each tree's step is taken, above 0. Default 0.1. A smaller rate
            /// needs more rounds and often scores better.
            #[track_caller]
            pub fn learning_rate(mut self, learning_rate: f64) -> Self {
                let ok: bool = learning_rate.is_finite() && learning_rate > 0.0;
                check(ok, "learning_rate", learning_rate, "a finite number above 0");
                self.params.learning_rate = learning_rate;
                self
            }

            /// The most rounds. Default 100. With early stopping the model may take fewer
            /// (`n_iter_`).
            #[track_caller]
            pub fn max_iter(mut self, max_iter: usize) -> Self {
                check(max_iter > 0, "max_iter", max_iter, "1 or more");
                self.params.max_iter = max_iter;
                self
            }

            /// The most leaves of each tree. Default 31. 0 is no limit, scikit-learn's
            /// `None`.
            #[track_caller]
            pub fn max_leaf_nodes(mut self, max_leaf_nodes: usize) -> Self {
                let ok: bool = max_leaf_nodes != 1;
                check(ok, "max_leaf_nodes", max_leaf_nodes, "2 or more, or 0 for no limit");
                self.params.max_leaf_nodes = (max_leaf_nodes > 0).then_some(max_leaf_nodes);
                self
            }

            /// The most splits between the root and a leaf of each tree. Default: no limit.
            #[track_caller]
            pub fn max_depth(mut self, max_depth: usize) -> Self {
                check(max_depth > 0, "max_depth", max_depth, "1 or more");
                self.params.max_depth = Some(max_depth);
                self
            }

            /// The fewest rows in a leaf. Default 20. On fewer than a few hundred rows, a
            /// smaller number lets the trees split at all.
            #[track_caller]
            pub fn min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
                let name: &str = "min_samples_leaf";
                check(min_samples_leaf >= 1, name, min_samples_leaf, "1 or more");
                self.params.min_samples_leaf = min_samples_leaf;
                self
            }

            /// The L2 penalty on the values of the leaves, 0 or more. Default 0.
            #[track_caller]
            pub fn l2_regularization(mut self, l2_regularization: f64) -> Self {
                check(
                    l2_regularization.is_finite() && l2_regularization >= 0.0,
                    "l2_regularization",
                    l2_regularization,
                    "a finite number of 0 or more",
                );
                self.params.l2_regularization = l2_regularization;
                self
            }

            /// The L1 penalty on the values of the leaves, 0 or more. Default 0: a leaf's Σg
            /// moves this much towards 0 first, so small leaves predict 0. LightGBM's and
            /// XGBoost's `reg_alpha`; scikit-learn has none.
            #[track_caller]
            pub fn l1_regularization(mut self, l1_regularization: f64) -> Self {
                check(
                    l1_regularization.is_finite() && l1_regularization >= 0.0,
                    "l1_regularization",
                    l1_regularization,
                    "a finite number of 0 or more",
                );
                self.params.l1_regularization = l1_regularization;
                self
            }

            /// The least a split must take off the loss to be made, 0 or more. Default 0.
            /// LightGBM's `min_split_gain`; XGBoost's `gamma` is half of it. scikit-learn has
            /// none.
            #[track_caller]
            pub fn min_split_gain(mut self, min_split_gain: f64) -> Self {
                check(
                    min_split_gain.is_finite() && min_split_gain >= 0.0,
                    "min_split_gain",
                    min_split_gain,
                    "a finite number of 0 or more",
                );
                self.params.min_split_gain = min_split_gain;
                self
            }

            /// The share of the features each node finds its split among, drawn anew for
            /// each node (rounded up, at least one). Default 1.0: every feature. scikit-learn's
            /// `max_features`, LightGBM's and XGBoost's `colsample_bynode`.
            #[track_caller]
            pub fn max_features(mut self, max_features: f64) -> Self {
                let ok: bool = max_features > 0.0 && max_features <= 1.0;
                check(ok, "max_features", max_features, "a share above 0.0 up to 1.0");
                self.params.max_features = max_features;
                self
            }

            /// The share of the training rows each round's trees grow on, drawn anew each
            /// round. Default 1.0: every row. The other rows still get the round's trees, so
            /// the gradients stay exact. LightGBM's and XGBoost's `subsample`; scikit-learn's
            /// histogram boosting has none.
            #[track_caller]
            pub fn subsample(mut self, subsample: f64) -> Self {
                let ok: bool = subsample > 0.0 && subsample <= 1.0;
                check(ok, "subsample", subsample, "a share above 0.0 up to 1.0");
                self.params.subsample = subsample;
                self
            }

            /// The features to split as categories: a split then sends a set of categories
            /// left, not the values below a threshold. Default: every text column
            /// (scikit-learn's `"from_dtype"`); `&[]` none. A categorical feature has at most
            /// `max_bins` categories, numbers or text.
            pub fn categorical_features(mut self, names: &[&str]) -> Self {
                self.params.categorical_features = Some(names.iter().map(|n| n.to_string()).collect());
                self
            }

            /// The most buckets per feature, 2 to 255. Default 255, like scikit-learn.
            #[track_caller]
            pub fn max_bins(mut self, max_bins: usize) -> Self {
                check((2..=255).contains(&max_bins), "max_bins", max_bins, "2 to 255");
                self.params.max_bins = max_bins;
                self
            }

            /// `true` holds `validation_fraction` of the rows out of training and stops
            /// when the loss on them has not got better for `n_iter_no_change` rounds;
            /// `false` runs every round on every row. Default: `true` above 10,000 rows,
            /// scikit-learn's `"auto"`.
            pub fn early_stopping(mut self, early_stopping: bool) -> Self {
                self.params.early_stopping = Some(early_stopping);
                self
            }

            /// The share of the rows held out for early stopping, above 0 and below 1.
            /// Default 0.1. A count (scikit-learn's int) is not taken.
            #[track_caller]
            pub fn validation_fraction(mut self, validation_fraction: f64) -> Self {
                let ok: bool = validation_fraction > 0.0 && validation_fraction < 1.0;
                let name: &str = "validation_fraction";
                check(ok, name, validation_fraction, "a share above 0 and below 1");
                self.params.validation_fraction = validation_fraction;
                self
            }

            /// Early stopping waits this many rounds for a better loss. Default 10.
            #[track_caller]
            pub fn n_iter_no_change(mut self, n_iter_no_change: usize) -> Self {
                let name: &str = "n_iter_no_change";
                check(n_iter_no_change >= 1, name, n_iter_no_change, "1 or more");
                self.params.n_iter_no_change = n_iter_no_change;
                self
            }

            /// A loss counts as better when it is lower by more than this. Default 1e-7.
            #[track_caller]
            pub fn tol(mut self, tol: f64) -> Self {
                check(tol.is_finite() && tol >= 0.0, "tol", tol, "a finite number of 0 or more");
                self.params.tol = tol;
                self
            }

            /// The seed of the rows held out for early stopping, of the rows each round draws
            /// (`subsample`) and of the features each node draws (`max_features`). Default 42.
            pub fn random_state(mut self, random_state: u64) -> Self {
                self.params.random_state = random_state;
                self
            }

        }
    };
}

boosted_model! {
    classifier
    /// Predicts a class with a sum of small trees, each one fitted to the errors of the trees
    /// before it: histogram gradient boosting, like scikit-learn's
    /// `HistGradientBoostingClassifier` and LightGBM.
    ///
    /// Each of the `max_iter` rounds grows one tree on the gradients of the log loss (one tree
    /// per class with more than two classes), leaf by leaf up to `max_leaf_nodes`, and adds
    /// `learning_rate` times its leaf values to the scores. `predict_proba` is the sigmoid or
    /// the softmax of the scores. The settings and their defaults are scikit-learn's.
    ///
    /// Above 10,000 rows it stops early by default: it holds 10% of the rows out, and stops when
    /// their loss has not got better for 10 rounds. `n_iter_` is the rounds it took, and the
    /// journal draws the two loss curves.
    ///
    /// The features are numbers or booleans in at most `max_bins` buckets, with nulls and NaN as
    /// they are, and text columns as categories (scikit-learn's `categorical_features=
    /// "from_dtype"`): a split then sends a set of categories left, found as LightGBM and
    /// scikit-learn find it, and a category `fit` never saw counts as missing. With at most 255
    /// distinct values per feature the buckets are scikit-learn's own, and so are the trees: the
    /// predictions then agree to about 1e-6.
    ///
    /// Beyond scikit-learn, from LightGBM and XGBoost: `subsample` (a share of the rows for each
    /// round), `l1_regularization` and `min_split_gain`. `max_features` draws each node's
    /// features as scikit-learn does, from a seed of rusmlai's own.
    ///
    /// scikit-learn settings with no setter: `verbose`, `scoring` (early stopping is on the
    /// loss). Not yet: `monotonic_cst`, `interaction_cst`, `warm_start`, and
    /// `validation_fraction` as a count.
    ///
    /// ```
    /// use rusmlai::prelude::*;
    /// let ds = Dataset::new(
    ///     df!(
    ///         "hours" => [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5],
    ///         "sleep" => [4.0, 5.0, 4.5, 5.5, 4.0, 5.0, 7.0, 8.0, 7.5, 8.5, 7.0, 8.0],
    ///         "passed" => ["no", "no", "no", "no", "no", "no", "yes", "yes", "yes", "yes", "yes", "yes"],
    ///     )
    ///     .unwrap(),
    ///     "",
    /// );
    /// // 12 rows: a leaf of 20 rows, the default, could never split.
    /// let mut model = GradientBoostingClassifier::new().min_samples_leaf(1);
    /// model.fit(&ds, "passed");
    /// assert_eq!(model.n_iter_, 100);
    /// assert_eq!(model.classes_, ["no", "yes"]);
    /// assert_eq!(model.score(&ds, "passed"), 1.0);
    /// let proba = model.predict_proba(&ds);
    /// assert!(proba.to_list("yes")[11] > 0.99);
    ///
    /// // A text column is a categorical feature: a split sends a set of its categories left.
    /// let course: Vec<&str> = ["math", "art", "bio"].iter().flat_map(|c| [*c; 12]).collect();
    /// let passed: Vec<&str> = course.iter().map(|c| if *c == "math" { "yes" } else { "no" }).collect();
    /// let courses = Dataset::new(df!("course" => course, "passed" => passed).unwrap(), "");
    /// let mut model = GradientBoostingClassifier::new().min_samples_leaf(1);
    /// model.fit(&courses, "passed");
    /// assert_eq!(model.is_categorical_, [true]);
    /// assert_eq!(model.score(&courses, "passed"), 1.0);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// from sklearn.ensemble import HistGradientBoostingClassifier
    /// ds = pd.DataFrame({
    ///     "hours": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5],
    ///     "sleep": [4.0, 5.0, 4.5, 5.5, 4.0, 5.0, 7.0, 8.0, 7.5, 8.5, 7.0, 8.0],
    ///     "passed": ["no", "no", "no", "no", "no", "no", "yes", "yes", "yes", "yes", "yes", "yes"],
    /// })
    /// x, y = ds[["hours", "sleep"]], ds["passed"]
    /// model = HistGradientBoostingClassifier(min_samples_leaf=1).fit(x, y)
    /// assert model.n_iter_ == 100
    /// assert model.classes_.tolist() == ["no", "yes"]
    /// assert model.score(x, y) == 1.0
    /// proba = model.predict_proba(x)
    /// assert proba[11, 1] > 0.99
    ///
    /// course = [c for c in ["math", "art", "bio"] for _ in range(12)]
    /// passed = ["yes" if c == "math" else "no" for c in course]
    /// courses = pd.DataFrame({"course": pd.Categorical(course), "passed": passed})
    /// model = HistGradientBoostingClassifier(min_samples_leaf=1).fit(courses[["course"]], courses["passed"])
    /// assert model.is_categorical_.tolist() == [True]
    /// assert model.score(courses[["course"]], courses["passed"]) == 1.0
    /// ```
    pub struct GradientBoostingClassifier(Params = Params::new("log_loss"));
}

boosting_settings!(GradientBoostingClassifier);

impl GradientBoostingClassifier {
    /// 100 rounds of trees of 31 leaves on the log loss, with a learning rate of 0.1.
    pub fn new() -> Self {
        Self::default()
    }

    /// How much each class counts: `"balanced"`, so that a rare class counts as much as a
    /// frequent one, or your own weights, `[("yes", 5.0), ("no", 1.0)]`. A class weight
    /// multiplies each row's gradient and hessian, as in scikit-learn. Default: every row
    /// counts once.
    #[track_caller]
    pub fn class_weight(mut self, class_weight: impl Into<ClassWeight>) -> Self {
        self.params.class_weight = Some(class_weight.into().checked(false));
        self
    }
}

boosted_model! {
    regressor
    /// Predicts a number with a sum of small trees, each one fitted to the errors of the trees
    /// before it: histogram gradient boosting, like scikit-learn's
    /// `HistGradientBoostingRegressor` and LightGBM.
    ///
    /// Each of the `max_iter` rounds grows one tree on the errors that are left (the gradients
    /// of the loss), leaf by leaf up to `max_leaf_nodes`, and adds `learning_rate` times its leaf
    /// values to the predictions, which start at the best constant: the mean for the squared
    /// error. The losses are scikit-learn's: `"squared_error"`, `"absolute_error"` (the median),
    /// `"quantile"`, `"poisson"` and `"gamma"` (`loss`). For the absolute error and a quantile,
    /// each leaf takes that quantile of its rows' errors after its tree grows, as in
    /// scikit-learn.
    ///
    /// The features (text as categories), early stopping, `n_iter_`, the settings from LightGBM
    /// and XGBoost, and the agreement with scikit-learn are those of
    /// [`GradientBoostingClassifier`].
    ///
    /// scikit-learn settings with no setter: `verbose`, `scoring`. Not yet: `monotonic_cst`,
    /// `interaction_cst`, `warm_start`.
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
    /// // 8 rows: a leaf of 20 rows, the default, could never split.
    /// let mut model = GradientBoostingRegressor::new().min_samples_leaf(1);
    /// model.fit(&ds, "price");
    /// assert_eq!(model.n_iter_, 100);
    /// assert!(model.score(&ds, "price") > 0.999);
    ///
    /// // Fewer rounds with a higher learning rate: each tree takes a larger step.
    /// let mut fast = GradientBoostingRegressor::new().min_samples_leaf(1).max_iter(10).learning_rate(0.5);
    /// fast.fit(&ds, "price");
    /// assert!(fast.score(&ds, "price") > 0.99);
    ///
    /// // The median instead of the mean, and a Poisson loss for counts.
    /// let mut median = GradientBoostingRegressor::new().loss("absolute_error").min_samples_leaf(1);
    /// median.fit(&ds, "price");
    /// assert!(median.score(&ds, "price") > 0.999);
    /// let mut counts = GradientBoostingRegressor::new().loss("poisson").min_samples_leaf(1);
    /// counts.fit(&ds, "price");
    /// assert!(counts.score(&ds, "price") > 0.999);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// from sklearn.ensemble import HistGradientBoostingRegressor
    /// ds = pd.DataFrame({
    ///     "area": [50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0],
    ///     "rooms": [1, 2, 2, 3, 3, 3, 4, 4],
    ///     "price": [150.0, 180.0, 200.0, 260.0, 270.0, 310.0, 400.0, 420.0],
    /// })
    /// x, y = ds[["area", "rooms"]], ds["price"]
    /// model = HistGradientBoostingRegressor(min_samples_leaf=1).fit(x, y)
    /// assert model.n_iter_ == 100
    /// assert model.score(x, y) > 0.999
    ///
    /// fast = HistGradientBoostingRegressor(min_samples_leaf=1, max_iter=10, learning_rate=0.5).fit(x, y)
    /// assert fast.score(x, y) > 0.99
    ///
    /// median = HistGradientBoostingRegressor(loss="absolute_error", min_samples_leaf=1).fit(x, y)
    /// assert median.score(x, y) > 0.999
    /// counts = HistGradientBoostingRegressor(loss="poisson", min_samples_leaf=1).fit(x, y)
    /// assert counts.score(x, y) > 0.999
    /// ```
    pub struct GradientBoostingRegressor(Params = Params::new("squared_error"));
}

boosting_settings!(GradientBoostingRegressor);

impl GradientBoostingRegressor {
    /// 100 rounds of trees of 31 leaves on the squared error, with a learning rate of 0.1.
    pub fn new() -> Self {
        Self::default()
    }

    /// What the trees lower, scikit-learn's names: `"squared_error"` (the default, the
    /// mean), `"absolute_error"` (the median), `"quantile"` (the quantile of `quantile`),
    /// `"poisson"` (counts: a target of 0 or more) or `"gamma"` (a target above 0). The last
    /// two predict through a log link, so a prediction is never below 0.
    #[track_caller]
    pub fn loss(mut self, loss: &str) -> Self {
        const LOSSES: [&str; 5] = ["squared_error", "absolute_error", "quantile", "poisson", "gamma"];
        let expected: &str = "\"squared_error\", \"absolute_error\", \"quantile\", \"poisson\" or \"gamma\"";
        check(LOSSES.contains(&loss), "loss", loss, expected);
        self.params.loss = loss.into();
        self
    }

    /// The quantile the `"quantile"` loss predicts, 0 to 1: 0.5 is the median, 0.9 a value
    /// that 90% of the targets are below.
    #[track_caller]
    pub fn quantile(mut self, quantile: f64) -> Self {
        check((0.0..=1.0).contains(&quantile), "quantile", quantile, "a number from 0 to 1");
        self.params.quantile = Some(quantile);
        self
    }
}

#[cfg(test)]
mod tests {
    use polars::prelude::{DataFrame, IntoLazy, lit};

    use super::*;
    use crate::{models::tree::tests::table, utils::col};

    fn assert_close(got: &[f64], want: &[f64], eps: f64) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < eps, "{got:?}, want {want:?}");
        }
    }

    fn first_values(predicted: Series) -> Vec<f64> {
        predicted.f64().unwrap().into_no_null_iter().collect()
    }

    fn leaves(trees: &[Tree]) -> usize {
        trees.iter().map(Tree::n_leaves).sum()
    }

    /// The 400 rows of `table()` with `hit`, 1 where the class is 1.
    fn binary() -> Dataset {
        let frame: DataFrame = (table().dataframe().clone().lazy())
            .with_column(col("class").eq(lit(1)).cast(DataType::Int64).alias("hit"))
            .collect()
            .unwrap();
        Dataset::new(frame, "").drop(&["class", "y"])
    }

    #[test]
    fn the_regressor_matches_scikit_learn() {
        // scikit-learn 1.9.1 (analysis/2026-11-17/boosting/parity.py).
        let ds: Dataset = table().drop(&["class"]);
        let first: Dataset = Dataset::new(ds.dataframe().slice(0, 4), "");
        let mut model = GradientBoostingRegressor::new().max_iter(20).min_samples_leaf(5);
        model.fit(&ds, "y");
        assert_eq!((model.n_iter_, leaves(&model.estimators_)), (20, 620));
        assert!(model.train_score_.is_empty(), "no early stopping on 400 rows");
        assert!((model.score(&ds, "y") - 0.9745349219620755).abs() < 1e-9);
        let want = [3.7246592603280133, 7.803292199923989, 20.433983080305193, 4.144797127998227];
        assert_close(&first_values(model.predict(&first)), &want, 1e-6);

        let mut deep = GradientBoostingRegressor::new()
            .max_iter(15)
            .min_samples_leaf(5)
            .max_depth(3)
            .max_leaf_nodes(0)
            .l2_regularization(1.0)
            .learning_rate(0.3);
        deep.fit(&ds, "y");
        assert_eq!(leaves(&deep.estimators_), 116);
        assert!((deep.score(&ds, "y") - 0.9842336133268225).abs() < 1e-9);
        let want = [2.6100593583034923, 6.648841615502295, 20.21722606262653, 2.885189680991498];
        assert_close(&first_values(deep.predict(&first)), &want, 1e-6);
    }

    /// In the first round every row of a class has the same gradient and every hessian is
    /// the same, so two splits that leave the same classes on each side tie in theory (a
    /// 91-row node: `b <= 4.5` and `c <= 4.5`). Their gains differ in the last bit by the
    /// order of the operations, so the engine computes the gain as scikit-learn does, and the
    /// trees are scikit-learn's.
    #[test]
    fn the_classifier_matches_scikit_learn_on_two_and_three_classes() {
        let ds: Dataset = binary();
        let mut model = GradientBoostingClassifier::new().max_iter(20).min_samples_leaf(5);
        model.fit(&ds, "hit");
        assert_eq!((model.classes_.clone(), leaves(&model.estimators_)), (vec!["0".into(), "1".into()], 620));
        assert_close(&model.baseline, &[-0.41589273127042337], 1e-12);
        assert_eq!(model.score(&ds, "hit"), 0.995);
        // The first tree up to the tie is scikit-learn's.
        let first: &Tree = &model.estimators_[0];
        let top = (first.feature()[..3].to_vec(), first.threshold()[..3].to_vec());
        assert_eq!(top, (vec![0, 0, 1], vec![77.5, 15.5, 8.5]));
        assert_close(&first.value()[3..4], &[-0.16597511235737983], 1e-9);
        let proba: Dataset = model.predict_proba(&Dataset::new(ds.dataframe().slice(0, 2), ""));
        assert_close(&proba.to_list("1"), &[0.05232012755361374, 0.05646410243408288], 1e-6);

        let ds: Dataset = table().drop(&["y"]);
        let mut model = GradientBoostingClassifier::new().max_iter(10).min_samples_leaf(5);
        model.fit(&ds, "class");
        assert_eq!((model.estimators_.len(), leaves(&model.estimators_)), (30, 930));
        let base = [-0.1468981306299441, 0.185807623195792, -0.038909492565848205];
        assert_close(&model.baseline, &base, 1e-12);
        assert_eq!(model.score(&ds, "class"), 0.9975);
        let proba: Dataset = model.predict_proba(&Dataset::new(ds.dataframe().slice(0, 2), ""));
        let want = [[0.7226868338755634, 0.04166549297302783, 0.2356476731514087],
            [0.8830528152447518, 0.06483992032299202, 0.052107264432256294]];
        for (c, class) in ["0", "1", "2"].iter().enumerate() {
            assert_close(&proba.to_list(class), &[want[0][c], want[1][c]], 1e-6);
        }
        assert_eq!(model.predict(&ds).dtype(), &DataType::Int64);
    }

    #[test]
    fn early_stopping_holds_rows_out_and_stops_when_they_stop_getting_better() {
        let ds: Dataset = table().drop(&["class"]);
        let mut model = GradientBoostingRegressor::new()
            .min_samples_leaf(5)
            .max_iter(500)
            .learning_rate(0.5)
            .early_stopping(true)
            .n_iter_no_change(5);
        model.fit(&ds, "y");
        assert!(model.n_iter_ < 500, "{} rounds", model.n_iter_);
        assert_eq!(model.estimators_.len(), model.n_iter_);
        assert_eq!(model.train_score_.len(), model.n_iter_ + 1);
        assert_eq!(model.validation_score_.len(), model.n_iter_ + 1);
        // The last 5 rounds did not beat the round before them.
        let v: &[f64] = &model.validation_score_;
        let reference: f64 = v[v.len() - 6] + 1e-7;
        assert!(v[v.len() - 5..].iter().all(|score| *score <= reference));
        // The training loss falls, and the journal gets both curves.
        assert!(model.train_score_.last() > model.train_score_.first());
        let (train, held) = (model.loss_curve().unwrap(), model.validation_curve().unwrap());
        assert_eq!((train.len(), held.len()), (model.n_iter_ + 1, model.n_iter_ + 1));
        assert!(train[0] > 0.0 && train[0] == -model.train_score_[0]);
    }

    #[test]
    fn learned_values_round_trip_and_a_broken_record_is_refused() {
        let ds: Dataset = table().drop(&["y"]);
        let mut trained = GradientBoostingClassifier::new().max_iter(5).min_samples_leaf(5);
        trained.fit(&ds, "class");
        let mut weights: Vec<u8> = Vec::new();
        trained.write_weights(&mut weights).unwrap();
        let mut loaded: GradientBoostingClassifier = GradientBoostingClassifier::new();
        loaded.try_load(trained.learned(), &weights, "class").unwrap();
        assert_eq!(loaded.estimators_, trained.estimators_);
        assert_eq!(loaded.predict_proba(&ds).to_list("2"), trained.predict_proba(&ds).to_list("2"));
        let mut again: GradientBoostingClassifier = GradientBoostingClassifier::new();
        again.try_set_params(trained.params()).unwrap();
        assert_eq!(again.params(), trained.params());

        // One tree short of 5 rounds of 3 classes.
        let cut: usize = weights.len() - {
            let mut last: Vec<u8> = Vec::new();
            trained.estimators_[14].write(&mut last).unwrap();
            last.len()
        };
        let err: String = GradientBoostingClassifier::new()
            .try_load(trained.learned(), &weights[..cut], "class")
            .unwrap_err()
            .to_string();
        assert!(err.contains("14 trees for 5 rounds of 3 scores"), "{err}");
        // A regressor's record is not a classifier's.
        let mut regressor = GradientBoostingRegressor::new().max_iter(2).min_samples_leaf(5);
        regressor.fit(&table().drop(&["class"]), "y");
        let mut bytes: Vec<u8> = Vec::new();
        regressor.write_weights(&mut bytes).unwrap();
        assert!(GradientBoostingClassifier::new().try_load(regressor.learned(), &bytes, "class").is_err());
        assert!(matches!(
            GradientBoostingRegressor::new().try_predict(&ds),
            Err(ModelError::NotFitted { .. })
        ));
    }

    #[test]
    #[should_panic(expected = "max_leaf_nodes = 1 is not valid")]
    fn a_bad_setting_panics_at_the_setter() {
        let _ = GradientBoostingRegressor::new().max_leaf_nodes(1);
    }

    /// `table()` plus `cat`, a text feature of 7 categories with nulls, `y2`, a target that
    /// depends on `cat`, and `pos`, a target above 0: analysis/2026-11-17/boosting/parity-extras.py.
    fn wide() -> Dataset {
        let ds: Dataset = table();
        let (a, b, c, y) = (ds.to_list("a"), ds.to_list("b"), ds.to_list("c"), ds.to_list("y"));
        let class: Vec<f64> = ds.to_list("class");
        let names = ["red", "green", "blue", "cyan", "gold", "pink", "teal"];
        let effect: [f64; 7] = [0.0, 9.0, -7.0, 15.0, 4.0, -12.0, 22.0];
        let score: Vec<f64> = y.iter().map(|y| y * 4.0).collect();
        let low: f64 = score.iter().copied().fold(f64::INFINITY, f64::min);
        let cat: Vec<Option<&str>> = (0..400).map(|i| (i % 23 != 0).then(|| names[i * 19 % 7])).collect();
        let y2: Vec<f64> = (0..400).map(|i| (score[i] + effect[i * 19 % 7]) / 4.0).collect();
        let pos: Vec<f64> = score.iter().map(|s| (s - low) / 10.0 + 1.0).collect();
        let class: Vec<i64> = class.iter().map(|c| *c as i64).collect();
        let frame = polars::df!("a" => a, "b" => b, "c" => c, "cat" => cat, "class" => class, "y" => y, "y2" => y2, "pos" => pos)
            .unwrap();
        Dataset::new(frame, "")
    }

    #[test]
    fn the_losses_match_scikit_learn() {
        // scikit-learn 1.9.1 (analysis/2026-11-17/boosting/parity-extras.py).
        let ds: Dataset = wide();
        type Case<'a> = (&'a str, Option<f64>, &'a str, usize, f64, [f64; 4]);
        let cases: [Case; 4] = [
            ("absolute_error", None, "y", 571, 0.9263521804472772,
                [6.584241787442192, 7.4960270891985115, 21.979907686741264, 5.424410312793735]),
            ("quantile", Some(0.8), "y", 532, 0.7344227105077279,
                [9.986688025766737, 13.941447198525978, 21.72305541870511, 12.44379704641397]),
            ("poisson", None, "pos", 620, 0.972669580449186,
                [4.356736157150976, 6.055084964866354, 11.325754641648615, 4.818960913963777]),
            ("gamma", None, "pos", 620, 0.9691250537659726,
                [3.286288676935615, 5.88333259008265, 11.220929397117207, 4.542311287941222]),
        ];
        for (loss, q, target, want_leaves, want_score, want) in cases {
            let data: Dataset = ds.select(&["a", "b", "c", target]);
            let mut model = GradientBoostingRegressor::new().loss(loss).max_iter(20).min_samples_leaf(5);
            if let Some(q) = q {
                model = model.quantile(q);
            }
            model.fit(&data, target);
            assert_eq!(leaves(&model.estimators_), want_leaves, "{loss}");
            assert!((model.score(&data, target) - want_score).abs() < 1e-9, "{loss}: {}", model.score(&data, target));
            let first: Dataset = Dataset::new(data.dataframe().slice(0, 4), "");
            assert_close(&first_values(model.predict(&first)), &want, 1e-6);
        }
    }

    #[test]
    fn categorical_features_match_scikit_learn() {
        // A text column is categorical by default, like a pandas category in scikit-learn.
        let ds: Dataset = wide().select(&["a", "b", "c", "cat", "y2"]);
        let first: Dataset = Dataset::new(ds.dataframe().slice(0, 4), "");
        let unseen: Dataset = {
            let mut frame: DataFrame = first.dataframe().clone();
            let cat = Series::new("cat".into(), [Some("violet"), None, Some("red"), Some("teal")]);
            frame.with_column(cat.into()).unwrap();
            Dataset::new(frame, "")
        };
        let mut model = GradientBoostingRegressor::new().max_iter(20).min_samples_leaf(5);
        model.fit(&ds, "y2");
        assert_eq!(model.is_categorical_, [false, false, false, true]);
        assert_eq!(leaves(&model.estimators_), 620);
        assert!((model.score(&ds, "y2") - 0.9740450324676879).abs() < 1e-9);
        let want = [4.594002914562685, 5.684409271356001, 24.548760640070384, 5.783674515449581];
        assert_close(&first_values(model.predict(&first)), &want, 1e-6);
        // A category fit never saw is a missing value, as in scikit-learn.
        let want = [4.594002914562685, 6.181852188925231, 23.74716243856518, 6.652011747063418];
        assert_close(&first_values(model.predict(&unseen)), &want, 1e-6);

        let mut deep = GradientBoostingRegressor::new()
            .max_iter(10)
            .min_samples_leaf(5)
            .max_leaf_nodes(0)
            .max_depth(3);
        deep.fit(&ds, "y2");
        assert_eq!(leaves(&deep.estimators_), 80);
        assert!((deep.score(&ds, "y2") - 0.7656642180642288).abs() < 1e-9);
        let want = [10.271305877395386, 9.766055706805869, 22.34761330010305, 9.678221735573018];
        assert_close(&first_values(deep.predict(&unseen)), &want, 1e-6);

        // Set splits survive the cache's bytes.
        let mut bytes: Vec<u8> = Vec::new();
        model.write_weights(&mut bytes).unwrap();
        let mut back = GradientBoostingRegressor::new();
        back.try_load(model.learned(), &bytes, "y2").unwrap();
        assert_eq!(back.predict(&unseen), model.predict(&unseen));
        assert!(model.estimators_.iter().any(|tree| tree.is_categorical().contains(&1)));
    }

    #[test]
    fn a_text_column_is_refused_unless_categorical() {
        let ds: Dataset = wide().select(&["a", "cat", "y"]);
        let mut model = GradientBoostingRegressor::new().categorical_features(&[]);
        let err: String = model.try_fit(&ds, "y").unwrap_err().to_string();
        assert!(err.contains("feature 'cat' is str, not a number"), "{err}");
        let few = GradientBoostingRegressor::new().max_bins(5).min_samples_leaf(5);
        let err: String = few.clone().try_fit(&ds, "y").unwrap_err().to_string();
        assert!(err.contains("categorical feature 'cat' has 7 categories; at most max_bins (5)"), "{err}");
    }

    #[test]
    fn class_weight_matches_scikit_learn() {
        // The baseline is the weighted log-odds.
        let ds: Dataset = binary();
        let first: Dataset = Dataset::new(ds.dataframe().slice(0, 4), "");
        let cases: [(ClassWeight, f64, f64, [f64; 4]); 2] = [
            ("balanced".into(), 0.0, 0.9975, [0.06512845501216484, 0.08288683319965778, 0.2306648943064915, 0.0696603572972091]),
            ([("0", 1.0), ("1", 4.0)].into(), 0.9704016298494671, 0.9675, [0.0973839136712914, 0.09320632165427688, 0.3294871095757591, 0.09739050222971819]),
        ];
        for (weight, baseline, score, want) in cases {
            let mut model = GradientBoostingClassifier::new().class_weight(weight).max_iter(20).min_samples_leaf(5);
            model.fit(&ds, "hit");
            assert!((model.baseline[0] - baseline).abs() < 1e-12, "{:?}", model.baseline);
            assert_eq!(model.score(&ds, "hit"), score);
            assert_close(&model.predict_proba(&first).to_list("1"), &want, 1e-6);
        }
    }

    #[test]
    fn row_weights_match_scikit_learn() {
        // scikit-learn's fit(sample_weight=w): a bag's bootstrap counts reach boosting so.
        let ds: Dataset = table().drop(&["class"]);
        let w: Vec<f32> = (0..400).map(|i| (i * 7 % 4) as f32).collect();
        let mut model = GradientBoostingRegressor::new().max_iter(20).min_samples_leaf(5);
        model.try_train_weighted(&ds, "y", &w).unwrap().unwrap();
        assert!((model.baseline[0] - 16.182916666666667).abs() < 1e-12);
        assert_eq!(leaves(&model.estimators_), 620);
        let first: Dataset = Dataset::new(ds.dataframe().slice(0, 4), "");
        let want = [3.4988384628980427, 7.6380736743159074, 21.056509603535503, 3.855300997613727];
        assert_close(&first_values(model.predict(&first)), &want, 1e-6);
    }

    #[test]
    fn l1_and_min_split_gain_change_the_leaves() {
        // One round, one split: x <= 4.5 leaves rows of y 1, 2, 3, 4 left and 11 .. 14 right.
        let ds: Dataset = Dataset::new(
            polars::df!("x" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], "y" => [1.0, 2.0, 3.0, 4.0, 11.0, 12.0, 13.0, 14.0]).unwrap(),
            "",
        );
        let fit = |model: GradientBoostingRegressor| {
            let mut model = model.max_iter(1).max_leaf_nodes(2).min_samples_leaf(1).learning_rate(1.0);
            model.fit(&ds, "y");
            first_values(model.predict(&ds))
        };
        // The mean is 7.5; Σg is +20 on the left and -20 on the right, over 4 rows.
        assert_close(&fit(GradientBoostingRegressor::new()), &[2.5, 2.5, 2.5, 2.5, 12.5, 12.5, 12.5, 12.5], 1e-12);
        // L1 of 8 moves each Σg 8 towards 0: steps of 12 / 4 = 3.
        let l1 = fit(GradientBoostingRegressor::new().l1_regularization(8.0));
        assert_close(&l1, &[4.5, 4.5, 4.5, 4.5, 10.5, 10.5, 10.5, 10.5], 1e-12);
        // The split takes 2 × 20² / 4 = 200 off the loss: a minimum above that keeps one leaf.
        let flat = fit(GradientBoostingRegressor::new().min_split_gain(201.0));
        assert_close(&flat, &[7.5; 8], 1e-12);
        assert_close(&fit(GradientBoostingRegressor::new().min_split_gain(199.0)), &[2.5, 2.5, 2.5, 2.5, 12.5, 12.5, 12.5, 12.5], 1e-12);
    }

    #[test]
    fn subsample_and_max_features_draw_from_the_seed() {
        let ds: Dataset = table().drop(&["class"]);
        let fit = |model: GradientBoostingRegressor| {
            let mut model = model.max_iter(20).min_samples_leaf(5);
            model.fit(&ds, "y");
            (first_values(model.predict(&ds)), model.score(&ds, "y"))
        };
        let plain = fit(GradientBoostingRegressor::new());
        // A share of 1 draws nothing.
        assert_eq!(fit(GradientBoostingRegressor::new().subsample(1.0).max_features(1.0)), plain);
        for model in [
            GradientBoostingRegressor::new().subsample(0.5),
            GradientBoostingRegressor::new().max_features(0.5),
        ] {
            let (once, score) = fit(model.clone());
            assert_eq!(fit(model.clone()), (once.clone(), score), "the same seed draws the same");
            assert_ne!(once, plain.0);
            assert!(score > 0.9, "{score}");
            assert_ne!(fit(model.random_state(7)).0, once);
        }
    }
}
