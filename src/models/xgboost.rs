//! [`XGBRegressor`] and [`XGBClassifier`]: gradient boosting with XGBoost's names, defaults
//! and split rules, on rusmlai's histogram tree engine (`models/hist.rs`).
//!
//! The engine is the one of [`GradientBoostingRegressor`](crate::models::GradientBoostingRegressor);
//! these types only map XGBoost's settings onto it ([`Library`]), so code ported from
//! `xgboost.XGBRegressor` keeps its settings and gets XGBoost's trees: depth-wise trees of
//! depth 6, a learning rate of 0.3, `reg_lambda` 1 and `min_child_weight` 1. A row goes left
//! when its value is below the split's value, as in XGBoost, and a node without missing
//! values sends them right.

use std::io::{self, Write};

use polars::prelude::{DataType, NamedFrom, Series};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    dataset::Dataset,
    metadata::models::CachedRun,
    models::{
        Classifier, Model, ModelError, Regressor, bad_learned,
        boosting::{
            Fitted, Learned, Library, Loss, Plan, boosted_model, categorical_columns, class_shares, fit_classifier,
            fit_regressor, scores_of,
        },
        check,
        hist::{Categories, Rules, Step},
        tree::{Tree, pick_classes, share_columns},
    },
};

/// XGBoost's settings, with its names and defaults (xgboost 3.4).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct XgbParams {
    objective: String,
    n_estimators: usize,
    /// 0 is no limit.
    max_depth: usize,
    /// 0 is no limit.
    max_leaves: usize,
    grow_policy: String,
    learning_rate: f64,
    min_child_weight: f64,
    gamma: f64,
    reg_alpha: f64,
    reg_lambda: f64,
    /// `None` is XGBoost's unset: 0 for the trees, and 0.7 in `count:poisson`'s hessian.
    max_delta_step: Option<f64>,
    subsample: f64,
    colsample_bytree: f64,
    colsample_bylevel: f64,
    colsample_bynode: f64,
    /// `None`: XGBoost's estimate, the constant with the lowest loss.
    base_score: Option<f64>,
    scale_pos_weight: f64,
    random_state: u64,
    max_bin: usize,
    huber_slope: f64,
    quantile_alpha: f64,
    tweedie_variance_power: f64,
}

impl XgbParams {
    fn new(objective: &str) -> Self {
        Self {
            objective: objective.into(),
            n_estimators: 100,
            max_depth: 6,
            max_leaves: 0,
            grow_policy: "depthwise".into(),
            learning_rate: 0.3,
            min_child_weight: 1.0,
            gamma: 0.0,
            reg_alpha: 0.0,
            reg_lambda: 1.0,
            max_delta_step: None,
            subsample: 1.0,
            colsample_bytree: 1.0,
            colsample_bylevel: 1.0,
            colsample_bynode: 1.0,
            base_score: None,
            scale_pos_weight: 1.0,
            random_state: 0,
            max_bin: 256,
            huber_slope: 1.0,
            quantile_alpha: 0.5,
            tweedie_variance_power: 1.5,
        }
    }

    /// `base_score` as a score: through the objective's link.
    fn base_margin(&self) -> Option<f64> {
        let score: f64 = self.base_score?;
        Some(match self.objective.as_str() {
            "binary:logistic" => (score / (1.0 - score)).ln(),
            "count:poisson" | "reg:gamma" | "reg:tweedie" => score.ln(),
            _ => score,
        })
    }
}

impl Library for XgbParams {
    fn plan(&self, _rows: usize) -> Plan {
        let leaves: Option<usize> = (self.max_leaves > 0).then_some(self.max_leaves);
        Plan {
            rounds: self.n_estimators,
            step: Step {
                l1: self.reg_alpha,
                // XGBoost keeps a split whose gain is above 1e-6 (`kRtEps`) and at least `gamma`.
                min_gain: self.gamma.max(1e-6),
                min_hessian: self.min_child_weight,
                max_delta: self.max_delta_step.unwrap_or(0.0),
                ..Step::new(self.reg_lambda, self.learning_rate)
            },
            max_depth: (self.max_depth > 0).then_some(self.max_depth),
            // ponytail: "depthwise" with a leaf limit grows best leaf first, as "lossguide"
            // does; XGBoost then splits by depth. A depth-ordered queue if anyone needs it.
            max_leaf_nodes: leaves,
            min_samples_leaf: 1,
            max_features: 1.0,
            by_tree: self.colsample_bytree,
            by_level: self.colsample_bylevel,
            by_node: self.colsample_bynode,
            subsample: self.subsample,
            subsample_freq: 1,
            rules: Rules::Xgboost,
            seed: self.random_state,
            early_stopping: false,
            validation_fraction: 0.1,
            n_iter_no_change: 10,
            tol: 0.0,
            base_score: self.base_margin(),
            symmetric: None,
        }
    }

    fn max_bins(&self) -> usize {
        // One bucket holds the missing values.
        self.max_bin - 1
    }

    fn categorical(&self, dataset: &Dataset, target: &str) -> Result<Vec<String>, ModelError> {
        categorical_columns(dataset, target, None)
    }

    fn regression<'a>(&self, y: &'a [f64]) -> Result<Loss<'a>, ModelError> {
        let refuse = |expected: &'static str| ModelError::InvalidSetting {
            setting: "objective",
            value: self.objective.clone(),
            expected,
        };
        Ok(match self.objective.as_str() {
            "reg:squarederror" => Loss::Squared(y),
            "reg:absoluteerror" => Loss::Absolute(y),
            "reg:quantileerror" => Loss::Quantile(y, self.quantile_alpha),
            "reg:pseudohubererror" => Loss::PseudoHuber(y, self.huber_slope),
            "count:poisson" if y.iter().all(|v| *v >= 0.0) => Loss::Poisson(y, self.max_delta_step.unwrap_or(0.7)),
            "count:poisson" => return Err(refuse("\"count:poisson\" only with a target of 0 or more")),
            "reg:gamma" if y.iter().all(|v| *v > 0.0) => Loss::Gamma(y),
            "reg:gamma" => return Err(refuse("\"reg:gamma\" only with a target above 0 in every row")),
            "reg:tweedie" if y.iter().all(|v| *v >= 0.0) => Loss::Tweedie(y, self.tweedie_variance_power),
            "reg:tweedie" => return Err(refuse("\"reg:tweedie\" only with a target of 0 or more")),
            _ => return Err(refuse(REGRESSION)),
        })
    }

    fn classification<'a>(&self, y: &'a [u32], classes: usize) -> Result<Loss<'a>, ModelError> {
        match (self.objective.as_str(), classes) {
            ("binary:logistic", 2) => Ok(Loss::Binary(y, [1.0, self.scale_pos_weight])),
            // Like XGBoost's scikit-learn wrapper: more than two classes take softprob.
            ("binary:logistic" | "multi:softprob" | "multi:softmax", k) => Ok(Loss::Classes(y, k, 2.0)),
            _ => Err(ModelError::InvalidSetting {
                setting: "objective",
                value: self.objective.clone(),
                expected: CLASSIFICATION,
            }),
        }
    }

    fn log_link(&self) -> bool {
        matches!(self.objective.as_str(), "count:poisson" | "reg:gamma" | "reg:tweedie")
    }
}

const REGRESSION: &str = "\"reg:squarederror\", \"reg:absoluteerror\", \"reg:quantileerror\", \
                          \"reg:pseudohubererror\", \"count:poisson\", \"reg:gamma\" or \"reg:tweedie\"";
const CLASSIFICATION: &str = "\"binary:logistic\", \"multi:softprob\" or \"multi:softmax\"";

/// A share above 0 up to 1.
#[track_caller]
fn share(name: &'static str, value: f64) {
    check(value > 0.0 && value <= 1.0, name, value, "a share above 0.0 up to 1.0");
}

/// A finite number of 0 or more.
#[track_caller]
fn at_least_zero(name: &'static str, value: f64) {
    check(value.is_finite() && value >= 0.0, name, value, "a finite number of 0 or more");
}

macro_rules! xgb_settings {
    ($model:ident) => {
        impl $model {
            /// The rounds of trees. Default 100.
            #[track_caller]
            pub fn n_estimators(mut self, n_estimators: usize) -> Self {
                check(n_estimators > 0, "n_estimators", n_estimators, "1 or more");
                self.params.n_estimators = n_estimators;
                self
            }

            /// The most splits between the root and a leaf. Default 6; 0 is no limit.
            pub fn max_depth(mut self, max_depth: usize) -> Self {
                self.params.max_depth = max_depth;
                self
            }

            /// The most leaves of a tree, 0 for no limit (the default). With a limit, a tree
            /// splits its best leaf first.
            #[track_caller]
            pub fn max_leaves(mut self, max_leaves: usize) -> Self {
                check(max_leaves != 1, "max_leaves", max_leaves, "2 or more, or 0 for no limit");
                self.params.max_leaves = max_leaves;
                self
            }

            /// `"depthwise"` (the default) or `"lossguide"`: with `max_leaves`, split the
            /// leaf whose split takes most off first.
            #[track_caller]
            pub fn grow_policy(mut self, grow_policy: &str) -> Self {
                let ok: bool = matches!(grow_policy, "depthwise" | "lossguide");
                check(ok, "grow_policy", grow_policy, "\"depthwise\" or \"lossguide\"");
                self.params.grow_policy = grow_policy.into();
                self
            }

            /// How much of each tree's step is taken, above 0. Default 0.3 (XGBoost's `eta`).
            #[track_caller]
            pub fn learning_rate(mut self, learning_rate: f64) -> Self {
                let ok: bool = learning_rate.is_finite() && learning_rate > 0.0;
                check(ok, "learning_rate", learning_rate, "a finite number above 0");
                self.params.learning_rate = learning_rate;
                self
            }

            /// The least sum of hessians in a child. Default 1: with the squared error, one
            /// row; with the log loss, about 4 rows where the model is unsure.
            #[track_caller]
            pub fn min_child_weight(mut self, min_child_weight: f64) -> Self {
                at_least_zero("min_child_weight", min_child_weight);
                self.params.min_child_weight = min_child_weight;
                self
            }

            /// The least a split must take off the loss. Default 0 (XGBoost's
            /// `min_split_loss`).
            #[track_caller]
            pub fn gamma(mut self, gamma: f64) -> Self {
                at_least_zero("gamma", gamma);
                self.params.gamma = gamma;
                self
            }

            /// The L1 penalty on the leaf values. Default 0.
            #[track_caller]
            pub fn reg_alpha(mut self, reg_alpha: f64) -> Self {
                at_least_zero("reg_alpha", reg_alpha);
                self.params.reg_alpha = reg_alpha;
                self
            }

            /// The L2 penalty on the leaf values. Default 1.
            #[track_caller]
            pub fn reg_lambda(mut self, reg_lambda: f64) -> Self {
                at_least_zero("reg_lambda", reg_lambda);
                self.params.reg_lambda = reg_lambda;
                self
            }

            /// The most a leaf value moves before the learning rate, 0 for no limit (the
            /// default). With `count:poisson` it also replaces the 0.7 that XGBoost adds to
            /// the log of the hessian.
            #[track_caller]
            pub fn max_delta_step(mut self, max_delta_step: f64) -> Self {
                at_least_zero("max_delta_step", max_delta_step);
                self.params.max_delta_step = Some(max_delta_step);
                self
            }

            /// The share of the rows each tree grows on, drawn anew for each round. Default 1.
            #[track_caller]
            pub fn subsample(mut self, subsample: f64) -> Self {
                share("subsample", subsample);
                self.params.subsample = subsample;
                self
            }

            /// The share of the features each tree draws. Default 1.
            #[track_caller]
            pub fn colsample_bytree(mut self, colsample_bytree: f64) -> Self {
                share("colsample_bytree", colsample_bytree);
                self.params.colsample_bytree = colsample_bytree;
                self
            }

            /// The share of the tree's features each depth draws. Default 1.
            #[track_caller]
            pub fn colsample_bylevel(mut self, colsample_bylevel: f64) -> Self {
                share("colsample_bylevel", colsample_bylevel);
                self.params.colsample_bylevel = colsample_bylevel;
                self
            }

            /// The share of the depth's features each node draws. Default 1.
            #[track_caller]
            pub fn colsample_bynode(mut self, colsample_bynode: f64) -> Self {
                share("colsample_bynode", colsample_bynode);
                self.params.colsample_bynode = colsample_bynode;
                self
            }

            /// Where every prediction starts. Default: the constant with the lowest loss (the
            /// mean, the median, the share of class 1, ...), as XGBoost estimates it.
            #[track_caller]
            pub fn base_score(mut self, base_score: f64) -> Self {
                check(base_score.is_finite(), "base_score", base_score, "a finite number");
                self.params.base_score = Some(base_score);
                self
            }

            /// The seed of `subsample` and the `colsample_*` draws. Default 0.
            pub fn random_state(mut self, random_state: u64) -> Self {
                self.params.random_state = random_state;
                self
            }

            /// The most buckets per feature, 2 to 65,536. Default 256; rusmlai keeps one of
            /// them for the missing values, so 256 gives 255 buckets of values.
            #[track_caller]
            pub fn max_bin(mut self, max_bin: usize) -> Self {
                check((2..=65_536).contains(&max_bin), "max_bin", max_bin, "2 to 65,536");
                self.params.max_bin = max_bin;
                self
            }
        }
    };
}

boosted_model! {
    regressor
    /// Gradient boosting with XGBoost's names, defaults and split rules: `xgboost.XGBRegressor`.
    ///
    /// Each round grows a depth-wise tree of depth 6 on the gradients of the objective, with
    /// a learning rate of 0.3, an L2 penalty `reg_lambda` of 1, and children of at least
    /// `min_child_weight` (1) of hessian. The predictions start from the constant with the
    /// lowest loss, as XGBoost estimates `base_score`. With at most 255 distinct values per
    /// feature, the buckets hold one value each, like XGBoost's, and the trees are XGBoost's,
    /// node for node: the predictions agree to 1e-3 or better (XGBoost keeps scores in f32,
    /// rusmlai in f64). Like XGBoost, two splits whose gains are equal as f32 go to the first
    /// feature.
    ///
    /// Objectives: `"reg:squarederror"` (the default), `"reg:absoluteerror"`,
    /// `"reg:quantileerror"` (`quantile_alpha`), `"reg:pseudohubererror"` (`huber_slope`),
    /// `"count:poisson"`, `"reg:gamma"` and `"reg:tweedie"` (`tweedie_variance_power`).
    ///
    /// XGBoost settings with no setter: `n_jobs`, `device`, `tree_method` (always `"hist"`),
    /// `verbosity`, `validate_parameters`, `callbacks`, `importance_type`, `missing` (NaN and
    /// nulls are missing), `eval_metric`. Not yet: `early_stopping_rounds` (it needs an
    /// `eval_set`), `booster="dart"`, `monotone_constraints`, `interaction_constraints`,
    /// XGBoost's own split of categories (a text column is split as LightGBM splits it), a
    /// depth-wise order for `max_leaves`, and XGBoost 3's hessian for `"reg:absoluteerror"`
    /// and `"reg:quantileerror"`, which falls as the residual grows (rusmlai's is 1, as in
    /// XGBoost 2: the leaves still take the median or the quantile, but the trees differ).
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
    /// let mut model = XGBRegressor::new();
    /// model.fit(&ds, "price");
    /// assert_eq!(model.n_iter_, 100);
    /// assert!(model.score(&ds, "price") > 0.999);
    ///
    /// // Stumps: trees of one split each, as XGBoost grows them.
    /// let mut stumps = XGBRegressor::new().max_depth(1).n_estimators(50);
    /// stumps.fit(&ds, "price");
    /// assert!(stumps.score(&ds, "price") > 0.9999);
    /// let first: f64 = stumps.predict(&ds).f64().unwrap().get(0).unwrap();
    /// assert!((first - 150.74615).abs() < 1e-3);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// from xgboost import XGBRegressor
    /// ds = pd.DataFrame({
    ///     "area": [50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0],
    ///     "rooms": [1, 2, 2, 3, 3, 3, 4, 4],
    ///     "price": [150.0, 180.0, 200.0, 260.0, 270.0, 310.0, 400.0, 420.0],
    /// })
    /// x, y = ds[["area", "rooms"]], ds["price"]
    /// model = XGBRegressor().fit(x, y)
    /// assert model.get_booster().num_boosted_rounds() == 100
    /// assert model.score(x, y) > 0.999
    ///
    /// stumps = XGBRegressor(max_depth=1, n_estimators=50).fit(x, y)
    /// assert stumps.score(x, y) > 0.9999
    /// assert abs(stumps.predict(x)[0] - 150.74615) < 1e-3
    /// ```
    pub struct XGBRegressor(XgbParams = XgbParams::new("reg:squarederror"));
}

xgb_settings!(XGBRegressor);

impl XGBRegressor {
    /// 100 rounds of trees of depth 6 on the squared error, with a learning rate of 0.3.
    pub fn new() -> Self {
        Self::default()
    }

    /// What the trees lower, XGBoost's names: `"reg:squarederror"` (the default),
    /// `"reg:absoluteerror"`, `"reg:quantileerror"`, `"reg:pseudohubererror"`,
    /// `"count:poisson"`, `"reg:gamma"` or `"reg:tweedie"`. The last three predict through a
    /// log link.
    #[track_caller]
    pub fn objective(mut self, objective: &str) -> Self {
        const OBJECTIVES: [&str; 7] = [
            "reg:squarederror",
            "reg:absoluteerror",
            "reg:quantileerror",
            "reg:pseudohubererror",
            "count:poisson",
            "reg:gamma",
            "reg:tweedie",
        ];
        check(OBJECTIVES.contains(&objective), "objective", objective, REGRESSION);
        self.params.objective = objective.into();
        self
    }

    /// The slope δ of `"reg:pseudohubererror"`, above 0. Default 1.
    #[track_caller]
    pub fn huber_slope(mut self, huber_slope: f64) -> Self {
        let ok: bool = huber_slope.is_finite() && huber_slope > 0.0;
        check(ok, "huber_slope", huber_slope, "a finite number above 0");
        self.params.huber_slope = huber_slope;
        self
    }

    /// The quantile `"reg:quantileerror"` predicts, above 0 and below 1. Default 0.5.
    #[track_caller]
    pub fn quantile_alpha(mut self, quantile_alpha: f64) -> Self {
        let ok: bool = quantile_alpha > 0.0 && quantile_alpha < 1.0;
        check(ok, "quantile_alpha", quantile_alpha, "a number above 0 and below 1");
        self.params.quantile_alpha = quantile_alpha;
        self
    }

    /// The variance power ρ of `"reg:tweedie"`, from 1 to 2. Default 1.5.
    #[track_caller]
    pub fn tweedie_variance_power(mut self, tweedie_variance_power: f64) -> Self {
        let ok: bool = (1.0..2.0).contains(&tweedie_variance_power);
        check(ok, "tweedie_variance_power", tweedie_variance_power, "1.0 up to below 2.0");
        self.params.tweedie_variance_power = tweedie_variance_power;
        self
    }
}

boosted_model! {
    classifier
    /// Gradient boosting with XGBoost's names, defaults and split rules: `xgboost.XGBClassifier`.
    ///
    /// Two classes boost the log loss (`"binary:logistic"`): one tree per round, on the
    /// log-odds of the second class. More classes boost the softmax (`"multi:softprob"`):
    /// one tree per class and round, with XGBoost's hessian 2 p (1 − p). The trees are
    /// [`XGBRegressor`]'s: depth 6, a learning rate of 0.3, `reg_lambda` 1 and
    /// `min_child_weight` 1. `scale_pos_weight` weighs the rows of the second class.
    ///
    /// XGBoost settings with no setter: those of [`XGBRegressor`]. Not yet: the same.
    ///
    /// ```
    /// use rusmlai::prelude::*;
    /// let ds = Dataset::new(
    ///     df!(
    ///         "hours" => [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5],
    ///         "passed" => [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    ///     )
    ///     .unwrap(),
    ///     "",
    /// );
    /// let mut model = XGBClassifier::new();
    /// model.fit(&ds, "passed");
    /// assert_eq!(model.classes_, ["0", "1"]);
    /// assert_eq!(model.score(&ds, "passed"), 1.0);
    /// // 0.866, as in XGBoost: few rows, and a learning rate of 0.3 over trees of one split.
    /// assert!(model.predict_proba(&ds).to_list("1")[15] > 0.85);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// from xgboost import XGBClassifier
    /// ds = pd.DataFrame({
    ///     "hours": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5],
    ///     "passed": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    /// })
    /// x, y = ds[["hours"]], ds["passed"]
    /// model = XGBClassifier().fit(x, y)
    /// assert model.classes_.tolist() == [0, 1]
    /// assert model.score(x, y) == 1.0
    /// assert model.predict_proba(x)[15, 1] > 0.85
    /// ```
    pub struct XGBClassifier(XgbParams = XgbParams::new("binary:logistic"));
}

xgb_settings!(XGBClassifier);

impl XGBClassifier {
    /// 100 rounds of trees of depth 6 on the log loss, with a learning rate of 0.3.
    pub fn new() -> Self {
        Self::default()
    }

    /// `"binary:logistic"` (the default; with more than two classes it is
    /// `"multi:softprob"`, as in XGBoost's scikit-learn wrapper), `"multi:softprob"` or
    /// `"multi:softmax"` (the same trees; `predict` is the same).
    #[track_caller]
    pub fn objective(mut self, objective: &str) -> Self {
        let ok: bool = matches!(objective, "binary:logistic" | "multi:softprob" | "multi:softmax");
        check(ok, "objective", objective, CLASSIFICATION);
        self.params.objective = objective.into();
        self
    }

    /// How much more a row of the second class counts, above 0: its gradient and hessian
    /// are multiplied by this. Default 1. XGBoost suggests (rows of class 0) / (rows of
    /// class 1) for unbalanced classes.
    #[track_caller]
    pub fn scale_pos_weight(mut self, scale_pos_weight: f64) -> Self {
        let ok: bool = scale_pos_weight.is_finite() && scale_pos_weight > 0.0;
        check(ok, "scale_pos_weight", scale_pos_weight, "a finite number above 0");
        self.params.scale_pos_weight = scale_pos_weight;
        self
    }
}

/// `save_model` / `try_save_model` on an XGBoost-named model.
macro_rules! xgb_export {
    ($model:ident) => {
        impl $model {
            /// Writes the fitted model as XGBoost's own JSON model, which Python reads with
            /// `xgboost.XGBRegressor().load_model(path)` (or `XGBClassifier`, or
            /// `xgboost.Booster(model_file=path)`) and which predicts what this model
            /// predicts. (xgboost: `model.save_model("model.json")`) Not yet: a model with
            /// categorical features. Panics with the error's message at your line.
            #[track_caller]
            pub fn save_model(&self, path: impl AsRef<std::path::Path>) {
                crate::error::OrFatal::or_fatal(self.try_save_model(path))
            }

            /// [`Self::save_model`] that returns the error.
            pub fn try_save_model(&self, path: impl AsRef<std::path::Path>) -> Result<(), ModelError> {
                if self.target.is_none() {
                    return Err(ModelError::NotFitted { model: stringify!($model) });
                }
                let model = crate::models::export::XgbModel {
                    trees: &self.estimators_,
                    baseline: &self.baseline,
                    names: &self.feature_names_in_,
                    objective: &self.params.objective,
                };
                crate::models::export::xgboost_json(&model, path.as_ref())
            }
        }
    };
}

xgb_export!(XGBRegressor);
xgb_export!(XGBClassifier);

#[cfg(test)]
pub(crate) mod tests {
    use polars::prelude::{DataFrame, IntoLazy, lit};

    use super::*;
    use crate::{models::tree::tests::table, utils::col};

    /// `table()` with `pos` (`y` moved above 0) and `hit` (1 where the class is 1).
    pub(crate) fn data() -> Dataset {
        let ds: Dataset = table();
        let low: f64 = ds.to_list("y").into_iter().fold(f64::INFINITY, f64::min);
        let frame: DataFrame = (ds.dataframe().clone().lazy())
            .with_column((col("y") - lit(low) + lit(1.0)).alias("pos"))
            .with_column(col("class").eq(lit(1)).cast(DataType::Int64).alias("hit"))
            .collect()
            .unwrap();
        Dataset::new(frame, "")
    }

    pub(crate) fn features(ds: &Dataset, keep: &str) -> Dataset {
        let drop: Vec<&str> = ["y", "pos", "hit", "class"].into_iter().filter(|c| *c != keep).collect();
        ds.drop(&drop)
    }

    /// Leaves, the mean prediction and the first four.
    pub(crate) type Want = (usize, f64, [f64; 4]);

    /// (leaves, mean prediction, the first four) against XGBoost's, from
    /// `analysis/2026-11-17/gbm-params/xgb-parity.py`.
    pub(crate) fn assert_like(name: &str, leaves: usize, preds: &[f64], want: Want, eps: f64) {
        let mean: f64 = preds.iter().sum::<f64>() / preds.len() as f64;
        // 0 leaves: the count is not compared.
        let ok: bool = (want.0 == 0 || leaves == want.0)
            && (mean - want.1).abs() < eps
            && preds[..4].iter().zip(want.2).all(|(p, w)| (p - w).abs() < eps);
        assert!(ok, "{name}: {leaves} leaves, mean {mean}, {:?}; want {want:?}", &preds[..4]);
    }

    #[test]
    fn regressors_match_xgboost() {
        let ds: Dataset = data();
        let cases: [(&str, XGBRegressor, &str, Want); 9] = [
            ("default", XGBRegressor::new(), "y", (227, 16.237516403198242, [4.879846572875977, 8.267618179321289, 19.68661880493164, 4.830348491668701])),
            ("shallow", XGBRegressor::new().max_depth(3).reg_lambda(0.0).min_child_weight(0.0), "y", (40, 16.235000610351562, [4.36831521987915, 8.202165603637695, 18.886581420898438, 6.421261310577393])),
            ("pseudohuber", XGBRegressor::new().objective("reg:pseudohubererror").huber_slope(2.0), "y", (5, -158.1951904296875, [-158.1951904296875; 4])),
            ("poisson", XGBRegressor::new().objective("count:poisson"), "pos", (315, 22.310043334960938, [14.123397827148438, 16.813804626464844, 24.75596809387207, 14.896202087402344])),
            ("gamma", XGBRegressor::new().objective("reg:gamma"), "pos", (221, 21.652259826660156, [7.480661392211914, 14.052299499511719, 26.649232864379883, 11.628669738769531])),
            ("tweedie", XGBRegressor::new().objective("reg:tweedie"), "pos", (284, 22.001188278198242, [9.070501327514648, 13.827332496643066, 26.71017837524414, 9.854898452758789])),
            ("penalties", XGBRegressor::new().gamma(50.0).reg_alpha(5.0).max_delta_step(1.0), "y", (12, 16.23499870300293, [14.734999656677246, 14.734999656677246, 17.734996795654297, 14.734999656677246])),
            ("lossguide", XGBRegressor::new().grow_policy("lossguide").max_leaves(8).max_depth(0), "y", (40, 16.234424591064453, [5.508053779602051, 9.841612815856934, 18.70755958557129, 5.98470401763916])),
            ("base_score", XGBRegressor::new().base_score(10.0), "y", (218, 15.075361251831055, [3.217115879058838, 6.974523067474365, 18.69886589050293, 4.110015869140625])),
        ];
        for (name, model, target, want) in cases {
            let mut model: XGBRegressor = model.n_estimators(5);
            let x: Dataset = features(&ds, target);
            model.fit(&x, target);
            let leaves: usize = model.estimators_.iter().map(Tree::n_leaves).sum();
            let preds: Vec<f64> = model.predict(&x).f64().unwrap().into_no_null_iter().collect();
            assert_like(name, leaves, &preds, want, 1e-3);
        }
    }

    /// XGBoost 3.4 gives these two a hessian that falls with the residual, and starts the
    /// absolute error elsewhere than the median: rusmlai keeps a hessian of 1 (XGBoost 2's),
    /// so the trees differ, but each leaf still takes its quantile.
    #[test]
    fn quantile_objectives_fit_their_quantile() {
        let ds: Dataset = data();
        let x: Dataset = features(&ds, "y");
        let y: Vec<f64> = ds.to_list("y");
        for (objective, q) in [("reg:absoluteerror", 0.5), ("reg:quantileerror", 0.8)] {
            let mut model: XGBRegressor = XGBRegressor::new().objective(objective).quantile_alpha(0.8);
            model.fit(&x, "y");
            let preds: Vec<f64> = model.predict(&x).f64().unwrap().into_no_null_iter().collect();
            let below: f64 = y.iter().zip(&preds).filter(|(y, p)| y <= p).count() as f64 / y.len() as f64;
            assert!((below - q).abs() < 0.1, "{objective}: {below} of the rows at or below");
        }
    }

    #[test]
    fn classifiers_match_xgboost() {
        let ds: Dataset = data();
        let cases: [(&str, XGBClassifier, &str, Want); 3] = [
            ("binary", XGBClassifier::new(), "hit", (109, 0.4020863473415375, [0.13216190040111542, 0.15577323734760284, 0.39467838406562805, 0.0959174707531929])),
            ("scale_pos_weight", XGBClassifier::new().scale_pos_weight(3.0), "hit", (116, 0.5134506821632385, [0.18782536685466766, 0.41244855523109436, 0.4601089060306549, 0.16392506659030914])),
            ("classes", XGBClassifier::new(), "class", (402, 0.3333333432674408, [0.6605177521705627, 0.10582298040390015, 0.23365925252437592, 0.6881757974624634])),
        ];
        for (name, model, target, want) in cases {
            let mut model: XGBClassifier = model.n_estimators(5);
            let x: Dataset = features(&ds, target);
            model.fit(&x, target);
            let leaves: usize = model.estimators_.iter().map(Tree::n_leaves).sum();
            let proba: Dataset = model.predict_proba(&x);
            let preds: Vec<f64> = match target {
                "hit" => proba.to_list("1"),
                // Row after row, class after class, as numpy's ravel.
                _ => {
                    let columns: Vec<Vec<f64>> = model.classes_.iter().map(|c| proba.to_list(c)).collect();
                    (0..x.height()).flat_map(|r| columns.iter().map(move |c| c[r])).collect()
                }
            };
            assert_like(name, leaves, &preds, want, 1e-5);
        }
    }
}
