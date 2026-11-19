//! [`LGBMRegressor`] and [`LGBMClassifier`]: gradient boosting with LightGBM's names,
//! defaults and split rules, on rusmlai's histogram tree engine (`models/hist.rs`).
//!
//! The engine is the one of [`GradientBoostingRegressor`](crate::models::GradientBoostingRegressor);
//! these types only map LightGBM's settings onto it ([`Library`]): leaf-wise trees of 31
//! leaves, a learning rate of 0.1, leaves of at least 20 rows (`min_child_samples`), counted as
//! LightGBM counts them, from each side's share of the hessian.

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
        tree::{ClassWeight, Tree, pick_classes, share_columns},
    },
};

/// LightGBM's settings, with its names and defaults (lightgbm 4.7).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct LgbmParams {
    /// `None`: `"regression"` for a regressor, `"binary"` or `"multiclass"` for a classifier.
    objective: Option<String>,
    n_estimators: usize,
    learning_rate: f64,
    num_leaves: usize,
    /// 0 or less is no limit.
    max_depth: i64,
    min_child_samples: usize,
    min_child_weight: f64,
    min_split_gain: f64,
    subsample: f64,
    subsample_freq: usize,
    colsample_bytree: f64,
    feature_fraction_bynode: f64,
    reg_alpha: f64,
    reg_lambda: f64,
    max_delta_step: f64,
    random_state: Option<u64>,
    max_bin: usize,
    min_data_in_bin: usize,
    boost_from_average: bool,
    alpha: f64,
    fair_c: f64,
    poisson_max_delta_step: f64,
    tweedie_variance_power: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    class_weight: Option<ClassWeight>,
    is_unbalance: bool,
    scale_pos_weight: f64,
}

impl Default for LgbmParams {
    fn default() -> Self {
        Self {
            objective: None,
            n_estimators: 100,
            learning_rate: 0.1,
            num_leaves: 31,
            max_depth: -1,
            min_child_samples: 20,
            min_child_weight: 1e-3,
            min_split_gain: 0.0,
            subsample: 1.0,
            subsample_freq: 0,
            colsample_bytree: 1.0,
            feature_fraction_bynode: 1.0,
            reg_alpha: 0.0,
            reg_lambda: 0.0,
            max_delta_step: 0.0,
            random_state: None,
            max_bin: 255,
            min_data_in_bin: 3,
            boost_from_average: true,
            alpha: 0.9,
            fair_c: 1.0,
            poisson_max_delta_step: 0.7,
            tweedie_variance_power: 1.5,
            class_weight: None,
            is_unbalance: false,
            scale_pos_weight: 1.0,
        }
    }
}

impl LgbmParams {
    /// The objective's main name, from LightGBM's own aliases.
    fn objective(&self, classifier: bool, classes: usize) -> &str {
        match self.objective.as_deref() {
            Some("regression" | "regression_l2" | "l2" | "mean_squared_error" | "mse" | "l2_root" | "root_mean_squared_error" | "rmse") => "regression",
            Some("regression_l1" | "l1" | "mean_absolute_error" | "mae") => "regression_l1",
            Some("mean_absolute_percentage_error" | "mape") => "mape",
            Some("softmax" | "multiclass") => "multiclass",
            Some(other) => other,
            None if !classifier => "regression",
            None if classes > 2 => "multiclass",
            None => "binary",
        }
    }
}

const REGRESSION: &str = "\"regression\", \"regression_l1\", \"huber\", \"fair\", \"poisson\", \
                          \"quantile\", \"mape\", \"gamma\" or \"tweedie\" (or one of their aliases)";
const CLASSIFICATION: &str = "\"binary\" (two classes) or \"multiclass\"";

impl Library for LgbmParams {
    fn plan(&self, _rows: usize) -> Plan {
        Plan {
            rounds: self.n_estimators,
            step: Step {
                l1: self.reg_alpha,
                min_gain: self.min_split_gain,
                min_hessian: self.min_child_weight,
                max_delta: self.max_delta_step,
                ..Step::new(self.reg_lambda, self.learning_rate)
            },
            max_depth: (self.max_depth > 0).then_some(self.max_depth as usize),
            max_leaf_nodes: Some(self.num_leaves),
            min_samples_leaf: self.min_child_samples,
            max_features: 1.0,
            by_tree: self.colsample_bytree,
            by_level: 1.0,
            by_node: self.feature_fraction_bynode,
            subsample: self.subsample,
            subsample_freq: self.subsample_freq,
            rules: Rules::Lightgbm,
            seed: self.random_state.unwrap_or(0),
            early_stopping: false,
            validation_fraction: 0.1,
            n_iter_no_change: 10,
            tol: 0.0,
            base_score: (!self.boost_from_average).then_some(0.0),
        }
    }

    fn max_bins(&self) -> usize {
        self.max_bin.min(255)
    }

    fn min_data_in_bin(&self) -> usize {
        self.min_data_in_bin
    }

    fn categorical(&self, dataset: &Dataset, target: &str) -> Result<Vec<String>, ModelError> {
        categorical_columns(dataset, target, None)
    }

    fn regression<'a>(&self, y: &'a [f64]) -> Result<Loss<'a>, ModelError> {
        let name: &str = self.objective(false, 0);
        let refuse = |expected: &'static str| ModelError::InvalidSetting {
            setting: "objective",
            value: name.into(),
            expected,
        };
        let at_least_zero: bool = y.iter().all(|v| *v >= 0.0);
        Ok(match name {
            "regression" => Loss::Squared(y),
            "regression_l1" => Loss::Absolute(y),
            "huber" => Loss::Huber(y, self.alpha),
            "fair" => Loss::Fair(y, self.fair_c),
            "quantile" => Loss::Quantile(y, self.alpha),
            "mape" => Loss::Mape(y),
            "poisson" if at_least_zero => Loss::Poisson(y, self.poisson_max_delta_step),
            "gamma" if y.iter().all(|v| *v > 0.0) => Loss::Gamma(y),
            "tweedie" if at_least_zero => Loss::Tweedie(y, self.tweedie_variance_power),
            "poisson" | "tweedie" => return Err(refuse("a target of 0 or more for \"poisson\" and \"tweedie\"")),
            "gamma" => return Err(refuse("a target above 0 in every row for \"gamma\"")),
            _ => return Err(refuse(REGRESSION)),
        })
    }

    fn classification<'a>(&self, y: &'a [u32], classes: usize) -> Result<Loss<'a>, ModelError> {
        match (self.objective(true, classes), classes) {
            ("binary", 2) => {
                // LightGBM's `is_unbalance`: the rarer class counts as much as the other.
                let positives: f64 = y.iter().filter(|c| **c == 1).count() as f64;
                let negatives: f64 = y.len() as f64 - positives;
                let mut labels: [f64; 2] = [1.0, 1.0];
                if self.is_unbalance && positives > 0.0 && negatives > 0.0 {
                    match positives > negatives {
                        true => labels[0] = positives / negatives,
                        false => labels[1] = negatives / positives,
                    }
                }
                labels[1] *= self.scale_pos_weight;
                Ok(Loss::Binary(y, labels))
            }
            ("multiclass", k) if k > 2 => Ok(Loss::Classes(y, k, k as f64 / (k as f64 - 1.0))),
            _ => Err(ModelError::InvalidSetting {
                setting: "objective",
                value: self.objective(true, classes).into(),
                expected: CLASSIFICATION,
            }),
        }
    }

    fn class_weight(&self) -> Option<&ClassWeight> {
        self.class_weight.as_ref()
    }

    fn log_link(&self) -> bool {
        matches!(self.objective(false, 0), "poisson" | "gamma" | "tweedie")
    }
}

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

macro_rules! lgbm_settings {
    ($model:ident) => {
        impl $model {
            /// The rounds of trees. Default 100.
            #[track_caller]
            pub fn n_estimators(mut self, n_estimators: usize) -> Self {
                check(n_estimators > 0, "n_estimators", n_estimators, "1 or more");
                self.params.n_estimators = n_estimators;
                self
            }

            /// How much of each tree's step is taken, above 0. Default 0.1.
            #[track_caller]
            pub fn learning_rate(mut self, learning_rate: f64) -> Self {
                let ok: bool = learning_rate.is_finite() && learning_rate > 0.0;
                check(ok, "learning_rate", learning_rate, "a finite number above 0");
                self.params.learning_rate = learning_rate;
                self
            }

            /// The most leaves of a tree, 2 or more. Default 31. A tree splits its best leaf
            /// first.
            #[track_caller]
            pub fn num_leaves(mut self, num_leaves: usize) -> Self {
                check(num_leaves >= 2, "num_leaves", num_leaves, "2 or more");
                self.params.num_leaves = num_leaves;
                self
            }

            /// The most splits between the root and a leaf; 0 or less (the default, -1) is no
            /// limit.
            pub fn max_depth(mut self, max_depth: i64) -> Self {
                self.params.max_depth = max_depth;
                self
            }

            /// The fewest rows in a leaf. Default 20. Like LightGBM, a side's rows are its
            /// share of the hessian times the node's rows, rounded.
            pub fn min_child_samples(mut self, min_child_samples: usize) -> Self {
                self.params.min_child_samples = min_child_samples;
                self
            }

            /// The least sum of hessians in a leaf. Default 1e-3.
            #[track_caller]
            pub fn min_child_weight(mut self, min_child_weight: f64) -> Self {
                at_least_zero("min_child_weight", min_child_weight);
                self.params.min_child_weight = min_child_weight;
                self
            }

            /// The least a split must take off the loss. Default 0.
            #[track_caller]
            pub fn min_split_gain(mut self, min_split_gain: f64) -> Self {
                at_least_zero("min_split_gain", min_split_gain);
                self.params.min_split_gain = min_split_gain;
                self
            }

            /// The share of the rows the trees grow on, drawn anew every `subsample_freq`
            /// rounds. Default 1. Like LightGBM, it does nothing while `subsample_freq` is 0.
            #[track_caller]
            pub fn subsample(mut self, subsample: f64) -> Self {
                share("subsample", subsample);
                self.params.subsample = subsample;
                self
            }

            /// How many rounds a draw of `subsample` lasts; 0 (the default) draws none.
            pub fn subsample_freq(mut self, subsample_freq: usize) -> Self {
                self.params.subsample_freq = subsample_freq;
                self
            }

            /// The share of the features each tree draws (LightGBM's `feature_fraction`), at
            /// least 2 of them. Default 1.
            #[track_caller]
            pub fn colsample_bytree(mut self, colsample_bytree: f64) -> Self {
                share("colsample_bytree", colsample_bytree);
                self.params.colsample_bytree = colsample_bytree;
                self
            }

            /// The share of the tree's features each node draws. Default 1.
            #[track_caller]
            pub fn feature_fraction_bynode(mut self, feature_fraction_bynode: f64) -> Self {
                share("feature_fraction_bynode", feature_fraction_bynode);
                self.params.feature_fraction_bynode = feature_fraction_bynode;
                self
            }

            /// The L1 penalty on the leaf values. Default 0.
            #[track_caller]
            pub fn reg_alpha(mut self, reg_alpha: f64) -> Self {
                at_least_zero("reg_alpha", reg_alpha);
                self.params.reg_alpha = reg_alpha;
                self
            }

            /// The L2 penalty on the leaf values. Default 0.
            #[track_caller]
            pub fn reg_lambda(mut self, reg_lambda: f64) -> Self {
                at_least_zero("reg_lambda", reg_lambda);
                self.params.reg_lambda = reg_lambda;
                self
            }

            /// The most a leaf value moves before the learning rate, 0 (the default) for no
            /// limit.
            #[track_caller]
            pub fn max_delta_step(mut self, max_delta_step: f64) -> Self {
                at_least_zero("max_delta_step", max_delta_step);
                self.params.max_delta_step = max_delta_step;
                self
            }

            /// The seed of `subsample` and the feature draws. Default: LightGBM's fixed seeds.
            pub fn random_state(mut self, random_state: u64) -> Self {
                self.params.random_state = Some(random_state);
                self
            }

            /// The most buckets per feature, 2 to 255. Default 255.
            #[track_caller]
            pub fn max_bin(mut self, max_bin: usize) -> Self {
                check((2..=255).contains(&max_bin), "max_bin", max_bin, "2 to 255");
                self.params.max_bin = max_bin;
                self
            }

            /// The fewest rows of a bucket: a value of fewer rows shares its bucket with the
            /// values above it. Default 3. Only a feature with at most `max_bin` distinct
            /// values is bucketed this way; one with more gets quantile buckets.
            #[track_caller]
            pub fn min_data_in_bin(mut self, min_data_in_bin: usize) -> Self {
                check(min_data_in_bin >= 1, "min_data_in_bin", min_data_in_bin, "1 or more");
                self.params.min_data_in_bin = min_data_in_bin;
                self
            }

            /// `true` (the default) starts every prediction from the constant with the lowest
            /// loss (the mean, the median, the log-odds of the second class, ...); `false`
            /// from 0.
            pub fn boost_from_average(mut self, boost_from_average: bool) -> Self {
                self.params.boost_from_average = boost_from_average;
                self
            }
        }
    };
}

boosted_model! {
    regressor
    /// Gradient boosting with LightGBM's names, defaults and split rules: `lightgbm.LGBMRegressor`.
    ///
    /// Each round grows a tree leaf by leaf, up to `num_leaves` (31), with a learning rate of
    /// 0.1 and leaves of at least `min_child_samples` (20) rows. The predictions start from
    /// the constant with the lowest loss (`boost_from_average`). A split's threshold lies
    /// halfway between the smallest value on its right and the value before it, as in
    /// LightGBM, and of two splits with the same gain the first feature wins, or on one
    /// feature the highest threshold. With at most 255 distinct values per feature, each with
    /// 3 rows or more, LightGBM's buckets hold one value each, like rusmlai's, and the trees
    /// are LightGBM's, node for node: the predictions agree to 1e-6 for every objective but
    /// `"quantile"`. LightGBM also splits a node whose rows all have one gradient, on
    /// rounding noise; rusmlai does not. Such a split changes no prediction, except with the
    /// losses whose leaves take a quantile of their rows (`"quantile"`, rarely
    /// `"regression_l1"` and `"mape"`).
    ///
    /// Objectives: `"regression"` (the default), `"regression_l1"`, `"huber"` (`alpha`),
    /// `"fair"` (`fair_c`), `"quantile"` (`alpha`), `"mape"`, `"poisson"`
    /// (`poisson_max_delta_step`), `"gamma"` and `"tweedie"` (`tweedie_variance_power`), and
    /// their LightGBM aliases (`"l2"`, `"mse"`, `"l1"`, `"mae"`, ...).
    ///
    /// LightGBM settings with no setter: `n_jobs` (`num_threads`), `verbose`, `device_type`,
    /// `importance_type`, `subsample_for_bin` (rusmlai bins from 200,000 rows too),
    /// `deterministic`, `force_col_wise`, `force_row_wise`, `histogram_pool_size`, the file,
    /// network and GPU settings. Not yet: `boosting_type` (`"dart"`, `"goss"`, `"rf"`),
    /// `extra_trees`, `path_smooth`, `monotone_constraints`, `interaction_constraints`,
    /// LightGBM's own settings for categories (`cat_smooth`, `cat_l2`, `max_cat_threshold`,
    /// `max_cat_to_onehot`, `min_data_per_group`: a text column is split as scikit-learn
    /// splits it), early stopping (it needs an `eval_set`), and LightGBM's bucket for 0 and
    /// its buckets of a feature with more than `max_bin` values (rusmlai's are quantiles of
    /// the rows).
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
    /// // 8 rows: leaves of 20 rows, the default, could never split.
    /// let mut model = LGBMRegressor::new().min_child_samples(1);
    /// model.fit(&ds, "price");
    /// assert_eq!(model.n_iter_, 100);
    /// // Buckets hold at least 3 rows, so not every pair of areas can be told apart.
    /// assert!(model.score(&ds, "price") > 0.95);
    ///
    /// // A bucket per value: every split is open.
    /// let mut fine = LGBMRegressor::new().min_child_samples(1).min_data_in_bin(1);
    /// fine.fit(&ds, "price");
    /// assert!(fine.score(&ds, "price") > 0.999);
    ///
    /// // The default leaves keep the mean of all rows: no split has 20 rows on each side.
    /// let mut flat = LGBMRegressor::new();
    /// flat.fit(&ds, "price");
    /// let first: f64 = flat.predict(&ds).f64().unwrap().get(0).unwrap();
    /// assert!((first - 273.75).abs() < 1e-9);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// from lightgbm import LGBMRegressor
    /// ds = pd.DataFrame({
    ///     "area": [50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0],
    ///     "rooms": [1, 2, 2, 3, 3, 3, 4, 4],
    ///     "price": [150.0, 180.0, 200.0, 260.0, 270.0, 310.0, 400.0, 420.0],
    /// })
    /// x, y = ds[["area", "rooms"]], ds["price"]
    /// model = LGBMRegressor(min_child_samples=1, verbose=-1).fit(x, y)
    /// assert model.n_estimators_ == 100
    /// assert model.score(x, y) > 0.95
    ///
    /// fine = LGBMRegressor(min_child_samples=1, min_data_in_bin=1, verbose=-1).fit(x, y)
    /// assert fine.score(x, y) > 0.999
    ///
    /// flat = LGBMRegressor(verbose=-1).fit(x, y)
    /// assert abs(flat.predict(x)[0] - 273.75) < 1e-9
    /// ```
    pub struct LGBMRegressor(LgbmParams = LgbmParams::default());
}

lgbm_settings!(LGBMRegressor);

impl LGBMRegressor {
    /// 100 rounds of trees of 31 leaves on the squared error, with a learning rate of 0.1.
    pub fn new() -> Self {
        Self::default()
    }

    /// What the trees lower, LightGBM's names: `"regression"` (the default),
    /// `"regression_l1"`, `"huber"`, `"fair"`, `"quantile"`, `"mape"`, `"poisson"`, `"gamma"`
    /// or `"tweedie"`, or an alias of one. The last three predict through a log link.
    #[track_caller]
    pub fn objective(mut self, objective: &str) -> Self {
        self.params.objective = Some(objective.into());
        let name: &str = self.params.objective(false, 0);
        const OBJECTIVES: [&str; 9] =
            ["regression", "regression_l1", "huber", "fair", "quantile", "mape", "poisson", "gamma", "tweedie"];
        check(OBJECTIVES.contains(&name), "objective", objective, REGRESSION);
        self
    }

    /// The α of `"huber"` (where the loss turns from squared to absolute) and of
    /// `"quantile"` (the quantile it predicts). Default 0.9.
    #[track_caller]
    pub fn alpha(mut self, alpha: f64) -> Self {
        check(alpha.is_finite() && alpha > 0.0, "alpha", alpha, "a finite number above 0");
        self.params.alpha = alpha;
        self
    }

    /// The `c` of `"fair"`, above 0. Default 1.
    #[track_caller]
    pub fn fair_c(mut self, fair_c: f64) -> Self {
        check(fair_c.is_finite() && fair_c > 0.0, "fair_c", fair_c, "a finite number above 0");
        self.params.fair_c = fair_c;
        self
    }

    /// What `"poisson"` adds to the log of its hessian, so that a leaf moves less. Default
    /// 0.7.
    #[track_caller]
    pub fn poisson_max_delta_step(mut self, poisson_max_delta_step: f64) -> Self {
        let ok: bool = poisson_max_delta_step.is_finite();
        check(ok, "poisson_max_delta_step", poisson_max_delta_step, "a finite number");
        self.params.poisson_max_delta_step = poisson_max_delta_step;
        self
    }

    /// The variance power ρ of `"tweedie"`, from 1 to 2. Default 1.5.
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
    /// Gradient boosting with LightGBM's names, defaults and split rules: `lightgbm.LGBMClassifier`.
    ///
    /// Two classes boost the log loss (`"binary"`): one tree per round, on the log-odds of the
    /// second class. More classes boost the softmax (`"multiclass"`): one tree per class and
    /// round, with LightGBM's hessian k / (k − 1) p (1 − p). The trees are [`LGBMRegressor`]'s.
    /// `class_weight` weighs each row by its class (scikit-learn's way, also in the start);
    /// `is_unbalance` and `scale_pos_weight` weigh the gradients of two classes only
    /// (LightGBM's way).
    ///
    /// LightGBM settings with no setter: those of [`LGBMRegressor`]. Not yet: the same, and
    /// `"multiclassova"`.
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
    /// // 16 rows: small leaves, and a bucket per value.
    /// let mut model = LGBMClassifier::new().min_child_samples(2).min_data_in_bin(1);
    /// model.fit(&ds, "passed");
    /// assert_eq!(model.classes_, ["0", "1"]);
    /// assert_eq!(model.score(&ds, "passed"), 1.0);
    /// assert!(model.predict_proba(&ds).to_list("1")[15] > 0.99);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// from lightgbm import LGBMClassifier
    /// ds = pd.DataFrame({
    ///     "hours": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5],
    ///     "passed": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    /// })
    /// x, y = ds[["hours"]], ds["passed"]
    /// model = LGBMClassifier(min_child_samples=2, min_data_in_bin=1, verbose=-1).fit(x, y)
    /// assert model.classes_.tolist() == [0, 1]
    /// assert model.score(x, y) == 1.0
    /// assert model.predict_proba(x)[15, 1] > 0.99
    /// ```
    pub struct LGBMClassifier(LgbmParams = LgbmParams::default());
}

lgbm_settings!(LGBMClassifier);

impl LGBMClassifier {
    /// 100 rounds of trees of 31 leaves on the log loss, with a learning rate of 0.1.
    pub fn new() -> Self {
        Self::default()
    }

    /// `"binary"` (two classes) or `"multiclass"` (`"softmax"`). Default: by the classes.
    #[track_caller]
    pub fn objective(mut self, objective: &str) -> Self {
        let ok: bool = matches!(objective, "binary" | "multiclass" | "softmax");
        check(ok, "objective", objective, CLASSIFICATION);
        self.params.objective = Some(objective.into());
        self
    }

    /// How much each class counts: `"balanced"`, or your own weights,
    /// `[("yes", 5.0), ("no", 1.0)]`. A class weight multiplies each row's weight, so the
    /// start moves too. Default: every class counts once.
    #[track_caller]
    pub fn class_weight(mut self, class_weight: impl Into<ClassWeight>) -> Self {
        self.params.class_weight = Some(class_weight.into().checked(false));
        self
    }

    /// `true` makes the rarer of two classes count as much as the other in the gradients
    /// (not in the start). Default `false`.
    pub fn is_unbalance(mut self, is_unbalance: bool) -> Self {
        self.params.is_unbalance = is_unbalance;
        self
    }

    /// How much more a row of the second class counts in the gradients, above 0. Default 1.
    #[track_caller]
    pub fn scale_pos_weight(mut self, scale_pos_weight: f64) -> Self {
        let ok: bool = scale_pos_weight.is_finite() && scale_pos_weight > 0.0;
        check(ok, "scale_pos_weight", scale_pos_weight, "a finite number above 0");
        self.params.scale_pos_weight = scale_pos_weight;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::xgboost::tests::{Want, assert_like, data, features};

    fn predictions(model: &impl Classifier, x: &Dataset, classes: &[String], two: bool) -> Vec<f64> {
        let proba: Dataset = model.predict_proba(x);
        if two {
            return proba.to_list("1");
        }
        // Row after row, class after class, as numpy's ravel.
        let columns: Vec<Vec<f64>> = classes.iter().map(|c| proba.to_list(c)).collect();
        (0..x.height()).flat_map(|r| columns.iter().map(move |c| c[r])).collect()
    }

    /// Against LightGBM's, from `analysis/2026-11-17/gbm-params/lgbm-parity.py`.
    #[test]
    fn regressors_match_lightgbm() {
        let ds: Dataset = data();
        let cases: [(&str, LGBMRegressor, &str, Want); 11] = [
            ("default", LGBMRegressor::new(), "y", (77, 16.235000004874657, [11.082056886561618, 12.463979637225467, 17.402013980204003, 10.429170009422187])),
            ("small leaves", LGBMRegressor::new().num_leaves(8).min_child_samples(5).max_depth(3), "y", (40, 16.235000008448722, [11.233792077158602, 12.992496725138174, 16.636637802844316, 10.797206064469615])),
            ("l1", LGBMRegressor::new().objective("regression_l1"), "y", (51, 16.249441906640627, [12.739838749999999, 12.78504625, 18.58999546875, 11.90617125])),
            ("huber", LGBMRegressor::new().objective("huber").alpha(2.0), "y", (63, 16.23873890288477, [15.239916192745342, 15.318782556254346, 17.113514344651875, 15.235000000000003])),
            ("fair", LGBMRegressor::new().objective("fair").fair_c(2.0), "y", (73, 16.179329513022005, [5.272310589701548, 11.16379409085482, 18.84893969510715, 1.1138364729777224])),
            ("mape", LGBMRegressor::new().objective("mape"), "pos", (83, 18.2905862125, [10.8239925, 14.0545925, 19.958077499999998, 15.115369999999999])),
            ("poisson", LGBMRegressor::new().objective("poisson"), "pos", (78, 22.248550048740103, [19.406361985348532, 20.129796751346433, 22.859059236454026, 19.031758044342634])),
            ("gamma", LGBMRegressor::new().objective("gamma"), "pos", (70, 21.337879902895256, [13.037060833249486, 16.154800639651025, 23.722415657662747, 14.452749760052845])),
            ("tweedie", LGBMRegressor::new().objective("tweedie"), "pos", (75, 21.84044503789031, [14.93904846434766, 17.83930309887154, 23.230877228706174, 15.593253030930143])),
            ("penalties", LGBMRegressor::new().reg_alpha(5.0).reg_lambda(3.0).min_split_gain(2.0).max_delta_step(1.0), "y", (12, 16.237484628216617, [15.734999999999998, 15.734999999999998, 16.735000000000007, 15.734999999999998])),
            ("no average", LGBMRegressor::new().boost_from_average(false), "y", (77, 6.648394852051512, [1.4954517100383735, 2.877374495978809, 7.8154088683149, 0.8425648518723737])),
        ];
        let mut failed: Vec<String> = Vec::new();
        for (name, model, target, want) in cases {
            let mut model: LGBMRegressor = model.n_estimators(5);
            let x: Dataset = features(&ds, target);
            model.fit(&x, target);
            let leaves: usize = model.estimators_.iter().map(Tree::n_leaves).sum();
            let preds: Vec<f64> = model.predict(&x).f64().unwrap().into_no_null_iter().collect();
            if let Err(e) = std::panic::catch_unwind(|| assert_like(name, leaves, &preds, want, 1e-6)) {
                failed.push(e.downcast_ref::<String>().cloned().unwrap_or_default());
            }
        }
        assert!(failed.is_empty(), "{}", failed.join("\n"));
    }

    /// LightGBM splits a node whose rows all have one gradient on rounding noise (the gains
    /// of its splits are its own gain to the last bit), which rusmlai does not copy. With the
    /// quantile loss each leaf then takes the quantile of fewer rows; the trees agree
    /// elsewhere, and the predictions still sit at the quantile.
    #[test]
    fn quantile_fits_its_quantile() {
        let ds: Dataset = data();
        let x: Dataset = features(&ds, "y");
        let y: Vec<f64> = ds.to_list("y");
        let mut model: LGBMRegressor = LGBMRegressor::new().objective("quantile").alpha(0.8);
        model.fit(&x, "y");
        let preds: Vec<f64> = model.predict(&x).f64().unwrap().into_no_null_iter().collect();
        let below: f64 = y.iter().zip(&preds).filter(|(y, p)| y <= p).count() as f64 / y.len() as f64;
        assert!((below - 0.8).abs() < 0.05, "{below} of the rows at or below");
    }

    #[test]
    fn classifiers_match_lightgbm() {
        let ds: Dataset = data();
        let cases: [(&str, LGBMClassifier, &str, Want); 4] = [
            ("binary", LGBMClassifier::new(), "hit", (75, 0.3988519517141624, [0.2381889760995229, 0.31926594239401257, 0.48038012283559683, 0.2381889760995229])),
            ("unbalance", LGBMClassifier::new().is_unbalance(true), "hit", (76, 0.4204280005485059, [0.241743110324076, 0.36366733473332274, 0.5167289783430324, 0.241743110324076])),
            ("scale_pos_weight", LGBMClassifier::new().scale_pos_weight(3.0), "hit", (74, 0.455163044734898, [0.29397853681459113, 0.4574719690391561, 0.5995727655757972, 0.25081308297521765])),
            // LightGBM splits two nodes whose rows all have one gradient, on rounding noise:
            // their leaves get the same value, so only the count differs.
            ("classes", LGBMClassifier::new(), "class", (0, 0.3333333333333333, [0.4673657606990974, 0.24980303879167495, 0.28283120050922766, 0.39989562022977854])),
        ];
        let mut failed: Vec<String> = Vec::new();
        for (name, model, target, want) in cases {
            let mut model: LGBMClassifier = model.n_estimators(5);
            let x: Dataset = features(&ds, target);
            model.fit(&x, target);
            let leaves: usize = model.estimators_.iter().map(Tree::n_leaves).sum();
            let preds: Vec<f64> = predictions(&model, &x, &model.classes_, target == "hit");
            if let Err(e) = std::panic::catch_unwind(|| assert_like(name, leaves, &preds, want, 1e-6)) {
                failed.push(e.downcast_ref::<String>().cloned().unwrap_or_default());
            }
        }
        assert!(failed.is_empty(), "{}", failed.join("\n"));
    }
}

