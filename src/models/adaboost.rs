//! [`AdaBoostClassifier`]: a weighted vote of small trees, each one grown on row weights that
//! put more weight on the rows that the trees before it got wrong (SAMME).

use std::io::{self, Write};

use polars::prelude::{DataType, Series};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    dataset::{Dataset, split::Rng},
    error::warn,
    metadata::models::CachedRun,
    models::{
        Classifier, Model, ModelError, bad_learned, check,
        design::column,
        hist::{Bins, Grown, Target, feature, grow},
        logistic::softmax,
        tree::{
            DecisionTreeClassifier, Tree, class_target, pick_classes, read_trees, share_columns,
            sum_leaves,
        },
    },
};

/// The settings of AdaBoost itself, with scikit-learn's names.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Params {
    n_estimators: usize,
    learning_rate: f64,
    random_state: u64,
}

/// The learned values that are small, as the model cache stores them in JSON. The trees are
/// bytes.
#[derive(Serialize, Deserialize)]
struct Learned {
    classes_: Vec<String>,
    feature_names_in_: Vec<String>,
    feature_importances_: Vec<f64>,
    estimator_weights_: Vec<f64>,
    estimator_errors_: Vec<f64>,
    dtype: DataType,
}

/// The class with the most weight in a leaf (the first one on a tie), like a tree's
/// `predict`.
fn leaf_class(weights: &[f64]) -> usize {
    (1..weights.len()).fold(0, |best, c| if weights[c] > weights[best] { c } else { best })
}

/// Predicts a class by a weighted vote of small trees, each one grown on row weights that
/// favour the rows the trees before it got wrong, like scikit-learn's `AdaBoostClassifier`.
///
/// The algorithm is SAMME, the one scikit-learn keeps. Each of the `n_estimators` rounds
/// grows a tree (a stump, one split, by default) on the current row weights. Its weighted
/// error `e` gives it the vote `learning_rate · (ln((1 − e) / e) + ln(classes − 1))`, and the
/// rows it got wrong weigh `exp(vote)` times more in the next round. A tree with no error
/// stops the rounds; a tree no better than chance is dropped, and stops them too.
///
/// `estimator` takes the tree with your settings, scikit-learn's `estimator`; only a
/// decision tree can be the base, because it takes row weights. The features are put into
/// buckets once for all rounds. `predict_proba` is scikit-learn's: the softmax of the votes
/// divided by `classes − 1`.
///
/// scikit-learn settings with no setter: `algorithm` (deprecated; SAMME is the only one
/// left). Not yet: another base model.
///
/// ```
/// use rusmlai::prelude::*;
/// let ds = Dataset::new(
///     df!(
///         "hours" => [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5],
///         "sleep" => [4.0, 5.0, 4.5, 5.5, 4.0, 8.0, 7.0, 4.0, 7.5, 8.5, 7.0, 8.0],
///         "passed" => ["no", "no", "no", "no", "no", "yes", "yes", "no", "yes", "yes", "yes", "yes"],
///     )
///     .unwrap(),
///     "",
/// );
/// let mut model = AdaBoostClassifier::new().n_estimators(10);
/// model.fit(&ds, "passed");
/// assert_eq!(model.classes_, ["no", "yes"]);
/// assert_eq!(model.score(&ds, "passed"), 1.0); // stumps on hours, then on sleep
/// assert_eq!(model.estimators_[0].node_count(), 3); // one split each
///
/// // Deeper trees with a smaller vote each.
/// let mut deep = AdaBoostClassifier::new()
///     .estimator(DecisionTreeClassifier::new().max_depth(2))
///     .learning_rate(0.5);
/// deep.fit(&ds, "passed");
/// assert_eq!(deep.estimator_errors_[0], 0.0); // one tree gets every row right
/// assert_eq!(deep.estimators_.len(), 1);
/// ```
///
/// Python:
///
/// ```python
/// import pandas as pd
/// from sklearn.ensemble import AdaBoostClassifier
/// from sklearn.tree import DecisionTreeClassifier
/// ds = pd.DataFrame({
///     "hours": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5],
///     "sleep": [4.0, 5.0, 4.5, 5.5, 4.0, 8.0, 7.0, 4.0, 7.5, 8.5, 7.0, 8.0],
///     "passed": ["no", "no", "no", "no", "no", "yes", "yes", "no", "yes", "yes", "yes", "yes"],
/// })
/// x, y = ds[["hours", "sleep"]], ds["passed"]
/// model = AdaBoostClassifier(n_estimators=10, random_state=42).fit(x, y)
/// assert model.classes_.tolist() == ["no", "yes"]
/// assert model.score(x, y) == 1.0
/// assert model.estimators_[0].tree_.node_count == 3
///
/// deep = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), learning_rate=0.5, random_state=42).fit(x, y)
/// assert deep.estimator_errors_[0] == 0.0
/// assert len(deep.estimators_) == 1
/// ```
#[derive(Debug, Clone)]
pub struct AdaBoostClassifier {
    params: Params,
    estimator: DecisionTreeClassifier,
    /// The classes `fit` saw, sorted: the order of `predict_proba`'s columns.
    pub classes_: Vec<String>,
    /// The feature columns `fit` saw.
    pub feature_names_in_: Vec<String>,
    /// How many features `fit` saw.
    pub n_features_in_: usize,
    /// The trees' importances, weighted by their votes, as shares that sum to 1.
    pub feature_importances_: Vec<f64>,
    /// The trees, in the order they were grown.
    pub estimators_: Vec<Tree>,
    /// The vote of each tree.
    pub estimator_weights_: Vec<f64>,
    /// The weighted error of each tree on the rows it was grown on.
    pub estimator_errors_: Vec<f64>,
    dtype: Option<DataType>,
    target: Option<String>,
    run: Option<CachedRun>,
}

impl Default for AdaBoostClassifier {
    fn default() -> Self {
        Self {
            params: Params {
                n_estimators: 50,
                learning_rate: 1.0,
                random_state: 42,
            },
            estimator: DecisionTreeClassifier::new().max_depth(1),
            classes_: Vec::new(),
            feature_names_in_: Vec::new(),
            n_features_in_: 0,
            feature_importances_: Vec::new(),
            estimators_: Vec::new(),
            estimator_weights_: Vec::new(),
            estimator_errors_: Vec::new(),
            dtype: None,
            target: None,
            run: None,
        }
    }
}

impl AdaBoostClassifier {
    /// 50 rounds of stumps (trees of one split), with a learning rate of 1.
    pub fn new() -> Self {
        Self::default()
    }

    /// The tree each round grows, with its settings. Default: a stump,
    /// `DecisionTreeClassifier::new().max_depth(1)`.
    pub fn estimator(mut self, estimator: DecisionTreeClassifier) -> Self {
        self.estimator = estimator;
        self
    }

    /// The most rounds. Default 50.
    #[track_caller]
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        check(n_estimators > 0, "n_estimators", n_estimators, "1 or more");
        self.params.n_estimators = n_estimators;
        self
    }

    /// What each tree's vote is multiplied by, above 0. Default 1.
    #[track_caller]
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        let ok: bool = learning_rate.is_finite() && learning_rate > 0.0;
        check(ok, "learning_rate", learning_rate, "a finite number above 0");
        self.params.learning_rate = learning_rate;
        self
    }

    /// The seed of the trees' feature draws (with `max_features` on the estimator). Default
    /// 42. A tree with every feature draws nothing.
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.params.random_state = random_state;
        self
    }

    /// The sum over the trees of each class's vote for each row, divided by the votes.
    fn votes(&self, dataset: &Dataset) -> Result<Vec<f64>, ModelError> {
        if self.target.is_none() {
            return Err(ModelError::NotFitted {
                model: "AdaBoostClassifier",
            });
        }
        let (k, weights): (usize, &[f64]) = (self.classes_.len(), &self.estimator_weights_);
        let against: f64 = -1.0 / (k as f64 - 1.0);
        let names: &[String] = &self.feature_names_in_;
        let mut votes: Vec<f64> = sum_leaves(&self.estimators_, dataset, names, k, |t, leaf, sum| {
            let class: usize = leaf_class(leaf);
            for (j, sum) in sum.iter_mut().enumerate() {
                *sum += if j == class { weights[t] } else { against * weights[t] };
            }
        })?;
        let total: f64 = weights.iter().sum();
        votes.iter_mut().for_each(|vote| *vote /= total);
        Ok(votes)
    }
}

impl Model for AdaBoostClassifier {
    fn try_train(&mut self, dataset: &Dataset, target: &str) -> Result<(), ModelError> {
        let (p, tree) = (&self.params, &self.estimator.params);
        let (classes, task): (Vec<String>, Target) =
            class_target(dataset, target, &tree.criterion)?;
        let Target::Classes { y, .. } = &task else {
            unreachable!("a classifier's target is classes")
        };
        let (n, k): (usize, usize) = (dataset.height(), classes.len());
        let bins: Bins = Bins::try_new(dataset, target, tree.max_bins)?;
        let series: Vec<Series> = (bins.names.iter())
            .map(|name| feature(dataset, name))
            .collect::<Result<_, _>>()?;
        let columns: Vec<&[f64]> = series.iter().map(column).collect();
        let mut leaves: Vec<u32> = vec![0; n];
        let mut weights: Vec<f64> = vec![1.0 / n as f64; n];
        let (mut trees, mut votes, mut errors): (Vec<Tree>, Vec<f64>, Vec<f64>) =
            (Vec::new(), Vec::new(), Vec::new());
        let mut importances: Vec<f64> = vec![0.0; bins.names.len()];
        let mut rng: Rng = Rng::new(p.random_state);
        for round in 0..p.n_estimators {
            let growth = tree.growth(bins.names.len(), rng.next_u64());
            let as_f32: Vec<f32> = weights.iter().map(|w| *w as f32).collect();
            let grown: Grown = grow(&bins, &task, &growth, Some(&as_f32), None);
            grown.tree.leaves(&columns, &mut leaves);
            let wrong: Vec<bool> = (leaves.iter().zip(y))
                .map(|(leaf, y)| {
                    leaf_class(&grown.tree.value()[*leaf as usize * k..][..k]) != *y as usize
                })
                .collect();
            let total: f64 = weights.iter().sum();
            let error: f64 = (weights.iter().zip(&wrong))
                .filter(|(_, wrong)| **wrong)
                .map(|(w, _)| w)
                .sum::<f64>()
                / total;
            // No error: this tree decides alone, with a vote of 1, like scikit-learn's.
            if error <= 0.0 {
                importances.iter_mut().zip(&grown.importances).for_each(|(sum, i)| *sum += i);
                trees.push(grown.tree);
                votes.push(1.0);
                errors.push(0.0);
                break;
            }
            // No better than chance: the tree has no vote, and more rounds won't help.
            if error >= 1.0 - 1.0 / k as f64 {
                if trees.is_empty() {
                    return Err(ModelError::InvalidSetting {
                        setting: "estimator",
                        value: format!("a tree with a weighted error of {error:.4}"),
                        expected: "a tree that does better than chance on the first round",
                    });
                }
                break;
            }
            let vote: f64 =
                p.learning_rate * (((1.0 - error) / error).ln() + (k as f64 - 1.0).ln());
            importances
                .iter_mut()
                .zip(&grown.importances)
                .for_each(|(sum, i)| *sum += vote * i);
            trees.push(grown.tree);
            votes.push(vote);
            errors.push(error);
            if round + 1 == p.n_estimators {
                break;
            }
            let boost: f64 = vote.exp();
            for (w, wrong) in weights.iter_mut().zip(&wrong) {
                if *wrong && *w > 0.0 {
                    *w *= boost;
                }
            }
            let sum: f64 = weights.iter().sum();
            if !sum.is_finite() {
                warn(format_args!(
                    "AdaBoostClassifier stopped after {} rounds: the row weights overflowed; \
                     lower learning_rate",
                    round + 1
                ));
                break;
            }
            weights.iter_mut().for_each(|w| *w /= sum);
        }
        let norm: f64 = votes.iter().sum();
        if norm > 0.0 {
            importances.iter_mut().for_each(|share| *share /= norm);
        }
        self.dtype = Some(dataset.column(target)?.dtype().clone());
        self.n_features_in_ = bins.names.len();
        (self.feature_names_in_, self.classes_) = (bins.names, classes);
        (self.estimators_, self.feature_importances_) = (trees, importances);
        (self.estimator_weights_, self.estimator_errors_) = (votes, errors);
        self.target = Some(target.into());
        Ok(())
    }

    fn params(&self) -> Value {
        let mut params: Value = serde_json::to_value(&self.params).unwrap_or_default();
        params["estimator"] = json!({
            "model": self.estimator.name(),
            "params": self.estimator.params(),
        });
        params
    }

    fn learned(&self) -> Value {
        serde_json::to_value(Learned {
            classes_: self.classes_.clone(),
            feature_names_in_: self.feature_names_in_.clone(),
            feature_importances_: self.feature_importances_.clone(),
            estimator_weights_: self.estimator_weights_.clone(),
            estimator_errors_: self.estimator_errors_.clone(),
            dtype: self.dtype.clone().unwrap_or(DataType::String),
        })
        .unwrap_or_default()
    }

    fn write_weights(&self, out: &mut dyn Write) -> io::Result<()> {
        self.estimators_.iter().try_for_each(|tree| tree.write(out))
    }

    fn try_set_params(&mut self, params: Value) -> Result<(), ModelError> {
        let own: Params = serde_json::from_value(params.clone())?;
        self.estimator
            .try_set_params(params["estimator"]["params"].clone())?;
        self.params = own;
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

    fn try_load(&mut self, learned: Value, weights: &[u8], target: &str) -> Result<(), ModelError> {
        let l: Learned = serde_json::from_value(learned)?;
        let (k, features): (usize, usize) = (l.classes_.len(), l.feature_names_in_.len());
        if k < 2 || l.feature_importances_.len() != features {
            return Err(bad_learned(format!(
                "{k} classes and {} importances for {features} features",
                l.feature_importances_.len()
            )));
        }
        let trees: Vec<Tree> = read_trees(weights, features, k)?;
        let rounds: [usize; 3] = [trees.len(), l.estimator_weights_.len(), l.estimator_errors_.len()];
        if rounds != [trees.len(); 3] {
            return Err(bad_learned(format!(
                "{} trees with {} votes and {} errors",
                rounds[0], rounds[1], rounds[2]
            )));
        }
        self.n_features_in_ = features;
        (self.feature_names_in_, self.feature_importances_) =
            (l.feature_names_in_, l.feature_importances_);
        (self.classes_, self.dtype, self.estimators_) = (l.classes_, Some(l.dtype), trees);
        (self.estimator_weights_, self.estimator_errors_) =
            (l.estimator_weights_, l.estimator_errors_);
        self.target = Some(target.into());
        Ok(())
    }

    fn run(&self) -> Option<&CachedRun> {
        self.run.as_ref()
    }

    fn set_run(&mut self, run: Option<CachedRun>) {
        self.run = run;
    }
}

impl Classifier for AdaBoostClassifier {
    /// The class with the most votes; the first one on a tie, like scikit-learn.
    fn try_predict(&self, dataset: &Dataset) -> Result<Series, ModelError> {
        pick_classes(
            &self.votes(dataset)?,
            &self.classes_,
            self.target.as_deref().unwrap_or_default(),
            self.dtype.as_ref().unwrap_or(&DataType::String),
        )
    }

    /// The softmax of each row's votes divided by `classes − 1`, like scikit-learn's.
    fn try_predict_proba(&self, dataset: &Dataset) -> Result<Dataset, ModelError> {
        let k: usize = self.classes_.len();
        let mut votes: Vec<f64> = self.votes(dataset)?;
        let mut shares: Vec<f64> = vec![0.0; k];
        for row in votes.chunks_mut(k) {
            row.iter_mut().for_each(|vote| *vote /= k as f64 - 1.0);
            softmax(row, &mut shares);
            row.copy_from_slice(&shares);
        }
        share_columns(dataset, &votes, &self.classes_, 1)
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

    fn fitted(model: AdaBoostClassifier, ds: &Dataset, target: &str) -> AdaBoostClassifier {
        let mut model: AdaBoostClassifier = model;
        model.fit(ds, target);
        model
    }

    #[test]
    fn samme_matches_scikit_learn_on_three_classes_and_two() {
        // scikit-learn 1.9.1 (analysis/2026-11-17/boosting/parity.py): the same for 10 seeds.
        // The tree weights are f32 here, so the votes agree to about 1e-6.
        let ds: Dataset = table().drop(&["y"]);
        let model = fitted(AdaBoostClassifier::new().n_estimators(10), &ds, "class");
        assert_close(
            &model.estimator_weights_,
            &[
                0.8333765218685952, 0.9738080483407558, 0.6440666038814544, 0.8210476641501327,
                0.5755500471386755, 0.301005587094484, 0.5071305672001699, 0.42832969247605696,
                0.3819210655072022, 0.2396382600531809,
            ],
            1e-5,
        );
        assert_close(
            &model.estimator_errors_[..3],
            &[0.4650000000000001, 0.4302917629718957, 0.5122676816305571],
            1e-6,
        );
        assert_eq!(model.score(&ds, "class"), 0.6);
        let first: Dataset = Dataset::new(ds.dataframe().slice(0, 1), "");
        let proba: Dataset = model.predict_proba(&first);
        let shares: Vec<f64> = ["0", "1", "2"].iter().map(|c| proba.to_list(c)[0]).collect();
        assert_close(&shares, &[0.3754996393627991, 0.34249887564413595, 0.28200148499306493], 1e-5);
        assert!((model.feature_importances_.iter().sum::<f64>() - 1.0).abs() < 1e-12);

        let frame: DataFrame = (table().dataframe().clone().lazy())
            .with_column(col("class").eq(lit(1)).alias("hit"))
            .collect()
            .unwrap();
        let binary: Dataset = Dataset::new(frame, "").drop(&["class", "y"]);
        let model = fitted(AdaBoostClassifier::new().n_estimators(10), &binary, "hit");
        assert_close(
            &model.estimator_weights_[..4],
            &[0.4158927312704228, 0.6289938260116825, 0.41355932439656207, 0.35162105993903375],
            1e-5,
        );
        assert_eq!(model.score(&binary, "hit"), 0.725);
        assert_eq!(model.classes_, ["false", "true"]);
        let first: Dataset = Dataset::new(binary.dataframe().slice(0, 1), "");
        assert_close(&model.predict_proba(&first).to_list("true"), &[0.37499612059935833], 1e-5);

        let deep = AdaBoostClassifier::new()
            .n_estimators(5)
            .learning_rate(0.5)
            .estimator(DecisionTreeClassifier::new().max_depth(2));
        let deep = fitted(deep, &ds, "class");
        assert_close(
            &deep.estimator_weights_,
            &[0.6506076759406572, 0.5991799856611955, 0.5408838538593868, 0.5408415297488152, 0.373752620756246],
            1e-5,
        );
        assert_eq!(deep.score(&ds, "class"), 0.785);
    }

    #[test]
    fn learned_values_round_trip_and_settings_hold_the_tree_s() {
        let ds: Dataset = table().drop(&["y"]);
        let trained = fitted(
            AdaBoostClassifier::new()
                .n_estimators(4)
                .estimator(DecisionTreeClassifier::new().max_depth(2).criterion("entropy")),
            &ds,
            "class",
        );
        assert_eq!(trained.params()["estimator"]["params"]["max_depth"], 2);
        let mut weights: Vec<u8> = Vec::new();
        trained.write_weights(&mut weights).unwrap();
        let mut loaded: AdaBoostClassifier = AdaBoostClassifier::new();
        loaded.try_set_params(trained.params()).unwrap();
        assert_eq!(loaded.params(), trained.params());
        loaded.try_load(trained.learned(), &weights, "class").unwrap();
        assert_eq!(loaded.predict(&ds), trained.predict(&ds));
        let mut learned: Value = trained.learned();
        learned["estimator_weights_"] = json!([1.0]);
        let err: String = AdaBoostClassifier::new()
            .try_load(learned, &weights, "class")
            .unwrap_err()
            .to_string();
        assert!(err.contains("4 trees with 1 votes"), "{err}");
        assert!(matches!(
            AdaBoostClassifier::new().try_predict(&ds),
            Err(ModelError::NotFitted { .. })
        ));
    }
}
