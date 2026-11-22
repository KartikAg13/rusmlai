//! [`BaggingClassifier`] and [`BaggingRegressor`]: copies of any model, each trained on its
//! own sample of the rows (and, if asked, of the features), and averaged.
//!
//! Unlike a random forest, the base model is yours to choose. A base model that takes row
//! weights (the trees) gets its sample as a count per row; any other trains on a copy of the
//! sampled rows. With decision trees, `RandomForest*` does the same work faster: it makes the
//! buckets once for all trees, where each copy in a bag makes its own.

use std::io::{self, Write};

use polars::prelude::{DataType, IdxCa, IdxSize, NamedFrom, Series};
use polars_core::runtime::THREAD_POOL;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    dataset::{Dataset, split::Rng},
    metadata::models::CachedRun,
    models::{
        Classifier, Model, ModelError, Regressor, bad_learned, check, label_target,
        tree::{pick_classes, share_columns},
    },
};
use crate::models::{share};

/// The settings of a bag, with scikit-learn's names.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Params {
    n_estimators: usize,
    max_samples: f64,
    max_features: f64,
    bootstrap: bool,
    random_state: u64,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            n_estimators: 10,
            max_samples: 1.0,
            max_features: 1.0,
            bootstrap: true,
            random_state: 42,
        }
    }
}

/// Trains `copy`, copy `t` of the bag, on its sample: `max_samples` of the rows, drawn with
/// repeats when `bootstrap`, and `max_features` of the features, drawn without repeats. A
/// model that takes row weights gets the sample as a count per row, like scikit-learn's
/// `sample_weight`; any other gets a copy of the drawn rows.
fn train_copy<M: Model>(
    copy: &mut M,
    dataset: &Dataset,
    target: &str,
    p: &Params,
    t: usize,
) -> Result<(), ModelError> {
    let mut rng: Rng = Rng::new(p.random_state ^ ((t as u64 + 1) << 32));
    let rows: usize = dataset.height();
    let draws: usize = ((p.max_samples * rows as f64) as usize).max(1);
    let picked: Option<Vec<IdxSize>> = if p.bootstrap {
        Some((0..draws).map(|_| (rng.next_u64() % rows as u64) as IdxSize).collect())
    } else if draws < rows {
        let mut all: Vec<IdxSize> = (0..rows as IdxSize).collect();
        rng.shuffle(&mut all);
        all.truncate(draws);
        Some(all)
    } else {
        None
    };
    let mut features: Vec<String> = dataset.columns();
    features.retain(|name| name != target);
    let keep: usize = ((p.max_features * features.len() as f64) as usize).max(1);
    if keep < features.len() {
        rng.shuffle(&mut features);
        features.truncate(keep);
    }
    // In column order, with the target, so a model sees the columns as they were.
    let names: Vec<String> = dataset.columns();
    let kept = names
        .iter()
        .filter(|name| *name == target || features.contains(name));
    let part: Dataset = dataset.try_select(&kept.map(String::as_str).collect::<Vec<&str>>())?;
    let Some(picked) = picked else {
        return copy.try_train(&part, target);
    };
    let mut counts: Vec<f32> = vec![0.0; rows];
    picked.iter().for_each(|row| counts[*row as usize] += 1.0);
    if let Some(trained) = copy.try_train_weighted(&part, target, &counts) {
        return trained;
    }
    let frame = part.dataframe.take(&IdxCa::from_vec("".into(), picked))?;
    copy.try_train(&part.derived(frame), target)
}

/// `p.n_estimators` copies of `base`, each trained on its own sample, on the Polars pool.
fn train<M: Model + Clone + Send + Sync>(
    base: &M,
    dataset: &Dataset,
    target: &str,
    p: &Params,
) -> Result<Vec<M>, ModelError> {
    THREAD_POOL.install(|| {
        (0..p.n_estimators)
            .into_par_iter()
            .map(|t| {
                let mut copy: M = base.clone();
                copy.set_run(None);
                train_copy(&mut copy, dataset, target, p, t)?;
                Ok(copy)
            })
            .collect()
    })
}

/// The settings of a bag and of its base model, for the cache key and the journal.
fn params_of<M: Model>(base: &M, p: &Params) -> Value {
    let mut params: Value = serde_json::to_value(p).unwrap_or_default();
    params["estimator"] = json!({ "model": base.name(), "params": base.params() });
    params
}

fn set_params<M: Model>(base: &mut M, p: &mut Params, params: Value) -> Result<(), ModelError> {
    let own: Params = serde_json::from_value(params.clone())?;
    base.try_set_params(params["estimator"]["params"].clone())?;
    *p = own;
    Ok(())
}

/// The weights of every copy, each after its length.
fn write_all<M: Model>(copies: &[M], out: &mut dyn Write) -> io::Result<()> {
    for copy in copies {
        let mut bytes: Vec<u8> = Vec::new();
        copy.write_weights(&mut bytes)?;
        out.write_all(&(bytes.len() as u64).to_le_bytes())?;
        out.write_all(&bytes)?;
    }
    Ok(())
}

/// The copies that [`write_all`] and each copy's `learned` gave back, as copies of `base`.
fn load_all<M: Model + Clone>(
    base: &M,
    learned: &Value,
    mut weights: &[u8],
    target: &str,
) -> Result<Vec<M>, ModelError> {
    let cut = || bad_learned("the weights of a bag are cut short".into());
    let copies: &Vec<Value> = learned["estimators_"]
        .as_array()
        .filter(|copies| !copies.is_empty())
        .ok_or_else(|| bad_learned("a bag with no fitted model".into()))?;
    copies
        .iter()
        .map(|learned| {
            let (len, rest) = weights.split_first_chunk::<8>().ok_or_else(cut)?;
            let len: usize = u64::from_le_bytes(*len) as usize;
            let (bytes, rest) = rest.split_at_checked(len).ok_or_else(cut)?;
            weights = rest;
            let mut copy: M = base.clone();
            copy.try_load(learned.clone(), bytes, target)?;
            Ok(copy)
        })
        .collect()
}

/// The settings both bags share, as setters on `$model`.
macro_rules! bagging_settings {
    ($model:ident, $bound:ident) => {
        impl<M: $bound + Clone + Send + Sync> $model<M> {
            /// The number of copies of the base model. Default 10.
            #[track_caller]
            pub fn n_estimators(mut self, n_estimators: usize) -> Self {
                check(n_estimators > 0, "n_estimators", n_estimators, "1 or more");
                self.params.n_estimators = n_estimators;
                self
            }

            /// The rows each copy trains on, as a share of the rows above 0 and at most 1.
            /// Default 1.0. A count (scikit-learn's int) is not taken.
            #[track_caller]
            pub fn max_samples(mut self, max_samples: f64) -> Self {
                share("max_samples", max_samples);
                self.params.max_samples = max_samples;
                self
            }

            /// The features each copy trains on, as a share of the features above 0 and at
            /// most 1, drawn once per copy. Default 1.0. A count is not taken.
            #[track_caller]
            pub fn max_features(mut self, max_features: f64) -> Self {
                share("max_features", max_features);
                self.params.max_features = max_features;
                self
            }

            /// `true` (the default) draws each copy's rows with repeats; `false` draws them
            /// without, so `max_samples(1.0)` then gives every copy every row.
            pub fn bootstrap(mut self, bootstrap: bool) -> Self {
                self.params.bootstrap = bootstrap;
                self
            }

            /// The seed of every draw. Default 42, so every run gives the same bag. A base
            /// model with a `random_state` of its own keeps it in every copy.
            pub fn random_state(mut self, random_state: u64) -> Self {
                self.params.random_state = random_state;
                self
            }
        }
    };
}

/// Predicts a class as the vote of several copies of one classifier, each trained on its own
/// sample of the rows, like scikit-learn's `BaggingClassifier`.
///
/// `new` takes the base model with its settings, where scikit-learn takes `estimator`.
/// `predict_proba` is the mean of the copies' probabilities and `predict` the class with the
/// largest mean. A class that a copy's sample did not hold counts as probability 0 there.
///
/// A base model that takes row weights (the decision trees) gets its sample as a count per
/// row, like scikit-learn's `sample_weight`, so `min_samples_leaf` counts rows, not repeats.
/// Any other model trains on a copy of the sampled rows. With decision trees as the base,
/// [`RandomForestClassifier`](crate::models::RandomForestClassifier) with `max_features(1.0)`
/// grows the same kind of trees faster: it puts the features into buckets once for all
/// trees, where each copy in a bag makes its own.
///
/// scikit-learn settings with no setter: `n_jobs` (the copies train on the Polars pool),
/// `verbose`. Not yet: `bootstrap_features`, `oob_score`, `warm_start`, and counts for
/// `max_samples` and `max_features`.
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
/// let mut model = BaggingClassifier::new(DecisionTreeClassifier::new().max_depth(2));
/// model.fit(&ds, "passed");
/// assert_eq!(model.estimators_.len(), 10);
/// assert_eq!(model.classes_, ["no", "yes"]);
/// assert_eq!(model.score(&ds, "passed"), 1.0);
/// assert_eq!(model.predict_proba(&ds).to_list("yes")[11], 1.0);
/// ```
///
/// Python:
///
/// ```python
/// import pandas as pd
/// from sklearn.ensemble import BaggingClassifier
/// from sklearn.tree import DecisionTreeClassifier
/// ds = pd.DataFrame({
///     "hours": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5],
///     "sleep": [4.0, 5.0, 4.5, 5.5, 4.0, 5.0, 7.0, 8.0, 7.5, 8.5, 7.0, 8.0],
///     "passed": ["no", "no", "no", "no", "no", "no", "yes", "yes", "yes", "yes", "yes", "yes"],
/// })
/// x, y = ds[["hours", "sleep"]], ds["passed"]
/// model = BaggingClassifier(DecisionTreeClassifier(max_depth=2), random_state=42).fit(x, y)
/// assert len(model.estimators_) == 10
/// assert model.classes_.tolist() == ["no", "yes"]
/// assert model.score(x, y) == 1.0
/// assert model.predict_proba(x)[11, 1] == 1.0
/// ```
#[derive(Debug, Clone)]
pub struct BaggingClassifier<M: Classifier> {
    estimator: M,
    params: Params,
    /// The fitted copies of the base model.
    pub estimators_: Vec<M>,
    /// The classes `fit` saw, sorted: the order of `predict_proba`'s columns.
    pub classes_: Vec<String>,
    dtype: Option<DataType>,
    target: Option<String>,
    run: Option<CachedRun>,
}

bagging_settings!(BaggingClassifier, Classifier);

impl<M: Classifier + Clone + Send + Sync> BaggingClassifier<M> {
    /// 10 copies of `estimator`, each on a bootstrap sample of the rows.
    pub fn new(estimator: M) -> Self {
        Self {
            estimator,
            params: Params::default(),
            estimators_: Vec::new(),
            classes_: Vec::new(),
            dtype: None,
            target: None,
            run: None,
        }
    }

    /// The sum of the copies' probabilities for each row, one number per class.
    fn shares(&self, dataset: &Dataset) -> Result<Vec<f64>, ModelError> {
        if self.target.is_none() {
            return Err(ModelError::NotFitted {
                model: "BaggingClassifier",
            });
        }
        let k: usize = self.classes_.len();
        let mut sums: Vec<f64> = vec![0.0; dataset.height() * k];
        for copy in &self.estimators_ {
            let proba: Dataset = copy.try_predict_proba(dataset)?;
            for (c, class) in self.classes_.iter().enumerate() {
                // A copy whose sample of copied rows had no row of this class gives it no
                // column.
                if proba.columns().contains(class) {
                    let share: Vec<f64> = proba.to_list(class);
                    sums.iter_mut().skip(c).step_by(k).zip(share).for_each(|(sum, p)| *sum += p);
                }
            }
        }
        Ok(sums)
    }
}

impl<M: Classifier + Clone + Send + Sync> Model for BaggingClassifier<M> {
    fn target(&self) -> Option<&str> {
        self.target.as_deref()
    }

    fn try_train(&mut self, dataset: &Dataset, target: &str) -> Result<(), ModelError> {
        let (classes, _): (Vec<String>, Series) = label_target(dataset, target)?;
        self.estimators_ = train(&self.estimator, dataset, target, &self.params)?;
        self.dtype = Some(dataset.column(target)?.dtype().clone());
        self.classes_ = classes;
        self.target = Some(target.into());
        Ok(())
    }

    fn params(&self) -> Value {
        params_of(&self.estimator, &self.params)
    }

    fn learned(&self) -> Value {
        json!({
            "classes_": self.classes_,
            "dtype": self.dtype.clone().unwrap_or(DataType::String),
            "estimators_": self.estimators_.iter().map(Model::learned).collect::<Vec<Value>>(),
        })
    }

    fn write_weights(&self, out: &mut dyn Write) -> io::Result<()> {
        write_all(&self.estimators_, out)
    }

    fn try_set_params(&mut self, params: Value) -> Result<(), ModelError> {
        set_params(&mut self.estimator, &mut self.params, params)
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
        let classes: Vec<String> = serde_json::from_value(learned["classes_"].clone())?;
        let dtype: DataType = serde_json::from_value(learned["dtype"].clone())?;
        if classes.len() < 2 {
            return Err(bad_learned(format!("a bag of {} classes", classes.len())));
        }
        self.estimators_ = load_all(&self.estimator, &learned, weights, target)?;
        (self.classes_, self.dtype) = (classes, Some(dtype));
        self.target = Some(target.into());
        Ok(())
    }

    fn run(&self) -> Option<&CachedRun> {
        self.run.as_ref()
    }

    fn set_run(&mut self, run: Option<CachedRun>) {
        self.run = run;
    }

    fn name(&self) -> &'static str {
        "BaggingClassifier"
    }
}

impl<M: Classifier + Clone + Send + Sync> Classifier for BaggingClassifier<M> {
    /// The class with the largest mean probability over the copies; the first one on a tie.
    fn try_predict(&self, dataset: &Dataset) -> Result<Series, ModelError> {
        pick_classes(
            &self.shares(dataset)?,
            &self.classes_,
            self.target.as_deref().unwrap_or_default(),
            self.dtype.as_ref().unwrap_or(&DataType::String),
        )
    }

    /// The mean of the copies' probabilities.
    fn try_predict_proba(&self, dataset: &Dataset) -> Result<Dataset, ModelError> {
        let copies: usize = self.estimators_.len();
        share_columns(dataset, &self.shares(dataset)?, &self.classes_, copies)
    }
}

/// Predicts a number as the mean of several copies of one regressor, each trained on its own
/// sample of the rows, like scikit-learn's `BaggingRegressor`.
///
/// `new` takes the base model with its settings, where scikit-learn takes `estimator`. The
/// settings and the cost are those of [`BaggingClassifier`]; with decision trees as the
/// base, [`RandomForestRegressor`](crate::models::RandomForestRegressor) is the faster way.
///
/// ```
/// use rusmlai::prelude::*;
/// let ds = Dataset::new(
///     df!(
///         "area" => [50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0],
///         "price" => [150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0, 360.0],
///     )
///     .unwrap(),
///     "",
/// );
/// // Any regressor: here ten straight lines, each through its own sample of the rows.
/// let mut model = BaggingRegressor::new(LinearRegression::new());
/// model.fit(&ds, "price");
/// assert_eq!(model.estimators_.len(), 10);
/// assert!((model.score(&ds, "price") - 1.0).abs() < 1e-9); // price = 3 × area
/// ```
///
/// Python:
///
/// ```python
/// import pandas as pd
/// from sklearn.ensemble import BaggingRegressor
/// from sklearn.linear_model import LinearRegression
/// ds = pd.DataFrame({
///     "area": [50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0],
///     "price": [150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0, 360.0],
/// })
/// model = BaggingRegressor(LinearRegression(), random_state=42).fit(ds[["area"]], ds["price"])
/// assert len(model.estimators_) == 10
/// assert abs(model.score(ds[["area"]], ds["price"]) - 1.0) < 1e-9
/// ```
#[derive(Debug, Clone)]
pub struct BaggingRegressor<M: Regressor> {
    estimator: M,
    params: Params,
    /// The fitted copies of the base model.
    pub estimators_: Vec<M>,
    target: Option<String>,
    run: Option<CachedRun>,
}

bagging_settings!(BaggingRegressor, Regressor);

impl<M: Regressor + Clone + Send + Sync> BaggingRegressor<M> {
    /// 10 copies of `estimator`, each on a bootstrap sample of the rows.
    pub fn new(estimator: M) -> Self {
        Self {
            estimator,
            params: Params::default(),
            estimators_: Vec::new(),
            target: None,
            run: None,
        }
    }
}

impl<M: Regressor + Clone + Send + Sync> Model for BaggingRegressor<M> {
    fn target(&self) -> Option<&str> {
        self.target.as_deref()
    }

    fn try_train(&mut self, dataset: &Dataset, target: &str) -> Result<(), ModelError> {
        self.estimators_ = train(&self.estimator, dataset, target, &self.params)?;
        self.target = Some(target.into());
        Ok(())
    }

    fn params(&self) -> Value {
        params_of(&self.estimator, &self.params)
    }

    fn learned(&self) -> Value {
        json!({ "estimators_": self.estimators_.iter().map(Model::learned).collect::<Vec<Value>>() })
    }

    fn write_weights(&self, out: &mut dyn Write) -> io::Result<()> {
        write_all(&self.estimators_, out)
    }

    fn try_set_params(&mut self, params: Value) -> Result<(), ModelError> {
        set_params(&mut self.estimator, &mut self.params, params)
    }

    fn try_evaluate(&self, dataset: &Dataset, target: &str, scoring: &str) -> Result<f64, ModelError> {
        crate::models::evaluate_regressor(self, dataset, target, scoring)
    }

    fn try_predict_method(&self, dataset: &Dataset, method: &str) -> Result<Dataset, ModelError> {
        crate::models::predict_regressor(self, dataset, method)
    }

    fn try_load(&mut self, learned: Value, weights: &[u8], target: &str) -> Result<(), ModelError> {
        self.estimators_ = load_all(&self.estimator, &learned, weights, target)?;
        self.target = Some(target.into());
        Ok(())
    }

    fn run(&self) -> Option<&CachedRun> {
        self.run.as_ref()
    }

    fn set_run(&mut self, run: Option<CachedRun>) {
        self.run = run;
    }

    fn name(&self) -> &'static str {
        "BaggingRegressor"
    }
}

impl<M: Regressor + Clone + Send + Sync> Regressor for BaggingRegressor<M> {
    /// The mean of the copies' predictions.
    fn try_predict(&self, dataset: &Dataset) -> Result<Series, ModelError> {
        let target: &String = self.target.as_ref().ok_or(ModelError::NotFitted {
            model: "BaggingRegressor",
        })?;
        let mut sums: Vec<f64> = vec![0.0; dataset.height()];
        for copy in &self.estimators_ {
            let predicted: Series = copy.try_predict(dataset)?;
            let values = predicted.f64()?.into_no_null_iter();
            sums.iter_mut().zip(values).for_each(|(sum, value)| *sum += value);
        }
        let copies: f64 = self.estimators_.len() as f64;
        sums.iter_mut().for_each(|sum| *sum /= copies);
        Ok(Series::new(target.into(), sums))
    }
}

#[cfg(test)]
mod tests {
    use polars::df;

    use super::*;
    use crate::models::{
        DecisionTreeClassifier, DecisionTreeRegressor, LinearRegression, LogisticRegression,
    };

    /// 300 rows: `a` and `b` decide the class and the number, with noise.
    fn table() -> Dataset {
        let mut rng: Rng = Rng::new(9);
        let mut draw = |levels: u64| -> Vec<f64> {
            (0..300).map(|_| (rng.next_u64() % levels) as f64).collect()
        };
        let (a, b, wobble, flip) = (draw(40), draw(25), draw(12), draw(8));
        let y: Vec<f64> = (0..300).map(|i| 2.0 * a[i] - b[i] + wobble[i]).collect();
        let class: Vec<&str> = (0..300)
            .map(|i| if (y[i] > 30.0) != (flip[i] == 0.0) { "high" } else { "low" })
            .collect();
        Dataset::new(df!("a" => a, "b" => b, "class" => class, "y" => y).unwrap(), "")
    }

    fn halves(ds: &Dataset, drop: &str) -> (Dataset, Dataset) {
        let frame = ds.drop(&[drop]).into_dataframe();
        (Dataset::new(frame.slice(0, 200), ""), Dataset::new(frame.slice(200, 100), ""))
    }

    #[test]
    fn every_row_once_and_every_feature_is_the_base_model() {
        let (train, test) = halves(&table(), "y");
        let whole = || BaggingClassifier::new(DecisionTreeClassifier::new()).bootstrap(false);
        let mut bag = whole().n_estimators(3);
        bag.fit(&train, "class");
        let mut one: DecisionTreeClassifier = DecisionTreeClassifier::new();
        one.fit(&train, "class");
        assert!(bag.estimators_.iter().all(|copy| copy.tree_ == one.tree_));
        assert_eq!(bag.predict(&test), one.predict(&test));
        assert_eq!(bag.predict_proba(&test).to_list("low"), one.predict_proba(&test).to_list("low"));
        assert_eq!(bag.name(), "BaggingClassifier");
        assert_eq!(bag.params()["estimator"]["model"], "DecisionTreeClassifier");
        assert_eq!(bag.params()["estimator"]["params"]["criterion"], "gini");
    }

    #[test]
    fn a_bag_of_trees_beats_one_tree_and_the_seed_decides() {
        let (train, test) = halves(&table(), "y");
        let fit = |bag: BaggingClassifier<DecisionTreeClassifier>| {
            let mut bag = bag;
            bag.fit(&train, "class");
            bag
        };
        let trees = || BaggingClassifier::new(DecisionTreeClassifier::new());
        let bag = fit(trees().n_estimators(40));
        let mut one: DecisionTreeClassifier = DecisionTreeClassifier::new();
        one.fit(&train, "class");
        let (bag_test, tree_test) = (bag.score(&test, "class"), one.score(&test, "class"));
        assert!(bag_test > tree_test, "bag {bag_test}, tree {tree_test}");

        let (a, again, b) = (fit(trees().random_state(1)), fit(trees().random_state(1)), fit(trees().random_state(2)));
        let trees_of = |bag: &BaggingClassifier<DecisionTreeClassifier>| {
            bag.estimators_.iter().map(|copy| copy.tree_.clone()).collect::<Vec<_>>()
        };
        assert_eq!(trees_of(&a), trees_of(&again), "the same seed, the same bag");
        assert_ne!(trees_of(&a), trees_of(&b));
        // A tree gets its bootstrap sample as counts: about 63% of the 200 rows, once each.
        // Half of the rows without repeats is 100.
        let drawn: u32 = a.estimators_[0].tree_.n_node_samples()[0];
        assert!((110..=140).contains(&drawn), "{drawn} of 200 rows");
        let half = fit(trees().max_samples(0.5).bootstrap(false));
        assert_eq!(half.estimators_[0].tree_.n_node_samples()[0], 100);
        // Half of the two features: each copy sees one.
        let narrow = fit(trees().max_features(0.5));
        assert!(narrow.estimators_.iter().all(|copy| copy.n_features_in_ == 1));
        let seen = |name: &str| narrow.estimators_.iter().any(|copy| copy.feature_names_in_ == [name]);
        assert!(seen("a") && seen("b"));
        assert!(narrow.score(&test, "class") > 0.5);
    }

    #[test]
    fn any_model_can_be_the_base_and_a_bag_round_trips() {
        let (train, test) = halves(&table(), "class");
        let mut lines = BaggingRegressor::new(LinearRegression::new()).n_estimators(5);
        lines.fit(&train, "y");
        let mut line: LinearRegression = LinearRegression::new();
        line.fit(&train, "y");
        // A linear model takes no weights, so each of the five lines trains on a copy of its
        // sampled rows: close to the one line through them all.
        assert!((lines.score(&test, "y") - line.score(&test, "y")).abs() < 0.02);
        assert_ne!(lines.estimators_[0].coef_, lines.estimators_[1].coef_);
        let mut trees = BaggingRegressor::new(DecisionTreeRegressor::new().max_depth(4));
        trees.fit(&train, "y");
        assert!(trees.score(&test, "y") > 0.8);

        // The copies' learned values and weights go through the cache's two parts.
        let mut weights: Vec<u8> = Vec::new();
        trees.write_weights(&mut weights).unwrap();
        let mut loaded = BaggingRegressor::new(DecisionTreeRegressor::new());
        loaded.try_set_params(trees.params()).unwrap();
        assert_eq!(loaded.params(), trees.params());
        loaded.try_load(trees.learned(), &weights, "y").unwrap();
        assert_eq!(loaded.predict(&test), trees.predict(&test));
        let cut: String = loaded
            .try_load(trees.learned(), &weights[..weights.len() - 1], "y")
            .unwrap_err()
            .to_string();
        assert!(cut.contains("cut short"), "{cut}");
        let mut lines_back = BaggingRegressor::new(LinearRegression::new());
        lines_back.try_load(lines.learned(), &[0; 40], "y").unwrap(); // five empty weights
        assert_eq!(lines_back.predict(&test), lines.predict(&test));

        let (train, test) = halves(&table(), "y");
        let mut logistic = BaggingClassifier::new(LogisticRegression::new()).n_estimators(4);
        logistic.fit(&train, "class");
        assert!(logistic.score(&test, "class") > 0.8);
        assert!(matches!(
            BaggingClassifier::new(LogisticRegression::new()).try_predict(&test),
            Err(ModelError::NotFitted { .. })
        ));
    }
}
