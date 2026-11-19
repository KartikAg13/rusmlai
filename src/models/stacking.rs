//! [`StackingClassifier`] and [`StackingRegressor`]: models that learn from the out-of-fold
//! predictions of other models, like scikit-learn's.
//!
//! `fit` gets, for each model, its out-of-fold predictions ([`cross_val_predict`]) and a copy
//! trained on all the rows; the final model then trains on the out-of-fold columns. `predict`
//! runs every model on the new rows and gives their predictions to the final model. The
//! models can be of different kinds, so they are held as `Box<dyn AnyModel>`.
//!
//! [`cross_val_predict`]: crate::models::cross_val_predict

use polars::prelude::{Column, DataFrame, Series};
use polars_core::runtime::THREAD_POOL;
use rayon::prelude::*;
use serde_json::{Map, Value, json};

use crate::{
    dataset::Dataset,
    error::OrFatal,
    metadata::models::CachedRun,
    models::{
        Classifier, LogisticRegression, Model, ModelError, Regressor, RidgeCV, bad_learned, check,
        folds::{Cv, oof},
        label_target, trainable,
    },
};

/// A model of any kind that a stack can hold and copy: every rusmlai model is one.
pub trait AnyModel: Model + Send + Sync {
    /// A copy in a box.
    fn clone_box(&self) -> Box<dyn AnyModel>;
}

impl<T: Model + Clone + Send + Sync + 'static> AnyModel for T {
    fn clone_box(&self) -> Box<dyn AnyModel> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn AnyModel> {
    fn clone(&self) -> Self {
        (**self).clone_box()
    }
}

impl Model for Box<dyn AnyModel> {
    fn try_train(&mut self, dataset: &Dataset, target: &str) -> Result<(), ModelError> {
        (**self).try_train(dataset, target)
    }
    fn try_train_weighted(&mut self, dataset: &Dataset, target: &str, weights: &[f32]) -> Option<Result<(), ModelError>> {
        (**self).try_train_weighted(dataset, target, weights)
    }
    fn params(&self) -> Value {
        (**self).params()
    }
    fn learned(&self) -> Value {
        (**self).learned()
    }
    fn write_weights(&self, out: &mut dyn std::io::Write) -> std::io::Result<()> {
        (**self).write_weights(out)
    }
    fn try_load(&mut self, learned: Value, weights: &[u8], target: &str) -> Result<(), ModelError> {
        (**self).try_load(learned, weights, target)
    }
    fn try_set_params(&mut self, params: Value) -> Result<(), ModelError> {
        (**self).try_set_params(params)
    }
    fn try_evaluate(&self, dataset: &Dataset, target: &str, scoring: &str) -> Result<f64, ModelError> {
        (**self).try_evaluate(dataset, target, scoring)
    }
    fn try_predict_method(&self, dataset: &Dataset, method: &str) -> Result<Dataset, ModelError> {
        (**self).try_predict_method(dataset, method)
    }
    fn is_classifier(&self) -> bool {
        (**self).is_classifier()
    }
    fn run(&self) -> Option<&CachedRun> {
        (**self).run()
    }
    fn set_run(&mut self, run: Option<CachedRun>) {
        (**self).set_run(run)
    }
    fn name(&self) -> &'static str {
        (**self).name()
    }
    fn loss_curve(&self) -> Option<Vec<f64>> {
        (**self).loss_curve()
    }
    fn validation_curve(&self) -> Option<Vec<f64>> {
        (**self).validation_curve()
    }
    fn try_fit(&mut self, dataset: &Dataset, target: &str) -> Result<(), ModelError> {
        (**self).try_fit(dataset, target)
    }
    fn try_load_run(&mut self, dataset: &Dataset, run: &str) -> Result<(), ModelError> {
        (**self).try_load_run(dataset, run)
    }
}

/// Names a stack's settings use, which a model can't take.
const RESERVED: [&str; 4] = ["final_estimator", "cv", "stack_method", "passthrough"];

/// What both stacks hold.
#[derive(Debug, Clone)]
struct Stack {
    estimators: Vec<(String, Box<dyn AnyModel>)>,
    final_estimator: Box<dyn AnyModel>,
    cv: Cv,
    stack_method: String,
    passthrough: bool,
    estimators_: Vec<Box<dyn AnyModel>>,
    final_estimator_: Option<Box<dyn AnyModel>>,
    stack_method_: Vec<String>,
    feature_names_in_: Vec<String>,
    target: Option<String>,
}

impl Stack {
    fn new(final_estimator: Box<dyn AnyModel>) -> Self {
        Self {
            estimators: Vec::new(),
            final_estimator,
            cv: Cv::Folds(5),
            stack_method: "auto".into(),
            passthrough: false,
            estimators_: Vec::new(),
            final_estimator_: None,
            stack_method_: Vec::new(),
            feature_names_in_: Vec::new(),
            target: None,
        }
    }

    #[track_caller]
    fn add(&mut self, name: &str, model: Box<dyn AnyModel>) {
        let free: bool = !name.is_empty()
            && !name.contains("__")
            && !RESERVED.contains(&name)
            && self.estimators.iter().all(|(other, _)| other != name);
        check(free, "name", name, "a new name, not empty, without \"__\", and not a setting of the stack");
        self.estimators.push((name.into(), model));
    }

    /// The method each model's columns come from: `stack_method`, or with `"auto"`
    /// `predict_proba` for a classifier and `predict` for a regressor.
    fn method(&self, model: &dyn AnyModel) -> String {
        match self.stack_method.as_str() {
            "auto" if model.is_classifier() => "predict_proba".into(),
            "auto" => "predict".into(),
            other => other.into(),
        }
    }

    fn train(&mut self, dataset: &Dataset, target: &str) -> Result<(), ModelError> {
        if self.estimators.is_empty() {
            return Err(ModelError::MissingSetting {
                setting: "estimators",
                reason: "add one model or more with .add(name, model)",
            });
        }
        self.cv.check()?;
        let reduced: Option<Dataset> = trainable(dataset, target)?;
        let dataset: &Dataset = reduced.as_ref().unwrap_or(dataset);
        let methods: Vec<String> = self.estimators.iter().map(|(_, m)| self.method(m.as_ref())).collect();
        let cv: Cv = self.cv;
        // Each model's out-of-fold columns and its copy trained on every row, all in parallel.
        let trained: Vec<(Dataset, Box<dyn AnyModel>)> = THREAD_POOL.install(|| {
            (self.estimators.par_iter().zip(&methods))
                .map(|((_, model), method)| {
                    let (columns, full) = rayon::join(
                        || oof(model, dataset, target, cv, method),
                        || {
                            let mut full: Box<dyn AnyModel> = model.clone();
                            full.set_run(None);
                            full.try_train(dataset, target).map(|()| full)
                        },
                    );
                    Ok((columns?, full?))
                })
                .collect::<Result<_, ModelError>>()
        })?;
        let (columns, fitted): (Vec<Dataset>, Vec<Box<dyn AnyModel>>) = trained.into_iter().unzip();
        self.feature_names_in_ = dataset.columns().into_iter().filter(|c| c != target).collect();
        let mut meta: DataFrame = self.meta(columns, dataset)?;
        meta.with_column(dataset.column(target)?.clone())?;
        let mut last: Box<dyn AnyModel> = self.final_estimator.clone();
        last.set_run(None);
        last.try_train(&Dataset::new(meta, ""), target)?;
        (self.estimators_, self.final_estimator_, self.stack_method_) = (fitted, Some(last), methods);
        self.target = Some(target.into());
        Ok(())
    }

    /// The final model's input: each model's columns (one for a prediction or the second
    /// class of two, else one per class, named `{model}_{class}`), then with `passthrough`
    /// the features themselves.
    fn meta(&self, outputs: Vec<Dataset>, dataset: &Dataset) -> Result<DataFrame, ModelError> {
        let height: usize = dataset.height();
        let mut columns: Vec<Column> = Vec::new();
        for ((name, _), output) in self.estimators.iter().zip(outputs) {
            let frame: DataFrame = output.into_dataframe();
            let all: &[Column] = frame.columns();
            let kept: &[Column] = if all.len() == 2 { &all[1..] } else { all };
            for column in kept {
                let label: String = if kept.len() == 1 {
                    name.clone()
                } else {
                    format!("{name}_{}", column.name())
                };
                columns.push(column.clone().with_name(label.into()));
            }
        }
        if self.passthrough {
            for feature in &self.feature_names_in_ {
                columns.push(dataset.column(feature)?.clone());
            }
        }
        Ok(DataFrame::new(height, columns)?)
    }

    fn transform(&self, dataset: &Dataset) -> Result<Dataset, ModelError> {
        if self.final_estimator_.is_none() {
            return Err(ModelError::NotFitted { model: "Stacking" });
        }
        let outputs: Vec<Dataset> = THREAD_POOL.install(|| {
            (self.estimators_.par_iter().zip(&self.stack_method_))
                .map(|(model, method)| model.try_predict_method(dataset, method))
                .collect::<Result<_, ModelError>>()
        })?;
        Ok(Dataset::new(self.meta(outputs, dataset)?, ""))
    }

    fn final_output(&self, dataset: &Dataset, method: &str) -> Result<Dataset, ModelError> {
        let meta: Dataset = self.transform(dataset)?;
        let last: &dyn AnyModel = self.final_estimator_.as_deref().ok_or(ModelError::NotFitted { model: "Stacking" })?;
        last.try_predict_method(&meta, method)
    }

    fn params(&self) -> Value {
        let mut params: Map<String, Value> = Map::new();
        for (name, model) in &self.estimators {
            params.insert(name.clone(), json!({ "model": model.name(), "params": model.params() }));
        }
        let last: &dyn AnyModel = self.final_estimator.as_ref();
        params.insert("final_estimator".into(), json!({ "model": last.name(), "params": last.params() }));
        params.insert("cv".into(), serde_json::to_value(self.cv).unwrap_or_default());
        params.insert("stack_method".into(), self.stack_method.clone().into());
        params.insert("passthrough".into(), self.passthrough.into());
        Value::Object(params)
    }

    fn set_params(&mut self, params: Value) -> Result<(), ModelError> {
        for (name, model) in &mut self.estimators {
            if let Some(inner) = params.get(name.as_str()).and_then(|p| p.get("params")) {
                model.try_set_params(inner.clone())?;
            }
        }
        if let Some(inner) = params.get("final_estimator").and_then(|p| p.get("params")) {
            self.final_estimator.try_set_params(inner.clone())?;
        }
        if let Some(cv) = params.get("cv") {
            self.cv = serde_json::from_value(cv.clone())?;
        }
        if let Some(method) = params.get("stack_method") {
            self.stack_method = serde_json::from_value(method.clone())?;
        }
        if let Some(passthrough) = params.get("passthrough") {
            self.passthrough = serde_json::from_value(passthrough.clone())?;
        }
        Ok(())
    }
}

/// The builder methods, `Model` impl and fitted values both stacks share.
macro_rules! stacking_model {
    ($name:ident, $kind:ident, $final_doc:literal) => {
        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl $name {
            /// Adds a model under `name`, like an entry of scikit-learn's `estimators`. The
            /// name labels its columns and its settings (`name__max_depth` in a search).
            #[track_caller]
            pub fn add<M: Model + Clone + Send + Sync + 'static>(mut self, name: &str, model: M) -> Self {
                self.stack.add(name, Box::new(model));
                self
            }

            #[doc = $final_doc]
            pub fn final_estimator<M: Model + Clone + Send + Sync + 'static>(mut self, model: M) -> Self {
                self.stack.final_estimator = Box::new(model);
                self
            }

            /// The folds of the out-of-fold predictions: a count (default 5) or a splitter
            /// such as `KFold::new().shuffle(true)`.
            #[track_caller]
            pub fn cv(mut self, cv: impl Into<Cv>) -> Self {
                let cv: Cv = cv.into();
                cv.check().or_fatal();
                self.stack.cv = cv;
                self
            }

            /// `true` also gives the final model the features themselves. Default `false`.
            pub fn passthrough(mut self, passthrough: bool) -> Self {
                self.stack.passthrough = passthrough;
                self
            }

            /// The models trained on all the rows, in the order they were added; empty before
            /// `fit`.
            pub fn estimators_(&self) -> &[Box<dyn AnyModel>] {
                &self.stack.estimators_
            }

            /// The final model, trained on the out-of-fold columns; `None` before `fit`.
            pub fn final_estimator_(&self) -> Option<&dyn AnyModel> {
                self.stack.final_estimator_.as_deref()
            }

            /// The method each model's columns came from.
            pub fn stack_method_(&self) -> &[String] {
                &self.stack.stack_method_
            }

            /// The feature columns `fit` saw.
            pub fn feature_names_in_(&self) -> &[String] {
                &self.stack.feature_names_in_
            }

            /// What the final model reads for `dataset`: each model's predictions, as columns
            /// named after it, like scikit-learn's `transform`.
            #[track_caller]
            pub fn transform(&self, dataset: &Dataset) -> Dataset {
                self.try_transform(dataset).or_fatal()
            }

            /// [`Self::transform`] that returns the error.
            pub fn try_transform(&self, dataset: &Dataset) -> Result<Dataset, ModelError> {
                self.stack.transform(dataset)
            }

            /// The final model's prediction, named after the target.
            fn predicted(&self, dataset: &Dataset) -> Result<Series, ModelError> {
                let out: Dataset = self.stack.final_output(dataset, "predict")?;
                let target: &str = self.stack.target.as_deref().unwrap_or("");
                Ok(out.into_dataframe().columns()[0].as_materialized_series().clone().with_name(target.into()))
            }
        }

        impl Model for $name {
            /// Every model's out-of-fold predictions and its copy on all the rows, then the
            /// final model on the out-of-fold columns.
            fn try_train(&mut self, dataset: &Dataset, target: &str) -> Result<(), ModelError> {
                self.check_target(dataset, target)?;
                self.stack.train(dataset, target)
            }

            /// A stack is not kept in the model cache: it only trains.
            fn try_fit(&mut self, dataset: &Dataset, target: &str) -> Result<(), ModelError> {
                self.try_train(dataset, target)
            }

            fn params(&self) -> Value {
                self.stack.params()
            }

            fn learned(&self) -> Value {
                json!({ "feature_names_in_": self.stack.feature_names_in_, "stack_method_": self.stack.stack_method_ })
            }

            fn try_load(&mut self, _: Value, _: &[u8], _: &str) -> Result<(), ModelError> {
                Err(bad_learned("a stack is not kept in the model cache".into()))
            }

            fn try_set_params(&mut self, params: Value) -> Result<(), ModelError> {
                self.stack.set_params(params)
            }

            fn try_evaluate(&self, dataset: &Dataset, target: &str, scoring: &str) -> Result<f64, ModelError> {
                paste_kind!($kind, evaluate)(self, dataset, target, scoring)
            }

            fn try_predict_method(&self, dataset: &Dataset, method: &str) -> Result<Dataset, ModelError> {
                paste_kind!($kind, predict)(self, dataset, method)
            }

            fn is_classifier(&self) -> bool {
                stringify!($kind) == "classifier"
            }

            fn run(&self) -> Option<&CachedRun> {
                None
            }

            fn set_run(&mut self, _: Option<CachedRun>) {}

            fn name(&self) -> &'static str {
                stringify!($name)
            }
        }
    };
}

/// `crate::models::{evaluate,predict}_{classifier,regressor}`.
macro_rules! paste_kind {
    (classifier, evaluate) => { crate::models::evaluate_classifier };
    (regressor, evaluate) => { crate::models::evaluate_regressor };
    (classifier, predict) => { crate::models::predict_classifier };
    (regressor, predict) => { crate::models::predict_regressor };
}

/// A classifier that learns from the class probabilities of other models, scikit-learn's
/// `StackingClassifier`: the method behind most Kaggle Playground wins (a logistic regression
/// on many models' out-of-fold predictions).
///
/// Each model added with [`add`](Self::add) gives its out-of-fold `predict_proba` on the
/// training rows ([`cross_val_predict`](crate::models::cross_val_predict) with `cv`, stratified
/// by default); with two classes only the second class's column is kept, as in scikit-learn.
/// The final model ([`LogisticRegression`] by default) trains on those columns. Each model
/// also trains on all the rows, and `predict` uses those copies. Every fold of every model
/// trains in parallel on the Polars pool.
///
/// scikit-learn settings with no setter yet: `stack_method` other than `"auto"` (rusmlai's
/// classifiers have no `decision_function`), `n_jobs` (the Polars pool), `verbose`. A stack is
/// not kept in the model cache.
///
/// ```
/// use rusmlai::prelude::*;
/// let x: Vec<f64> = (0..60).map(|i| i as f64).collect();
/// let z: Vec<f64> = (0..60).map(|i| ((i * 7) % 11) as f64).collect();
/// let y: Vec<&str> = (0..60).map(|i| if i % 3 == 0 || i > 40 { "yes" } else { "no" }).collect();
/// let ds = Dataset::new(df!("x" => x, "z" => z, "y" => y).unwrap(), "");
/// let mut stack = StackingClassifier::new()
///     .add("tree", DecisionTreeClassifier::new().max_depth(3))
///     .add("logistic", LogisticRegression::new())
///     .cv(StratifiedKFold::new().shuffle(true));
/// stack.fit(&ds, "y");
/// assert_eq!(stack.transform(&ds).columns(), ["tree", "logistic"]); // P(yes) of each
/// assert_eq!(stack.classes_, ["no", "yes"]);
/// assert!(stack.score(&ds, "y") > 0.7); // scikit-learn's stack scores 0.733 here too
/// assert_eq!(stack.predict_proba(&ds).columns(), ["no", "yes"]);
/// ```
///
/// Python:
///
/// ```python
/// import numpy as np
/// import pandas as pd
/// from sklearn.ensemble import StackingClassifier
/// from sklearn.linear_model import LogisticRegression
/// from sklearn.model_selection import StratifiedKFold
/// from sklearn.tree import DecisionTreeClassifier
/// i = np.arange(60)
/// ds = pd.DataFrame({"x": i * 1.0, "z": (i * 7 % 11) * 1.0, "y": np.where((i % 3 == 0) | (i > 40), "yes", "no")})
/// stack = StackingClassifier(
///     [("tree", DecisionTreeClassifier(max_depth=3, random_state=42)), ("logistic", LogisticRegression())],
///     cv=StratifiedKFold(5, shuffle=True, random_state=42),
/// ).fit(ds[["x", "z"]], ds["y"])
/// assert stack.transform(ds[["x", "z"]]).shape == (60, 2)
/// assert stack.classes_.tolist() == ["no", "yes"]
/// assert stack.score(ds[["x", "z"]], ds["y"]) > 0.7
/// ```
#[derive(Debug, Clone)]
pub struct StackingClassifier {
    stack: Stack,
    /// The classes of the target, sorted, learned by `fit`.
    pub classes_: Vec<String>,
}

stacking_model!(StackingClassifier, classifier, "The model that learns from the other models' columns, a classifier. Default `LogisticRegression::new()`, as in scikit-learn.");

impl StackingClassifier {
    /// A stack with no models yet and a `LogisticRegression` on top, 5 stratified folds.
    pub fn new() -> Self {
        Self {
            stack: Stack::new(Box::new(LogisticRegression::new())),
            classes_: Vec::new(),
        }
    }

    fn check_target(&mut self, dataset: &Dataset, target: &str) -> Result<(), ModelError> {
        if !self.stack.final_estimator.is_classifier() {
            return Err(ModelError::InvalidSetting {
                setting: "final_estimator",
                value: self.stack.final_estimator.name().into(),
                expected: "a classifier",
            });
        }
        self.classes_ = label_target(dataset, target)?.0;
        Ok(())
    }
}

impl Classifier for StackingClassifier {
    fn try_predict(&self, dataset: &Dataset) -> Result<Series, ModelError> {
        self.predicted(dataset)
    }

    fn try_predict_proba(&self, dataset: &Dataset) -> Result<Dataset, ModelError> {
        self.stack.final_output(dataset, "predict_proba")
    }
}

/// A regressor that learns from the predictions of other models, scikit-learn's
/// `StackingRegressor`.
///
/// Each model added with [`add`](Self::add) gives its out-of-fold predictions on the training
/// rows ([`cross_val_predict`](crate::models::cross_val_predict) with `cv`, 5 folds in row
/// order by default); the final model ([`RidgeCV`] by default, as in scikit-learn) trains on
/// those columns. Each model also trains on all the rows, and `predict` uses those copies.
///
/// scikit-learn settings with no setter yet: `n_jobs` (the Polars pool), `verbose`. A stack
/// is not kept in the model cache.
///
/// ```
/// use rusmlai::prelude::*;
/// let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
/// let y: Vec<f64> = x.iter().map(|x| if *x < 25.0 { 2.0 * x } else { 80.0 - x }).collect();
/// let ds = Dataset::new(df!("x" => x, "y" => y).unwrap(), "");
/// let mut stack = StackingRegressor::new()
///     .add("linear", LinearRegression::new())
///     .add("tree", DecisionTreeRegressor::new().max_depth(4))
///     .cv(KFold::new().shuffle(true));
/// stack.fit(&ds, "y");
/// assert_eq!(stack.transform(&ds).columns(), ["linear", "tree"]);
/// assert!(stack.score(&ds, "y") > 0.95);
/// ```
///
/// Python:
///
/// ```python
/// import numpy as np
/// import pandas as pd
/// from sklearn.ensemble import StackingRegressor
/// from sklearn.linear_model import LinearRegression
/// from sklearn.model_selection import KFold
/// from sklearn.tree import DecisionTreeRegressor
/// x = np.arange(50.0)
/// ds = pd.DataFrame({"x": x, "y": np.where(x < 25, 2 * x, 80 - x)})
/// stack = StackingRegressor(
///     [("linear", LinearRegression()), ("tree", DecisionTreeRegressor(max_depth=4, random_state=42))],
///     cv=KFold(5, shuffle=True, random_state=42),
/// ).fit(ds[["x"]], ds["y"])
/// assert stack.transform(ds[["x"]]).shape == (50, 2)
/// assert stack.score(ds[["x"]], ds["y"]) > 0.95
/// ```
#[derive(Debug, Clone)]
pub struct StackingRegressor {
    stack: Stack,
}

stacking_model!(StackingRegressor, regressor, "The model that learns from the other models' columns, a regressor. Default `RidgeCV::new()`, as in scikit-learn.");

impl StackingRegressor {
    /// A stack with no models yet and a `RidgeCV` on top, 5 folds.
    pub fn new() -> Self {
        Self {
            stack: Stack::new(Box::new(RidgeCV::new())),
        }
    }

    fn check_target(&mut self, _: &Dataset, _: &str) -> Result<(), ModelError> {
        if self.stack.final_estimator.is_classifier() {
            return Err(ModelError::InvalidSetting {
                setting: "final_estimator",
                value: self.stack.final_estimator.name().into(),
                expected: "a regressor",
            });
        }
        Ok(())
    }
}

impl Regressor for StackingRegressor {
    fn try_predict(&self, dataset: &Dataset) -> Result<Series, ModelError> {
        self.predicted(dataset)
    }
}

#[cfg(test)]
mod tests {
    use polars::df;

    use super::*;
    use crate::models::{DecisionTreeClassifier, StratifiedKFold};

    fn toy() -> Dataset {
        let x: Vec<f64> = (0..60).map(|i| i as f64).collect();
        let z: Vec<f64> = (0..60).map(|i| ((i * 7) % 11) as f64).collect();
        let y: Vec<&str> = (0..60).map(|i| if i % 3 == 0 || i > 40 { "yes" } else { "no" }).collect();
        Dataset::new(df!("x" => x, "z" => z, "y" => y).unwrap(), "")
    }

    /// scikit-learn 1.9 on the same rows: score 0.7333, final coefficients
    /// [1.22099503, 1.0920786], intercept -1.05280424 (its Newton stops 2e-4 sooner).
    #[test]
    fn a_classifier_stack_matches_scikit_learn() {
        let ds: Dataset = toy();
        let mut stack = StackingClassifier::new()
            .add("tree", DecisionTreeClassifier::new().max_depth(3))
            .add("logistic", LogisticRegression::new())
            .cv(StratifiedKFold::new().shuffle(true));
        stack.try_train(&ds, "y").unwrap();
        assert!((stack.try_score(&ds, "y").unwrap() - 0.7333333333333333).abs() < 1e-12);
        let last: Value = stack.final_estimator_().unwrap().learned();
        let coef: Vec<f64> = serde_json::from_value(last["coef_"][0].clone()).unwrap();
        assert!((coef[0] - 1.22099503).abs() < 1e-3 && (coef[1] - 1.0920786).abs() < 1e-3, "{coef:?}");
        let meta: Dataset = stack.transform(&ds);
        let want = [(1.0, 0.14822042), (0.28125, 0.1673451), (0.28125, 0.1704119), (0.5, 0.1917553)];
        for (i, (tree, logistic)) in want.iter().enumerate() {
            assert!((meta.to_list("tree")[i] - tree).abs() < 1e-12);
            assert!((meta.to_list("logistic")[i] - logistic).abs() < 1e-6);
        }
    }

    #[test]
    fn a_regressor_stack_matches_scikit_learn() {
        use crate::models::{DecisionTreeRegressor, KFold, LinearRegression};
        let x: Vec<f64> = (0..50).map(f64::from).collect();
        let y: Vec<f64> = x.iter().map(|x| if *x < 25.0 { 2.0 * x } else { 80.0 - x }).collect();
        let ds: Dataset = Dataset::new(df!("x" => x, "y" => y).unwrap(), "");
        let mut stack = StackingRegressor::new()
            .add("linear", LinearRegression::new())
            .add("tree", DecisionTreeRegressor::new().max_depth(4))
            .cv(KFold::new().shuffle(true));
        stack.try_train(&ds, "y").unwrap();
        assert!((stack.try_score(&ds, "y").unwrap() - 0.9662124199003568).abs() < 1e-9);
        let last: Value = stack.final_estimator_().unwrap().learned();
        assert_eq!(last["alpha_"], 10.0);
        let coef: Vec<f64> = serde_json::from_value(last["ridge"]["coef_"].clone()).unwrap();
        assert!((coef[0] + 0.1804999511779109).abs() < 1e-9 && (coef[1] - 1.0521494471724318).abs() < 1e-9);
    }

    #[test]
    fn a_stack_reaches_its_models_settings_and_refuses_bad_ones() {
        let stack = StackingClassifier::new().add("tree", DecisionTreeClassifier::new());
        let mut search = crate::models::GridSearchCV::new(stack)
            .param_grid("tree__max_depth", [1, 3])
            .param_grid("final_estimator__C", [0.1, 1.0])
            .cv(3);
        search.try_train(&toy(), "y").unwrap();
        assert_eq!(search.cv_results_.height(), 4);
        let empty = StackingClassifier::new();
        assert!(matches!(empty.clone().try_train(&toy(), "y"), Err(ModelError::MissingSetting { .. })));
        let wrong = StackingClassifier::new().add("t", DecisionTreeClassifier::new()).final_estimator(RidgeCV::new());
        assert!(matches!(wrong.clone().try_train(&toy(), "y"), Err(ModelError::InvalidSetting { .. })));
    }
}
