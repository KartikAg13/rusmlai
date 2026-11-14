//! [`Pipeline`] and [`ModelStacker`]: preprocessing and a model, fitted and applied together.

use polars::prelude::Series;
use serde_json::Value;

use crate::{
    dataset::Dataset,
    models::{Classifier, Model, ModelError, Regressor},
    preprocess::{ColumnTransformer, Transform},
};

/// The model part of a [`Pipeline`]. In v1 it holds one model and passes every call to it.
/// Later it stacks several models, like scikit-learn's `StackingRegressor`, with the same API.
///
/// ```
/// use rusmlai::prelude::*;
/// let model = ModelStacker::new(DummyRegressor::new());
/// ```
#[derive(Debug, Clone)]
pub struct ModelStacker<M> {
    /// The model.
    pub model: M,
}

impl<M> ModelStacker<M> {
    /// Wraps one model.
    pub fn new(model: M) -> Self {
        Self { model }
    }
}

/// Every [`Model`] method that a wrapper passes to its model `$m` unchanged.
macro_rules! pass_to_model {
    ($m:ident) => {
        fn params(&self) -> Value {
            self.$m.params()
        }

        fn learned(&self) -> Value {
            self.$m.learned()
        }

        fn try_load(&mut self, learned: Value, target: &str) -> Result<(), ModelError> {
            self.$m.try_load(learned, target)
        }

        fn try_set_params(&mut self, params: Value) -> Result<(), ModelError> {
            self.$m.try_set_params(params)
        }

        fn run(&self) -> Option<&str> {
            self.$m.run()
        }

        fn set_run(&mut self, run: Option<String>) {
            self.$m.set_run(run)
        }

        fn name(&self) -> &'static str {
            self.$m.name()
        }

        fn loss_curve(&self) -> Option<Vec<f64>> {
            self.$m.loss_curve()
        }

        fn validation_loss_curve(&self) -> Option<Vec<f64>> {
            self.$m.validation_loss_curve()
        }
    };
}

impl<M: Model> Model for ModelStacker<M> {
    fn try_train(&mut self, dataset: &Dataset, target: &str) -> Result<(), ModelError> {
        self.model.try_train(dataset, target)
    }

    /// The model's own fit, so the cache keys on the model, not on this wrapper.
    fn try_fit(&mut self, dataset: &Dataset, target: &str) -> Result<(), ModelError> {
        self.model.try_fit(dataset, target)
    }

    fn try_load_run(&mut self, dataset: &Dataset, run: &str) -> Result<(), ModelError> {
        self.model.try_load_run(dataset, run)
    }

    pass_to_model!(model);
}

impl<M: Regressor> Regressor for ModelStacker<M> {
    fn try_predict(&self, dataset: &Dataset) -> Result<Series, ModelError> {
        self.model.try_predict(dataset)
    }

    fn try_score(&self, dataset: &Dataset, target: &str) -> Result<f64, ModelError> {
        self.model.try_score(dataset, target)
    }
}

impl<M: Classifier> Classifier for ModelStacker<M> {
    fn try_predict(&self, dataset: &Dataset) -> Result<Series, ModelError> {
        self.model.try_predict(dataset)
    }

    fn try_predict_proba(&self, dataset: &Dataset) -> Result<Dataset, ModelError> {
        self.model.try_predict_proba(dataset)
    }

    fn try_score(&self, dataset: &Dataset, target: &str) -> Result<f64, ModelError> {
        self.model.try_score(dataset, target)
    }
}

/// A [`ColumnTransformer`] and a [`ModelStacker`], like scikit-learn's `Pipeline`.
///
/// - `fit(&train, target)` fits the preprocessing on every column but the target, then fits
///   the model on the features and the target.
/// - `predict(&test)` transforms `test` with what `fit` learned, then predicts. A target column
///   in `test` is ignored, and so is any other column that `fit` didn't see.
/// - `score(&test, target)` is R² for a regressor.
///
/// A `Pipeline` is the same kind of model as its `ModelStacker`, so it is a [`Regressor`] or a
/// [`Classifier`] when the model is one.
///
/// ```
/// use polars::df;
/// use rusmlai::{prelude::*, steps};
/// let ds = Dataset::new(
///     df!(
///         "hours" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
///         "course" => ["a", "b", "a", "b", "a", "b", "a", "b"],
///         "score" => [50.0, 55.0, 60.0, 62.0, 70.0, 71.0, 80.0, 85.0],
///     )
///     .unwrap(),
///     "",
/// );
/// let (train, test) = train_test_split(&ds, 0.25, 42, "");
/// let prep = ColumnTransformer::new()
///     .add("num", &["hours"], steps![StandardScaler::new()])
///     .add("cat", &["course"], steps![OneHotEncoder::new()]);
/// let mut pipe = Pipeline::new(prep, ModelStacker::new(DummyRegressor::new()));
/// pipe.fit(&train, "score");
/// assert_eq!(pipe.predict(&test).len(), 2);
/// let r2: f64 = pipe.score(&test, "score"); // one number for every row scores at most 0
/// assert!(r2 <= 0.0);
/// ```
///
/// Python:
///
/// ```python
/// import pandas as pd
/// from sklearn.compose import ColumnTransformer
/// from sklearn.dummy import DummyRegressor
/// from sklearn.model_selection import train_test_split
/// from sklearn.pipeline import Pipeline
/// from sklearn.preprocessing import OneHotEncoder, StandardScaler
/// ds = pd.DataFrame({
///     "hours": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
///     "course": ["a", "b", "a", "b", "a", "b", "a", "b"],
///     "score": [50.0, 55.0, 60.0, 62.0, 70.0, 71.0, 80.0, 85.0],
/// })
/// train, test = train_test_split(ds, test_size=0.25, random_state=42)
/// prep = ColumnTransformer([("num", StandardScaler(), ["hours"]), ("cat", OneHotEncoder(), ["course"])])
/// pipe = Pipeline([("prep", prep), ("model", DummyRegressor())])
/// pipe.fit(train.drop(columns="score"), train["score"])
/// assert len(pipe.predict(test.drop(columns="score"))) == 2
/// assert pipe.score(test.drop(columns="score"), test["score"]) <= 0.0
/// ```
#[derive(Debug)]
pub struct Pipeline<M> {
    /// The preprocessing, `named_steps["prep"]` in scikit-learn.
    pub prep: ColumnTransformer,
    /// The model, `named_steps["model"]` in scikit-learn.
    pub model: ModelStacker<M>,
}

impl<M> Pipeline<M> {
    /// Preprocessing, then the model.
    pub fn new(prep: ColumnTransformer, model: ModelStacker<M>) -> Self {
        Self { prep, model }
    }
}

impl<M: Model> Pipeline<M> {
    /// Fits the preprocessing on every column but the target, and gives the model's input: the
    /// features and the target.
    fn fit_prep(&mut self, dataset: &Dataset, target: &str) -> Result<Dataset, ModelError> {
        // The target never goes into the preprocessing, so `&[]` never selects it.
        let x: Dataset = dataset.try_drop(&[target])?;
        let features: Dataset = self.prep.try_fit_transform(&x)?;
        Ok(features.try_with_column(dataset, target)?)
    }

    /// The model's input for `score`: the preprocessed rows and the target.
    fn scored(&self, dataset: &Dataset, target: &str) -> Result<Dataset, ModelError> {
        let features: Dataset = self.prep.try_transform(dataset)?;
        Ok(features.try_with_column(dataset, target)?)
    }
}

impl<M: Model> Model for Pipeline<M> {
    fn try_train(&mut self, dataset: &Dataset, target: &str) -> Result<(), ModelError> {
        let input: Dataset = self.fit_prep(dataset, target)?;
        self.model.try_train(&input, target)
    }

    /// The model's fit goes through the cache with the preprocessed data, so the same
    /// preprocessing and model on the same rows load instead of training.
    fn try_fit(&mut self, dataset: &Dataset, target: &str) -> Result<(), ModelError> {
        let input: Dataset = self.fit_prep(dataset, target)?;
        self.model.try_fit(&input, target)
    }

    /// Fits the preprocessing on `dataset` (from the cached stats, so it costs little), then
    /// loads the run's model, which trained on the preprocessed columns.
    fn try_load_run(&mut self, dataset: &Dataset, run: &str) -> Result<(), ModelError> {
        let record = crate::metadata::models::find_run(dataset, run)?;
        let input: Dataset = self.fit_prep(dataset, &record.target)?;
        self.model.try_load_run(&input, run)
    }

    pass_to_model!(model);
}

impl<M: Regressor> Regressor for Pipeline<M> {
    fn try_predict(&self, dataset: &Dataset) -> Result<Series, ModelError> {
        let features: Dataset = self.prep.try_transform(dataset)?;
        self.model.try_predict(&features)
    }

    /// Scores the model on the preprocessed rows, so a score on the training rows is recorded
    /// as one.
    fn try_score(&self, dataset: &Dataset, target: &str) -> Result<f64, ModelError> {
        self.model.try_score(&self.scored(dataset, target)?, target)
    }
}

impl<M: Classifier> Classifier for Pipeline<M> {
    fn try_predict(&self, dataset: &Dataset) -> Result<Series, ModelError> {
        let features: Dataset = self.prep.try_transform(dataset)?;
        self.model.try_predict(&features)
    }

    fn try_predict_proba(&self, dataset: &Dataset) -> Result<Dataset, ModelError> {
        let features: Dataset = self.prep.try_transform(dataset)?;
        self.model.try_predict_proba(&features)
    }

    /// Scores the model on the preprocessed rows, like the regressor's `score`.
    fn try_score(&self, dataset: &Dataset, target: &str) -> Result<f64, ModelError> {
        self.model.try_score(&self.scored(dataset, target)?, target)
    }
}

#[cfg(test)]
mod tests {
    use polars::df;

    use super::*;
    use crate::{metrics::try_r2_score, models::DummyRegressor, preprocess::StandardScaler, steps};

    #[test]
    fn fit_learns_from_train_rows_and_predict_ignores_the_target() {
        let train: Dataset = Dataset::new(
            df!("x" => [1.0, 2.0, 3.0], "y" => [1.0, 2.0, 6.0]).unwrap(),
            "",
        );
        let test: Dataset = Dataset::new(df!("x" => [4.0, 5.0], "y" => [3.0, 7.0]).unwrap(), "");
        let prep: ColumnTransformer =
            ColumnTransformer::new().add("num", &[], steps![StandardScaler::new()]);
        let mut pipe = Pipeline::new(prep, ModelStacker::new(DummyRegressor::new()));
        pipe.fit(&train, "y");
        // `&[]` read `x` only, at fit and at predict, though `test` has `y` too.
        assert_eq!(pipe.prep.get_feature_names_out(), ["num__x"]);
        let predicted: Series = pipe.predict(&test);
        assert_eq!(predicted.name().as_str(), "y");
        assert_eq!(
            predicted
                .f64()
                .unwrap()
                .into_no_null_iter()
                .collect::<Vec<f64>>(),
            [3.0, 3.0]
        );
        let y: &Series = test.dataframe.column("y").unwrap().as_materialized_series();
        assert_eq!(pipe.score(&test, "y"), try_r2_score(y, &predicted).unwrap());
    }

    #[test]
    fn errors_name_the_problem() {
        let ds: Dataset = Dataset::new(df!("x" => [1.0, 2.0], "y" => [1.0, 2.0]).unwrap(), "");
        let prep: ColumnTransformer =
            ColumnTransformer::new().add("num", &[], steps![StandardScaler::new()]);
        let mut pipe = Pipeline::new(prep, ModelStacker::new(DummyRegressor::new()));
        assert!(pipe.try_predict(&ds).is_err());
        assert!(matches!(
            pipe.try_fit(&ds, "nope"),
            Err(ModelError::Dataset(_))
        ));
    }
}
