//! Every model: [`Model`], the trait they share, [`Regressor`] and [`Classifier`] on top of it,
//! [`ModelError`], and one file per model family ([`dummy`], [`linear`], [`logistic`]).

use std::fmt::Debug;

pub(crate) mod design;
pub mod dummy;
pub mod linear;
pub mod logistic;

pub use dummy::DummyRegressor;
pub use linear::LinearRegression;
pub use logistic::LogisticRegression;

use indexmap::IndexMap;
use polars::{
    error::PolarsError,
    prelude::{DataType, NamedFrom, Series},
};
use serde_json::Value;
use thiserror::Error;

use crate::{
    dataset::{Dataset, error::DatasetError},
    error::OrFatal,
    metadata::{dataset::stats::NumericStats, models},
    metrics::{try_accuracy_score, try_log_loss, try_regression_scores},
    preprocess::{LabelEncoder, Transform, error::PreprocessError},
};

/// What can fail while fitting, predicting or scoring a model.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ModelError {
    /// A column lookup failed, for example a text target.
    #[error(transparent)]
    Dataset(#[from] DatasetError),

    /// The preprocessing of a [`Pipeline`](crate::pipeline::Pipeline) failed.
    #[error(transparent)]
    Preprocess(#[from] PreprocessError),

    /// A Polars query failed.
    #[error(transparent)]
    Polars(#[from] PolarsError),

    /// `predict` or `score` ran before `fit`.
    #[error("{model} is not fitted: call fit first")]
    NotFitted {
        /// The model's name.
        model: &'static str,
    },

    /// The target has nulls, so a model can't learn from those rows.
    #[error(
        "target '{column}' has {nulls} nulls; drop those rows or fill them before fit, like scikit-learn"
    )]
    TargetNulls {
        /// The target column.
        column: String,
        /// How many nulls.
        nulls: usize,
    },

    /// A setting that another setting needs is missing.
    #[error("{setting} must be set: {reason}")]
    MissingSetting {
        /// The setting.
        setting: &'static str,
        /// Why it is needed.
        reason: &'static str,
    },

    /// The model got no feature column.
    #[error("the dataset has no feature columns besides the target '{target}'")]
    NoFeatures {
        /// The target column.
        target: String,
    },

    /// A feature is text (or another type that is not a number).
    #[error(
        "feature '{column}' is {dtype}, not a number: encode it with OneHotEncoder or OrdinalEncoder in the ColumnTransformer"
    )]
    NotNumeric {
        /// The column.
        column: String,
        /// Its dtype.
        dtype: String,
    },

    /// A feature has nulls.
    #[error(
        "feature '{column}' has {nulls} nulls: fill them with SimpleImputer in the ColumnTransformer"
    )]
    FeatureNulls {
        /// The column.
        column: String,
        /// How many nulls.
        nulls: usize,
    },

    /// A column has NaN or infinite values.
    #[error("column '{column}' has NaN or infinite values; a model needs finite numbers")]
    NotFinite {
        /// The column.
        column: String,
    },

    /// A setting got a value it doesn't take.
    #[error("{setting} = {value} is not valid; use {expected}")]
    InvalidSetting {
        /// The setting.
        setting: &'static str,
        /// The value given.
        value: String,
        /// What it takes.
        expected: &'static str,
    },

    /// A classifier's target has one class only, so there is nothing to tell apart.
    #[error("target '{column}' has one class only ({class}); a classifier needs two or more")]
    OneClass {
        /// The target column.
        column: String,
        /// The class.
        class: String,
    },

    /// A label that is not one of the classes, for example in `log_loss`.
    #[error("'{class}' is not one of the classes {classes:?}")]
    UnknownClass {
        /// The label.
        class: String,
        /// The classes.
        classes: Vec<String>,
    },

    /// The linear solve failed.
    #[error("the eigendecomposition of XᵀX did not converge")]
    Solve,

    /// A metric got true values and predictions of different lengths.
    #[error("y_true has {y_true} values and y_pred has {y_pred}; they must have the same length")]
    LengthMismatch {
        /// Values in `y_true`.
        y_true: usize,
        /// Values in `y_pred`.
        y_pred: usize,
    },

    /// A metric got nulls, which have no error to measure.
    #[error("{input} has {nulls} nulls; a metric needs a value in every row")]
    MetricNulls {
        /// `y_true` or `y_pred`.
        input: &'static str,
        /// How many nulls.
        nulls: usize,
    },

    /// A metric got no values.
    #[error("y_true and y_pred are empty")]
    Empty,

    /// The dataset was made in memory, so it has no CSV next to which a model cache or a
    /// journal could live.
    #[error("this dataset has no CSV, so it has no model cache: load it with read_csv")]
    NoCache,

    /// A file of the model cache or the journal couldn't be written.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// `load_run` got a run that the model cache doesn't have.
    #[error(
        "run '{run}' is not in the model cache, which has {runs} runs (r01 to r{runs:02}); open ds.journal() to see them"
    )]
    RunNotFound {
        /// The run asked for.
        run: String,
        /// How many runs the cache has.
        runs: usize,
    },

    /// `load_run` got a run that this model or this data can't take.
    #[error("run '{run}' can't be loaded here: {reason}")]
    RunMismatch {
        /// The run asked for.
        run: String,
        /// Why not.
        reason: String,
    },

    /// Learned values that don't fit this model, for example from a cache line of an older
    /// version.
    #[error("these learned values don't fit the model: {0}")]
    Learned(#[from] serde_json::Error),
}

/// What every model shares: training, the settings and learned values that the model cache
/// ([`ModelCache`]) stores, and `fit` / `load_run` through that cache. [`Regressor`] and
/// [`Classifier`] add `predict` and `score`.
///
/// On a dataset that came from a CSV, `fit` goes through the model cache: a fit with the same
/// model, settings, features, target and data loads what it learned instead of training, and
/// every fit and score is a line of the journal (`Dataset::journal`).
///
/// A model implements `try_train` (its own training) and the methods the cache needs:
/// `params`, `learned`, `try_load`, `try_set_params`, and `run` / `set_run`.
///
/// [`ModelCache`]: crate::metadata::models::ModelCache
pub trait Model: Debug {
    /// Trains on `dataset`, where `target` is the column to predict, without the cache.
    fn try_train(&mut self, dataset: &Dataset, target: &str) -> Result<(), ModelError>;

    /// The model's settings, by scikit-learn name: part of the cache key, and the journal's
    /// hyperparameter columns.
    fn params(&self) -> Value;

    /// What `fit` learned, by scikit-learn name (`coef_`, `intercept_`, ...).
    fn learned(&self) -> Value;

    /// Puts back what [`Model::learned`] gave, as if `fit` had run on `target`.
    fn try_load(&mut self, learned: Value, target: &str) -> Result<(), ModelError>;

    /// Puts back settings that [`Model::params`] gave.
    fn try_set_params(&mut self, params: Value) -> Result<(), ModelError>;

    /// The cache id of the last fit; `None` before `fit` or without a cache.
    fn run(&self) -> Option<&str>;

    /// Sets the cache id of the last fit.
    fn set_run(&mut self, run: Option<String>);

    /// The model's name, scikit-learn's class name.
    fn name(&self) -> &'static str {
        let full: &'static str = std::any::type_name::<Self>();
        full.rsplit("::").next().unwrap_or(full)
    }

    /// The training loss per round, for models that train in rounds.
    fn loss_curve(&self) -> Option<Vec<f64>> {
        None
    }

    /// The validation loss per round, for models that hold rows out while they train.
    fn validation_loss_curve(&self) -> Option<Vec<f64>> {
        None
    }

    /// Learns from `dataset`, where `target` is the column to predict, through the model cache.
    fn try_fit(&mut self, dataset: &Dataset, target: &str) -> Result<(), ModelError> {
        models::fit(self, dataset, target)
    }

    /// Makes this model the one of an earlier run of the journal: its settings and what it
    /// learned, read from the model cache of `dataset`'s CSV. `run` is the journal's label
    /// (`"r03"`) or the run's id. `dataset` must have the features the run trained on.
    fn try_load_run(&mut self, dataset: &Dataset, run: &str) -> Result<(), ModelError> {
        models::load_run(self, dataset, run)
    }

    /// [`Model::try_load_run`] that panics with the error's message at your line.
    ///
    /// ```no_run
    /// use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/s6e1/train.csv").drop(&["id"]);
    /// let (train, test) = train_test_split(&ds, 0.2, 42, "");
    /// let mut model = LinearRegression::new();
    /// model.load_run(&train, "r03"); // the settings and weights of run r03 in ds.journal()
    /// let predicted = model.predict(&test);
    /// ```
    ///
    /// Python has no model journal; the closest is reloading a pickled model:
    ///
    /// ```python
    /// # model = joblib.load("models/r03.joblib")
    /// ```
    #[track_caller]
    fn load_run(&mut self, dataset: &Dataset, run: &str) {
        self.try_load_run(dataset, run).or_fatal()
    }

    /// Learns from `dataset`. Panics with the error's message at your line.
    #[track_caller]
    fn fit(&mut self, dataset: &Dataset, target: &str) {
        self.try_fit(dataset, target).or_fatal()
    }
}

/// A model that predicts a number, like scikit-learn's regressors.
///
/// `fit` takes the whole dataset and the name of the target column: every other column is a
/// feature. `predict` gives one value per row, and `score` is R² ([`r2_score`]). The cache
/// records each score with MSE, RMSE and MAE.
///
/// [`r2_score`]: crate::metrics::r2_score
pub trait Regressor: Model {
    /// One prediction per row of `dataset`, as a `Float64` series named after the target.
    fn try_predict(&self, dataset: &Dataset) -> Result<Series, ModelError>;

    /// R² of the predictions against the `target` column of `dataset`. The cache records it
    /// with MSE, RMSE and MAE, for the journal.
    fn try_score(&self, dataset: &Dataset, target: &str) -> Result<f64, ModelError> {
        let predicted: Series = self.try_predict(dataset)?;
        let y: &Series = dataset.column(target)?.as_materialized_series();
        let (r2, mse, mae): (f64, f64, f64) = try_regression_scores(y, &predicted)?;
        let metrics: IndexMap<String, f64> = IndexMap::from([
            ("r2".to_string(), r2),
            ("mse".to_string(), mse),
            ("rmse".to_string(), mse.sqrt()),
            ("mae".to_string(), mae),
        ]);
        models::score(self, dataset, metrics);
        Ok(r2)
    }

    /// One prediction per row. Panics with the error's message at your line.
    #[track_caller]
    fn predict(&self, dataset: &Dataset) -> Series {
        self.try_predict(dataset).or_fatal()
    }

    /// R² on `dataset`. Panics with the error's message at your line.
    #[track_caller]
    fn score(&self, dataset: &Dataset, target: &str) -> f64 {
        self.try_score(dataset, target).or_fatal()
    }
}

/// A model that predicts a class, like scikit-learn's classifiers.
///
/// `fit` takes the target column as it is: text, integers or booleans. The classes are its
/// distinct values, sorted like [`LabelEncoder`](crate::preprocess::LabelEncoder) sorts them
/// (`classes_`). `predict` gives one class per row, in the target's own type, `predict_proba`
/// one probability column per class, named after the class, and `score` is the accuracy
/// ([`accuracy_score`]). The cache records each score with the log loss ([`log_loss`]).
///
/// [`accuracy_score`]: crate::metrics::accuracy_score
/// [`log_loss`]: crate::metrics::log_loss
pub trait Classifier: Model {
    /// One class per row of `dataset`, as a series of the target's type, named after it.
    fn try_predict(&self, dataset: &Dataset) -> Result<Series, ModelError>;

    /// The probability of each class for each row: one `Float64` column per class, named after
    /// the class, in `classes_` order. Each row sums to 1.
    fn try_predict_proba(&self, dataset: &Dataset) -> Result<Dataset, ModelError>;

    /// The share of rows whose `target` the model predicts. The cache records it with the log
    /// loss, for the journal.
    fn try_score(&self, dataset: &Dataset, target: &str) -> Result<f64, ModelError> {
        let y: &Series = dataset.column(target)?.as_materialized_series();
        let accuracy: f64 = try_accuracy_score(y, &self.try_predict(dataset)?)?;
        let metrics: IndexMap<String, f64> = IndexMap::from([
            ("accuracy".to_string(), accuracy),
            (
                "log_loss".to_string(),
                try_log_loss(y, &self.try_predict_proba(dataset)?)?,
            ),
        ]);
        models::score(self, dataset, metrics);
        Ok(accuracy)
    }

    /// One class per row. Panics with the error's message at your line.
    #[track_caller]
    fn predict(&self, dataset: &Dataset) -> Series {
        self.try_predict(dataset).or_fatal()
    }

    /// One probability column per class. Panics with the error's message at your line.
    #[track_caller]
    fn predict_proba(&self, dataset: &Dataset) -> Dataset {
        self.try_predict_proba(dataset).or_fatal()
    }

    /// The accuracy on `dataset`. Panics with the error's message at your line.
    #[track_caller]
    fn score(&self, dataset: &Dataset, target: &str) -> f64 {
        self.try_score(dataset, target).or_fatal()
    }
}

/// The cached stats of a numeric target with no nulls, the check every regressor's `fit` makes.
pub(crate) fn numeric_target<'a>(
    dataset: &'a Dataset,
    target: &str,
) -> Result<&'a NumericStats, ModelError> {
    let nulls: usize = dataset.try_column_info(target)?.null_count;
    if nulls > 0 {
        return Err(ModelError::TargetNulls {
            column: target.into(),
            nulls,
        });
    }
    Ok(dataset.try_numeric_stats(target)?)
}

/// The classes of a classifier's target with no nulls, sorted like [`LabelEncoder`], and the
/// target as their codes 0, 1, 2, ... in `Float64`: the check every classifier's `fit` makes.
pub(crate) fn label_target(
    dataset: &Dataset,
    target: &str,
) -> Result<(Vec<String>, Series), ModelError> {
    let nulls: usize = dataset.try_column_info(target)?.null_count;
    if nulls > 0 {
        return Err(ModelError::TargetNulls {
            column: target.into(),
            nulls,
        });
    }
    let y: Dataset = dataset.try_select(&[target])?;
    let mut encoder: LabelEncoder = LabelEncoder::new();
    encoder.try_fit(&y)?;
    if encoder.classes_.len() < 2 {
        return Err(ModelError::OneClass {
            column: target.into(),
            class: encoder.classes_.first().cloned().unwrap_or_default(),
        });
    }
    let codes: Series = encoder
        .try_transform(&y)?
        .column(target)?
        .as_materialized_series()
        .cast(&DataType::Float64)?
        .rechunk();
    Ok((encoder.classes_, codes))
}

/// `classes` as a series of the target's type `dtype`: what a classifier's `predict` returns.
/// Polars can't cast text to `Boolean`, so booleans are read by hand.
pub(crate) fn class_labels(
    name: &str,
    classes: &[String],
    dtype: &DataType,
) -> Result<Series, ModelError> {
    if dtype.is_bool() {
        let values: Vec<bool> = classes.iter().map(|c| c == "true").collect();
        return Ok(Series::new(name.into(), values));
    }
    Ok(Series::new(name.into(), classes).cast(dtype)?)
}
