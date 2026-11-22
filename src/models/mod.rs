//! Every model: [`Model`], the trait they share, [`Regressor`] and [`Classifier`] on top of it,
//! [`ModelError`], and one file per model family ([`dummy`], [`linear`], [`logistic`], [`tree`], [`forest`], [`bagging`]).

use std::fmt::Debug;

pub mod adaboost;
pub mod bagging;
pub mod boosting;
pub mod catboost;
pub(crate) mod design;
pub mod dummy;
pub(crate) mod export;
pub mod folds;
pub mod forest;
pub mod hill;
pub(crate) mod hist;
pub mod search;
pub mod stacking;
pub mod summary;
pub mod lightgbm;
pub mod linear;
pub mod persist;
pub mod logistic;
pub mod tree;
pub mod xgboost;

pub use adaboost::AdaBoostClassifier;
pub use bagging::{BaggingClassifier, BaggingRegressor};
pub use search::{GridSearchCV, cross_val_score, try_cross_val_score};
pub use summary::OlsSummary;
pub use stacking::{AnyModel, StackingClassifier, StackingRegressor};
pub use boosting::{GradientBoostingClassifier, GradientBoostingRegressor};
pub use catboost::{CatBoostClassifier, CatBoostRegressor};
pub use dummy::DummyRegressor;
pub use folds::{Cv, KFold, StratifiedKFold, cross_val_predict, try_cross_val_predict};
pub use hill::HillClimbing;
pub use forest::{ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor};
pub use lightgbm::{LGBMClassifier, LGBMRegressor};
pub use linear::{ElasticNet, Lasso, LinearRegression, Ridge, RidgeCV};
pub use logistic::LogisticRegression;
pub use tree::{
    ClassWeight, DecisionTreeClassifier, DecisionTreeRegressor, compute_sample_weight,
    try_compute_sample_weight,
};
pub use xgboost::{XGBClassifier, XGBRegressor};

use indexmap::IndexMap;
use polars::{
    error::PolarsError,
    prelude::{DataFrame, DataType, IntoLazy, NamedFrom, Series, lit},
};
use serde_json::Value;
use thiserror::Error;

use crate::{
    dataset::{Dataset, error::DatasetError},
    error::{OrFatal, warn},
    metadata::{
        dataset::stats::NumericStats,
        models::{self, CachedRun},
    },
    metrics::{
        try_accuracy_score, try_balanced_accuracy_score, try_log_loss, try_regression_scores,
        try_roc_auc_score,
    },
    preprocess::{LabelEncoder, Transform, error::PreprocessError},
    utils::col,
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

    /// The rows of a fold could not be taken.
    #[error(transparent)]
    Split(#[from] crate::dataset::split::SplitError),

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
        "target '{column}' has {nulls} {}; drop those rows or fill them before fit, like scikit-learn",
        nulls_word(*.nulls)
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
        "feature '{column}' has {nulls} {}: fill them with SimpleImputer in the ColumnTransformer",
        nulls_word(*.nulls)
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

    /// A categorical feature has more categories than a split can tell apart.
    #[error(
        "categorical feature '{column}' has {categories} categories; at most max_bins ({max_bins}) fit: group the rare ones, or leave the column out of categorical_features"
    )]
    TooManyCategories {
        /// The column.
        column: String,
        /// Its categories (at least).
        categories: usize,
        /// The most it may have.
        max_bins: usize,
    },

    /// A categorical feature was text at `fit` and is not now, or the other way around.
    #[error("categorical feature '{column}' is {dtype} now and was {was} at fit")]
    CategoryType {
        /// The column.
        column: String,
        /// Its dtype now.
        dtype: String,
        /// "text" or "numbers".
        was: &'static str,
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
    #[error(
        "target '{column}' has continuous values (like {value}), not classes: use a regressor such as LinearRegression, or cut it into classes first, like scikit-learn"
    )]
    ContinuousTarget { column: String, value: f64 },

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
    #[error("{input} has {nulls} {}; a metric needs a value in every row", nulls_word(*.nulls))]
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
    #[error(
        "there is no model cache: this dataset has no CSV (load it with read_csv), or the cache is off (set_cache)"
    )]
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
/// `params`, `learned`, `try_load`, `try_set_params`, and `run` / `set_run`; a tree model also
/// `write_weights`.
///
/// [`ModelCache`]: crate::metadata::models::ModelCache
pub trait Model: Debug {
    /// Trains on `dataset`, where `target` is the column to predict, without the cache.
    fn try_train(&mut self, dataset: &Dataset, target: &str) -> Result<(), ModelError>;

    /// [`Model::try_train`] where row `i` counts `weights[i]` times (0 leaves it out), like
    /// scikit-learn's `sample_weight`; `None` when the model takes no weights, which is the
    /// default. `Bagging*` gives a bootstrap sample this way to the models that take it, the
    /// trees, and copies the rows for the others.
    fn try_train_weighted(
        &mut self,
        _dataset: &Dataset,
        _target: &str,
        _weights: &[f32],
    ) -> Option<Result<(), ModelError>> {
        None
    }

    /// The model's settings, by scikit-learn name: part of the cache key, and the journal's
    /// hyperparameter columns.
    fn params(&self) -> Value;

    /// What `fit` learned, by scikit-learn name (`coef_`, `intercept_`, ...).
    fn learned(&self) -> Value;

    /// Writes what `fit` learned that is too large for JSON, as bytes: the nodes of a tree
    /// model. The cache keeps them in a binary file next to [`Model::learned`]. Most models
    /// write nothing.
    fn write_weights(&self, _out: &mut dyn std::io::Write) -> std::io::Result<()> {
        Ok(())
    }

    /// Puts back what [`Model::learned`] gave and the bytes [`Model::write_weights`] wrote
    /// (empty for most models), as if `fit` had run on `target`.
    fn try_load(&mut self, learned: Value, weights: &[u8], target: &str) -> Result<(), ModelError>;

    /// Puts back settings that [`Model::params`] gave.
    fn try_set_params(&mut self, params: Value) -> Result<(), ModelError>;

    /// The score of the fitted model on `dataset` by scikit-learn's scorer `scoring`: higher
    /// is better. `""` is the model's own score, R² or the accuracy. A regressor takes
    /// `"r2"`, `"neg_mean_squared_error"`, `"neg_root_mean_squared_error"` and
    /// `"neg_mean_absolute_error"`, a classifier `"accuracy"`, `"balanced_accuracy"`,
    /// `"roc_auc"` (a binary target) and `"neg_log_loss"`. Unlike
    /// `score`, it records nothing in the model cache: a search scores many models that are
    /// never kept.
    fn try_evaluate(&self, dataset: &Dataset, target: &str, scoring: &str) -> Result<f64, ModelError>;

    /// The fitted model's predictions by scikit-learn's method name, as a table: `"predict"`
    /// gives one column (named after the target), and a classifier's `"predict_proba"` one
    /// column per class. What `cross_val_predict` and the stacking models call, since they
    /// hold models of any kind.
    fn try_predict_method(&self, dataset: &Dataset, method: &str) -> Result<Dataset, ModelError>;

    /// Whether the model predicts classes. A search then gives each fold the same share of
    /// each class, like scikit-learn's `StratifiedKFold`.
    fn is_classifier(&self) -> bool {
        false
    }

    /// The cache record of the last fit (its id and the cache that holds it); `None` before
    /// `fit` or without a cache.
    fn run(&self) -> Option<&CachedRun>;

    /// Sets the cache record of the last fit.
    fn set_run(&mut self, run: Option<CachedRun>);

    /// The model's name, scikit-learn's class name.
    fn name(&self) -> &'static str {
        let full: &'static str = std::any::type_name::<Self>();
        full.rsplit("::").next().unwrap_or(full)
    }

    /// The training loss per round, for models that train in rounds.
    fn loss_curve(&self) -> Option<Vec<f64>> {
        None
    }

    /// The loss per round on the rows the model held out of training, for models that stop
    /// early on them (gradient boosting). The journal draws it next to the training loss.
    fn validation_curve(&self) -> Option<Vec<f64>> {
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
    /// Python has no model journal. The closest is to save each model yourself with `joblib.dump`
    /// and read it back with `joblib.load`.
    #[track_caller]
    fn load_run(&mut self, dataset: &Dataset, run: &str) {
        self.try_load_run(dataset, run).or_fatal()
    }

    /// Learns from `dataset`. Panics with the error's message at your line.
    #[track_caller]
    fn fit(&mut self, dataset: &Dataset, target: &str) {
        self.try_fit(dataset, target).or_fatal()
    }

    /// The target column `fit` saw; `None` before `fit`.
    fn target(&self) -> Option<&str> {
        None
    }

    /// What a saved file holds besides the model: a `Pipeline`'s fitted preprocessing.
    fn prep_state(&self) -> Result<Option<Value>, ModelError> {
        Ok(None)
    }

    /// Puts back what [`Model::prep_state`] gave.
    fn try_restore_prep(&mut self, _state: Value) -> Result<(), ModelError> {
        Ok(())
    }

    /// Saves the fitted model to one file at `path`: its settings, what it learned and its
    /// target, and for a `Pipeline` its fitted preprocessing. Read it back with
    /// [`Model::load`]. The file holds numbers and names, never code. Panics with the error's
    /// message at your line: not fitted, or the file can't be written.
    ///
    /// ```
    /// use rusmlai::prelude::*;
    /// let ds = Dataset::new(df!("x" => [1.0, 2.0, 3.0, 4.0], "y" => [2.0, 4.1, 5.9, 8.0]).unwrap(), "");
    /// let mut model = LinearRegression::new();
    /// model.fit(&ds, "y");
    /// let path = std::env::temp_dir().join("rusmlai-doc-linear.model");
    /// model.save(&path);
    /// let loaded = LinearRegression::load(&path);
    /// assert_eq!(loaded.coef_, model.coef_);
    /// assert_eq!(loaded.predict(&ds), model.predict(&ds));
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import os, tempfile
    /// import joblib
    /// import pandas as pd
    /// from sklearn.linear_model import LinearRegression
    /// ds = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.1, 5.9, 8.0]})
    /// model = LinearRegression().fit(ds[["x"]], ds["y"])
    /// path = os.path.join(tempfile.gettempdir(), "rusmlai-doc-linear.joblib")
    /// joblib.dump(model, path)
    /// loaded = joblib.load(path)  # a pickle: it can run code, so load only your own files
    /// assert (loaded.coef_ == model.coef_).all()
    /// ```
    #[track_caller]
    fn save(&self, path: impl AsRef<std::path::Path>)
    where
        Self: Sized,
    {
        persist::try_save(self, path.as_ref()).or_fatal()
    }

    /// [`Model::save`] that returns the error.
    fn try_save(&self, path: impl AsRef<std::path::Path>) -> Result<(), ModelError>
    where
        Self: Sized,
    {
        persist::try_save(self, path.as_ref())
    }

    /// The model that [`Model::save`] wrote to `path`, as it was. Panics with the error's
    /// message at your line: no such file, not a saved model, or a model of another kind.
    #[track_caller]
    fn load(path: impl AsRef<std::path::Path>) -> Self
    where
        Self: Sized + Default,
    {
        persist::try_load(path.as_ref()).or_fatal()
    }

    /// [`Model::load`] that returns the error.
    fn try_load_from(path: impl AsRef<std::path::Path>) -> Result<Self, ModelError>
    where
        Self: Sized + Default,
    {
        persist::try_load(path.as_ref())
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
    /// loss, for the journal. A label the model never saw in `fit` has no probability, so then
    /// the log loss is left out with a warning, and the accuracy still counts, like
    /// scikit-learn's `score`.
    fn try_score(&self, dataset: &Dataset, target: &str) -> Result<f64, ModelError> {
        let y: &Series = dataset.column(target)?.as_materialized_series();
        let accuracy: f64 = try_accuracy_score(y, &self.try_predict(dataset)?)?;
        let mut metrics: IndexMap<String, f64> =
            IndexMap::from([("accuracy".to_string(), accuracy)]);
        match self
            .try_predict_proba(dataset)
            .and_then(|proba| try_log_loss(y, &proba))
        {
            Ok(loss) => {
                metrics.insert("log_loss".to_string(), loss);
            }
            Err(e) => warn(format_args!("the log loss was not recorded: {e}")),
        }
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

/// The scorer names a regressor takes in [`Model::try_evaluate`].
const REGRESSION_SCORERS: &str = "\"r2\", \"neg_mean_squared_error\", \"neg_root_mean_squared_error\" or \"neg_mean_absolute_error\" for a regressor";

/// The scorer names a classifier takes in [`Model::try_evaluate`].
const CLASSIFICATION_SCORERS: &str = "\"accuracy\", \"balanced_accuracy\", \"roc_auc\" or \"neg_log_loss\" for a classifier";

/// [`Model::try_evaluate`] of a regressor.
pub(crate) fn evaluate_regressor<R: Regressor + ?Sized>(
    model: &R,
    dataset: &Dataset,
    target: &str,
    scoring: &str,
) -> Result<f64, ModelError> {
    let invalid = || ModelError::InvalidSetting {
        setting: "scoring",
        value: scoring.into(),
        expected: REGRESSION_SCORERS,
    };
    if !matches!(
        scoring,
        "" | "r2" | "neg_mean_squared_error" | "neg_root_mean_squared_error" | "neg_mean_absolute_error"
    ) {
        return Err(invalid());
    }
    let predicted: Series = model.try_predict(dataset)?;
    let y: &Series = dataset.column(target)?.as_materialized_series();
    let (r2, mse, mae): (f64, f64, f64) = try_regression_scores(y, &predicted)?;
    Ok(match scoring {
        "neg_mean_squared_error" => -mse,
        "neg_root_mean_squared_error" => -mse.sqrt(),
        "neg_mean_absolute_error" => -mae,
        _ => r2,
    })
}

/// [`Model::try_evaluate`] of a classifier.
pub(crate) fn evaluate_classifier<C: Classifier + ?Sized>(
    model: &C,
    dataset: &Dataset,
    target: &str,
    scoring: &str,
) -> Result<f64, ModelError> {
    let y: &Series = dataset.column(target)?.as_materialized_series();
    match scoring {
        "" | "accuracy" => try_accuracy_score(y, &model.try_predict(dataset)?),
        "neg_log_loss" => Ok(-try_log_loss(y, &model.try_predict_proba(dataset)?)?),
        "balanced_accuracy" => try_balanced_accuracy_score(y, &model.try_predict(dataset)?),
        "roc_auc" => {
            // The score of the greater class: predict_proba's columns are the sorted classes.
            let proba: Dataset = model.try_predict_proba(dataset)?;
            if proba.width() != 2 {
                return Err(ModelError::InvalidSetting {
                    setting: "scoring",
                    value: format!("roc_auc with {} classes", proba.width()),
                    expected: "a binary target for \"roc_auc\"",
                });
            }
            let last: String = proba.columns()[1].clone();
            try_roc_auc_score(y, proba.column(&last)?.as_materialized_series())
        }
        _ => Err(ModelError::InvalidSetting {
            setting: "scoring",
            value: scoring.into(),
            expected: CLASSIFICATION_SCORERS,
        }),
    }
}

/// [`Model::try_predict_method`] of a regressor.
pub(crate) fn predict_regressor<R: Regressor + ?Sized>(
    model: &R,
    dataset: &Dataset,
    method: &str,
) -> Result<Dataset, ModelError> {
    match method {
        "predict" => one_column(model.try_predict(dataset)?),
        _ => Err(ModelError::InvalidSetting {
            setting: "method",
            value: method.into(),
            expected: "\"predict\" for a regressor",
        }),
    }
}

/// [`Model::try_predict_method`] of a classifier.
pub(crate) fn predict_classifier<C: Classifier + ?Sized>(
    model: &C,
    dataset: &Dataset,
    method: &str,
) -> Result<Dataset, ModelError> {
    match method {
        "predict" => one_column(model.try_predict(dataset)?),
        "predict_proba" => model.try_predict_proba(dataset),
        _ => Err(ModelError::InvalidSetting {
            setting: "method",
            value: method.into(),
            expected: "\"predict\" or \"predict_proba\" for a classifier",
        }),
    }
}

/// A prediction as a table of one column.
fn one_column(predicted: Series) -> Result<Dataset, ModelError> {
    let frame: DataFrame = DataFrame::new(predicted.len(), vec![predicted.into()])?;
    Ok(Dataset::new(frame, ""))
}

/// The cached stats of a numeric target with no nulls, NaN or infinity, the check every
/// regressor's `fit` makes (scikit-learn: "Input y contains NaN"). A sum is finite only when
/// every value is, so the cached mean is the check.
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
    let stats: &NumericStats = dataset.try_numeric_stats(target)?;
    if !stats.mean.is_finite() {
        return Err(ModelError::NotFinite {
            column: target.into(),
        });
    }
    Ok(stats)
}

/// `dataset` without its identifier and constant columns ([`ColumnRole::left_out`]), and one
/// warning that names them, or `None` when it has none. The target always stays. `fit`,
/// [`GridSearchCV`] and [`cross_val_score`] train on what this keeps.
///
/// [`ColumnRole::left_out`]: crate::metadata::dataset::role::ColumnRole::left_out
pub(crate) fn trainable(dataset: &Dataset, target: &str) -> Result<Option<Dataset>, ModelError> {
    let (mut kept, mut out): (Vec<String>, Vec<(String, &'static str)>) = (Vec::new(), Vec::new());
    for name in dataset.columns() {
        let role = dataset.try_column_info(&name)?.role();
        match name != target && role.left_out() {
            true => out.push((name, role.as_str())),
            false => kept.push(name),
        }
    }
    let Some((first, _)) = out.first() else {
        return Ok(None);
    };
    let named: Vec<String> = out.iter().map(|(name, role)| format!("{name} ({role})")).collect();
    warn(format_args!(
        "fit leaves out {}: such columns teach a model nothing about new rows. To learn from \
         one anyway: ds.set_role(\"{first}\", ColumnRole::Feature)",
        named.join(", ")
    ));
    let kept: Vec<&str> = kept.iter().map(String::as_str).collect();
    Ok(Some(dataset.try_select(&kept)?))
}

/// Panics at the caller's line with [`ModelError::InvalidSetting`] unless `ok`.
#[track_caller]
pub(crate) fn check(ok: bool, setting: &'static str, value: impl ToString, expected: &'static str) {
    if !ok {
        Err::<(), ModelError>(ModelError::InvalidSetting {
            setting,
            value: value.to_string(),
            expected,
        })
        .or_fatal();
    }
}

/// [`check`] that `value` is a finite number above 0.
#[track_caller]
pub(crate) fn above_zero(name: &'static str, value: f64) {
    check(value.is_finite() && value > 0.0, name, value, "a finite number above 0");
}

/// [`check`] that `value` is a finite number of 0 or more.
#[track_caller]
pub(crate) fn at_least_zero(name: &'static str, value: f64) {
    check(value.is_finite() && value >= 0.0, name, value, "a finite number of 0 or more");
}

/// [`check`] that `value` is a share above 0 up to 1.
#[track_caller]
pub(crate) fn share(name: &'static str, value: f64) {
    check(value > 0.0 && value <= 1.0, name, value, "a share above 0.0 up to 1.0");
}

/// "null" or "nulls", for the error messages.
fn nulls_word(n: usize) -> &'static str {
    if n == 1 { "null" } else { "nulls" }
}

/// The error of learned values from the cache that don't fit together, like 3 coefficients for
/// 2 features: such a record is skipped and the model trains instead.
pub(crate) fn bad_learned(why: String) -> ModelError {
    ModelError::Learned(<serde_json::Error as serde::de::Error>::custom(why))
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
    // A float target with a fraction is a regression target, as in scikit-learn's
    // `type_of_target`; whole numbers like 0.0 / 1.0 are classes.
    if dataset.column(target)?.dtype().is_float() {
        let t = col(target);
        let fraction: DataFrame = dataset
            .dataframe
            .clone()
            .lazy()
            .select([t
                .clone()
                .filter((t.clone() % lit(1.0)).neq(lit(0.0)).and(t.is_not_nan()))
                .first()])
            .collect()?;
        if let Some(value) = fraction.columns()[0].f64()?.get(0) {
            return Err(ModelError::ContinuousTarget {
                column: target.into(),
                value,
            });
        }
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
