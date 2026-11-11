//! Every model: [`Regressor`], the trait they share, [`ModelError`], and one file per model
//! family ([`dummy`] today).

use std::fmt::Debug;

pub mod dummy;

pub use dummy::DummyRegressor;

use polars::{error::PolarsError, prelude::Series};
use thiserror::Error;

use crate::{
    dataset::{Dataset, error::DatasetError},
    error::OrFatal,
    metadata::dataset::stats::NumericStats,
    metrics::try_r2_score,
    preprocess::error::PreprocessError,
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
}

/// A model that predicts a number, like scikit-learn's regressors.
///
/// `fit` takes the whole dataset and the name of the target column: every other column is a
/// feature. `predict` gives one value per row, and `score` is R² ([`r2_score`]).
///
/// [`r2_score`]: crate::metrics::r2_score
pub trait Regressor: Debug {
    /// Learns from `dataset`, where `target` is the column to predict.
    fn try_fit(&mut self, dataset: &Dataset, target: &str) -> Result<(), ModelError>;

    /// One prediction per row of `dataset`, as a `Float64` series named after the target.
    fn try_predict(&self, dataset: &Dataset) -> Result<Series, ModelError>;

    /// R² of the predictions against the `target` column of `dataset`.
    fn try_score(&self, dataset: &Dataset, target: &str) -> Result<f64, ModelError> {
        let predicted: Series = self.try_predict(dataset)?;
        try_r2_score(dataset.column(target)?.as_materialized_series(), &predicted)
    }

    /// Learns from `dataset`. Panics with the error's message at your line.
    #[track_caller]
    fn fit(&mut self, dataset: &Dataset, target: &str) {
        self.try_fit(dataset, target).or_fatal()
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
