use polars::error::PolarsError;
use thiserror::Error;

use crate::dataset::error::DatasetError;

/// Why a chart or a chart option failed. The `try_*` chart methods return it.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PlotError {
    #[error("unknown estimator '{0}'. Use one of: mean, median, sum, min, max, std, count")]
    UnknownEstimator(String),

    #[error("x must name a column")]
    EmptyX,

    #[error(
        "estimator '{0}' needs a column of values to aggregate: pass one, or count rows instead"
    )]
    EstimatorNeedsY(String),

    #[error("nothing to plot: no rows of {0} have values (nulls, NaN and infinities are left out)")]
    EmptyResult(String),

    #[error("nothing to plot: {0}")]
    Undefined(String),

    #[error("{0} has a single value, so it has no shape to draw")]
    NoSpread(String),

    #[error("bins must be between 1 and 1000, got {0}")]
    InvalidBins(usize),

    #[error("hue needs an x column: pass x, or use the hue column as x")]
    HueNeedsX,

    #[error("{option}(): {reason}")]
    InvalidOption {
        option: &'static str,
        reason: String,
    },

    #[error("{option}() does not work on a {chart} chart")]
    OptionNotAllowed {
        option: &'static str,
        chart: &'static str,
    },

    #[error(transparent)]
    Dataset(#[from] DatasetError),

    #[error(transparent)]
    Polars(#[from] PolarsError),
}
