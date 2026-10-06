use polars::error::PolarsError;
use thiserror::Error;

use crate::dataset::error::DatasetError;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PlotError {
    #[error("unknown estimator: {0}")]
    UnknownEstimator(String),

    #[error("x must name a column")]
    EmptyX,

    #[error("estimator '{0}' needs a y column. Pass y, or use count(x, hue) to count rows")]
    EstimatorNeedsY(String),

    #[error("nothing to plot: no rows are left in {0} after nulls are dropped")]
    EmptyResult(String),

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
