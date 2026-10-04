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

    #[error(transparent)]
    Dataset(#[from] DatasetError),

    #[error(transparent)]
    Polars(#[from] PolarsError),
}
