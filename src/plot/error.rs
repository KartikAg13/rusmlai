use thiserror::Error;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PlotError {
    #[error("unknown estimator: {0}")]
    UnknownEstimator(String),
}
