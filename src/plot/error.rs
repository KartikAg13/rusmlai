use thiserror::Error;

#[derive(Debug, Error)]
pub enum PlotError {
    #[error("unknown estimator: {0}")]
    UnknownEstimator(String),
}
