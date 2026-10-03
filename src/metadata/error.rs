use polars::error::PolarsError;
use thiserror::Error;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MetadataError {
    #[error(transparent)]
    Polars(#[from] PolarsError),
}
