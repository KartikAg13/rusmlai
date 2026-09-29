use polars::error::PolarsError;
use thiserror::Error;

#[derive(Debug, Error)]

pub enum MetadataError {
    #[error(transparent)]
    Polars(#[from] PolarsError),
}
