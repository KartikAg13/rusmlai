use polars::error::PolarsError;
use thiserror::Error;

#[derive(Debug, Error)]

pub enum MetadataError {
    #[error("datatype is not supported: {0}")]
    UnsupportedDatatype(String),

    #[error(transparent)]
    Polars(#[from] PolarsError),
}
