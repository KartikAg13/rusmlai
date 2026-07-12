use polars::error::PolarsError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DataError {
    #[error("file not found at location '{0}'")]
    FileNotFound(String),

    #[error("column '{0}' not found in dataset")]
    ColumnNotFound(String),

    #[error("unsupported datatype '{0}' found in metadata")]
    UnsupportedDatatype(String),

    #[error(transparent)]
    Polars(#[from] PolarsError),
}
