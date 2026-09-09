use polars::error::PolarsError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DatasetError {
    #[error("file not found at '{path}' (current working directory: '{cwd}')")]
    FileNotFound { path: String, cwd: String },

    #[error("column '{requested}' not found in dataset. Available columns: {available}")]
    ColumnNotFound {
        requested: String,
        available: String,
    },

    #[error("unsupported datatype '{0}' found in metadata")]
    UnsupportedDatatype(String),

    #[error("no columns passed to Dataset::select")]
    EmptySelect,

    #[error("column '{column}' is not numeric (found {actual_dtype})")]
    ColumnNotNumeric {
        column: String,
        actual_dtype: String,
    },

    #[error("column '{column}' is not categorical (found {actual_dtype})")]
    ColumnNotCategorical {
        column: String,
        actual_dtype: String,
    },

    #[error(transparent)]
    Polars(#[from] PolarsError),
}
