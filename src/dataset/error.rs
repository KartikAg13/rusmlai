//! [`DatasetError`]: why reading or changing a [`Dataset`](crate::dataset::Dataset) failed.

use polars::error::PolarsError;
use thiserror::Error;

use crate::metadata::error::MetadataError;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DatasetError {
    #[error("file not found at '{absolute}' (given '{path}', relative to '{cwd}')")]
    FileNotFound {
        path: String,
        absolute: String,
        cwd: String,
    },

    #[error(
        "column '{requested}' not found in dataset.{} Available columns: {available}",
        crate::utils::did_you_mean(.requested, .available)
    )]
    ColumnNotFound {
        requested: String,
        available: String,
    },

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

    #[error("sheet '{sheet}' not found. Sheets: {available}")]
    SheetNotFound { sheet: String, available: String },

    #[error("could not read {what}: {reason}")]
    Read { what: String, reason: String },

    #[error("could not write {path}: {reason}")]
    Write { path: String, reason: String },

    /// An option got a value it doesn't take.
    #[error("{option} = '{value}' is not valid; use {expected}")]
    InvalidOption {
        option: &'static str,
        value: String,
        expected: &'static str,
    },

    #[error(transparent)]
    Metadata(#[from] MetadataError),

    #[error(transparent)]
    Polars(#[from] PolarsError),
}
