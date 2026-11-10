//! [`PreprocessError`].

use polars::error::PolarsError;
use thiserror::Error;

use crate::dataset::error::DatasetError;

/// What can fail while fitting or applying a transformer.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PreprocessError {
    /// A stats lookup failed, for example a text column given to a scaler.
    #[error(transparent)]
    Dataset(#[from] DatasetError),

    /// A Polars query failed.
    #[error(transparent)]
    Polars(#[from] PolarsError),

    /// `transform` saw a column that `fit` never learned.
    #[error(
        "{transformer} was not fitted on column '{column}': call fit first, on data that has this column"
    )]
    NotFitted {
        /// The transformer's name.
        transformer: &'static str,
        /// The column it doesn't know.
        column: String,
    },

    /// A [`ColumnTransformer::add`](crate::preprocess::ColumnTransformer::add) names a column
    /// that is neither an input column nor the output of an earlier add.
    #[error(
        "add '{add}' reads column '{column}', which is not an input column or an earlier add's output. Available: {available}"
    )]
    UnknownColumn {
        /// The add's name.
        add: String,
        /// The missing column.
        column: String,
        /// The columns it could read.
        available: String,
    },

    /// An option got a value it doesn't take.
    #[error("{option} = '{value}' is not valid; use {expected}")]
    InvalidOption {
        /// The option's name.
        option: &'static str,
        /// The value given.
        value: String,
        /// What it takes.
        expected: String,
    },

    /// `transform` gave an encoder a value that its `fit` never saw.
    #[error(
        "output column '{column}' got a value that fit never saw. To allow a category, use .handle_unknown(\"ignore\") on a OneHotEncoder, or .handle_unknown(\"use_encoded_value\").unknown_value(..) on an OrdinalEncoder"
    )]
    UnknownAtTransform {
        /// The output column that failed.
        column: String,
    },

    /// An encoder found categories it was not told about.
    #[error("{transformer} found unknown categories [{categories}] in column '{column}'. {hint}")]
    UnknownCategory {
        /// The encoder's name.
        transformer: &'static str,
        /// The column.
        column: String,
        /// The unknown categories.
        categories: String,
        /// How to fix it.
        hint: &'static str,
    },
}
