//! [`MetadataError`].

use polars::error::PolarsError;
use thiserror::Error;

/// What can fail while computing or loading metadata.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MetadataError {
    /// A Polars query failed.
    #[error(transparent)]
    Polars(#[from] PolarsError),
}
