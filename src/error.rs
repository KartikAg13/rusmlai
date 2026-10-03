use std::fmt::Display;

use thiserror::Error;

use crate::{dataset::error::DatasetError, metadata::error::MetadataError, plot::error::PlotError};

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RusMLAIErrorInner {
    #[error(transparent)]
    Data(#[from] DatasetError),

    #[error(transparent)]
    Metadata(#[from] MetadataError),

    #[error(transparent)]
    Plot(#[from] PlotError),
}

/// Turns the `try_*` result into the panicking public API.
pub trait OrFatal<T> {
    /// Returns the value, or panics with the error's message at the user's call site
    /// (every public wrapper is `#[track_caller]`, so the location points at their code).
    fn or_fatal(self) -> T;
}

impl<T, E: Display> OrFatal<T> for Result<T, E> {
    #[track_caller]
    fn or_fatal(self) -> T {
        match self {
            Ok(value) => value,
            Err(e) => panic!("{e}"),
        }
    }
}

/// Non-fatal problems (a cache file that couldn't be written, ...) go to stderr.
pub(crate) fn warn(msg: impl Display) {
    eprintln!("[rusmlai warning] {msg}");
}
