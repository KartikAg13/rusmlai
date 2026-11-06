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

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    #[test]
    fn or_fatal_returns_the_value() {
        assert_eq!(Ok::<u8, String>(3).or_fatal(), 3);
    }

    #[test]
    fn or_fatal_panics_with_the_message_at_the_callers_line() {
        // The hook is global while it is set; no other unit test panics on purpose.
        let seen: Arc<Mutex<Option<(String, u32)>>> = Arc::default();
        let record = Arc::clone(&seen);
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let message: String = info
                .payload()
                .downcast_ref::<String>()
                .cloned()
                .unwrap_or_default();
            if message == "boom" {
                *record.lock().unwrap() = info.location().map(|l| (l.file().to_string(), l.line()));
            }
        }));
        let line: u32 = line!() + 1;
        let caught = std::panic::catch_unwind(|| Err::<(), &str>("boom").or_fatal());
        std::panic::set_hook(previous);
        assert!(caught.is_err());
        assert_eq!(*seen.lock().unwrap(), Some((file!().to_string(), line)));
    }
}
