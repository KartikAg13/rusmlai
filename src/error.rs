//! [`OrFatal`], which turns a `try_*` result into the panicking public API, and the `warn` channel for non-fatal problems.

use std::{
    fmt::Display,
    sync::{Mutex, PoisonError},
};

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

/// The warnings of this run so far, in order, each once.
static WARNINGS: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// The setting `set_warnings` and `RUSMLAI_WARNINGS` share.
const WARNINGS_VAR: &str = "RUSMLAI_WARNINGS";

/// A non-fatal problem or a choice rusmlai made for you (columns left out of a fit, a
/// category `fit` never saw, a cache file that couldn't be written): printed to stderr once
/// per run, however often it happens, unless `set_warnings(false)`. [`take_warnings`] reads
/// them either way.
pub(crate) fn warn(msg: impl Display) {
    let msg: String = msg.to_string();
    let mut seen = WARNINGS.lock().unwrap_or_else(PoisonError::into_inner);
    if seen.contains(&msg) {
        return;
    }
    let quiet: bool = crate::utils::setting(WARNINGS_VAR)
        .is_some_and(|v| ["off", "0", "false"].contains(&v.to_ascii_lowercase().as_str()));
    if !quiet {
        eprintln!("[rusmlai warning] {msg}");
    }
    seen.push(msg);
}

/// `false` stops printing rusmlai's warnings (a notebook keeps it for every later cell);
/// `true`, the default, prints each warning once per run. The environment variable
/// `RUSMLAI_WARNINGS=off` does the same. [`take_warnings`] still collects them.
///
/// ```
/// use rusmlai::prelude::*;
/// set_warnings(false);
/// let ds = Dataset::new(df!("x" => [1.0, 1.0, 1.0], "y" => [0.0, 1.0, 2.0]).unwrap(), "");
/// let _ = KBinsDiscretizer::new().strategy("uniform").fit_transform(&ds.select(&["x"]));
/// assert!(take_warnings().iter().any(|w| w.contains("constant")));
/// set_warnings(true);
/// ```
///
/// Python: `warnings.filterwarnings("ignore")`, and `warnings.catch_warnings(record=True)` to
/// read them.
pub fn set_warnings(on: bool) {
    crate::utils::set_setting(WARNINGS_VAR, if on { "on" } else { "off" });
}

/// Every warning since the last call (each once), and forgets them, so the next repeat
/// prints again. For tests and notebooks that want to check what rusmlai decided.
pub fn take_warnings() -> Vec<String> {
    std::mem::take(&mut *WARNINGS.lock().unwrap_or_else(PoisonError::into_inner))
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
