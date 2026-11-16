//! Helpers every module shares.

use std::sync::{Mutex, PoisonError};

use polars::prelude::{Expr, PlSmallStr};

use crate::display::in_notebook;

/// What `set_theme`, `set_palette` and `set_cache` chose in a program, by variable name.
static SETTINGS: Mutex<Vec<(&'static str, String)>> = Mutex::new(Vec::new());

/// Sets a rusmlai-wide setting. In a program it stays in this process's memory, so no thread
/// can race on the environment. In a notebook it goes into the environment variable `var`:
/// evcxr builds each cell with its own copy of rusmlai's statics, and the environment is the
/// only state the cells share.
pub(crate) fn set_setting(var: &'static str, value: &str) {
    if in_notebook() {
        // SAFETY: changing the environment is unsound only while another thread reads it. evcxr
        // runs one cell at a time, and Polars' threads wait idle between the queries of a cell.
        unsafe { std::env::set_var(var, value) };
    } else {
        let mut settings = SETTINGS.lock().unwrap_or_else(PoisonError::into_inner);
        settings.retain(|(name, _)| *name != var);
        settings.push((var, value.to_string()));
    }
}

/// A setting: what [`set_setting`] chose in this program, else the environment variable `var`
/// (set in a notebook, or by the shell).
pub(crate) fn setting(var: &str) -> Option<String> {
    let settings = SETTINGS.lock().unwrap_or_else(PoisonError::into_inner);
    settings
        .iter()
        .find(|(name, _)| *name == var)
        .map(|(_, value)| value.clone())
        .or_else(|| std::env::var(var).ok())
}

/// The column with exactly this name. Polars' own `col` reads `"*"` as every column and
/// `"^...$"` as a regex, so a CSV whose header holds such a name would select the wrong
/// columns. Every query in this crate goes through this function instead.
pub(crate) fn col(name: impl Into<PlSmallStr>) -> Expr {
    Expr::Column(name.into())
}
