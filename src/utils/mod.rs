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

/// The edit distance between two names (insert, delete or change one character, or swap
/// two neighbours: the optimal string alignment distance), ignoring case.
fn distance(a: &str, b: &str) -> usize {
    let (a, b): (Vec<char>, Vec<char>) = (a.to_lowercase().chars().collect(), b.to_lowercase().chars().collect());
    let mut d: Vec<Vec<usize>> = vec![vec![0; b.len() + 1]; a.len() + 1];
    (0..=a.len()).for_each(|i| d[i][0] = i);
    (0..=b.len()).for_each(|j| d[0][j] = j);
    for i in 1..=a.len() {
        for j in 1..=b.len() {
            let change: usize = usize::from(a[i - 1] != b[j - 1]);
            d[i][j] = (d[i - 1][j] + 1).min(d[i][j - 1] + 1).min(d[i - 1][j - 1] + change);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                d[i][j] = d[i][j].min(d[i - 2][j - 2] + 1);
            }
        }
    }
    d[a.len()][b.len()]
}

/// " Did you mean 'x'?" for the name of `available` (a comma-separated list) closest to
/// `wanted`, when it is close: at most a third of the name's characters apart, or the same
/// name in another case. Empty otherwise.
pub(crate) fn did_you_mean(wanted: &str, available: &str) -> String {
    let best: Option<(usize, &str)> = available
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty() && *name != wanted)
        .map(|name| (distance(wanted, name), name))
        .min_by_key(|(d, _)| *d);
    match best {
        Some((d, name)) if d <= (wanted.chars().count() / 3).max(1) => format!(" Did you mean '{name}'?"),
        _ => String::new(),
    }
}

#[cfg(test)]
mod did_you_mean_tests {
    use super::*;

    #[test]
    fn a_close_name_is_suggested_and_a_far_one_is_not() {
        assert_eq!(did_you_mean("nmae", "id,name,age"), " Did you mean 'name'?");
        assert_eq!(did_you_mean("CGPA", "id,cgpa"), " Did you mean 'cgpa'?");
        assert_eq!(did_you_mean("study_hour", "study_hours,age"), " Did you mean 'study_hours'?");
        assert_eq!(did_you_mean("grade", "id,name,age"), "");
        assert_eq!(did_you_mean("x", ""), "");
    }
}
