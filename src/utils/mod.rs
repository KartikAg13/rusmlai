//! Helpers every module shares.

use polars::prelude::{Expr, PlSmallStr};

/// The column with exactly this name. Polars' own `col` reads `"*"` as every column and
/// `"^...$"` as a regex, so a CSV whose header holds such a name would select the wrong
/// columns. Every query in this crate goes through this function instead.
pub(crate) fn col(name: impl Into<PlSmallStr>) -> Expr {
    Expr::Column(name.into())
}
