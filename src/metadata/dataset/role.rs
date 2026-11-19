//! What each column is for: a feature, or a column a model should leave out (an id, a
//! constant), decided from the stats `read_csv` computes anyway.

use polars::datatypes::DataType;
use serde::{Deserialize, Serialize};

use crate::metadata::dataset::stats::ColumnStats;

/// Fewest distinct values before a high share of distinct values says "identifier": on a
/// small table every column can be unique by chance.
const MIN_UNIQUE: usize = 50;
/// Share of distinct values among the non-null ones that makes an identifier.
const ID_RATIO: f64 = 0.95;
/// Mean length of a text value, in characters, from which a near-unique text column is free
/// text (a comment, a description) rather than a key (a name, an email).
const LONG_TEXT: f64 = 30.0;

/// What a column is for. `read_csv` decides it from the column's stats; [`Dataset::role`]
/// shows it, [`Dataset::set_role`] changes it, and `fit` leaves out the `Identifier` and
/// `Constant` columns, with a warning that names them.
///
/// | Role | When |
/// |---|---|
/// | `Constant` | one value only (and no nulls), or no values at all |
/// | `Datetime` | a date or time type, or text whose values read as dates (`2024-05-31`) |
/// | `Identifier` | integers or text with a value per row (95% distinct, 50 or more), integers in a dense range; or a name like `id`, `*_id`, `index`, `uuid`, `roll*`, `email`, `phone` with half of its values distinct |
/// | `FreeText` | near-unique text with a mean length of 30 characters or more |
/// | `Feature` | everything else, and every float column that is not constant |
///
/// [`Dataset::role`]: crate::dataset::Dataset::role
/// [`Dataset::set_role`]: crate::dataset::Dataset::set_role
///
/// ```
/// use rusmlai::prelude::*;
/// let ds = read_csv("tests/fixtures/data/students.csv");
/// assert_eq!(ds.role("id"), ColumnRole::Identifier); // named id, a value per row
/// assert_eq!(ds.role("cgpa"), ColumnRole::Feature);
/// assert_eq!(ds.role("name"), ColumnRole::Feature); // 12 names are too few to tell
/// ```
///
/// Python has no column roles: you drop such columns by hand, `ds.drop(columns=["id"])`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ColumnRole {
    /// A column a model learns from.
    Feature,
    /// A key with a value per row: a row number, an id, a name, an email.
    Identifier,
    /// One value in every row.
    Constant,
    /// Long, mostly distinct text.
    FreeText,
    /// Dates or times.
    Datetime,
}

impl ColumnRole {
    /// The role's name, as `info()` shows it.
    pub fn as_str(self) -> &'static str {
        match self {
            ColumnRole::Feature => "feature",
            ColumnRole::Identifier => "identifier",
            ColumnRole::Constant => "constant",
            ColumnRole::FreeText => "free text",
            ColumnRole::Datetime => "datetime",
        }
    }

    /// Whether `fit` leaves the column out: an identifier or a constant teaches a model
    /// nothing that holds for new rows.
    pub fn left_out(self) -> bool {
        matches!(self, ColumnRole::Identifier | ColumnRole::Constant)
    }
}

/// Whether a column's name says it is a key.
fn id_like(name: &str) -> bool {
    let name: String = name.trim().to_lowercase();
    matches!(name.as_str(), "id" | "idx" | "index" | "row" | "rowid")
        || name.ends_with("_id")
        || name.ends_with(" id")
        || name.starts_with("id_")
        || name.starts_with("unnamed:")
        || name.starts_with("roll")
        || ["uuid", "guid", "email", "e-mail", "phone"].iter().any(|word| name.contains(word))
}

/// Whether `text` starts like a date: `2024-05-31`, `2024/05/31` or `31/05/2024`.
fn date_like(text: &str) -> bool {
    let b: &[u8] = text.as_bytes();
    let digits = |range: std::ops::Range<usize>| range.clone().all(|i| b.get(i).is_some_and(u8::is_ascii_digit));
    let sep = |i: usize| matches!(b.get(i), Some(b'-' | b'/' | b'.'));
    (digits(0..4) && sep(4) && digits(5..7) && sep(7) && digits(8..10))
        || (digits(0..2) && sep(2) && digits(3..5) && sep(5) && digits(6..10))
}

/// The role of a column from its name, dtype, nulls, row count and stats.
pub(crate) fn detect(name: &str, dtype: &DataType, null_count: usize, rows: usize, stats: &ColumnStats) -> ColumnRole {
    let values: usize = rows.saturating_sub(null_count);
    let (unique, range): (Option<usize>, Option<f64>) = match stats {
        ColumnStats::Numeric(n) => {
            let one: bool = n.min == n.max && !n.min.is_nan();
            (n.unique_count.or(one.then_some(1)), Some(n.max - n.min + 1.0))
        }
        ColumnStats::Categorical(c) => (Some(c.unique_count), None),
    };
    if values == 0 || unique == Some(0) || (unique == Some(1) && null_count == 0) {
        return ColumnRole::Constant;
    }
    if dtype.is_temporal() {
        return ColumnRole::Datetime;
    }
    let text: bool = matches!(dtype, DataType::String | DataType::Categorical(..) | DataType::Enum(..));
    let (Some(unique), true) = (unique, text || dtype.is_integer()) else {
        return ColumnRole::Feature;
    };
    let share: f64 = unique as f64 / values as f64;
    if let ColumnStats::Categorical(c) = stats {
        let top: Vec<&str> = c.categories.iter().take(5).map(|(v, _)| v.as_str()).collect();
        if text && !top.is_empty() && top.iter().all(|v| date_like(v)) {
            return ColumnRole::Datetime;
        }
        let long: bool = text && {
            let sample = c.categories.iter().take(1000);
            let (chars, count) = sample.fold((0usize, 0usize), |(n, k), (v, _)| (n + v.chars().count(), k + 1));
            count > 0 && chars as f64 / count as f64 >= LONG_TEXT
        };
        if long && unique >= MIN_UNIQUE && share >= 0.5 {
            return ColumnRole::FreeText;
        }
    }
    // Integers in a dense range: a row number or an id, not a measure that happens to be
    // distinct (prices in cents spread far wider than the row count).
    let dense: bool = range.is_none_or(|span| span <= 1.5 * unique as f64);
    if (unique >= MIN_UNIQUE && share >= ID_RATIO && dense) || (id_like(name) && share >= 0.5) {
        return ColumnRole::Identifier;
    }
    ColumnRole::Feature
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::dataset::stats::{CategoricalStats, NumericStats};

    fn ints(unique: usize, min: f64, max: f64) -> ColumnStats {
        ColumnStats::Numeric(NumericStats { unique_count: Some(unique), min, max, ..Default::default() })
    }

    fn text(values: &[&str], rows_each: u64) -> ColumnStats {
        ColumnStats::Categorical(CategoricalStats {
            unique_count: values.len(),
            categories: values.iter().map(|v| (v.to_string(), rows_each)).collect(),
            ..Default::default()
        })
    }

    #[test]
    fn ids_constants_dates_and_text_are_told_apart() {
        let int: &DataType = &DataType::Int64;
        let str_: &DataType = &DataType::String;
        // A row number of 1,000 rows, under any name.
        assert_eq!(detect("n", int, 0, 1000, &ints(1000, 0.0, 999.0)), ColumnRole::Identifier);
        // Distinct integers spread far wider than the rows: a measure.
        assert_eq!(detect("cents", int, 0, 1000, &ints(1000, 0.0, 1e7)), ColumnRole::Feature);
        // Small tables: only the name decides.
        assert_eq!(detect("hours", int, 0, 8, &ints(8, 1.0, 8.0)), ColumnRole::Feature);
        assert_eq!(detect("student_id", int, 0, 8, &ints(8, 1.0, 8.0)), ColumnRole::Identifier);
        assert_eq!(detect("Unnamed: 0", int, 0, 8, &ints(8, 0.0, 7.0)), ColumnRole::Identifier);
        // A repeated code is a feature even with an id-like name.
        assert_eq!(detect("zone_id", int, 0, 1000, &ints(40, 1.0, 40.0)), ColumnRole::Feature);
        // One value; one value with nulls is not constant (the nulls say something).
        assert_eq!(detect("k", int, 0, 10, &ints(1, 3.0, 3.0)), ColumnRole::Constant);
        assert_eq!(detect("k", int, 2, 10, &ints(1, 3.0, 3.0)), ColumnRole::Feature);
        let floats = ColumnStats::Numeric(NumericStats { min: 2.5, max: 2.5, ..Default::default() });
        assert_eq!(detect("f", &DataType::Float64, 0, 10, &floats), ColumnRole::Constant);
        assert_eq!(detect("empty", &DataType::Float64, 10, 10, &floats), ColumnRole::Constant);
        assert_eq!(detect("d", &DataType::Date, 0, 10, &ints(10, 0.0, 9.0)), ColumnRole::Datetime);

        let names: Vec<String> = (0..60).map(|i| format!("person {i}")).collect();
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        assert_eq!(detect("name", str_, 0, 60, &text(&names, 1)), ColumnRole::Identifier);
        let notes: Vec<String> = (0..60).map(|i| format!("a long comment about the visit, number {i}")).collect();
        let notes: Vec<&str> = notes.iter().map(String::as_str).collect();
        assert_eq!(detect("notes", str_, 0, 60, &text(&notes, 1)), ColumnRole::FreeText);
        assert_eq!(detect("day", str_, 0, 60, &text(&["2024-05-31", "2024-06-01"], 30)), ColumnRole::Datetime);
        assert_eq!(detect("city", str_, 0, 60, &text(&["Pune", "Delhi"], 30)), ColumnRole::Feature);
    }
}
