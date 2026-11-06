//! [`ColumnInfo`]: dtype, null count and the lazily computed stats of one column.

use polars::{
    datatypes::DataType,
    frame::{DataFrame, column::Column},
};
use std::sync::OnceLock;

use crate::metadata::{
    dataset::{dto::ColumnInfoDTO, stats::ColumnStats},
    error::MetadataError,
};

/// What is known about one column: dtype, null count and stats.
///
/// ```
/// use rusmlai::prelude::*;
/// use polars::prelude::DataType;
/// let ds = read_csv("tests/fixtures/data/students.csv");
/// let info = ds.try_column_info("cgpa").unwrap();
/// assert_eq!(info.dtype, DataType::Float64);
/// assert_eq!(info.null_count, 1);
/// assert!(info.stats.get().is_some()); // read_csv computes every column
/// ```
///
/// [`ColumnInfo::new`] leaves the stats empty, and [`ColumnInfo::ensure_stats`] fills them once:
///
/// ```
/// use polars::prelude::Column;
/// use rusmlai::metadata::dataset::{info::ColumnInfo, stats::ColumnStats};
/// let column = Column::new("x".into(), [1.0, 2.0, 3.0]);
/// let info = ColumnInfo::new(&column);
/// assert!(info.stats.get().is_none());
/// let Ok(ColumnStats::Numeric(stats)) = info.ensure_stats(&column) else { panic!() };
/// assert_eq!(stats.median, 2.0);
/// ```
#[derive(Debug, Clone)]
pub struct ColumnInfo {
    /// The column's Polars dtype.
    pub dtype: DataType,
    /// Null values in the column.
    pub null_count: usize,
    /// The stats, empty until computed. Set at most once.
    pub stats: OnceLock<ColumnStats>,
}

impl ColumnInfo {
    /// Dtype and null count of `column`, with no stats yet.
    pub fn new(column: &Column) -> Self {
        ColumnInfo {
            dtype: column.dtype().clone(),
            null_count: column.null_count(),
            stats: OnceLock::new(),
        }
    }

    /// Dtype and null count of `column`, with `stats` already computed.
    pub fn with_stats(column: &Column, stats: ColumnStats) -> Self {
        ColumnInfo {
            stats: OnceLock::from(stats),
            ..ColumnInfo::new(column)
        }
    }

    /// The column's stats, computed now if they aren't there yet.
    pub fn ensure_stats(&self, column: &Column) -> Result<&ColumnStats, MetadataError> {
        if let Some(stats) = self.stats.get() {
            return Ok(stats);
        }
        let dataframe: DataFrame = DataFrame::new(column.len(), vec![column.clone()])?;
        let stats: ColumnStats = ColumnStats::compute_stats(&dataframe)?.remove(0);
        Ok(self.stats.get_or_init(|| stats))
    }
}

impl From<&ColumnInfo> for ColumnInfoDTO {
    fn from(value: &ColumnInfo) -> Self {
        Self {
            dtype: value.dtype.clone(),
            null_count: value.null_count,
            stats: value.stats.get().cloned(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_stats_computes_missing_stats() {
        let column: Column = Column::new("x".into(), [1.0, 3.0, 2.0]);
        let info: ColumnInfo = ColumnInfo::new(&column);
        let stats: &ColumnStats = info.ensure_stats(&column).unwrap();
        assert!(matches!(stats, ColumnStats::Numeric(s) if s.median == 2.0));
    }
}
