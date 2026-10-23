use polars::{
    datatypes::DataType,
    frame::{DataFrame, column::Column},
};
use std::sync::OnceLock;

use crate::metadata::{
    dataset::{dto::ColumnInfoDTO, stats::ColumnStats},
    error::MetadataError,
};

#[derive(Debug, Clone)]
pub struct ColumnInfo {
    pub dtype: DataType,
    pub null_count: usize,

    pub stats: OnceLock<ColumnStats>,
}

impl ColumnInfo {
    pub fn new(column: &Column) -> Self {
        ColumnInfo {
            dtype: column.dtype().clone(),
            null_count: column.null_count(),
            stats: OnceLock::new(),
        }
    }

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
