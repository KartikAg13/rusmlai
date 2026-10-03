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
        let info: ColumnInfo = ColumnInfo::new(column);
        let _ = info.stats.set(stats);
        info
    }

    pub fn initialize(&self, column: &Column) -> Result<(), MetadataError> {
        if self.stats.get().is_none() {
            let dataframe: DataFrame = DataFrame::new(column.len(), vec![column.clone()])?;
            if let Some(stats) = ColumnStats::compute_stats(&dataframe)?.pop() {
                let _ = self.stats.set(stats);
            }
        }
        Ok(())
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
