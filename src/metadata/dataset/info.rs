use polars::{datatypes::DataType, frame::column::Column};
use std::sync::OnceLock;

use crate::metadata::{
    dataset::{
        dto::ColumnInfoDTO,
        stats::{CategoricalStats, ColumnStats, NumericStats},
    },
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

    pub fn initialize(&self, column: &Column) -> Result<(), MetadataError> {
        if self.stats.get().is_none() {
            let computed: ColumnStats = if self.dtype.is_numeric() {
                ColumnStats::Numeric(NumericStats::compute_stats(column)?)
            } else {
                ColumnStats::Categorical(CategoricalStats::compute_stats(column)?)
            };

            let _ = self.stats.set(computed);
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
