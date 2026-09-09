use polars::{datatypes::DataType, frame::column::Column};
use std::sync::OnceLock;

use crate::{
    dataset::{
        error::DatasetError,
        utils::{option_to_oncelock, parse_dtype},
    },
    fatal,
    metadata::{
        dataset::stats::{CategoricalStats, ColumnStats, NumericStats},
        dto::ColumnInfoDto,
    },
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

    pub fn initialize(&self, column: &Column) -> Result<(), DatasetError> {
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

impl From<ColumnInfoDto> for ColumnInfo {
    fn from(dto: ColumnInfoDto) -> Self {
        Self {
            dtype: parse_dtype(&dto.dtype)
                .unwrap_or_else(|e| fatal!(e, "src/data/dataset/metadata/info.rs (fromw)")),
            null_count: dto.null_count,
            stats: option_to_oncelock(dto.stats),
        }
    }
}
