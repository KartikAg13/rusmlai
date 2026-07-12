use polars::{
    datatypes::DataType,
    frame::{DataFrame, column::Column},
};
use std::sync::OnceLock;

use crate::{
    data::{
        dataset::metadata::{
            dto::ColumnInfoDto,
            stats::{CategoricalStats, ColumnStats, NumericStats},
        },
        error::DataError,
        utils::{option_to_oncelock, parse_dtype},
    },
    fatal,
};

#[derive(Clone)]
pub struct ColumnInfo {
    pub name: String,
    pub dtype: DataType,
    pub null_count: usize,

    pub stats: OnceLock<ColumnStats>,
}

impl ColumnInfo {
    pub fn new(column: &Column) -> Self {
        ColumnInfo {
            name: column.name().to_string(),
            dtype: column.dtype().clone(),
            null_count: column.null_count(),
            stats: OnceLock::new(),
        }
    }

    pub fn stats(&self, df: DataFrame) -> Result<&ColumnStats, DataError> {
        if let Some(s) = self.stats.get() {
            return Ok(s);
        }
        let computed: ColumnStats = if self.dtype.is_numeric() {
            ColumnStats::Numeric(NumericStats::compute_stats(&df, &self.name)?)
        } else {
            ColumnStats::Categorical(CategoricalStats::compute_stats(&df, &self.name)?)
        };
        Ok(self.stats.get_or_init(|| computed))
    }
}

impl From<ColumnInfoDto> for ColumnInfo {
    fn from(dto: ColumnInfoDto) -> Self {
        Self {
            name: dto.name,
            dtype: parse_dtype(&dto.dtype).unwrap_or_else(|e| fatal!(e)),
            null_count: dto.null_count,
            stats: option_to_oncelock(dto.stats),
        }
    }
}
