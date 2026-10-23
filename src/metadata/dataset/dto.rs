use std::sync::OnceLock;

use indexmap::IndexMap;
use polars::datatypes::DataType;
use serde::{Deserialize, Serialize};

use crate::metadata::dataset::{info::ColumnInfo, stats::ColumnStats};

#[derive(Serialize, Deserialize)]
pub struct DatasetCacheDTO {
    pub fingerprint: (u64, u64),
    pub columns: IndexMap<String, ColumnInfoDTO>,
}

#[derive(Serialize, Deserialize)]
pub struct ColumnInfoDTO {
    pub dtype: DataType,
    pub null_count: usize,
    pub stats: Option<ColumnStats>,
}

impl From<ColumnInfoDTO> for ColumnInfo {
    fn from(dto: ColumnInfoDTO) -> Self {
        Self {
            dtype: dto.dtype,
            null_count: dto.null_count,
            stats: dto.stats.map_or_else(OnceLock::new, OnceLock::from),
        }
    }
}
