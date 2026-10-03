use indexmap::IndexMap;
use polars::datatypes::DataType;
use serde::{Deserialize, Serialize};

use crate::metadata::{
    dataset::{info::ColumnInfo, stats::ColumnStats},
    utils::option_to_oncelock,
};

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
            stats: option_to_oncelock(dto.stats),
        }
    }
}
