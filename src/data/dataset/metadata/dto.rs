use serde::{Deserialize, Serialize};

use crate::data::dataset::metadata::{Metadata, info::ColumnInfo, stats::ColumnStats};

#[derive(Serialize, Deserialize)]
pub struct ColumnInfoDto {
    pub name: String,
    pub dtype: String,
    pub null_count: usize,
    pub stats: Option<ColumnStats>,
}

#[derive(Serialize, Deserialize)]
pub struct MetadataDto {
    pub source_size: u64,
    pub source_mtime: u64,
    pub height: usize,
    pub width: usize,
    pub columns: Vec<ColumnInfoDto>,
}

impl From<ColumnInfo> for ColumnInfoDto {
    fn from(info: ColumnInfo) -> Self {
        Self {
            name: info.name,
            dtype: info.dtype.to_string(),
            null_count: info.null_count,
            stats: info.stats.get().cloned(),
        }
    }
}

impl From<Metadata> for MetadataDto {
    fn from(data: Metadata) -> Self {
        Self {
            source_size: 0,
            source_mtime: 0,
            height: data.height,
            width: data.width,
            columns: data
                .columns
                .into_iter()
                .map(|c| ColumnInfoDto::from((*c).clone()))
                .collect(),
        }
    }
}
