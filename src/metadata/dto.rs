use std::ffi::OsString;

use serde::{Deserialize, Serialize};

use crate::metadata::{Metadata, dataset::dto::DatasetCacheDTO};

#[derive(Serialize, Deserialize)]
pub struct MetadataDTO {
    pub source_size: u64,
    pub source_mtime: u64,
    pub filepath: String,
    pub height: usize,
    pub width: usize,
    #[serde(flatten)]
    pub dataset_cache: DatasetCacheDTO,
}

impl From<MetadataDTO> for Metadata {
    fn from(dto: MetadataDTO) -> Self {
        Self {
            filepath: Some(OsString::from(dto.filepath)),
            height: dto.height,
            width: dto.width,
            dataset_cache: dto.dataset_cache.into(),
        }
    }
}
