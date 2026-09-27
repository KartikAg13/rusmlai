use std::{
    fs::{create_dir_all, read, write},
    io,
    path::{Path, PathBuf},
    sync::Arc,
};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::{
    fatal,
    metadata::{
        Metadata,
        constants::{CACHE_DIRECTORY, CACHE_EXTENSION},
        dataset::{DatasetCache, info::ColumnInfo, stats::ColumnStats},
        utils::{file_fingerprint, option_to_oncelock, parse_dtype},
    },
};

#[derive(Serialize, Deserialize)]
pub struct ColumnInfoDTO {
    pub dtype: String,
    pub null_count: usize,
    pub stats: Option<ColumnStats>,
}

#[derive(Serialize, Deserialize)]
pub struct DatasetCacheDTO {
    pub columns: IndexMap<String, ColumnInfoDTO>,
}

impl From<ColumnInfoDTO> for ColumnInfo {
    fn from(dto: ColumnInfoDTO) -> Self {
        Self {
            dtype: parse_dtype(&dto.dtype)
                .unwrap_or_else(|e| fatal!(e, "src/metadata/dataset/dto.rs (from)")),
            null_count: dto.null_count,
            stats: option_to_oncelock(dto.stats),
        }
    }
}

impl From<DatasetCacheDTO> for DatasetCache {
    fn from(value: DatasetCacheDTO) -> Self {
        Self {
            columns: value
                .columns
                .into_iter()
                .map(|(name, info)| (name, Arc::new(info.into())))
                .collect(),
        }
    }
}

impl DatasetCacheDTO {
    fn cache_path(path: &Path) -> Option<PathBuf> {
        let parent: &Path = path.parent()?;
        let name: &str = path.file_stem()?.to_str()?;
        Some(
            parent
                .join(CACHE_DIRECTORY)
                .join(format!("{name}.{CACHE_EXTENSION}")),
        )
    }

    pub fn try_load_cache(filename: impl AsRef<Path>) -> Option<Metadata> {
        let path: &Path = filename.as_ref();
        let cache_path: PathBuf = Self::cache_path(path)?;
        let (size, mtime): (u64, u64) = file_fingerprint(path).ok()?;
        let bytes: Vec<u8> = read(&cache_path).ok()?;
        let dto: MetadataDTO = serde_json::from_slice(&bytes).ok()?;
        if dto.source_size != size || dto.source_mtime != mtime {
            return None;
        }
        let mut metadata: Metadata = dto.into();
        metadata.filepath = Some(path.as_os_str().to_os_string());
        Some(metadata)
    }

    pub fn try_write_cache(metadata: &Metadata) -> io::Result<()> {
        let path: &Path = match metadata.filepath.as_deref() {
            Some(p) => Path::new(p),
            None => return Ok(()),
        };
        let cache_path: PathBuf = match Self::cache_path(path) {
            Some(p) => p,
            None => return Ok(()),
        };
        if let Some(meta_dir) = cache_path.parent() {
            create_dir_all(meta_dir)?;
        }
        let (size, mtime): (u64, u64) = file_fingerprint(path)?;
        let mut dto: MetadataDTO = MetadataDTO::from(metadata);
        dto.source_size = size;
        dto.source_mtime = mtime;
        let json: Vec<u8> = serde_json::to_vec_pretty(&dto)?;
        write(cache_path, json)
    }
}
