use std::{
    ffi::OsStr,
    fs::{create_dir_all, read, write},
    io,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::data::{
    constants::{CACHE_DIRECTORY, CACHE_EXTENSION},
    dataset::metadata::{Metadata, info::ColumnInfo, stats::ColumnStats},
    utils::file_fingerprint,
};

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
    pub filepath: String,
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

impl From<&Metadata> for MetadataDto {
    fn from(data: &Metadata) -> Self {
        Self {
            source_size: 0,
            source_mtime: 0,
            filepath: data.filepath.clone().unwrap().into_string().unwrap(),
            height: data.height,
            width: data.width,
            columns: data
                .columns
                .clone()
                .into_iter()
                .map(|c| ColumnInfoDto::from((*c).clone()))
                .collect(),
        }
    }
}

impl MetadataDto {
    pub fn try_load_cache(filename: impl AsRef<Path>) -> Option<Metadata> {
        let path: &Path = filename.as_ref();
        let parent: &Path = path.parent()?;
        let meta_dir: PathBuf = parent.join(CACHE_DIRECTORY);
        let name: &str = path.file_stem()?.to_str()?;
        let cache_path: PathBuf = meta_dir.join(format!("{name}.json"));
        let (size, mtime): (u64, u64) = file_fingerprint(path).ok()?;
        let bytes: Vec<u8> = read(&cache_path).ok()?;
        let dto: MetadataDto = serde_json::from_slice(&bytes).ok()?;
        if dto.source_size != size || dto.source_mtime != mtime {
            return None;
        }
        Some(dto.into())
    }

    pub fn try_write_cache(metadata: &Metadata) -> io::Result<()> {
        let filename: &OsStr = metadata.filepath.as_deref().unwrap();
        let path: &Path = Path::new(filename);
        let parent: &Path = path.parent().unwrap();
        let meta_dir: PathBuf = parent.join(CACHE_DIRECTORY);
        create_dir_all(&meta_dir)?;
        let (size, mtime): (u64, u64) = file_fingerprint(path)?;
        let mut dto: MetadataDto = MetadataDto::from(metadata);
        dto.source_size = size;
        dto.source_mtime = mtime;
        let json: Vec<u8> = serde_json::to_vec_pretty(&dto)?;
        let filename: &OsStr = path.file_stem().unwrap();
        let cache_path: PathBuf = meta_dir.join(filename).with_extension(CACHE_EXTENSION);
        write(cache_path, json)
    }
}
