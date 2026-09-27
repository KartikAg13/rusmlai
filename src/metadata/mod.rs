use indexmap::IndexMap;
use polars::frame::DataFrame;
use std::{ffi::OsString, path::Path, sync::Arc};

use crate::metadata::{
    dataset::{DatasetCache, dto::DatasetCacheDTO, info::ColumnInfo},
    dto::MetadataDTO,
};

pub mod constants;
pub mod dataset;
pub mod dto;
pub mod error;
pub mod plot;
pub mod utils;

#[derive(Debug, Clone)]
pub struct Metadata {
    pub filepath: Option<OsString>,
    pub height: usize,
    pub width: usize,
    pub dataset_cache: DatasetCache,
    pub plot_cache: PlotCache,
}

impl From<&Metadata> for MetadataDTO {
    fn from(metadata: &Metadata) -> Self {
        Self {
            source_size: 0,
            source_mtime: 0,
            filepath: metadata
                .filepath
                .as_deref()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default(),
            height: metadata.height,
            width: metadata.width,
            dataset_cache: DatasetCacheDTO::from(&metadata.dataset_cache),
        }
    }
}

impl Metadata {
    pub fn new(dataframe: &DataFrame, path: impl AsRef<Path>) -> Self {
        let width: usize = dataframe.width();
        let mut data: IndexMap<String, Arc<ColumnInfo>> = IndexMap::with_capacity(width);
        for column in dataframe.columns() {
            let name: String = column.name().to_string();
            let info: ColumnInfo = ColumnInfo::new(column);
            info.initialize(column).unwrap();
            data.insert(name, Arc::new(info));
        }
        Self {
            filepath: Some(path.as_ref().as_os_str().to_os_string()),
            height: dataframe.height(),
            width: width,
            dataset_cache: DatasetCache::new(data),
        }
    }

    pub fn load_cache(path: impl AsRef<Path>) -> Option<Metadata> {
        DatasetCacheDTO::try_load_cache(path)
    }

    pub fn write_cache(&self) {
        DatasetCacheDTO::try_write_cache(&self).unwrap()
    }

    pub fn select(&self, selected: &[&str]) -> Metadata {
        let columns = selected
            .iter()
            .filter_map(|name| {
                self.dataset_cache
                    .columns
                    .get(*name)
                    .map(|info| ((*name).to_string(), Arc::clone(info)))
            })
            .collect();
        Metadata {
            filepath: self.filepath.clone(),
            height: self.height,
            width: selected.len(),
            dataset_cache: DatasetCache::new(columns),
        }
    }

    pub fn try_info(&self) {
        self.dataset_cache.info_table(self.height, self.width);
    }

    pub fn try_describe(&self) {
        self.dataset_cache.describe_table(self.height);
    }

    pub fn try_metadata(&self, skip: &[&str]) {
        self.dataset_cache.metadata_table(
            skip,
            self.filepath
                .clone()
                .unwrap()
                .into_string()
                .unwrap()
                .as_str(),
            self.height,
            self.width,
        );
    }
}
