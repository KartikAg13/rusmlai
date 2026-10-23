use indexmap::IndexMap;
use polars::frame::DataFrame;
use std::{
    ffi::OsString,
    fs, io,
    path::{Path, PathBuf},
    sync::{Arc, PoisonError, RwLock},
};

use crate::{
    display::TableView,
    error::{OrFatal, warn},
    metadata::{
        constants::{DATASET_CACHE, PLOTS_CACHE},
        dataset::{
            DatasetCache,
            dto::{ColumnInfoDTO, DatasetCacheDTO},
            info::ColumnInfo,
            stats::ColumnStats,
        },
        error::MetadataError,
        plot::PlotCache,
        utils::{append_jsonl, cache_path, file_fingerprint, print_saved, read_jsonl, write_jsonl},
    },
    plot::{result::PlotResult, spec::PlotSpec},
};

pub mod constants;
pub mod dataset;
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

impl Metadata {
    #[track_caller]
    pub fn new(dataframe: &DataFrame, path: impl AsRef<Path>) -> Self {
        Self::try_new(dataframe, path).or_fatal()
    }

    pub fn try_new(dataframe: &DataFrame, path: impl AsRef<Path>) -> Result<Self, MetadataError> {
        let width: usize = dataframe.width();
        let stats: Vec<ColumnStats> = ColumnStats::compute_stats(dataframe)?;
        let data: IndexMap<String, Arc<ColumnInfo>> = dataframe
            .columns()
            .iter()
            .zip(stats)
            .map(|(column, stats)| {
                let info: ColumnInfo = ColumnInfo::with_stats(column, stats);
                (column.name().to_string(), Arc::new(info))
            })
            .collect();
        Ok(Self {
            filepath: Some(path.as_ref().as_os_str().to_os_string()),
            height: dataframe.height(),
            width,
            dataset_cache: DatasetCache::new(data),
            plot_cache: PlotCache::default(),
        })
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
            plot_cache: PlotCache::new(Arc::clone(&self.plot_cache.entries)),
        }
    }

    #[track_caller]
    pub fn load(dataframe: &DataFrame, path: &Path) -> Self {
        Self::try_load(dataframe, path).or_fatal()
    }

    pub fn try_load(dataframe: &DataFrame, path: &Path) -> Result<Self, MetadataError> {
        let fingerprint: (u64, u64) = file_fingerprint(path).unwrap_or_default();
        let columns_path: PathBuf = cache_path(path, DATASET_CACHE);
        let plots_path: PathBuf = cache_path(path, PLOTS_CACHE);

        let columns: Option<IndexMap<String, Arc<ColumnInfo>>> = fs::read(&columns_path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<DatasetCacheDTO>(&bytes).ok())
            .filter(|dto| dto.fingerprint == fingerprint && dto.columns.len() == dataframe.width())
            .map(|dto| {
                dto.columns
                    .into_iter()
                    .map(|(name, d)| (name, Arc::new(d.into())))
                    .collect()
            });
        let mut metadata: Metadata = match columns {
            Some(c) => Self {
                filepath: Some(path.as_os_str().to_os_string()),
                height: dataframe.height(),
                width: dataframe.width(),
                dataset_cache: DatasetCache::new(c),
                plot_cache: PlotCache::default(),
            },
            None => {
                let m: Metadata = Self::try_new(dataframe, path)?;
                let dto: DatasetCacheDTO = DatasetCacheDTO {
                    fingerprint,
                    columns: m
                        .dataset_cache
                        .columns
                        .iter()
                        .map(|(n, i)| (n.clone(), ColumnInfoDTO::from(i.as_ref())))
                        .collect(),
                };
                let written: io::Result<()> = serde_json::to_vec_pretty(&dto)
                    .map_err(io::Error::from)
                    .and_then(|json| {
                        fs::create_dir_all(columns_path.parent().unwrap())?;
                        fs::write(&columns_path, json)
                    });
                match written {
                    Ok(()) => print_saved(&columns_path),
                    Err(e) => warn(format_args!(
                        "could not write {}: {e}",
                        columns_path.display()
                    )),
                }
                m
            }
        };

        match read_jsonl::<(PlotSpec, PlotResult)>(&plots_path, fingerprint) {
            Some(entries) => {
                metadata.plot_cache =
                    PlotCache::new(Arc::new(RwLock::new(entries.into_iter().collect())))
            }
            None => {
                if let Err(e) = write_jsonl::<()>(&plots_path, fingerprint, &[]) {
                    warn(format_args!(
                        "could not write {}: {e}",
                        plots_path.display()
                    ));
                }
            }
        }
        Ok(metadata)
    }

    pub fn plot<E>(
        &self,
        spec: PlotSpec,
        compute: impl FnOnce() -> Result<PlotResult, E>,
    ) -> Result<PlotResult, E> {
        let entries = &self.plot_cache.entries;
        if let Some(hit) = entries
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&spec)
        {
            return Ok(hit.clone());
        }
        let result: PlotResult = compute()?;
        if let Some(path) = &self.filepath {
            let plots_path: PathBuf = cache_path(Path::new(path), PLOTS_CACHE);
            // No file: the dataset wasn't loaded by `read_csv` (`Dataset::new`), so it has no cache.
            if let Err(e) = append_jsonl(&plots_path, &(&spec, &result))
                && e.kind() != io::ErrorKind::NotFound
            {
                warn(format_args!(
                    "could not update {}: {e}",
                    plots_path.display()
                ));
            }
        }
        entries
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(spec, result.clone());
        Ok(result)
    }

    pub fn try_info(&self) -> TableView {
        self.dataset_cache.info_table(self.height, self.width)
    }

    pub fn try_describe(&self) -> TableView {
        self.dataset_cache.describe_table(self.height)
    }

    pub fn try_metadata(&self, skip: &[&str]) -> TableView {
        let path: String = self
            .filepath
            .as_deref()
            .map_or("-".into(), |p| p.to_string_lossy().into_owned());
        self.dataset_cache
            .metadata_table(skip, &path, self.height, self.width)
    }
}
