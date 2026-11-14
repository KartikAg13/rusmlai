//! [`Metadata`]: every cache of a dataset in one place. `dataset_cache` holds the stats of each
//! column, `plot_cache` the numbers behind each chart drawn so far. Both are saved next to the
//! CSV, in `metadata/{stem}.json` and `metadata/{stem}.plots.jsonl`, and are trusted only while
//! the CSV's size and modification time match.

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
pub mod journal;
pub mod models;
pub mod plot;
pub mod utils;

/// Every cache of one [`Dataset`](crate::dataset::Dataset): the column stats and the chart numbers.
///
/// `read_csv` builds it with [`Metadata::load`]; you reach it as `ds.metadata`.
///
/// ```
/// use rusmlai::prelude::*;
/// let ds = read_csv("tests/fixtures/data/students.csv");
/// assert_eq!(ds.metadata.width, 6);
/// assert_eq!(ds.metadata.height, ds.dataframe.height());
/// ```
///
/// `select` keeps the stats it needs and shares them, without a copy:
///
/// ```
/// use rusmlai::prelude::*;
/// use std::sync::Arc;
/// let ds = read_csv("tests/fixtures/data/students.csv");
/// let part = ds.select(&["cgpa"]);
/// let before = &ds.metadata.dataset_cache.columns["cgpa"];
/// let after = &part.metadata.dataset_cache.columns["cgpa"];
/// assert!(Arc::ptr_eq(before, after));
/// ```
///
/// A dataset made in memory has no file, so it never reads or writes a cache:
///
/// ```
/// use polars::df;
/// use rusmlai::prelude::*;
/// let ds = Dataset::new(df!("x" => [1, 2, 3]).unwrap(), "");
/// assert_eq!(ds.metadata.dataset_cache.columns.len(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct Metadata {
    /// The CSV it came from: the name `metadata()` shows, and charts auto-save next to it.
    /// Splits and transforms keep it. The stats and chart files are written only for a
    /// `read_csv` dataset; the model cache of every dataset that has a path goes next to it.
    pub filepath: Option<OsString>,
    /// Row count.
    pub height: usize,
    /// Column count.
    pub width: usize,
    /// Per-column dtype, null count and stats, in column order.
    pub dataset_cache: DatasetCache,
    /// The numbers behind every chart drawn so far, keyed by [`PlotSpec`].
    pub plot_cache: PlotCache,
}

impl Metadata {
    /// Computes the stats of every column; writes no file. Panics on a Polars error.
    #[track_caller]
    pub fn new(dataframe: &DataFrame, path: impl AsRef<Path>) -> Self {
        Self::try_new(dataframe, path).or_fatal()
    }

    /// [`Metadata::new`] that returns the error.
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

    /// The metadata of `selected` only, in that order. Shares the stats and the plot cache;
    /// names that are not columns are skipped.
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
            plot_cache: self.plot_cache.clone(),
        }
    }

    /// Reads `{csv dir}/metadata/{stem}.json` and `.plots.jsonl` when their fingerprint matches
    /// the CSV, else computes the stats and writes the files. A stale or broken file is a miss,
    /// and a failed write is only a warning.
    #[track_caller]
    pub fn load(dataframe: &DataFrame, path: &Path) -> Self {
        Self::try_load(dataframe, path).or_fatal()
    }

    /// [`Metadata::load`] that returns the error.
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

        // The file is set only when it is there to append to, so a read-only folder warns once.
        match read_jsonl::<(PlotSpec, PlotResult)>(&plots_path, fingerprint) {
            Some(entries) => {
                metadata.plot_cache.entries = Arc::new(RwLock::new(entries.into_iter().collect()));
                metadata.plot_cache.file = Some(plots_path);
            }
            None => match write_jsonl::<()>(&plots_path, fingerprint, &[]) {
                Ok(()) => metadata.plot_cache.file = Some(plots_path),
                Err(e) => warn(format_args!(
                    "could not write {}: {e}",
                    plots_path.display()
                )),
            },
        }
        Ok(metadata)
    }

    /// The cached result for `spec`, or `compute()` on a miss. A miss is stored in memory and
    /// appended to the `.plots.jsonl` file of a dataset that `read_csv` loaded. An error is
    /// returned and not cached.
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
        // No file: the dataset wasn't loaded by `read_csv`, so it has no cache on disk.
        if let Some(plots_path) = &self.plot_cache.file
            && let Err(e) = append_jsonl(plots_path, &(&spec, &result))
        {
            warn(format_args!(
                "could not update {}: {e}",
                plots_path.display()
            ));
        }
        entries
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(spec, result.clone());
        Ok(result)
    }

    /// The table behind [`Dataset::info`](crate::dataset::Dataset::info).
    pub fn try_info(&self) -> TableView {
        self.dataset_cache.info_table(self.height, self.width)
    }

    /// The table behind [`Dataset::describe`](crate::dataset::Dataset::describe).
    pub fn try_describe(&self) -> TableView {
        self.dataset_cache.describe_table(self.height)
    }

    /// The table behind [`Dataset::metadata`](crate::dataset::Dataset::metadata), without the
    /// columns in `skip`.
    pub fn try_metadata(&self, skip: &[&str]) -> TableView {
        let path: String = self
            .filepath
            .as_deref()
            .map_or("-".into(), |p| p.to_string_lossy().into_owned());
        self.dataset_cache
            .metadata_table(skip, &path, self.height, self.width)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use polars::df;

    use super::*;
    use crate::{dataset::Dataset, plot::error::PlotError};

    fn spec() -> PlotSpec {
        PlotSpec::Corrplot {
            columns: vec!["x".into()],
        }
    }

    fn result() -> PlotResult {
        PlotResult::Series {
            labels: vec!["a".into()],
            values: vec![1.0],
        }
    }

    fn metadata() -> Metadata {
        let frame: DataFrame = df!("x" => [1.0, 2.0]).unwrap();
        Metadata {
            filepath: None,
            ..Metadata::new(&frame, "")
        }
    }

    #[test]
    fn plot_computes_on_a_miss_and_never_on_a_hit() {
        let m: Metadata = metadata();
        let calls: Cell<u32> = Cell::new(0);
        let compute = || {
            calls.set(calls.get() + 1);
            Ok::<_, PlotError>(result())
        };
        m.plot(spec(), compute).unwrap();
        m.plot(spec(), compute).unwrap();
        assert_eq!(calls.get(), 1);
        // `select` shares the cache, so the hit carries over.
        m.select(&["x"]).plot(spec(), compute).unwrap();
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn plot_does_not_cache_an_error() {
        let m: Metadata = metadata();
        let failed: Result<PlotResult, PlotError> = m.plot(spec(), || Err(PlotError::HueNeedsX));
        assert!(failed.is_err());
        assert!(m.plot_cache.entries.read().unwrap().is_empty());
    }

    #[test]
    fn transformed_metadata_has_no_file_and_still_prints() {
        let ds: Dataset = Dataset::new(df!("x" => [1.0, 2.0], "s" => ["a", "b"]).unwrap(), "");
        let changed: Dataset = Dataset::new(df!("x" => [5.0, 6.0], "s" => ["a", "b"]).unwrap(), "");
        let m: Metadata =
            ds.metadata
                .dataset_cache
                .after_transform(&changed, vec!["x".into()], false);
        assert!(m.filepath.is_none() && m.plot_cache.entries.read().unwrap().is_empty());
        assert!(Arc::ptr_eq(
            &m.dataset_cache.columns["s"],
            &ds.metadata.dataset_cache.columns["s"]
        ));
        assert!(!Arc::ptr_eq(
            &m.dataset_cache.columns["x"],
            &ds.metadata.dataset_cache.columns["x"]
        ));
        let text: String = m.try_metadata(&[]).to_string();
        assert!(
            text.starts_with("Filepath: -") && text.contains("mean = 5.50"),
            "{text}"
        );
    }
}
