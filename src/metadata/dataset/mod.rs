//! Per-column metadata: [`DatasetCache`] holds one [`ColumnInfo`] per column.

use std::sync::Arc;

use indexmap::IndexMap;
use polars::frame::{DataFrame, column::Column};

use crate::{
    dataset::Dataset,
    display::TableView,
    error::OrFatal,
    metadata::{
        Metadata,
        dataset::{info::ColumnInfo, stats::ColumnStats},
        error::MetadataError,
        plot::PlotCache,
    },
};

pub mod dto;
pub mod info;
pub mod stats;

/// The [`ColumnInfo`] of every column, in column order.
///
/// ```
/// use rusmlai::prelude::*;
/// let ds = read_csv("tests/fixtures/data/students.csv");
/// let names: Vec<&String> = ds.metadata.dataset_cache.columns.keys().collect();
/// assert_eq!(names, ["id", "name", "age", "cgpa", "batch", "passed"]);
/// assert_eq!(ds.metadata.dataset_cache.columns["cgpa"].null_count, 1);
/// ```
#[derive(Debug, Clone)]
pub struct DatasetCache {
    /// Column name to its info. `Arc`, so `select` and `drop` share it.
    pub columns: IndexMap<String, Arc<ColumnInfo>>,
}

impl DatasetCache {
    /// Wraps an existing map.
    pub fn new(columns: IndexMap<String, Arc<ColumnInfo>>) -> Self {
        Self { columns }
    }

    /// The metadata of `new_dataset`, made from this one. Recomputes the stats of
    /// `affected_columns` and of new columns only, or of every column if the row count changed.
    /// The result has no file and an empty plot cache.
    #[track_caller]
    pub fn after_transform(
        &self,
        new_dataset: &Dataset,
        affected_columns: Vec<String>,
        row_count_changed: bool,
    ) -> Metadata {
        self.try_after_transform(new_dataset, affected_columns, row_count_changed)
            .or_fatal()
    }

    /// [`DatasetCache::after_transform`] that returns the error.
    pub fn try_after_transform(
        &self,
        new_dataset: &Dataset,
        affected_columns: Vec<String>,
        row_count_changed: bool,
    ) -> Result<Metadata, MetadataError> {
        let dataframe: &DataFrame = &new_dataset.dataframe;
        let stale: Vec<Column> = dataframe
            .columns()
            .iter()
            .filter(|column| {
                let name: &str = column.name().as_str();
                row_count_changed
                    || affected_columns.iter().any(|a| a == name)
                    || !self.columns.contains_key(name)
            })
            .cloned()
            .collect();
        let names: Vec<String> = stale.iter().map(|c| c.name().to_string()).collect();
        let mut fresh: IndexMap<String, ColumnStats> = names
            .into_iter()
            .zip(ColumnStats::compute_stats(&DataFrame::new(
                dataframe.height(),
                stale,
            )?)?)
            .collect();

        let data: IndexMap<String, Arc<ColumnInfo>> = dataframe
            .columns()
            .iter()
            .map(|column| {
                let name: String = column.name().to_string();
                let info: Arc<ColumnInfo> = match fresh.swap_remove(&name) {
                    Some(stats) => Arc::new(ColumnInfo::with_stats(column, stats)),
                    None => Arc::clone(&self.columns[&name]),
                };
                (name, info)
            })
            .collect();

        Ok(Metadata {
            filepath: None,
            height: new_dataset.dataframe.height(),
            width: new_dataset.dataframe.width(),
            dataset_cache: DatasetCache::new(data),
            plot_cache: PlotCache::default(),
        })
    }

    /// One row per column: position, name, non-null count, dtype.
    pub fn info_table(&self, height: usize, width: usize) -> TableView {
        let rows: Vec<Vec<String>> = self
            .columns
            .iter()
            .enumerate()
            .map(|(i, (name, info))| {
                vec![
                    i.to_string(),
                    name.clone(),
                    (height - info.null_count).to_string(),
                    format!("{:?}", info.dtype),
                ]
            })
            .collect();
        TableView::new(
            &format!("Dataset: {height} rows x {width} columns"),
            &["#", "columns", "non-null count", "dtypes"].map(String::from),
            &rows,
        )
    }

    /// One row per numeric column whose stats are computed: count, mean, std, min, quartiles, max.
    pub fn describe_table(&self, height: usize) -> TableView {
        let rows: Vec<Vec<String>> = self
            .columns
            .iter()
            .filter_map(|(name, info)| match info.stats.get() {
                Some(ColumnStats::Numeric(s)) => Some(vec![
                    name.clone(),
                    (height - info.null_count).to_string(),
                    format!("{:.4}", s.mean),
                    format!("{:.4}", s.std),
                    format!("{:.4}", s.min),
                    format!("{:.4}", s.quantile_25),
                    format!("{:.4}", s.median),
                    format!("{:.4}", s.quantile_75),
                    format!("{:.4}", s.max),
                ]),
                _ => None,
            })
            .collect();
        let caption: &str = if rows.is_empty() {
            "No numeric columns with computed stats."
        } else {
            ""
        };
        TableView::new(
            caption,
            &[
                "columns", "count", "mean", "std", "min", "25%", "50%", "75%", "max",
            ]
            .map(String::from),
            &rows,
        )
    }

    /// Every cached stat of every column not in `skip`, under a caption with the file and shape.
    pub fn metadata_table(
        &self,
        skip: &[&str],
        filepath: &str,
        height: usize,
        width: usize,
    ) -> TableView {
        let mut rows: Vec<Vec<String>> = Vec::new();
        for (name, info) in &self.columns {
            if skip.contains(&name.as_str()) {
                continue;
            }
            let (mean_mode, min_max, std_var, quantile, unique_top3): (
                String,
                String,
                String,
                String,
                String,
            ) = match info.stats.get() {
                None => ("-".into(), "-".into(), "-".into(), "-".into(), "-".into()),
                Some(ColumnStats::Numeric(s)) => (
                    format!("mean = {:.2}\nmedian = {:.2}", s.mean, s.median),
                    format!("min = {:.2}\nmax = {:.2}", s.min, s.max),
                    format!("std = {:.2}\nvariance = {:.2}", s.std, s.variance),
                    format!(
                        "25% = {:.2}\n50% = {:.2}\n75% = {:.2}",
                        s.quantile_25, s.median, s.quantile_75
                    ),
                    "-".to_string(),
                ),
                Some(ColumnStats::Categorical(s)) => {
                    let top3: Vec<String> = s
                        .categories
                        .iter()
                        .take(3)
                        .map(|(name, count)| format!("'{name}': {count}"))
                        .collect();
                    (
                        format!("mode = '{}'", s.mode),
                        "-".to_string(),
                        "-".to_string(),
                        "-".to_string(),
                        format!(
                            "unique = {}\ntop3:\n  {}",
                            s.unique_count,
                            top3.join("\n  ")
                        ),
                    )
                }
            };
            rows.push(vec![
                name.clone(),
                format!("{:?}", info.dtype),
                info.null_count.to_string(),
                mean_mode,
                min_max,
                std_var,
                quantile,
                unique_top3,
            ]);
        }
        TableView::new(
            &format!("Filepath: {filepath}\nDataset: {height} rows x {width} columns"),
            &[
                "columns",
                "dtypes",
                "nulls",
                "mean/median/mode",
                "min/max",
                "std/variance",
                "quantiles",
                "unique/top3",
            ]
            .map(String::from),
            &rows,
        )
    }
}
