use comfy_table::{Table, presets::UTF8_FULL};
use polars::frame::{DataFrame, column::Column};
use std::{ffi::OsString, path::Path, sync::Arc};

use crate::data::dataset::{
    Dataset,
    metadata::{dto::MetadataDto, info::ColumnInfo, stats::ColumnStats},
};

pub mod dto;
pub mod info;
pub mod stats;

pub struct Metadata {
    pub filepath: Option<OsString>,
    pub height: usize,
    pub width: usize,
    pub columns: Vec<Arc<ColumnInfo>>,
}

impl From<MetadataDto> for Metadata {
    fn from(dto: MetadataDto) -> Self {
        Self {
            filepath: Some(OsString::from(dto.filepath)),
            height: dto.height,
            width: dto.width,
            columns: dto
                .columns
                .into_iter()
                .map(|c| Arc::new(c.into()))
                .collect(),
        }
    }
}

impl Metadata {
    pub fn new(dataframe: &DataFrame, path: impl AsRef<Path>) -> Self {
        let width: usize = dataframe.width();
        let mut data: Vec<Arc<ColumnInfo>> = Vec::with_capacity(width);
        for column in dataframe.columns() {
            let info: ColumnInfo = ColumnInfo::new(&column);
            info.initialize(column).unwrap();
            data.push(Arc::new(info));
        }
        Self {
            filepath: Some(path.as_ref().as_os_str().to_os_string()),
            height: dataframe.height(),
            width: width,
            columns: data,
        }
    }

    pub fn load_cache(path: impl AsRef<Path>) -> Option<Metadata> {
        MetadataDto::try_load_cache(path)
    }

    pub fn write_cache(&self) {
        MetadataDto::try_write_cache(&self).unwrap()
    }

    pub fn after_transform(
        &self,
        new_dataset: &Dataset,
        affected_columns: Vec<String>,
        row_count_changed: bool,
    ) -> Metadata {
        let columns: Vec<Arc<ColumnInfo>> = new_dataset
            .df
            .get_column_names()
            .iter()
            .map(|name| {
                let name_s: String = name.to_string();
                let is_affected: bool = row_count_changed || affected_columns.contains(&name_s);

                if !is_affected {
                    if let Some(existing) = self.columns.iter().find(|c| c.name == name_s) {
                        return Arc::clone(existing);
                    }
                }
                let column: &Column = new_dataset.df.column(name).unwrap();
                Arc::new(ColumnInfo::new(column))
            })
            .collect();

        Metadata {
            filepath: None,
            height: new_dataset.df.height(),
            width: new_dataset.df.width(),
            columns,
        }
    }

    pub fn try_info(&self) {
        let mut table: Table = Table::new();
        table.load_preset(UTF8_FULL);
        table.set_header(vec!["#", "columns", "non-null count", "dtypes"]);

        for (i, col) in self.columns.iter().enumerate() {
            let non_null: usize = self.height - col.null_count;
            table.add_row(vec![
                i.to_string(),
                col.name.clone(),
                format!("{}", non_null),
                format!("{:?}", col.dtype),
            ]);
        }

        println!(
            "Dataset: {} rows x {} columns\n{table}",
            self.height, self.width
        );
    }

    pub fn try_describe(&self) {
        let mut table: Table = Table::new();
        table.load_preset(UTF8_FULL);
        table.set_header(vec![
            "columns", "count", "mean", "std", "min", "25%", "50%", "75%", "max",
        ]);

        let mut any_numeric: bool = false;

        for col in &self.columns {
            if let Some(ColumnStats::Numeric(s)) = col.stats.get() {
                any_numeric = true;
                let count: usize = self.height - col.null_count;
                table.add_row(vec![
                    col.name.clone(),
                    count.to_string(),
                    format!("{:.4}", s.mean),
                    format!("{:.4}", s.std),
                    format!("{:.4}", s.min),
                    format!("{:.4}", s.quantile_25),
                    format!("{:.4}", s.median),
                    format!("{:.4}", s.quantile_75),
                    format!("{:.4}", s.max),
                ]);
            }
        }

        if !any_numeric {
            println!("No numeric columns with computed stats.");
            return;
        }

        println!("{table}");
    }

    pub fn try_metadata(&self, skip: &[&str]) {
        let mut table: Table = Table::new();
        table.load_preset(UTF8_FULL);
        table.set_header(vec![
            "columns",
            "dtypes",
            "nulls",
            "mean/median/mode",
            "min/max",
            "std/variance",
            "quantiles",
            "unique/top3",
        ]);

        for col in &self.columns {
            if skip.contains(&col.name.as_str()) {
                continue;
            }
            let (mean_mode, min_max, std_var, quantile, unique_top3): (
                String,
                String,
                String,
                String,
                String,
            ) = match col.stats.get() {
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
            table.add_row(vec![
                col.name.clone(),
                format!("{:?}", col.dtype),
                col.null_count.to_string(),
                mean_mode,
                min_max,
                std_var,
                quantile,
                unique_top3,
            ]);
        }

        println!(
            "\nFilepath: {}",
            self.filepath.clone().unwrap().into_string().unwrap()
        );
        println!(
            "Dataset: {} rows x {} columns\n\n{table}",
            self.height, self.width
        );
    }
}
