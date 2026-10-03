use std::path::Path;

use polars::frame::DataFrame;

use crate::{
    dataset::{error::DatasetError, reader::Reader},
    display::TableView,
    error::OrFatal,
    metadata::{
        Metadata,
        dataset::info::ColumnInfo,
        dataset::stats::{CategoricalStats, ColumnStats, NumericStats},
    },
};

pub mod error;
pub mod reader;

pub struct Dataset {
    pub dataframe: DataFrame,
    pub metadata: Metadata,
}

impl Dataset {
    #[track_caller]
    pub fn new(dataframe: DataFrame, path: impl AsRef<Path>) -> Self {
        let metadata: Metadata = Metadata::new(&dataframe, path);
        Self {
            dataframe,
            metadata,
        }
    }

    fn try_read_csv(path: impl AsRef<Path>) -> Result<Dataset, DatasetError> {
        let path: &Path = path.as_ref();
        let dataframe: DataFrame = Reader::new(path).finish()?;
        let metadata: Metadata = Metadata::try_load(&dataframe, path)?;
        Ok(Dataset {
            dataframe,
            metadata,
        })
    }

    fn try_select(&self, columns: &[&str]) -> Result<Dataset, DatasetError> {
        if columns.is_empty() {
            return Err(DatasetError::EmptySelect);
        }
        for column in columns {
            if !self.metadata.dataset_cache.columns.contains_key(*column) {
                return Err(DatasetError::ColumnNotFound {
                    requested: column.to_string(),
                    available: self.columns().join(","),
                });
            }
        }
        Ok(Self {
            dataframe: self.dataframe.select(columns)?,
            metadata: self.metadata.select(columns),
        })
    }

    fn try_drop(&self, columns: &[&str]) -> Result<Dataset, DatasetError> {
        for column in columns {
            if !self.metadata.dataset_cache.columns.contains_key(*column) {
                return Err(DatasetError::ColumnNotFound {
                    requested: column.to_string(),
                    available: self.columns().join(","),
                });
            }
        }

        let remaining: Vec<&str> = self
            .metadata
            .dataset_cache
            .columns
            .keys()
            .map(String::as_str)
            .filter(|name| !columns.contains(name))
            .collect();

        self.try_select(&remaining)
    }

    pub fn try_column_info(&self, column_name: &str) -> Result<&ColumnInfo, DatasetError> {
        match self
            .metadata
            .dataset_cache
            .columns
            .get(column_name)
            .map(|c| c.as_ref())
        {
            Some(c) => Ok(c),
            None => Err(DatasetError::ColumnNotFound {
                requested: column_name.to_string(),
                available: self.columns().join(","),
            }),
        }
    }

    pub fn try_numeric_stats(&self, column_name: &str) -> Result<NumericStats, DatasetError> {
        let info: &ColumnInfo = self.try_column_info(column_name)?;
        match info.stats.get().unwrap() {
            ColumnStats::Categorical(_) => Err(DatasetError::ColumnNotNumeric {
                column: column_name.to_string(),
                actual_dtype: info.dtype.to_string(),
            }),
            ColumnStats::Numeric(n) => Ok(n.clone()),
        }
    }

    pub fn try_categorical_stats(
        &self,
        column_name: &str,
    ) -> Result<CategoricalStats, DatasetError> {
        let info: &ColumnInfo = self.try_column_info(column_name)?;
        match info.stats.get().unwrap() {
            ColumnStats::Numeric(_) => Err(DatasetError::ColumnNotCategorical {
                column: column_name.to_string(),
                actual_dtype: info.dtype.to_string(),
            }),
            ColumnStats::Categorical(c) => Ok(c.clone()),
        }
    }

    pub fn metadata(&self, skip_columns: &[&str]) {
        self.metadata.try_metadata(skip_columns).show()
    }

    pub fn info(&self) {
        self.metadata.try_info().show()
    }

    pub fn describe(&self) {
        self.metadata.try_describe().show()
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.metadata.height, self.metadata.width)
    }

    pub fn height(&self) -> usize {
        self.metadata.height
    }

    pub fn width(&self) -> usize {
        self.metadata.width
    }

    pub fn head(&self, n: usize) {
        frame_view(&self.dataframe.head(Some(n))).show()
    }

    pub fn tail(&self, n: usize) {
        frame_view(&self.dataframe.tail(Some(n))).show()
    }

    pub fn columns(&self) -> Vec<String> {
        self.metadata
            .dataset_cache
            .columns
            .keys()
            .cloned()
            .collect()
    }

    pub fn dtypes(&self) -> Vec<String> {
        self.metadata
            .dataset_cache
            .columns
            .values()
            .map(|info| info.dtype.to_string())
            .collect()
    }

    #[track_caller]
    pub fn select(&self, columns: &[&str]) -> Dataset {
        self.try_select(columns).or_fatal()
    }

    #[track_caller]
    pub fn drop(&self, columns: &[&str]) -> Dataset {
        self.try_drop(columns).or_fatal()
    }
}

/// Rows of a (small) frame as a table, with Polars' `shape: (rows, cols)` caption.
fn frame_view(frame: &DataFrame) -> TableView {
    let headers: Vec<String> = frame
        .get_column_names()
        .iter()
        .map(|name| name.to_string())
        .collect();
    let rows: Vec<Vec<String>> = (0..frame.height())
        .map(|i| {
            frame
                .columns()
                .iter()
                .map(|c| c.get(i).map_or("?".into(), |v| v.str_value().into_owned()))
                .collect()
        })
        .collect();
    let caption: String = format!("shape: ({}, {})", frame.height(), frame.width());
    TableView::new(&caption, &headers, &rows)
}

#[track_caller]
pub fn read_csv(path: impl AsRef<Path>) -> Dataset {
    Dataset::try_read_csv(path).or_fatal()
}
