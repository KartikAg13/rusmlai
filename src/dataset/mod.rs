use std::path::Path;

use polars::frame::DataFrame;

use crate::{
    dataset::{error::DatasetError, reader::Reader},
    fatal,
    metadata::{
        Metadata,
        info::ColumnInfo,
        stats::{CategoricalStats, ColumnStats, NumericStats},
    },
};

pub mod error;
pub mod reader;
pub mod utils;

pub struct Dataset {
    pub dataframe: DataFrame,
    pub metadata: Metadata,
}

impl Dataset {
    pub fn new(dataframe: DataFrame, path: impl AsRef<Path>) -> Self {
        let metadata: Metadata = Metadata::new(&dataframe, path);
        Self {
            dataframe: dataframe,
            metadata: metadata,
        }
    }

    fn try_read_csv(path: impl AsRef<Path>) -> Result<Dataset, DatasetError> {
        let path: &Path = path.as_ref();
        let dataframe: DataFrame = Reader::new(path).finish()?;
        let metadata: Metadata = match Metadata::load_cache(path) {
            Some(m) => {
                if m.width == dataframe.width() && m.height == dataframe.height() {
                    m
                } else {
                    let m: Metadata = Metadata::new(&dataframe, path);
                    let _ = m.write_cache();
                    m
                }
            }
            _ => {
                let m: Metadata = Metadata::new(&dataframe, path);
                let _ = m.write_cache();
                m
            }
        };
        Ok(Dataset {
            dataframe: dataframe,
            metadata: metadata,
        })
    }

    fn try_select(&self, columns: &[&str]) -> Result<Dataset, DatasetError> {
        if columns.is_empty() {
            return Err(DatasetError::EmptySelect);
        }
        for column in columns {
            if !self.metadata.columns.contains_key(*column) {
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
            if !self.metadata.columns.contains_key(*column) {
                return Err(DatasetError::ColumnNotFound {
                    requested: column.to_string(),
                    available: self.columns().join(","),
                });
            }
        }

        let remaining: Vec<&str> = self
            .metadata
            .columns
            .keys()
            .map(String::as_str)
            .filter(|name| !columns.contains(name))
            .collect();

        self.try_select(&remaining)
    }

    pub fn try_column_info(&self, column_name: &str) -> Result<&ColumnInfo, DatasetError> {
        match self.metadata.columns.get(column_name).map(|c| c.as_ref()) {
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
        self.metadata.try_metadata(skip_columns);
    }

    pub fn info(&self) {
        self.metadata.try_info();
    }

    pub fn describe(&self) {
        self.metadata.try_describe();
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
        println!("{}", self.dataframe.head(Some(n)));
    }

    pub fn tail(&self, n: usize) {
        println!("{}", self.dataframe.tail(Some(n)));
    }

    pub fn columns(&self) -> Vec<String> {
        self.metadata.columns.keys().cloned().collect()
    }

    pub fn dtypes(&self) -> Vec<String> {
        self.metadata
            .columns
            .values()
            .map(|info| info.dtype.to_string())
            .collect()
    }

    pub fn select(&self, columns: &[&str]) -> Dataset {
        self.try_select(columns)
            .unwrap_or_else(|e| fatal!(e, "src/data/dataset/mod.rs (select)"))
    }

    pub fn drop(&self, columns: &[&str]) -> Dataset {
        self.try_drop(columns)
            .unwrap_or_else(|e| fatal!(e, "src/data/dataset/mod.rs (drop)"))
    }
}

pub fn read_csv(path: impl AsRef<Path>) -> Dataset {
    Dataset::try_read_csv(path).unwrap_or_else(|e| fatal!(e, "src/data/dataset/mod.rs (read_csv)"))
}
