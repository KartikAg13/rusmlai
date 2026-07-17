use std::path::Path;

use polars::frame::DataFrame;

use crate::{
    data::{dataset::metadata::Metadata, error::DataError, reader::Reader},
    fatal,
};

pub mod metadata;

pub struct Dataset {
    df: DataFrame,
    metadata: Metadata,
}

impl Dataset {
    pub fn new(dataframe: DataFrame, path: impl AsRef<Path>) -> Self {
        let metadata: Metadata = Metadata::new(&dataframe, path);
        Self {
            df: dataframe,
            metadata,
        }
    }

    fn try_read_csv(path: impl AsRef<Path>) -> Result<Dataset, DataError> {
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
            df: dataframe,
            metadata,
        })
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
        println!("{}", self.df.head(Some(n)));
    }

    pub fn tail(&self, n: usize) {
        println!("{}", self.df.tail(Some(n)));
    }

    pub fn columns(&self) -> Vec<String> {
        self.metadata
            .columns
            .iter()
            .map(|c| c.name.as_str().into())
            .collect()
    }

    pub fn dtypes(&self) -> Vec<String> {
        self.metadata
            .columns
            .iter()
            .map(|c| c.dtype.to_string())
            .collect()
    }
}

pub fn read_csv(path: impl AsRef<Path>) -> Dataset {
    Dataset::try_read_csv(path).unwrap_or_else(|e| fatal!(e, "src/data/dataset/mod.rs (read_csv)"))
}
