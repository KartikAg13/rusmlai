use std::path::Path;

use polars::frame::DataFrame;

use crate::{
    data::{dataset::metadata::Metadata, reader::Reader},
    fatal,
};

pub mod metadata;

pub struct Dataset {
    pub df: DataFrame,
    pub metadata: Metadata,
}

impl Dataset {
    pub fn new(dataframe: DataFrame) -> Self {
        let metadata: Metadata = Metadata::new(&dataframe);
        Self {
            df: dataframe,
            metadata,
        }
    }

    pub fn read_csv(path: impl AsRef<Path>) -> Dataset {
        Dataset::new(Reader::new(path).finish().unwrap_or_else(|e| fatal!(e)))
    }

    pub fn shape(&self) -> (usize, usize) {
        self.df.shape()
    }
}
