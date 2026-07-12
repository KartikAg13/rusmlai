use std::path::{Path, PathBuf};

use polars::{
    frame::DataFrame,
    io::{
        SerReader,
        csv::read::{CsvParseOptions, CsvReadOptions},
    },
};

use crate::data::error::DataError;

pub mod parse;
pub mod read;

pub struct Reader {
    pub path: PathBuf,
    pub read_options: CsvReadOptions,
    pub parse_options: CsvParseOptions,
}

impl Reader {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            read_options: CsvReadOptions::default(),
            parse_options: CsvParseOptions::default(),
        }
    }

    pub fn finish(self) -> Result<DataFrame, DataError> {
        if !self.path.is_file() {
            return Err(DataError::FileNotFound(self.path.display().to_string()));
        }
        Ok(self
            .read_options
            .with_parse_options(self.parse_options)
            .try_into_reader_with_file_path(Some(self.path))?
            .finish()?)
    }
}
