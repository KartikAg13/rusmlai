use std::path::{Path, PathBuf};

use polars::{
    frame::DataFrame,
    io::{
        SerReader,
        csv::read::{CsvParseOptions, CsvReadOptions},
    },
};

use crate::dataset::error::DatasetError;

/// Reads a CSV with Polars options, for files that [`read_csv`](crate::prelude::read_csv)'s
/// defaults don't fit (another separator, no header, ...). Returns a Polars `DataFrame`;
/// [`Dataset::new`](crate::prelude::Dataset::new) turns it into a `Dataset`.
///
/// # Examples
///
/// Basic: the same defaults as `read_csv`:
/// ```
/// use rusmlai::dataset::reader::Reader;
///
/// let frame = Reader::new("tests/fixtures/data/students.csv").finish()?;
/// assert_eq!(frame.shape(), (12, 6));
/// # Ok::<(), rusmlai::prelude::DatasetError>(())
/// ```
///
/// Advanced: read only some columns, then compute the stats:
/// ```
/// use polars::prelude::CsvReadOptions;
/// use rusmlai::{dataset::reader::Reader, prelude::*};
///
/// let options = CsvReadOptions::default().with_columns(Some(["cgpa".into(), "passed".into()].into()));
/// let frame = Reader::new("tests/fixtures/data/students.csv").with_read_options(options).finish()?;
/// let ds = Dataset::new(frame, "tests/fixtures/data/students.csv");
/// assert_eq!(ds.columns(), ["cgpa", "passed"]);
/// # Ok::<(), DatasetError>(())
/// ```
///
/// Edge case: a semicolon file read with the default comma gives one column:
/// ```
/// use polars::prelude::CsvParseOptions;
/// use rusmlai::dataset::reader::Reader;
/// # let dir = std::env::temp_dir().join("rusmlai-reader-doc");
/// # std::fs::create_dir_all(&dir).unwrap();
/// let path = dir.join("semi.csv");
/// std::fs::write(&path, "a;b\n1;2\n").unwrap();
/// assert_eq!(Reader::new(&path).finish()?.width(), 1);
/// let semicolon = CsvParseOptions::default().with_separator(b';');
/// assert_eq!(Reader::new(&path).with_parse_options(semicolon).finish()?.width(), 2);
/// # Ok::<(), rusmlai::prelude::DatasetError>(())
/// ```
pub struct Reader {
    pub path: PathBuf,
    pub read_options: CsvReadOptions,
    pub parse_options: CsvParseOptions,
}

impl Reader {
    /// A reader with Polars' default options: comma separated, a header row, dtypes inferred.
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            read_options: CsvReadOptions::default(),
            parse_options: CsvParseOptions::default(),
        }
    }

    /// Replaces the read options (columns, rows, header, schema...).
    pub fn with_read_options(mut self, read: CsvReadOptions) -> Self {
        self.read_options = read;
        self
    }

    /// Replaces the parse options (separator, quote char, decimal comma...).
    pub fn with_parse_options(mut self, parse: CsvParseOptions) -> Self {
        self.parse_options = parse;
        self
    }

    /// Reads the file.
    ///
    /// # Errors
    ///
    /// [`DatasetError::FileNotFound`] when there is no file at the path, and
    /// [`DatasetError::Polars`] when Polars cannot parse it.
    pub fn finish(self) -> Result<DataFrame, DatasetError> {
        if !self.path.is_file() {
            let cwd: String = match std::env::current_dir() {
                Ok(p) => p.display().to_string(),
                Err(_) => "unknown".to_string(),
            };
            return Err(DatasetError::FileNotFound {
                path: self.path.display().to_string(),
                cwd,
            });
        }
        Ok(self
            .read_options
            .with_parse_options(self.parse_options)
            .try_into_reader_with_file_path(Some(self.path))?
            .finish()?)
    }
}
