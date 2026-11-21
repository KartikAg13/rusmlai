//! [`Dataset`]: a Polars table and its cached column stats, read with [`read_csv`].
//! Column selection, shape, the pandas-style printouts (`head`, `info`, `describe`) and
//! information gain live here. Charts are in [`crate::plot`].

use std::{fs::File, path::Path, sync::Arc};

use polars::{
    datatypes::DataType,
    frame::{DataFrame, column::Column},
    io::{
        SerReader, SerWriter,
        csv::{
            read::{CsvParseOptions, CsvReadOptions},
            write::CsvWriter,
        },
    },
    lazy::{
        dsl::{DslPlan, Engine, Expr, len, lit, when},
        frame::{IntoLazy, LazyFrame},
    },
};

use crate::utils::col;
use crate::{
    dataset::error::DatasetError,
    display::TableView,
    error::OrFatal,
    metadata::{
        Metadata,
        cache_on,
        constants::{DATASET_CACHE, PLOTS_CACHE},
        dataset::{
            dto::DatasetCacheDTO,
            info::ColumnInfo,
            role::ColumnRole,
            stats::{CategoricalStats, ColumnStats, NumericStats},
        },
        plot::PlotCache,
        utils::{cache_path, file_fingerprint, print_saved, unwrap_f64, write_cache, write_jsonl},
    },
};

pub mod clean;
pub mod error;
pub mod read;
pub mod split;

/// Equal-width bins of a numeric column in `information_gain`.
const BINS: i64 = 256;

/// A table plus its cached column stats. (pandas: `DataFrame`)
///
/// [`Dataset::dataframe`] is the Polars table, read-only, so the cached stats always describe
/// it. To change the rows, change a copy with Polars and make a new `Dataset` of it with
/// [`Dataset::new`]. `metadata` holds the stats of every column and the plot cache.
/// [`read_csv`] loads both, and reads the stats from `metadata/` next to the CSV when they are
/// there. `select` and `drop` share the stats with the new `Dataset`, so they copy no numbers.
/// `println!("{ds}")` prints the table like `print(df)` in pandas.
///
/// ```
/// # use rusmlai::prelude::*;
/// let ds = read_csv("tests/fixtures/data/students.csv");
/// println!("{ds}"); // shape: (12, 6), then the first and last rows
/// assert!(format!("{ds}").contains("Aarav Shah"));
/// ```
///
/// Python:
///
/// ```python
/// import pandas as pd
/// ds = pd.read_csv("tests/fixtures/data/students.csv")
/// print(ds)
/// assert "Aarav Shah" in str(ds)
/// ```
pub struct Dataset {
    pub(crate) dataframe: DataFrame,
    pub metadata: Metadata,
}

impl std::fmt::Display for Dataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.dataframe, f)
    }
}

impl std::fmt::Debug for Dataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.dataframe, f)
    }
}

impl Dataset {
    /// A `Dataset` from a Polars `DataFrame` you already have. (pandas: `pd.DataFrame(data)`)
    ///
    /// The stats of every column are computed now, and nothing is read from disk. `path` is
    /// the CSV this data belongs to: `metadata()` shows it, charts auto-save in its folder, and
    /// model fits use its model cache. `""` means none, so nothing is written either.
    ///
    /// # Panics
    ///
    /// If Polars cannot compute the stats.
    ///
    /// # Examples
    ///
    /// Basic: a small table built in code:
    /// ```
    /// use rusmlai::prelude::*;
    ///
    /// let frame = df!("x" => [1.0, 2.0, 3.0], "label" => ["a", "b", "a"]).unwrap();
    /// let ds = Dataset::new(frame, "");
    /// assert_eq!(ds.shape(), (3, 2));
    /// ```
    ///
    /// Advanced: the stats are ready at once:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let frame = df!("x" => [1.0, 2.0, 3.0, 10.0]).unwrap();
    /// let ds = Dataset::new(frame, "");
    /// let x = ds.try_numeric_stats("x")?;
    /// assert_eq!((x.mean, x.median), (4.0, 2.5));
    /// # Ok::<(), DatasetError>(())
    /// ```
    ///
    /// Edge case: a table with no rows gets NaN stats, not a panic:
    /// ```
    /// # use rusmlai::polars::prelude::*;
    /// # use rusmlai::prelude::*;
    /// let frame = df!("x" => Vec::<f64>::new()).unwrap();
    /// let ds = Dataset::new(frame, "");
    /// assert!(ds.try_numeric_stats("x").unwrap().mean.is_nan());
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.DataFrame({"x": [1.0, 2.0, 3.0], "label": ["a", "b", "a"]})
    /// assert ds.shape == (3, 2)
    /// ```
    #[track_caller]
    pub fn new(dataframe: DataFrame, path: impl AsRef<Path>) -> Self {
        let metadata: Metadata = Metadata::new(&dataframe, path);
        Self {
            dataframe,
            metadata,
        }
    }

    /// [`Dataset::select`] that returns an error instead of panicking.
    ///
    /// # Errors
    ///
    /// - [`DatasetError::EmptySelect`] when `columns` is empty.
    /// - [`DatasetError::ColumnNotFound`] for an unknown name. The message lists the columns
    ///   that exist.
    ///
    /// # Examples
    ///
    /// Basic:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let scores = ds.try_select(&["cgpa", "passed"])?;
    /// assert_eq!(scores.columns(), ["cgpa", "passed"]);
    /// # Ok::<(), DatasetError>(())
    /// ```
    ///
    /// Advanced: a name from user input, with the error handled:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// match ds.try_select(&["grade"]) {
    ///     Err(DatasetError::ColumnNotFound { available, .. }) => assert!(available.contains("cgpa")),
    ///     _ => unreachable!(),
    /// }
    /// ```
    ///
    /// Edge case: an empty list is an error, not an empty table:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// assert!(matches!(ds.try_select(&[]), Err(DatasetError::EmptySelect)));
    /// ```
    pub fn try_select(&self, columns: &[&str]) -> Result<Dataset, DatasetError> {
        if columns.is_empty() {
            return Err(DatasetError::EmptySelect);
        }
        let picked: Vec<Column> = columns
            .iter()
            .map(|name| self.column(name).cloned())
            .collect::<Result<Vec<Column>, DatasetError>>()?;
        Ok(Self {
            dataframe: DataFrame::new(self.dataframe.height(), picked)?,
            metadata: self.metadata.select(columns),
        })
    }

    /// [`Dataset::drop`] that returns an error instead of panicking.
    ///
    /// # Errors
    ///
    /// - [`DatasetError::ColumnNotFound`] for an unknown name.
    /// - [`DatasetError::EmptySelect`] when every column is dropped.
    ///
    /// # Examples
    ///
    /// Basic:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let features = ds.try_drop(&["id", "name"])?;
    /// assert_eq!(features.columns(), ["age", "cgpa", "batch", "passed"]);
    /// # Ok::<(), DatasetError>(())
    /// ```
    ///
    /// Advanced: an empty list keeps every column:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// assert_eq!(ds.try_drop(&[])?.columns(), ds.columns());
    /// # Ok::<(), DatasetError>(())
    /// ```
    ///
    /// Edge case: a typo is an error, not a silent no-op (pandas `errors="raise"`):
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// assert!(matches!(ds.try_drop(&["nmae"]), Err(DatasetError::ColumnNotFound { .. })));
    /// ```
    pub fn try_drop(&self, columns: &[&str]) -> Result<Dataset, DatasetError> {
        for column in columns {
            self.try_column_info(column)?;
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

    /// A column's dtype, null count and stats, as cached.
    ///
    /// # Errors
    ///
    /// [`DatasetError::ColumnNotFound`] for an unknown name.
    ///
    /// # Examples
    ///
    /// Basic:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let age = ds.try_column_info("age")?;
    /// assert_eq!(age.null_count, 1);
    /// # Ok::<(), DatasetError>(())
    /// ```
    ///
    /// Advanced: branch on the dtype:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let numeric: Vec<String> = ds
    ///     .columns()
    ///     .into_iter()
    ///     .filter(|c| ds.try_column_info(c).is_ok_and(|info| info.dtype.is_numeric()))
    ///     .collect();
    /// assert_eq!(numeric, ["id", "age", "cgpa"]);
    /// ```
    ///
    /// Edge case: names are case-sensitive:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// assert!(ds.try_column_info("Age").is_err());
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// assert ds["age"].isna().sum() == 1
    /// assert ds.select_dtypes("number").columns.tolist() == ["id", "age", "cgpa"]
    /// ```
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

    /// What `column` is for ([`ColumnRole`]): `fit` leaves out identifiers and constants, with
    /// a warning that names them. `read_csv` decides it from the stats; computes the column's
    /// stats first when they are missing.
    ///
    /// ```
    /// use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// assert_eq!(ds.role("id"), ColumnRole::Identifier);
    /// assert_eq!(ds.role("age"), ColumnRole::Feature);
    /// // The two sides of a split keep the roles of the whole table.
    /// let (train, _) = train_test_split(&ds, 0.25, 42, "");
    /// assert_eq!(train.role("id"), ColumnRole::Identifier);
    /// ```
    ///
    /// Python has no column roles: you drop such columns yourself, `ds.drop(columns=["id"])`.
    #[track_caller]
    pub fn role(&self, column: &str) -> ColumnRole {
        self.try_role(column).or_fatal()
    }

    /// [`Dataset::role`] that returns an error instead of panicking.
    ///
    /// # Errors
    ///
    /// [`DatasetError::ColumnNotFound`] when there is no such column.
    pub fn try_role(&self, column: &str) -> Result<ColumnRole, DatasetError> {
        let info: &ColumnInfo = self.try_column_info(column)?;
        if info.role.get().is_none() && info.stats.get().is_none() {
            info.ensure_stats(self.dataframe.column(column)?)?;
        }
        Ok(info.role())
    }

    /// Sets what `column` is for, in place of the role `read_csv` found. Only this dataset
    /// changes (its splits and `to_csv` keep it); the table it came from keeps its own.
    ///
    /// ```
    /// use rusmlai::prelude::*;
    /// let mut ds = read_csv("tests/fixtures/data/students.csv").select(&["id", "age", "cgpa"]);
    /// ds.set_role("id", ColumnRole::Feature); // learn from the id after all
    /// assert_eq!(ds.role("id"), ColumnRole::Feature);
    /// ds.set_role("age", ColumnRole::Identifier); // and leave age out of every fit
    /// assert_eq!(ds.role("age"), ColumnRole::Identifier);
    /// assert_eq!(read_csv("tests/fixtures/data/students.csv").role("id"), ColumnRole::Identifier);
    /// ```
    ///
    /// Python has no column roles: you choose the feature columns yourself, `ds[["id", "cgpa"]]`.
    #[track_caller]
    pub fn set_role(&mut self, column: &str, role: ColumnRole) {
        self.try_set_role(column, role).or_fatal()
    }

    /// [`Dataset::set_role`] that returns an error instead of panicking.
    ///
    /// # Errors
    ///
    /// [`DatasetError::ColumnNotFound`] when there is no such column.
    pub fn try_set_role(&mut self, column: &str, role: ColumnRole) -> Result<(), DatasetError> {
        let info: ColumnInfo = self.try_column_info(column)?.with_role(Some(role));
        // `insert` on a key that is there keeps its place in the column order.
        self.metadata.dataset_cache.columns.insert(column.to_string(), Arc::new(info));
        Ok(())
    }

    /// Writes the table to a CSV file at `path` (tabs for a `.tsv`) and makes that file this
    /// dataset's: its stats and an empty chart cache are written next to it, in `metadata/`,
    /// so the next `read_csv` of it computes nothing. From then on its charts and models are
    /// cached next to the new file. Roles set with [`Dataset::set_role`] are kept.
    /// (pandas: `df.to_csv(path, index=False)`)
    ///
    /// ```
    /// use rusmlai::prelude::*;
    /// let dir = std::env::temp_dir().join("rusmlai-doc-to-csv");
    /// std::fs::create_dir_all(&dir).unwrap();
    /// let mut ds = read_csv("tests/fixtures/data/students.csv").drop(&["name"]);
    /// let path = dir.join("students-clean.csv");
    /// ds.to_csv(&path);
    /// assert!(dir.join("metadata/students-clean.json").is_file());
    /// let back = read_csv(&path); // reads the stats back: nothing to compute
    /// assert_eq!(back.shape(), ds.shape());
    /// assert_eq!(back.try_numeric_stats("cgpa").unwrap().median, 7.4);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import os
    /// import tempfile
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv").drop(columns=["name"])
    /// path = os.path.join(tempfile.gettempdir(), "rusmlai-doc-to-csv", "students-clean.csv")
    /// os.makedirs(os.path.dirname(path), exist_ok=True)
    /// ds.to_csv(path, index=False)
    /// back = pd.read_csv(path)
    /// assert back.shape == ds.shape
    /// assert back["cgpa"].median() == 7.4
    /// ```
    #[track_caller]
    pub fn to_csv(&mut self, path: impl AsRef<Path>) {
        self.try_to_csv(path).or_fatal()
    }

    /// [`Dataset::to_csv`] that returns an error instead of panicking. A cache file that
    /// can't be written is only a warning.
    ///
    /// # Errors
    ///
    /// [`DatasetError::Write`] when the CSV can't be written, [`DatasetError::Polars`] when
    /// Polars fails.
    pub fn try_to_csv(&mut self, path: impl AsRef<Path>) -> Result<(), DatasetError> {
        let path: &Path = path.as_ref();
        let failed = |e: &dyn std::fmt::Display| DatasetError::Write {
            path: path.display().to_string(),
            reason: e.to_string(),
        };
        self.try_fill_stats()?;
        let tab: bool = path.extension().is_some_and(|e| e.eq_ignore_ascii_case("tsv"));
        // Polars writes no date or time type to CSV: they go out as ISO text, like pandas.
        let mut frame: DataFrame = self.dataframe.clone();
        for name in self.columns() {
            if frame.column(&name)?.dtype().is_temporal() {
                let text: Column = frame.column(&name)?.cast(&DataType::String)?;
                frame.with_column(text)?;
            }
        }
        let mut file: File = File::create(path).map_err(|e| failed(&e))?;
        CsvWriter::new(&mut file)
            .include_header(true)
            .with_separator(if tab { b'\t' } else { b',' })
            .finish(&mut frame)?;
        file.sync_all().map_err(|e| failed(&e))?;
        drop(file);
        self.metadata.filepath = Some(path.as_os_str().to_os_string());
        self.metadata.plot_cache = PlotCache::default();
        if !cache_on() {
            return Ok(());
        }
        // What `read_csv` will make of each column: one it reads back as another type (a
        // date, Polars categories) keeps no stats there, and gets them on first use.
        let read_back: DataFrame = CsvReadOptions::default()
            .with_n_rows(Some(1))
            .with_parse_options(CsvParseOptions::default().with_separator(if tab { b'\t' } else { b',' }))
            .try_into_reader_with_file_path(Some(path.to_path_buf()))?
            .finish()?;
        let columns = (self.metadata.dataset_cache.columns.iter())
            .map(|(name, info)| {
                let dtype: Option<&DataType> = read_back.column(name).ok().map(|c| c.dtype());
                let kept: Arc<ColumnInfo> = match dtype {
                    Some(dtype) if dtype != &info.dtype => Arc::new(ColumnInfo {
                        dtype: dtype.clone(),
                        stats: Default::default(),
                        ..info.with_role(info.role.get().copied())
                    }),
                    _ => Arc::clone(info),
                };
                (name.clone(), kept)
            })
            .collect();
        let fingerprint: (u64, u64) = file_fingerprint(path).map_err(|e| failed(&e))?;
        let stats_path = cache_path(path, DATASET_CACHE);
        let json: Vec<u8> = serde_json::to_vec_pretty(&DatasetCacheDTO::new(fingerprint, &columns))
            .map_err(|e| failed(&e))?;
        match write_cache(&stats_path, &json) {
            Ok(()) => print_saved(&stats_path),
            Err(e) => crate::error::warn(format_args!("could not write {}: {e}", stats_path.display())),
        }
        let plots_path = cache_path(path, PLOTS_CACHE);
        match write_jsonl::<()>(&plots_path, fingerprint, &[]) {
            Ok(()) => self.metadata.plot_cache.file = Some(plots_path),
            Err(e) => crate::error::warn(format_args!("could not write {}: {e}", plots_path.display())),
        }
        Ok(())
    }

    /// Mean, median, std, variance, min, max and quartiles of a number column.
    /// (pandas: `df[col].describe()`)
    ///
    /// Read from the cache, so it costs nothing after `read_csv`. Nulls are left out, std and
    /// variance use `ddof = 1` like pandas, and the quartiles are linear like
    /// `np.quantile`.
    ///
    /// # Errors
    ///
    /// - [`DatasetError::ColumnNotFound`] for an unknown name.
    /// - [`DatasetError::ColumnNotNumeric`] for a text or true/false column.
    ///
    /// # Examples
    ///
    /// Basic:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let age = ds.try_numeric_stats("age")?;
    /// assert_eq!((age.min, age.median, age.max), (19.0, 21.0, 24.0));
    /// # Ok::<(), DatasetError>(())
    /// ```
    ///
    /// Advanced: an interquartile-range outlier rule:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let cgpa = ds.try_numeric_stats("cgpa")?;
    /// let iqr: f64 = cgpa.quantile_75 - cgpa.quantile_25;
    /// let (low, high) = (cgpa.quantile_25 - 1.5 * iqr, cgpa.quantile_75 + 1.5 * iqr);
    /// assert!(low < cgpa.min && cgpa.max < high); // no outliers here
    /// # Ok::<(), DatasetError>(())
    /// ```
    ///
    /// Edge case: a text column is an error that names its dtype:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let err = ds.try_numeric_stats("batch").unwrap_err();
    /// assert_eq!(err.to_string(), "column 'batch' is not numeric (found str)");
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// age = ds["age"]
    /// assert (age.min(), age.median(), age.max()) == (19.0, 21.0, 24.0)
    ///
    /// cgpa = ds["cgpa"]
    /// q25, q75 = cgpa.quantile([0.25, 0.75])
    /// iqr = q75 - q25
    /// assert q25 - 1.5 * iqr < cgpa.min() and cgpa.max() < q75 + 1.5 * iqr  # no outliers here
    /// ```
    pub fn try_numeric_stats(&self, column_name: &str) -> Result<&NumericStats, DatasetError> {
        let (info, stats): (&ColumnInfo, &ColumnStats) = self.try_stats(column_name)?;
        match stats {
            ColumnStats::Categorical(_) => Err(DatasetError::ColumnNotNumeric {
                column: column_name.to_string(),
                actual_dtype: info.dtype.to_string(),
            }),
            ColumnStats::Numeric(n) => Ok(n),
        }
    }

    /// Value counts, unique count, mode and entropy of a text or true/false column.
    /// (pandas: `df[col].value_counts()`)
    ///
    /// Categories are sorted by count, most common first, and ties sort by name, so the mode
    /// is the same on every run. Nulls are not a category. Entropy is in nats, like
    /// `scipy.stats.entropy` of the counts.
    ///
    /// # Errors
    ///
    /// - [`DatasetError::ColumnNotFound`] for an unknown name.
    /// - [`DatasetError::ColumnNotCategorical`] for a number column.
    ///
    /// # Examples
    ///
    /// Basic:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let passed = ds.try_categorical_stats("passed")?;
    /// assert_eq!(passed.categories, [("true".to_string(), 8), ("false".to_string(), 4)]);
    /// # Ok::<(), DatasetError>(())
    /// ```
    ///
    /// Advanced: three batches of four rows: a tie, broken by name, and the largest entropy:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let batch = ds.try_categorical_stats("batch")?;
    /// assert_eq!((batch.mode.as_str(), batch.unique_count), ("B13", 3));
    /// assert!((batch.entropy - 3f64.ln()).abs() < 1e-12);
    /// # Ok::<(), DatasetError>(())
    /// ```
    ///
    /// Edge case: a null name is left out, so 11 names are counted from 12 rows:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let name = ds.try_categorical_stats("name")?;
    /// assert_eq!(name.unique_count, 11);
    /// assert!(name.categories.iter().any(|(n, _)| n == "Zoë Müller"));
    /// # Ok::<(), DatasetError>(())
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// assert ds["passed"].value_counts().to_dict() == {True: 8, False: 4}
    /// assert ds["batch"].mode()[0] == "B13"  # pandas also breaks ties by name
    /// assert ds["name"].nunique() == 11  # the null is left out, as in rusmlai
    /// ```
    pub fn try_categorical_stats(
        &self,
        column_name: &str,
    ) -> Result<&CategoricalStats, DatasetError> {
        let (info, stats): (&ColumnInfo, &ColumnStats) = self.try_stats(column_name)?;
        match stats {
            ColumnStats::Numeric(_) => Err(DatasetError::ColumnNotCategorical {
                column: column_name.to_string(),
                actual_dtype: info.dtype.to_string(),
            }),
            ColumnStats::Categorical(c) => Ok(c),
        }
    }

    /// The column's info and stats; stats that aren't there yet are computed now.
    pub fn try_stats(
        &self,
        column_name: &str,
    ) -> Result<(&ColumnInfo, &ColumnStats), DatasetError> {
        let info: &ColumnInfo = self.try_column_info(column_name)?;
        let stats: &ColumnStats = match info.stats.get() {
            Some(stats) => stats,
            None => info.ensure_stats(self.column(column_name)?)?,
        };
        Ok((info, stats))
    }

    /// Computes the stats that are still missing (the columns a transform made), in one pass
    /// for all of them.
    pub(crate) fn try_fill_stats(&self) -> Result<(), DatasetError> {
        let missing: Vec<Column> = self
            .dataframe
            .columns()
            .iter()
            .filter(|column| {
                self.try_column_info(column.name().as_str())
                    .is_ok_and(|info| info.stats.get().is_none())
            })
            .cloned()
            .collect();
        if missing.is_empty() {
            return Ok(());
        }
        let frame: DataFrame = DataFrame::new(self.dataframe.height(), missing)?;
        let stats: Vec<ColumnStats> = ColumnStats::compute_stats(&frame)?;
        for (column, stats) in frame.columns().iter().zip(stats) {
            self.try_column_info(column.name().as_str())?.fill(column, stats);
        }
        Ok(())
    }

    /// A column as a group-by key: text as it is, a number as its bin among `BINS` equal bins
    /// between its finite min and max. NaN and ±inf become null, so they are left out.
    fn try_key(&self, column: &str) -> Result<Expr, DatasetError> {
        let stats: &NumericStats = match self.try_numeric_stats(column) {
            Ok(stats) => stats,
            Err(DatasetError::ColumnNotNumeric { .. }) => return Ok(col(column)),
            Err(e) => return Err(e),
        };
        // The cached min and max keep ±inf; the bins need the finite ones.
        let (lo, hi): (f64, f64) = if stats.min.is_finite() && stats.max.is_finite() {
            (stats.min, stats.max)
        } else {
            let edges: DataFrame = self
                .dataframe
                .clone()
                .lazy()
                .select([col(column).cast(DataType::Float64)])
                .filter(col(column).is_finite())
                .select([col(column).min().alias("lo"), col(column).max().alias("hi")])
                .collect()?;
            (unwrap_f64(&edges, "lo")?, unwrap_f64(&edges, "hi")?)
        };
        let span: f64 = (hi - lo).max(1e-12); // a constant column: every value in bin 0
        let bin: Expr = ((col(column).cast(DataType::Float64) - lit(lo)) / lit(span)
            * lit(BINS as f64))
        .cast(DataType::Int64);
        Ok(when(bin.clone().gt(lit(BINS - 1)))
            .then(lit(BINS - 1))
            .otherwise(bin))
    }

    /// [`Dataset::information_gain`] that returns an error instead of panicking.
    ///
    /// One Polars run does the counting: for each column, one group-by of (column, target),
    /// then the two single-column counts from that small table.
    /// gain = H(column) + H(target) - H(both). Nothing is cached, so each call counts again.
    ///
    /// # Errors
    ///
    /// [`DatasetError::ColumnNotFound`] when `target` doesn't exist.
    ///
    /// # Examples
    ///
    /// Basic:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let gains = ds.select(&["cgpa", "batch", "passed"]).try_information_gain("passed")?;
    /// assert_eq!(gains[0].0, "cgpa");
    /// # Ok::<(), DatasetError>(())
    /// ```
    ///
    /// Advanced: keep the columns that carry at least 0.1 nats:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv").drop(&["id", "name"]);
    /// let useful: Vec<String> = ds
    ///     .try_information_gain("passed")?
    ///     .into_iter()
    ///     .filter(|(_, gain)| *gain >= 0.1)
    ///     .map(|(column, _)| column)
    ///     .collect();
    /// assert!(useful.contains(&"cgpa".to_string()));
    /// # Ok::<(), DatasetError>(())
    /// ```
    ///
    /// Edge case: a target that doesn't exist:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// assert!(matches!(
    ///     ds.try_information_gain("grade"),
    ///     Err(DatasetError::ColumnNotFound { .. })
    /// ));
    /// ```
    pub fn try_information_gain(&self, target: &str) -> Result<Vec<(String, f64)>, DatasetError> {
        let y: Expr = self.try_key(target)?;
        let features: Vec<String> = self.columns().into_iter().filter(|c| c != target).collect();
        let lazy: LazyFrame = self.dataframe.clone().lazy();
        let mut plans: Vec<DslPlan> = Vec::with_capacity(features.len() * 3);
        for feature in &features {
            let joint: LazyFrame = lazy
                .clone()
                .select([self.try_key(feature)?.alias("x"), y.clone().alias("y")])
                .drop_nulls(None)
                .group_by([col("x"), col("y")])
                .agg([len().alias("n")])
                .cache();
            for key in ["x", "y"] {
                plans.push(
                    joint
                        .clone()
                        .group_by([col(key)])
                        .agg([col("n").sum()])
                        .logical_plan,
                );
            }
            plans.push(joint.logical_plan);
        }
        let counts: Vec<DataFrame> = LazyFrame::collect_all_with_engine(
            plans,
            Engine::Auto,
            lazy.get_current_optimizations(),
        )?;
        let mut gains: Vec<(String, f64)> = features
            .into_iter()
            .zip(counts.chunks(3))
            .map(|(feature, c)| {
                let gain: f64 = entropy(&c[0])? + entropy(&c[1])? - entropy(&c[2])?;
                Ok((feature, gain.max(0.0))) // rounding can leave -1e-16 for unrelated columns
            })
            .collect::<Result<_, DatasetError>>()?;
        gains.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(gains)
    }

    /// A column found by comparing names, not through Polars' name index. evcxr builds each
    /// notebook cell with its own copy of Polars, and an index made by the cell that read the
    /// table gives wrong answers in later cells ("not found" for a column that exists).
    pub(crate) fn column(&self, name: &str) -> Result<&Column, DatasetError> {
        self.dataframe
            .columns()
            .iter()
            .find(|column| column.name().as_str() == name)
            .ok_or_else(|| DatasetError::ColumnNotFound {
                requested: name.to_string(),
                available: self.columns().join(","),
            })
    }

    /// Prints every cached stat of every column in one table: dtype, nulls, mean / median /
    /// mode, min / max, std / variance, quartiles, and the top 3 categories.
    ///
    /// Prints HTML in an evcxr notebook and text elsewhere. `skip_columns` leaves columns out
    /// of the table, for example a long free-text column. The table itself, for tests, is
    /// `ds.metadata.try_metadata(skip)`.
    ///
    /// # Examples
    ///
    /// Basic:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.metadata(&[]);
    /// ```
    ///
    /// Advanced: leave the id and name columns out:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.metadata(&["id", "name"]);
    /// let table: String = ds.metadata.try_metadata(&["id", "name"]).to_string();
    /// assert!(table.contains("cgpa") && !table.contains("Zoë"));
    /// ```
    #[track_caller]
    pub fn metadata(&self, skip_columns: &[&str]) {
        self.try_fill_stats().or_fatal(); // a transformed column has no stats until asked
        self.metadata.try_metadata(skip_columns).show()
    }

    /// Prints each column's position, name, non-null count and dtype. (pandas: `df.info()`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.info();
    /// assert!(ds.metadata.try_info().to_string().contains("12 rows x 6 columns"));
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// ds.info()
    /// ```
    pub fn info(&self) {
        self.metadata.try_info().show()
    }

    /// Prints count, mean, std, min, quartiles and max of every number column.
    /// (pandas: `df.describe()`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.describe();
    /// assert!(!ds.metadata.try_describe().to_string().contains("batch")); // text: left out
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// assert "batch" not in ds.describe().columns  # text: left out
    /// ```
    #[track_caller]
    pub fn describe(&self) {
        self.try_fill_stats().or_fatal(); // a transformed column has no stats until asked
        self.metadata.try_describe().show()
    }

    /// (rows, columns). (pandas: `df.shape`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// assert_eq!(read_csv("tests/fixtures/data/students.csv").shape(), (12, 6));
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// assert ds.shape == (12, 6)
    /// ```
    pub fn shape(&self) -> (usize, usize) {
        (self.metadata.height, self.metadata.width)
    }

    /// The number of rows. (pandas: `len(df)`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// assert_eq!(read_csv("tests/fixtures/data/students.csv").height(), 12);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// assert len(ds) == 12
    /// ```
    pub fn height(&self) -> usize {
        self.metadata.height
    }

    /// The number of columns. (pandas: `len(df.columns)`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// assert_eq!(read_csv("tests/fixtures/data/students.csv").width(), 6);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// assert ds.shape[1] == 6
    /// ```
    pub fn width(&self) -> usize {
        self.metadata.width
    }

    /// Prints the first `n` rows. (pandas: `df.head(n)`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// read_csv("tests/fixtures/data/students.csv").head(5);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// print(ds.head(5))
    /// ```
    pub fn head(&self, n: usize) {
        frame_view(&self.dataframe.head(Some(n))).show()
    }

    /// Prints the last `n` rows. (pandas: `df.tail(n)`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// read_csv("tests/fixtures/data/students.csv").tail(3);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// print(ds.tail(3))
    /// ```
    pub fn tail(&self, n: usize) {
        frame_view(&self.dataframe.tail(Some(n))).show()
    }

    /// The column names, in order. (pandas: `df.columns`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// assert_eq!(ds.columns(), ["id", "name", "age", "cgpa", "batch", "passed"]);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// assert ds.columns.tolist() == ["id", "name", "age", "cgpa", "batch", "passed"]
    /// ```
    pub fn columns(&self) -> Vec<String> {
        self.metadata
            .dataset_cache
            .columns
            .keys()
            .cloned()
            .collect()
    }

    /// The Polars dtype of each column, in order. (pandas: `df.dtypes`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// assert_eq!(ds.dtypes(), ["i64", "str", "i64", "f64", "str", "bool"]);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// # pandas makes `age` float64 because it has a null; rusmlai keeps i64 and the null.
    /// assert ds.dtypes.astype(str).tolist() == ["int64", "str", "float64", "float64", "str", "bool"]
    /// ```
    pub fn dtypes(&self) -> Vec<String> {
        self.metadata
            .dataset_cache
            .columns
            .values()
            .map(|info| info.dtype.to_string())
            .collect()
    }

    /// The Polars table, to read or to copy for Polars operations that rusmlai doesn't have.
    /// It is read-only, so the cached stats stay true; make a new `Dataset` of a changed copy.
    ///
    /// ```
    /// use rusmlai::{polars::prelude::*, prelude::*};
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let passed: DataFrame = ds.dataframe().clone().lazy().filter(col("passed")).collect().unwrap();
    /// let ds = Dataset::new(passed, "tests/fixtures/data/students.csv");
    /// assert_eq!(ds.height(), 8);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// assert len(ds[ds["passed"]]) == 8
    /// ```
    pub fn dataframe(&self) -> &DataFrame {
        &self.dataframe
    }

    /// The Polars table, without the stats.
    pub fn into_dataframe(self) -> DataFrame {
        self.dataframe
    }

    /// A `Dataset` from a table and metadata computed before, for the benches, which time one
    /// step without recomputing the stats. `metadata` must describe `dataframe`.
    #[doc(hidden)]
    pub fn from_parts(dataframe: DataFrame, metadata: Metadata) -> Self {
        Self {
            dataframe,
            metadata,
        }
    }

    /// [`Dataset::to_list`] that returns an error instead of panicking.
    ///
    /// # Errors
    ///
    /// - [`DatasetError::ColumnNotFound`] for an unknown name.
    /// - [`DatasetError::ColumnNotNumeric`] for a text column.
    pub fn try_to_list(&self, column: &str) -> Result<Vec<f64>, DatasetError> {
        let values: &Column = self.column(column)?;
        if !(values.dtype().is_primitive_numeric() || values.dtype() == &DataType::Boolean) {
            return Err(DatasetError::ColumnNotNumeric {
                column: column.into(),
                actual_dtype: values.dtype().to_string(),
            });
        }
        let floats: Column = values.cast(&DataType::Float64)?;
        Ok(floats
            .f64()?
            .iter()
            .map(|v| v.unwrap_or(f64::NAN))
            .collect())
    }

    /// The values of a number or true/false column as `f64`, in row order. A null becomes NaN,
    /// and `true` / `false` become 1 / 0. (pandas: `df[col].to_list()`)
    ///
    /// # Panics
    ///
    /// For an unknown or text column. [`Dataset::try_to_list`] returns these as errors.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let cgpa: Vec<f64> = ds.to_list("cgpa");
    /// assert_eq!(cgpa[..3], [8.1, 7.4, 6.2]);
    /// assert!(cgpa[5].is_nan()); // the null
    /// ```
    ///
    /// Edge case: a text column has no numbers:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// read_csv("tests/fixtures/data/students.csv").to_list("batch"); // panics: column 'batch' is not numeric (found str)
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import math
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// cgpa = ds["cgpa"].to_list()
    /// assert cgpa[:3] == [8.1, 7.4, 6.2]
    /// assert math.isnan(cgpa[5])
    /// ```
    #[track_caller]
    pub fn to_list(&self, column: &str) -> Vec<f64> {
        self.try_to_list(column).or_fatal()
    }

    /// A new `Dataset` with these columns, in this order. (pandas: `df[["a", "b"]]`)
    ///
    /// The stats and the plot cache are shared with `self`, so nothing is computed again.
    ///
    /// # Panics
    ///
    /// On an empty list or an unknown column. [`Dataset::try_select`] returns these as errors.
    ///
    /// # Examples
    ///
    /// Basic:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// assert_eq!(ds.select(&["cgpa", "age"]).columns(), ["cgpa", "age"]);
    /// ```
    ///
    /// Advanced: the stats are the same objects, not copies:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let small = ds.select(&["cgpa"]);
    /// let (a, b) = (&ds.metadata.dataset_cache.columns["cgpa"], &small.metadata.dataset_cache.columns["cgpa"]);
    /// assert!(std::sync::Arc::ptr_eq(a, b));
    /// ```
    ///
    /// Edge case: an unknown column panics and lists the columns that exist:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// read_csv("tests/fixtures/data/students.csv").select(&["grade"]); // panics, listing the columns
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// assert ds[["cgpa", "age"]].columns.tolist() == ["cgpa", "age"]
    /// ```
    #[track_caller]
    pub fn select(&self, columns: &[&str]) -> Dataset {
        self.try_select(columns).or_fatal()
    }

    /// A new `Dataset` without these columns. (pandas: `df.drop(columns=[...])`)
    ///
    /// The other columns keep their order and share their stats with `self`.
    ///
    /// # Panics
    ///
    /// On an unknown column, or when nothing would be left. [`Dataset::try_drop`] returns
    /// these as errors.
    ///
    /// # Examples
    ///
    /// Basic:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// assert_eq!(ds.drop(&["id", "name"]).width(), 4);
    /// ```
    ///
    /// Advanced: features and target, sklearn style:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let x = ds.drop(&["id", "name", "passed"]);
    /// let y = ds.select(&["passed"]);
    /// assert_eq!((x.width(), y.width(), x.height()), (3, 1, 12));
    /// ```
    ///
    /// Edge case: dropping every column panics:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.drop(&["id", "name", "age", "cgpa", "batch", "passed"]);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// assert ds.drop(columns=["id", "name"]).shape[1] == 4
    ///
    /// X, y = ds.drop(columns=["id", "name", "passed"]), ds["passed"]  # y is a Series in pandas
    /// assert (X.shape[1], X.shape[0]) == (3, 12)
    /// ```
    #[track_caller]
    pub fn drop(&self, columns: &[&str]) -> Dataset {
        self.try_drop(columns).or_fatal()
    }

    /// How much each column tells about `target`, in nats, highest first: how far the
    /// target's entropy drops once the column is known (information gain, also called mutual
    /// information). (scikit-learn: `mutual_info_score`, or `mutual_info_classif` with
    /// `discrete_features=True`)
    ///
    /// Text and true/false columns are counted as they are. Numbers are cut into 256 equal
    /// bins between their lowest and highest finite value. For each column, rows where it or
    /// the target has no value (null, NaN, ±inf) are left out. A column with a different
    /// value in every row, like an id, scores as high as the target's own entropy, so leave
    /// ids out.
    ///
    /// # Panics
    ///
    /// If `target` doesn't exist. [`Dataset::try_information_gain`] returns it as an error.
    ///
    /// # Examples
    ///
    /// Basic: rank the columns that tell the most about `passed`:
    /// ```
    /// use rusmlai::prelude::*;
    ///
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// for (column, gain) in ds.drop(&["id", "name"]).information_gain("passed") {
    ///     println!("{column}: {gain:.3}");
    /// }
    /// ```
    ///
    /// Advanced: an id column tops the list with the target's whole entropy, which is why
    /// ids should be dropped first:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let h: f64 = ds.try_categorical_stats("passed").unwrap().entropy;
    /// let gains = ds.information_gain("passed");
    /// let id: f64 = gains.iter().find(|(c, _)| c == "id").unwrap().1;
    /// assert!((id - h).abs() < 1e-12);
    /// ```
    ///
    /// Edge case: a target that doesn't exist:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// read_csv("tests/fixtures/data/students.csv").information_gain("grade"); // panics: column 'grade' not found
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// from sklearn.metrics import mutual_info_score
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// gains = {}
    /// for column in ["age", "cgpa", "batch"]:
    ///     rows = ds[[column, "passed"]].dropna()  # rusmlai leaves out each column's missing rows too
    ///     x = rows[column]
    ///     if x.dtype.kind in "if":
    ///         x = pd.cut(x, 256)  # 256 equal bins, like rusmlai; mutual_info_classif would guess instead
    ///     gains[column] = mutual_info_score(x, rows["passed"])
    /// for column, gain in sorted(gains.items(), key=lambda g: -g[1]):
    ///     print(f"{column}: {gain:.3f}")
    /// ```
    #[track_caller]
    pub fn information_gain(&self, target: &str) -> Vec<(String, f64)> {
        self.try_information_gain(target).or_fatal()
    }
}

/// Entropy of the group sizes in column "n".
fn entropy(counts: &DataFrame) -> Result<f64, DatasetError> {
    let n: Vec<f64> = counts
        .column("n")?
        .cast(&DataType::Float64)?
        .f64()?
        .into_no_null_iter()
        .collect();
    Ok(CategoricalStats::entropy(n.iter().copied()))
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

/// Reads a CSV file into a [`Dataset`] and loads or computes its column stats.
/// (pandas: `pd.read_csv`)
///
/// The first read computes the stats of every column and writes them to
/// `metadata/{stem}.json` next to the CSV. Later reads load that file in microseconds, as long
/// as the CSV's size and modification time are unchanged. A stale or broken cache file is
/// computed again, never an error. Each saved file prints `[rusmlai] saved <path>`.
///
/// The cache is trusted like code, because only the size and time are checked. For a folder
/// that came from someone else with its `metadata/`, call
/// [`set_cache(false)`](crate::metadata::set_cache) first.
///
/// # Panics
///
/// If the file doesn't exist or Polars cannot parse it. [`try_read_csv`] returns these as
/// errors.
///
/// # Examples
///
/// Basic:
/// ```
/// use rusmlai::prelude::*;
///
/// let ds = read_csv("tests/fixtures/data/students.csv");
/// assert_eq!(ds.shape(), (12, 6));
/// ```
///
/// Advanced: the second read comes from the cache, with the same stats:
/// ```
/// # use rusmlai::prelude::*;
/// let first = read_csv("tests/fixtures/data/students.csv");
/// let again = read_csv("tests/fixtures/data/students.csv");
/// assert_eq!(
///     first.try_numeric_stats("cgpa").unwrap().median,
///     again.try_numeric_stats("cgpa").unwrap().median,
/// );
/// ```
///
/// Advanced: other options (a separator, some columns, no header row) are Polars' own: read
/// the file through `rusmlai::polars` and make a `Dataset` of the table, under the file's path
/// so charts and models still use its folder:
/// ```
/// use rusmlai::{polars::prelude::*, prelude::*};
/// # let dir = std::env::temp_dir().join("rusmlai-read-options-doc");
/// # std::fs::create_dir_all(&dir).unwrap();
/// let path = dir.join("semi.csv");
/// std::fs::write(&path, "a;b\n1;2\n").unwrap();
/// let frame: DataFrame = CsvReadOptions::default()
///     .with_parse_options(CsvParseOptions::default().with_separator(b';'))
///     .try_into_reader_with_file_path(Some(path.clone()))?
///     .finish()?;
/// let ds = Dataset::new(frame, &path);
/// assert_eq!(ds.columns(), ["a", "b"]);
/// # Ok::<(), PolarsError>(())
/// ```
///
/// A `.tsv` file is read with tabs (pandas: `read_csv(path, sep="\t")`):
/// ```
/// # use rusmlai::prelude::*;
/// assert_eq!(read_csv("tests/fixtures/data/students.tsv").shape(), (12, 6));
/// ```
///
/// Edge case: a missing file panics with the path and the working directory:
/// ```should_panic
/// rusmlai::prelude::read_csv("does/not/exist.csv"); // panics: file not found at 'does/not/exist.csv'
/// ```
///
/// Python:
///
/// ```python
/// import os
/// import tempfile
/// import pandas as pd
/// ds = pd.read_csv("tests/fixtures/data/students.csv")
/// assert ds.shape == (12, 6)
///
/// path = os.path.join(tempfile.mkdtemp(), "semi.csv")
/// with open(path, "w") as f:
///     f.write("a;b\n1;2\n")
/// assert pd.read_csv(path, sep=";").columns.tolist() == ["a", "b"]
/// assert pd.read_csv("tests/fixtures/data/students.tsv", sep="\t").shape == (12, 6)
/// ```
#[track_caller]
pub fn read_csv(path: impl AsRef<Path>) -> Dataset {
    try_read_csv(path).or_fatal()
}

/// [`read_csv`] that returns an error instead of panicking.
///
/// # Errors
///
/// - [`DatasetError::FileNotFound`] when there is no file at `path`. The message gives the
///   absolute path that was looked up and the working directory, because relative paths start there.
/// - [`DatasetError::Polars`] when Polars cannot parse the file.
///
/// A cache file that cannot be written is only a warning on stderr.
///
/// # Examples
///
/// Basic:
/// ```
/// # use rusmlai::prelude::*;
/// let ds = try_read_csv("tests/fixtures/data/students.csv")?;
/// assert_eq!(ds.height(), 12);
/// # Ok::<(), DatasetError>(())
/// ```
///
/// Advanced: fall back to a second path:
/// ```
/// # use rusmlai::prelude::*;
/// let ds = try_read_csv("data/train.csv")
///     .or_else(|_| try_read_csv("tests/fixtures/data/students.csv"))?;
/// assert_eq!(ds.width(), 6);
/// # Ok::<(), DatasetError>(())
/// ```
///
/// Edge case: rows with missing fields at the end are filled with nulls:
/// ```
/// # use rusmlai::prelude::*;
/// let ds = try_read_csv("tests/fixtures/data/ragged.csv")?;
/// assert_eq!(ds.shape(), (5, 3));
/// assert_eq!(ds.try_column_info("temp")?.null_count, 2);
/// # Ok::<(), DatasetError>(())
/// ```
/// `Ok` if `path` is a file, else [`DatasetError::FileNotFound`] with the absolute path that
/// was looked up (`..` folded away) and the working directory, because a relative path starts there.
pub(crate) fn existing(path: &Path) -> Result<(), DatasetError> {
    if path.is_file() {
        return Ok(());
    }
    let cwd: std::path::PathBuf = std::env::current_dir().unwrap_or_default();
    let mut absolute: std::path::PathBuf = std::path::PathBuf::new();
    for part in cwd.join(path).components() {
        match part {
            std::path::Component::ParentDir => {
                absolute.pop();
            }
            std::path::Component::CurDir => {}
            other => absolute.push(other),
        }
    }
    Err(DatasetError::FileNotFound {
        path: path.display().to_string(),
        absolute: absolute.display().to_string(),
        cwd: cwd.display().to_string(),
    })
}

pub fn try_read_csv(path: impl AsRef<Path>) -> Result<Dataset, DatasetError> {
    let path: &Path = path.as_ref();
    existing(path)?;
    let tab: bool = path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("tsv"));
    // One chunk per column: Polars leaves one per read thread (129 for 630,000 rows), and a
    // model would copy every column into one piece on every fit and predict.
    let dataframe: DataFrame = CsvReadOptions::default()
        .with_rechunk(true)
        .with_parse_options(CsvParseOptions::default().with_separator(if tab {
            b'\t'
        } else {
            b','
        }))
        .try_into_reader_with_file_path(Some(path.to_path_buf()))?
        .finish()?;
    let metadata: Metadata = Metadata::try_load(&dataframe, path)?;
    Ok(Dataset {
        dataframe,
        metadata,
    })
}

impl Dataset {
    /// `dataframe`, whose rows line up with this dataset's, as a dataset whose stats are
    /// computed on first use, with this dataset's path: a model's output, like
    /// `predict_proba`.
    pub(crate) fn derived(&self, dataframe: DataFrame) -> Dataset {
        let columns: indexmap::IndexMap<String, std::sync::Arc<ColumnInfo>> = dataframe
            .columns()
            .iter()
            .map(|c| {
                (
                    c.name().to_string(),
                    std::sync::Arc::new(ColumnInfo::new(c)),
                )
            })
            .collect();
        let metadata: Metadata = Metadata {
            filepath: self.metadata.filepath.clone(),
            height: dataframe.height(),
            width: dataframe.width(),
            dataset_cache: crate::metadata::dataset::DatasetCache::new(columns),
            plot_cache: crate::metadata::plot::PlotCache::default(),
        };
        Dataset {
            dataframe,
            metadata,
        }
    }

    /// This dataset with the column `name` of `source` added, or replaced. The column keeps its
    /// stats from `source`, so a model's `fit` reads the target's cached stats. The rows must
    /// line up.
    pub(crate) fn try_with_column(
        &self,
        source: &Dataset,
        name: &str,
    ) -> Result<Dataset, DatasetError> {
        let column: Column = source.column(name)?.clone();
        let info: std::sync::Arc<ColumnInfo> =
            std::sync::Arc::clone(&source.metadata.dataset_cache.columns[name]);
        let mut dataframe: DataFrame = self.dataframe.clone();
        dataframe.with_column(column)?;
        let mut metadata: Metadata = self.metadata.clone();
        metadata
            .dataset_cache
            .columns
            .insert(name.to_string(), info);
        metadata.width = dataframe.width();
        Ok(Dataset {
            dataframe,
            metadata,
        })
    }
}

#[cfg(test)]
mod tests {
    use polars::df;

    use super::*;

    fn small() -> Dataset {
        Dataset::new(df!("x" => [1.0, 2.0], "s" => ["a", "b"]).unwrap(), "")
    }

    #[test]
    fn information_gain_matches_sklearn() {
        // sklearn mutual_info_score: `same` gives ln 2, `noise` gives 0; `num` bins like `same`.
        let frame: DataFrame = df!(
            "same" => ["a", "a", "b", "b"],
            "noise" => ["u", "v", "u", "v"],
            "num" => [0.0, f64::NAN, 1.0, f64::INFINITY],
            "y" => ["yes", "yes", "no", "no"]
        )
        .unwrap();
        let gains: Vec<(String, f64)> = Dataset::new(frame, "").try_information_gain("y").unwrap();
        let gain = |name: &str| gains.iter().find(|g| g.0 == name).unwrap().1;
        assert!((gain("same") - std::f64::consts::LN_2).abs() < 1e-12);
        assert!((gain("num") - std::f64::consts::LN_2).abs() < 1e-12);
        assert!(gain("noise").abs() < 1e-12);
        assert_eq!(gains[2].0, "noise");
    }

    #[test]
    fn information_gain_bins_a_numeric_target_and_a_constant_column() {
        let frame: DataFrame = df!(
            "y" => [0.0, 0.0, 10.0, 10.0],
            "c" => [5.0, 5.0, 5.0, 5.0],
            "s" => ["a", "a", "b", "b"]
        )
        .unwrap();
        let gains: Vec<(String, f64)> = Dataset::new(frame, "").try_information_gain("y").unwrap();
        assert_eq!(gains[0].0, "s");
        assert!((gains[0].1 - std::f64::consts::LN_2).abs() < 1e-12);
        assert_eq!(gains[1], ("c".to_string(), 0.0));
    }

    #[test]
    fn errors_name_the_problem() {
        let ds: Dataset = small();
        assert!(matches!(ds.try_select(&[]), Err(DatasetError::EmptySelect)));
        match ds.try_select(&["z"]) {
            Err(DatasetError::ColumnNotFound {
                requested,
                available,
            }) => {
                assert_eq!((requested.as_str(), available.as_str()), ("z", "x,s"))
            }
            _ => panic!("expected ColumnNotFound"),
        }
        assert!(matches!(
            ds.try_numeric_stats("s"),
            Err(DatasetError::ColumnNotNumeric { .. })
        ));
        assert!(matches!(
            ds.try_categorical_stats("x"),
            Err(DatasetError::ColumnNotCategorical { .. })
        ));
        assert!(matches!(
            ds.try_drop(&["x", "s"]),
            Err(DatasetError::EmptySelect)
        ));
        let missing: DatasetError = try_read_csv("a/../no/such.csv").unwrap_err();
        let cwd: std::path::PathBuf = std::env::current_dir().unwrap();
        assert!(
            missing
                .to_string()
                .contains(&cwd.join("no/such.csv").display().to_string()),
            "{missing}"
        );
    }

    #[test]
    fn select_keeps_the_order_given() {
        let ds: Dataset = small();
        let picked: Dataset = ds.select(&["s", "x"]);
        assert_eq!(picked.columns(), ["s", "x"]);
        assert_eq!(picked.dataframe.get_column_names(), ["s", "x"]);
        assert_eq!(picked.shape(), (2, 2));
    }

    #[test]
    fn missing_stats_are_computed_on_first_use() {
        let mut ds: Dataset = small();
        let info: ColumnInfo = ColumnInfo::new(ds.dataframe.column("x").unwrap());
        ds.metadata
            .dataset_cache
            .columns
            .insert("x".into(), std::sync::Arc::new(info));
        assert_eq!(ds.try_numeric_stats("x").unwrap().mean, 1.5);
    }

    #[test]
    fn fill_stats_computes_what_a_transform_left_empty() {
        use crate::preprocess::{StandardScaler, Transform};
        let scaled: Dataset = StandardScaler::new().fit_transform(&small().select(&["x"]));
        let empty: String = scaled.metadata.try_describe().to_string();
        assert!(empty.starts_with("No numeric columns"), "{empty}");
        scaled.try_fill_stats().unwrap();
        let table: String = scaled.metadata.try_describe().to_string();
        assert!(
            table.contains("-1.0000") && table.contains("1.0000"),
            "{table}"
        );
        scaled.try_fill_stats().unwrap(); // nothing missing: no work, no error
    }

    #[test]
    fn frame_view_shows_nulls_and_the_shape() {
        let frame: DataFrame = df!("a" => [Some(1), None]).unwrap();
        let text: String = frame_view(&frame).to_string();
        assert!(
            text.starts_with("shape: (2, 1)") && text.contains("null"),
            "{text}"
        );
    }
}
