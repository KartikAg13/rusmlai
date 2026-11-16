//! Readers besides [`read_csv`](crate::prelude::read_csv): Parquet, Excel and SQL. Each one
//! gives a [`Dataset`] with the stats of every column, like `read_csv`.

use std::{borrow::Cow, fs::File, path::Path};

use calamine::{Data, Reader, open_workbook_auto};
use polars::{
    frame::DataFrame,
    io::{SerReader, parquet::read::ParquetReader},
    prelude::{Column, NamedFrom, Series},
};

use crate::{
    dataset::{Dataset, error::DatasetError, existing},
    error::OrFatal,
    metadata::Metadata,
};

/// One value of a sheet or a database row, before its column gets a type.
#[derive(Debug, Clone, PartialEq)]
enum Cell {
    Null,
    Int(i64),
    Float(f64),
    Bool(bool),
    Text(String),
}

/// The column names and the rows of a sheet or a query.
type Rows = (Vec<String>, Vec<Vec<Cell>>);

/// A whole number that an `f64` holds exactly, so it can be an `Int64`.
fn whole(v: f64) -> bool {
    v.fract() == 0.0 && v.abs() < 9.0e15
}

/// One column of cells, typed like pandas and `read_csv`: whole numbers are `Int64`, other
/// numbers `Float64`, true / false `Boolean`, anything else (or a mix) text. A null stays null.
fn typed(name: &str, cells: &[Cell]) -> Column {
    let mut values = cells.iter().filter(|c| **c != Cell::Null).peekable();
    let some: bool = values.peek().is_some();
    let all =
        |test: fn(&Cell) -> bool| some && cells.iter().filter(|c| **c != Cell::Null).all(test);
    let series: Series =
        if all(|c| matches!(c, Cell::Int(_)) || matches!(c, Cell::Float(v) if whole(*v))) {
            let ints: Vec<Option<i64>> = cells
                .iter()
                .map(|c| match c {
                    Cell::Int(v) => Some(*v),
                    Cell::Float(v) => Some(*v as i64),
                    _ => None,
                })
                .collect();
            Series::new(name.into(), ints)
        } else if all(|c| matches!(c, Cell::Int(_) | Cell::Float(_))) {
            let floats: Vec<Option<f64>> = cells
                .iter()
                .map(|c| match c {
                    Cell::Int(v) => Some(*v as f64),
                    Cell::Float(v) => Some(*v),
                    _ => None,
                })
                .collect();
            Series::new(name.into(), floats)
        } else if all(|c| matches!(c, Cell::Bool(_))) {
            let bools: Vec<Option<bool>> = cells
                .iter()
                .map(|c| match c {
                    Cell::Bool(v) => Some(*v),
                    _ => None,
                })
                .collect();
            Series::new(name.into(), bools)
        } else {
            let texts: Vec<Option<String>> = cells
                .iter()
                .map(|c| match c {
                    Cell::Null => None,
                    Cell::Int(v) => Some(v.to_string()),
                    Cell::Float(v) => Some(v.to_string()),
                    Cell::Bool(v) => Some(v.to_string()),
                    Cell::Text(v) => Some(v.clone()),
                })
                .collect();
            Series::new(name.into(), texts)
        };
    series.into()
}

/// The table of `rows`. A repeated name gets `.1`, `.2`, ... like pandas.
fn frame((names, rows): Rows) -> Result<DataFrame, DatasetError> {
    let mut unique: Vec<String> = Vec::with_capacity(names.len());
    for name in names {
        let mut candidate: String = name.clone();
        let mut n: usize = 0;
        while unique.contains(&candidate) {
            n += 1;
            candidate = format!("{name}.{n}");
        }
        unique.push(candidate);
    }
    let columns: Vec<Column> = unique
        .iter()
        .enumerate()
        .map(|(j, name)| {
            let cells: Vec<Cell> = rows
                .iter()
                .map(|row| row.get(j).cloned().unwrap_or(Cell::Null))
                .collect();
            typed(name, &cells)
        })
        .collect();
    Ok(DataFrame::new(rows.len(), columns)?)
}

/// `dataframe` with its stats computed now, under `path` (charts and models use its folder).
fn dataset(dataframe: DataFrame, path: &str) -> Result<Dataset, DatasetError> {
    let metadata: Metadata = Metadata::try_new(&dataframe, path)?;
    Ok(Dataset {
        dataframe,
        metadata,
    })
}

/// [`read_parquet`] that returns an error instead of panicking.
///
/// # Errors
///
/// [`DatasetError::FileNotFound`] when there is no file at `path`, and
/// [`DatasetError::Polars`] when Polars cannot read it.
pub fn try_read_parquet(path: impl AsRef<Path>) -> Result<Dataset, DatasetError> {
    let path: &Path = path.as_ref();
    existing(path)?;
    // One chunk per column, like `read_csv`: Parquet comes in one chunk per row group.
    let dataframe: DataFrame = ParquetReader::new(File::open(path).map_err(|e| read(path, e))?)
        .set_rechunk(true)
        .finish()?;
    let metadata: Metadata = Metadata::try_load(&dataframe, path)?;
    Ok(Dataset {
        dataframe,
        metadata,
    })
}

/// Reads a Parquet file into a [`Dataset`], with the stats cached next to it like
/// [`read_csv`](crate::prelude::read_csv) (`metadata/{name}.parquet.json`). The column types
/// come from the file. (pandas: `pd.read_parquet`)
///
/// # Panics
///
/// If the file doesn't exist or isn't Parquet. [`try_read_parquet`] returns these as errors.
///
/// # Examples
///
/// Basic:
/// ```
/// use rusmlai::prelude::*;
///
/// let ds = read_parquet("tests/fixtures/data/students.parquet");
/// assert_eq!(ds.shape(), (12, 6));
/// ```
///
/// Advanced: the file keeps its types, so `age` stays whole numbers with one null:
/// ```
/// # use rusmlai::prelude::*;
/// let ds = read_parquet("tests/fixtures/data/students.parquet");
/// assert_eq!(ds.dtypes(), ["i64", "str", "i64", "f64", "str", "bool"]);
/// assert_eq!(ds.try_column_info("age").unwrap().null_count, 1);
/// ```
///
/// Edge case: a missing file panics with the path and the working directory:
/// ```should_panic
/// rusmlai::prelude::read_parquet("does/not/exist.parquet"); // panics: file not found at ...
/// ```
///
/// Python:
///
/// ```python
/// import pandas as pd
/// ds = pd.read_parquet("tests/fixtures/data/students.parquet")
/// assert ds.shape == (12, 6)
/// assert ds["age"].isna().sum() == 1
/// ```
#[track_caller]
pub fn read_parquet(path: impl AsRef<Path>) -> Dataset {
    try_read_parquet(path).or_fatal()
}

/// A [`DatasetError::Read`] of `path`.
fn read(path: &Path, reason: impl ToString) -> DatasetError {
    DatasetError::Read {
        what: path.display().to_string(),
        reason: reason.to_string(),
    }
}

/// A sheet cell. A date is `2024-01-05` (with ` 13:45:00` when it has a time), a duration its
/// number of days, and an error cell (`#DIV/0!`) null.
fn excel(cell: &Data) -> Cell {
    match cell {
        Data::Int(v) => Cell::Int(*v),
        Data::Float(v) => Cell::Float(*v),
        Data::Bool(v) => Cell::Bool(*v),
        Data::String(v) | Data::DateTimeIso(v) | Data::DurationIso(v) => Cell::Text(v.clone()),
        Data::DateTime(v) if v.is_datetime() => v.as_datetime().map_or(Cell::Null, |t| {
            let text: String = t.to_string();
            Cell::Text(text.strip_suffix(" 00:00:00").unwrap_or(&text).to_string())
        }),
        Data::DateTime(v) => Cell::Float(v.as_f64()),
        Data::Error(_) | Data::Empty => Cell::Null,
    }
}

/// [`read_excel`] that returns an error instead of panicking.
///
/// # Errors
///
/// - [`DatasetError::FileNotFound`] when there is no file at `path`.
/// - [`DatasetError::SheetNotFound`] for an unknown sheet. The message lists the sheets.
/// - [`DatasetError::Read`] when the file isn't a workbook.
pub fn try_read_excel(path: impl AsRef<Path>, sheet_name: &str) -> Result<Dataset, DatasetError> {
    let path: &Path = path.as_ref();
    existing(path)?;
    let mut book = open_workbook_auto(path).map_err(|e| read(path, e))?;
    let sheets: Vec<String> = book.sheet_names();
    let sheet: &str = match sheet_name {
        "" => sheets.first().map_or("", String::as_str),
        name => name,
    };
    if !sheets.iter().any(|s| s == sheet) {
        return Err(DatasetError::SheetNotFound {
            sheet: sheet.into(),
            available: sheets.join(", "),
        });
    }
    let range = book.worksheet_range(sheet).map_err(|e| read(path, e))?;
    let mut lines = range.rows();
    let names: Vec<String> = lines.next().map_or(Vec::new(), |header| {
        (header.iter().enumerate())
            .map(|(i, cell)| match cell {
                Data::Empty => format!("Unnamed: {i}"),
                cell => cell.to_string(),
            })
            .collect()
    });
    let rows: Vec<Vec<Cell>> = lines.map(|line| line.iter().map(excel).collect()).collect();
    dataset(frame((names, rows))?, &path.to_string_lossy())
}

/// Reads one sheet of an Excel or OpenDocument workbook (`.xlsx`, `.xlsm`, `.xlsb`, `.xls`,
/// `.ods`) into a [`Dataset`]. The first row names the columns. (pandas: `pd.read_excel`)
///
/// `sheet_name = ""` reads the first sheet, like pandas' `sheet_name=0`. Each column gets a
/// type like `read_csv` gives: whole numbers are `i64`, other numbers `f64`, true / false
/// `bool`, and anything else, dates included (`2024-01-05`), text. An empty cell is null. A
/// workbook has several sheets, so the stats are computed on every read, not cached.
///
/// # Panics
///
/// If the file or the sheet doesn't exist, or the file isn't a workbook. [`try_read_excel`]
/// returns these as errors.
///
/// # Examples
///
/// Basic: the first sheet:
/// ```
/// use rusmlai::prelude::*;
///
/// let ds = read_excel("tests/fixtures/data/students.xlsx", "");
/// assert_eq!(ds.shape(), (12, 6));
/// assert_eq!(ds.dtypes(), ["i64", "str", "i64", "f64", "str", "bool"]);
/// ```
///
/// Advanced: a sheet by name, and the old `.xls` format:
/// ```
/// # use rusmlai::prelude::*;
/// let top = read_excel("tests/fixtures/data/students.xlsx", "top");
/// assert_eq!(top.columns(), ["name", "cgpa"]);
/// let old = read_excel("tests/fixtures/data/students.xls", "");
/// assert_eq!(old.shape(), (12, 6));
/// ```
///
/// Edge case: an unknown sheet lists the sheets there are:
/// ```
/// # use rusmlai::prelude::*;
/// let err = try_read_excel("tests/fixtures/data/students.xlsx", "grades").unwrap_err();
/// assert_eq!(err.to_string(), "sheet 'grades' not found. Sheets: students, top");
/// ```
///
/// Python:
///
/// ```python
/// import pandas as pd
/// ds = pd.read_excel("tests/fixtures/data/students.xlsx")
/// assert ds.shape == (12, 6)
/// top = pd.read_excel("tests/fixtures/data/students.xlsx", sheet_name="top")
/// assert top.columns.tolist() == ["name", "cgpa"]
/// old = pd.read_excel("tests/fixtures/data/students.xls")  # needs xlrd
/// assert old.shape == (12, 6)
/// ```
#[track_caller]
pub fn read_excel(path: impl AsRef<Path>, sheet_name: &str) -> Dataset {
    try_read_excel(path, sheet_name).or_fatal()
}

/// `con` without its password, for messages: `postgresql://user:secret@host/db` shows as
/// `postgresql://…@host/db`.
fn redacted(con: &str) -> Cow<'_, str> {
    match (con.find("://"), con.rfind('@')) {
        (Some(scheme), Some(at)) if at > scheme => {
            format!("{}…{}", &con[..scheme + 3], &con[at..]).into()
        }
        _ => con.into(),
    }
}

/// The rows of `sql` on the SQLite file `path`, opened read-only.
fn sqlite(sql: &str, path: &str) -> Result<Rows, String> {
    use rusqlite::{Connection, OpenFlags, types::ValueRef};
    let db = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|e| e.to_string())?;
    let mut statement = db.prepare(sql).map_err(|e| e.to_string())?;
    let names: Vec<String> = statement.column_names().map(str::to_string).collect();
    let mut found = statement.query([]).map_err(|e| e.to_string())?;
    let mut rows: Vec<Vec<Cell>> = Vec::new();
    while let Some(row) = found.next().map_err(|e| e.to_string())? {
        let mut cells: Vec<Cell> = Vec::with_capacity(names.len());
        for (i, name) in names.iter().enumerate() {
            cells.push(match row.get_ref(i).map_err(|e| e.to_string())? {
                ValueRef::Null => Cell::Null,
                ValueRef::Integer(v) => Cell::Int(v),
                ValueRef::Real(v) => Cell::Float(v),
                ValueRef::Text(v) => Cell::Text(String::from_utf8_lossy(v).into_owned()),
                ValueRef::Blob(_) => {
                    return Err(format!(
                        "column '{name}' holds binary data (BLOB): leave it out of the query"
                    ));
                }
            });
        }
        rows.push(cells);
    }
    Ok((names, rows))
}

/// The rows of `sql` on the PostgreSQL server at `url`, without TLS.
fn postgres(sql: &str, url: &str) -> Result<Rows, String> {
    use postgres::{Client, NoTls};
    let mut client: Client = Client::connect(url, NoTls).map_err(|e| e.to_string())?;
    let statement = client.prepare(sql).map_err(|e| e.to_string())?;
    let names: Vec<String> = (statement.columns().iter())
        .map(|c| c.name().to_string())
        .collect();
    let found = client.query(&statement, &[]).map_err(|e| e.to_string())?;
    let rows = found.iter().map(|row| {
        (row.columns().iter().enumerate())
            .map(|(i, column)| {
                let cell = match column.type_().name() {
                    "bool" => row.try_get::<_, Option<bool>>(i).map(|v| v.map(Cell::Bool)),
                    "int2" => row
                        .try_get::<_, Option<i16>>(i)
                        .map(|v| v.map(|v| Cell::Int(v.into()))),
                    "int4" => row
                        .try_get::<_, Option<i32>>(i)
                        .map(|v| v.map(|v| Cell::Int(v.into()))),
                    "int8" => row.try_get::<_, Option<i64>>(i).map(|v| v.map(Cell::Int)),
                    "float4" => row
                        .try_get::<_, Option<f32>>(i)
                        .map(|v| v.map(|v| Cell::Float(v.into()))),
                    "float8" => row.try_get::<_, Option<f64>>(i).map(|v| v.map(Cell::Float)),
                    "text" | "varchar" | "bpchar" | "name" => row
                        .try_get::<_, Option<String>>(i)
                        .map(|v| v.map(Cell::Text)),
                    other => {
                        let name: &str = column.name();
                        return Err(format!(
                            "column '{name}' has the type {other}: cast it in the query, \
                             like {name}::float8 or {name}::text"
                        ));
                    }
                };
                cell.map(|c| c.unwrap_or(Cell::Null))
                    .map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<Cell>, String>>()
    });
    Ok((names, rows.collect::<Result<_, _>>()?))
}

/// The rows of `sql` on the MySQL server at `url`, without TLS.
fn mysql(sql: &str, url: &str) -> Result<Rows, String> {
    use mysql::{Conn, Opts, Value, consts::ColumnType as T, prelude::Queryable};
    let opts: Opts = Opts::from_url(url).map_err(|e| e.to_string())?;
    let mut conn: Conn = Conn::new(opts).map_err(|e| e.to_string())?;
    let found = conn.query_iter(sql).map_err(|e| e.to_string())?;
    let columns = found.columns();
    let names: Vec<String> = (columns.as_ref().iter())
        .map(|c| c.name_str().into_owned())
        .collect();
    let kinds: Vec<T> = columns.as_ref().iter().map(|c| c.column_type()).collect();
    let mut rows: Vec<Vec<Cell>> = Vec::new();
    for row in found {
        let row = row.map_err(|e| e.to_string())?;
        let cells = (0..names.len()).map(|i| match row.as_ref(i).unwrap_or(&Value::NULL) {
            Value::NULL => Cell::Null,
            Value::Int(v) => Cell::Int(*v),
            Value::UInt(v) => i64::try_from(*v).map_or(Cell::Float(*v as f64), Cell::Int),
            Value::Float(v) => Cell::Float((*v).into()),
            Value::Double(v) => Cell::Float(*v),
            // The text protocol sends every value as text: the column type says what it is.
            Value::Bytes(v) => {
                let text: String = String::from_utf8_lossy(v).into_owned();
                match kinds[i] {
                    T::MYSQL_TYPE_TINY
                    | T::MYSQL_TYPE_SHORT
                    | T::MYSQL_TYPE_INT24
                    | T::MYSQL_TYPE_LONG
                    | T::MYSQL_TYPE_LONGLONG
                    | T::MYSQL_TYPE_YEAR => text.parse().map_or(Cell::Text(text), Cell::Int),
                    T::MYSQL_TYPE_FLOAT
                    | T::MYSQL_TYPE_DOUBLE
                    | T::MYSQL_TYPE_DECIMAL
                    | T::MYSQL_TYPE_NEWDECIMAL => {
                        text.parse().map_or(Cell::Text(text), Cell::Float)
                    }
                    _ => Cell::Text(text),
                }
            }
            other => Cell::Text(other.as_sql(true).trim_matches('\'').to_string()),
        });
        rows.push(cells.collect());
    }
    Ok((names, rows))
}

/// [`read_sql`] that returns an error instead of panicking.
///
/// # Errors
///
/// - [`DatasetError::FileNotFound`] when `con` is a SQLite file that doesn't exist. No empty
///   database is created.
/// - [`DatasetError::Read`] when the connection or the query fails. The message shows `con`
///   without its password.
pub fn try_read_sql(sql: &str, con: &str) -> Result<Dataset, DatasetError> {
    let server: bool = ["postgresql://", "postgres://", "mysql://"]
        .iter()
        .any(|scheme| con.starts_with(scheme));
    if !server {
        existing(Path::new(con))?;
    }
    let rows: Rows = match con {
        _ if con.starts_with("mysql://") => mysql(sql, con),
        _ if server => postgres(sql, con),
        _ => sqlite(sql, con),
    }
    .map_err(|reason| DatasetError::Read {
        what: redacted(con).into_owned(),
        reason,
    })?;
    dataset(frame(rows)?, if server { "" } else { con })
}

/// The result of a SQL query as a [`Dataset`], with the stats of every column.
/// (pandas: `pd.read_sql(sql, con)`)
///
/// `con` is a SQLite file, or a server URL: `postgresql://user:password@host:5432/db` or
/// `mysql://user:password@host:3306/db` (no TLS). A SQLite file is opened read-only, and its
/// folder holds the chart and model files, like a CSV's; a server's dataset is in memory. A
/// query's rows can change between reads, so the stats are not cached.
///
/// Each column gets a type like `read_csv` gives: whole numbers are `i64`, other numbers
/// `f64`, and text stays text. SQLite has no true / false type, so a boolean column is 0 / 1.
/// PostgreSQL types other than numbers, booleans and text (`numeric`, dates) fail with a hint
/// to cast them in the query (`price::float8`, `day::text`); MySQL gives them as text.
///
/// # Panics
///
/// If the database can't be reached or the query fails. [`try_read_sql`] returns these as
/// errors.
///
/// # Examples
///
/// Basic: a SQLite file:
/// ```
/// use rusmlai::prelude::*;
///
/// let ds = read_sql("SELECT * FROM students", "tests/fixtures/data/students.db");
/// assert_eq!(ds.shape(), (12, 6));
/// ```
///
/// Advanced: a query, and a server:
/// ```no_run
/// # use rusmlai::prelude::*;
/// let passed = read_sql(
///     "SELECT age, cgpa FROM students WHERE passed = 1",
///     "tests/fixtures/data/students.db",
/// );
/// let ds = read_sql("SELECT * FROM students", "postgresql://me:secret@localhost:5432/school");
/// let ds = read_sql("SELECT * FROM students", "mysql://me:secret@localhost:3306/school");
/// ```
///
/// Edge case: a SQLite file that doesn't exist is an error, not a new empty database:
/// ```
/// # use rusmlai::prelude::*;
/// assert!(try_read_sql("SELECT 1", "does/not/exist.db").is_err());
/// assert!(!std::path::Path::new("does/not/exist.db").exists());
/// ```
///
/// Python:
///
/// ```python
/// import sqlite3
/// import pandas as pd
/// con = sqlite3.connect("tests/fixtures/data/students.db")
/// ds = pd.read_sql("SELECT * FROM students", con)
/// assert ds.shape == (12, 6)
/// ```
#[track_caller]
pub fn read_sql(sql: &str, con: &str) -> Dataset {
    try_read_sql(sql, con).or_fatal()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dtype(cells: &[Cell]) -> String {
        typed("x", cells).dtype().to_string()
    }

    #[test]
    fn a_column_gets_the_narrowest_type_that_holds_every_cell() {
        use Cell::*;
        assert_eq!(dtype(&[Int(1), Float(2.0), Null]), "i64");
        assert_eq!(dtype(&[Int(1), Float(2.5)]), "f64");
        assert_eq!(dtype(&[Bool(true), Null]), "bool");
        assert_eq!(dtype(&[Int(1), Text("a".into())]), "str");
        assert_eq!(dtype(&[Bool(true), Int(1)]), "str");
        assert_eq!(dtype(&[Null, Null]), "str"); // like an empty CSV column
        assert_eq!(dtype(&[Float(1e300)]), "f64"); // too big for i64
    }

    #[test]
    fn repeated_names_are_numbered_like_pandas() {
        let names: Vec<String> = ["a", "a", "a"].map(String::from).to_vec();
        let frame: DataFrame = frame((names, vec![vec![Cell::Int(1); 3]])).unwrap();
        let columns: Vec<&str> = frame.columns().iter().map(|c| c.name().as_str()).collect();
        assert_eq!(columns, ["a", "a.1", "a.2"]);
    }

    #[test]
    fn a_password_never_reaches_a_message() {
        assert_eq!(
            redacted("postgresql://me:secret@db:5432/x"),
            "postgresql://…@db:5432/x"
        );
        assert_eq!(redacted("data/school.db"), "data/school.db");
    }
}
