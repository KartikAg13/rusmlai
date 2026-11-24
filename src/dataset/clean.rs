//! pandas' everyday cleaning on a [`Dataset`]: `dropna`, `fillna`, `astype`, `rename`,
//! `query`, `drop_duplicates`, `sample`, `sort_values` and `value_counts`. Each gives a new
//! dataset; a column the step leaves unchanged keeps its cached stats, a changed one computes
//! them on first use.

use std::sync::Arc;

use indexmap::IndexMap;
use polars::prelude::{
    Column, DataFrame, DataType, Expr, IdxCa, IdxSize, IntoLazy, LazyFrame, NULL, NamedFrom, SortMultipleOptions, UniqueKeepStrategy,
    len, lit,
};

use crate::{
    dataset::{Dataset, error::DatasetError},
    error::OrFatal,
    metadata::{
        Metadata,
        dataset::{DatasetCache, info::ColumnInfo},
        plot::PlotCache,
    },
    models::folds::Mt19937,
    utils::col,
};

/// One value to fill with: a number, a whole number, text or true / false.
#[derive(Debug, Clone, PartialEq)]
pub enum Scalar {
    Float(f64),
    Int(i64),
    Text(String),
    Bool(bool),
}

impl From<f64> for Scalar {
    fn from(v: f64) -> Self {
        Scalar::Float(v)
    }
}

impl From<i64> for Scalar {
    fn from(v: i64) -> Self {
        Scalar::Int(v)
    }
}

impl From<i32> for Scalar {
    fn from(v: i32) -> Self {
        Scalar::Int(i64::from(v))
    }
}

impl From<&str> for Scalar {
    fn from(v: &str) -> Self {
        Scalar::Text(v.into())
    }
}

impl From<bool> for Scalar {
    fn from(v: bool) -> Self {
        Scalar::Bool(v)
    }
}

impl Scalar {
    pub(super) fn lit(&self) -> Expr {
        match self {
            Scalar::Float(v) => lit(*v),
            Scalar::Int(v) => lit(*v),
            Scalar::Text(v) => lit(v.clone()),
            Scalar::Bool(v) => lit(*v),
        }
    }
}

/// The Polars type of a pandas `astype` name.
fn dtype_of(name: &str) -> Result<DataType, DatasetError> {
    Ok(match name {
        "int64" | "int" | "Int64" => DataType::Int64,
        "int32" | "Int32" => DataType::Int32,
        "float64" | "float" | "Float64" => DataType::Float64,
        "float32" | "Float32" => DataType::Float32,
        "str" | "string" | "object" => DataType::String,
        "bool" | "boolean" => DataType::Boolean,
        _ => {
            return Err(DatasetError::InvalidOption {
                option: "astype",
                value: name.into(),
                expected: "\"int64\", \"int32\", \"float64\", \"float32\", \"str\" or \"bool\"",
            });
        }
    })
}

impl Dataset {
    /// `frame` as a dataset with this one's path (a step's result, or a model's output like
    /// `predict_proba`); the columns `kept` says are unchanged keep their stats, the others
    /// compute theirs on first use.
    pub(crate) fn reshaped(&self, frame: DataFrame, kept: impl Fn(&str) -> bool) -> Dataset {
        let columns: IndexMap<String, Arc<ColumnInfo>> = (frame.columns().iter())
            .map(|c| {
                let name: &str = c.name().as_str();
                let info: Arc<ColumnInfo> = match (kept(name), self.metadata.dataset_cache.columns.get(name)) {
                    (true, Some(info)) => Arc::clone(info),
                    _ => Arc::new(ColumnInfo::new(c)),
                };
                (name.to_string(), info)
            })
            .collect();
        let metadata: Metadata = Metadata {
            filepath: self.metadata.filepath.clone(),
            height: frame.height(),
            width: frame.width(),
            dataset_cache: DatasetCache::new(columns),
            plot_cache: PlotCache::default(),
        };
        Dataset::from_parts(frame, metadata)
    }

    /// Every named column exists; `&[]` is every column.
    pub(super) fn names(&self, columns: &[&str]) -> Result<Vec<String>, DatasetError> {
        match columns.is_empty() {
            true => Ok(self.columns()),
            false => columns.iter().map(|c| self.column(c).map(|_| c.to_string())).collect(),
        }
    }

    pub(super) fn lazy(&self) -> LazyFrame {
        self.dataframe.clone().lazy()
    }

    /// Whether a value of `name` is there: not null, and not NaN in a float column (pandas'
    /// `notna`).
    pub(super) fn present(&self, name: &str) -> Result<Expr, DatasetError> {
        let there: Expr = col(name).is_not_null();
        Ok(match self.column(name)?.dtype().is_float() {
            true => there.and(col(name).is_not_nan()),
            false => there,
        })
    }

    /// `name` with NaN as null, so Polars skips it where pandas skips NaN.
    pub(super) fn value(&self, name: &str) -> Result<Expr, DatasetError> {
        Ok(match self.column(name)?.dtype().is_float() {
            true => col(name).fill_nan(lit(NULL).cast(DataType::Float64)),
            false => col(name),
        })
    }

    /// The rows without a missing value (null, or NaN in a float column) in any of `subset`;
    /// `&[]` checks every column. (pandas: `df.dropna(subset=[...])`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// assert_eq!(ds.dropna(&[]).height(), 9); // a name, an age and a cgpa are missing
    /// assert_eq!(ds.dropna(&["age"]).height(), 11);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// assert len(ds.dropna()) == 9
    /// assert len(ds.dropna(subset=["age"])) == 11
    /// ```
    #[track_caller]
    pub fn dropna(&self, subset: &[&str]) -> Dataset {
        self.try_dropna(subset).or_fatal()
    }

    /// [`Dataset::dropna`], or the unknown column.
    pub fn try_dropna(&self, subset: &[&str]) -> Result<Dataset, DatasetError> {
        let mut keep: Expr = lit(true);
        for name in self.names(subset)? {
            keep = keep.and(self.present(&name)?);
        }
        let frame: DataFrame = self.lazy().filter(keep).collect()?;
        Ok(self.reshaped(frame, |_| false))
    }

    /// Missing values (null, or NaN in a float column) filled, one value per column. A whole
    /// number column filled with a fraction becomes a float column, as in pandas. (pandas:
    /// `df.fillna({"age": 0, "name": "unknown"})`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let filled = ds.fillna(&[("age", 0.into()), ("name", "unknown".into())]);
    /// assert_eq!(filled.to_list("age")[6], 0.0);
    /// assert_eq!(filled.value_counts("name", false).height(), 12);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// filled = ds.fillna({"age": 0, "name": "unknown"})
    /// assert filled["age"].tolist()[6] == 0.0
    /// assert len(filled["name"].value_counts()) == 12
    /// ```
    #[track_caller]
    pub fn fillna(&self, values: &[(&str, Scalar)]) -> Dataset {
        self.try_fillna(values).or_fatal()
    }

    /// [`Dataset::fillna`], or the unknown column, or a value of the wrong kind.
    pub fn try_fillna(&self, values: &[(&str, Scalar)]) -> Result<Dataset, DatasetError> {
        let mut filled: Vec<Expr> = Vec::new();
        for (name, value) in values {
            // Only the passes with something to fill: Polars keeps the null count, and a NaN
            // test reads the column without writing it.
            let column: &Column = self.column(name)?;
            let float: bool = matches!(column.dtype(), DataType::Float32 | DataType::Float64);
            let nans: bool = float && column.as_materialized_series().is_nan().is_ok_and(|nan| nan.any());
            let mut expr: Expr = col(*name);
            if column.null_count() > 0 {
                expr = expr.fill_null(value.lit());
            }
            if nans {
                expr = expr.fill_nan(value.lit());
            }
            filled.push(expr.alias(*name));
        }
        let frame: DataFrame = self.lazy().with_columns(filled).collect()?;
        Ok(self.reshaped(frame, |c| !values.iter().any(|(name, _)| *name == c)))
    }

    /// Columns changed to another type: `"int64"`, `"int32"`, `"float64"`, `"float32"`,
    /// `"str"` or `"bool"` (pandas' names). A value that does not convert fails, as in pandas.
    /// (pandas: `df.astype({"age": "float64"})`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let coded = ds.astype(&[("passed", "int64"), ("age", "float64")]);
    /// assert_eq!(coded.to_list("passed").iter().sum::<f64>(), 8.0);
    /// assert_eq!(coded.dtypes()[2], "f64");
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// coded = ds.astype({"passed": "int64", "age": "float64"})
    /// assert coded["passed"].sum() == 8
    /// assert str(coded["age"].dtype) == "float64"
    /// ```
    #[track_caller]
    pub fn astype(&self, types: &[(&str, &str)]) -> Dataset {
        self.try_astype(types).or_fatal()
    }

    /// [`Dataset::astype`], or the unknown column or type, or a value that does not convert.
    pub fn try_astype(&self, types: &[(&str, &str)]) -> Result<Dataset, DatasetError> {
        let mut cast: Vec<Expr> = Vec::new();
        for (name, dtype) in types {
            self.column(name)?;
            cast.push(col(*name).strict_cast(dtype_of(dtype)?).alias(*name));
        }
        let frame: DataFrame = self.lazy().with_columns(cast).collect()?;
        Ok(self.reshaped(frame, |c| !types.iter().any(|(name, _)| *name == c)))
    }

    /// Columns renamed, old name to new; every column keeps its stats. (pandas:
    /// `df.rename(columns={"cgpa": "gpa"})`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let renamed = ds.rename(&[("cgpa", "gpa")]);
    /// assert_eq!(renamed.columns()[3], "gpa");
    /// assert_eq!(renamed.try_numeric_stats("gpa").unwrap().max, 9.0);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// renamed = ds.rename(columns={"cgpa": "gpa"})
    /// assert renamed.columns[3] == "gpa"
    /// assert renamed["gpa"].max() == 9.0
    /// ```
    #[track_caller]
    pub fn rename(&self, names: &[(&str, &str)]) -> Dataset {
        self.try_rename(names).or_fatal()
    }

    /// [`Dataset::rename`], or the unknown column.
    pub fn try_rename(&self, names: &[(&str, &str)]) -> Result<Dataset, DatasetError> {
        let mut frame: DataFrame = self.dataframe.clone();
        for (old, new) in names {
            self.column(old)?;
            frame.rename(old, (*new).into())?;
        }
        let mut columns: IndexMap<String, Arc<ColumnInfo>> = IndexMap::new();
        for (name, info) in &self.metadata.dataset_cache.columns {
            let renamed: &str = names.iter().find(|(old, _)| old == name).map_or(name.as_str(), |(_, new)| new);
            columns.insert(renamed.to_string(), Arc::clone(info));
        }
        let mut metadata: Metadata = self.metadata.clone();
        metadata.dataset_cache = DatasetCache::new(columns);
        metadata.plot_cache = PlotCache::default();
        Ok(Dataset::from_parts(frame, metadata))
    }

    /// The rows where `condition`, a Polars expression, is true (a null counts as false).
    /// (pandas: `df[df["age"] > 21]`, or `df.query("age > 21")`)
    ///
    /// ```
    /// use rusmlai::polars::prelude::{col, lit};
    /// use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let older = ds.query(col("age").gt(lit(21)));
    /// assert_eq!(older.height(), 4);
    /// let both = ds.query(col("age").gt(lit(21)).and(col("passed")));
    /// assert_eq!(both.height(), 1);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// assert len(ds[ds["age"] > 21]) == 4
    /// assert len(ds.query("age > 21 and passed")) == 1
    /// ```
    #[track_caller]
    pub fn query(&self, condition: Expr) -> Dataset {
        self.try_query(condition).or_fatal()
    }

    /// [`Dataset::query`], or why Polars could not run the condition.
    pub fn try_query(&self, condition: Expr) -> Result<Dataset, DatasetError> {
        let frame: DataFrame = self.lazy().filter(condition).collect()?;
        Ok(self.reshaped(frame, |_| false))
    }

    /// The rows that are not repeats of an earlier row (`keep = "first"`, pandas' default), of
    /// a later one (`"last"`), or with no repeat at all (`"none"`, pandas' `keep=False`),
    /// comparing the columns of `subset` (`&[]`: all). The rows keep their order.
    /// (pandas: `df.drop_duplicates(subset=[...], keep="first")`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// assert_eq!(ds.drop_duplicates(&["batch"], "first").to_list("id"), [0.0, 1.0, 2.0]);
    /// assert_eq!(ds.drop_duplicates(&["batch"], "last").to_list("id"), [9.0, 10.0, 11.0]);
    /// assert_eq!(ds.drop_duplicates(&[], "first").height(), 12);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// assert ds.drop_duplicates(subset=["batch"])["id"].tolist() == [0, 1, 2]
    /// assert ds.drop_duplicates(subset=["batch"], keep="last")["id"].tolist() == [9, 10, 11]
    /// assert len(ds.drop_duplicates()) == 12
    /// ```
    #[track_caller]
    pub fn drop_duplicates(&self, subset: &[&str], keep: &str) -> Dataset {
        self.try_drop_duplicates(subset, keep).or_fatal()
    }

    /// [`Dataset::drop_duplicates`], or the unknown column or `keep`.
    pub fn try_drop_duplicates(&self, subset: &[&str], keep: &str) -> Result<Dataset, DatasetError> {
        let strategy: UniqueKeepStrategy = match keep {
            "first" => UniqueKeepStrategy::First,
            "last" => UniqueKeepStrategy::Last,
            "none" => UniqueKeepStrategy::None,
            _ => {
                return Err(DatasetError::InvalidOption {
                    option: "keep",
                    value: keep.into(),
                    expected: "\"first\", \"last\" or \"none\" (pandas' keep=False)",
                });
            }
        };
        let by: Vec<Expr> = self.names(subset)?.iter().map(col).collect();
        let frame: DataFrame = self.lazy().unique_stable_generic(Some(by), strategy).collect()?;
        Ok(self.reshaped(frame, |_| false))
    }

    /// `n` rows drawn without replacement, in the order drawn: the rows pandas draws with the
    /// same `random_state` (numpy's `RandomState(random_state).permutation`, which rusmlai
    /// reproduces). For a share, pass `(share * rows).round()`. (pandas:
    /// `df.sample(n, random_state=42)`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// assert_eq!(ds.sample(3, 42).to_list("id"), [10.0, 9.0, 0.0]);
    /// assert_eq!(ds.sample(12, 7).height(), 12); // every row, shuffled
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// assert ds.sample(3, random_state=42)["id"].tolist() == [10, 9, 0]
    /// assert len(ds.sample(12, random_state=7)) == 12
    /// ```
    #[track_caller]
    pub fn sample(&self, n: usize, random_state: u64) -> Dataset {
        self.try_sample(n, random_state).or_fatal()
    }

    /// [`Dataset::sample`], or more rows asked than there are.
    pub fn try_sample(&self, n: usize, random_state: u64) -> Result<Dataset, DatasetError> {
        if n > self.height() {
            return Err(DatasetError::InvalidOption {
                option: "n",
                value: n.to_string(),
                expected: "at most the number of rows (sample draws without replacement)",
            });
        }
        let mut order: Vec<IdxSize> = (0..self.height() as IdxSize).collect();
        Mt19937::new(random_state).shuffle(&mut order);
        let picked: IdxCa = IdxCa::new("".into(), &order[..n]);
        let frame: DataFrame = self.dataframe.take(&picked)?;
        Ok(self.reshaped(frame, |_| false))
    }

    /// The rows sorted by the columns `by`, each ascending or not (`ascending`: one value for
    /// every column, one per column, or `&[]` for ascending). Missing values go last and ties
    /// keep their order. (pandas: `df.sort_values(by, ascending=False)`; pandas' default sort
    /// of one column does not keep the order of ties.)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let best = ds.sort_values(&["cgpa"], &[false]);
    /// assert_eq!(best.to_list("id")[..3], [3.0, 11.0, 8.0]);
    /// assert_eq!(best.to_list("id")[11], 5.0); // the missing cgpa, last
    /// let grouped = ds.sort_values(&["batch", "cgpa"], &[true, false]);
    /// assert_eq!(grouped.to_list("id")[..4], [3.0, 0.0, 6.0, 9.0]);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// best = ds.sort_values("cgpa", ascending=False)
    /// assert best["id"].tolist()[:3] == [3, 11, 8]
    /// assert best["id"].tolist()[11] == 5
    /// grouped = ds.sort_values(["batch", "cgpa"], ascending=[True, False])
    /// assert grouped["id"].tolist()[:4] == [3, 0, 6, 9]
    /// ```
    #[track_caller]
    pub fn sort_values(&self, by: &[&str], ascending: &[bool]) -> Dataset {
        self.try_sort_values(by, ascending).or_fatal()
    }

    /// [`Dataset::sort_values`], or an unknown column, or `ascending` of another length.
    pub fn try_sort_values(&self, by: &[&str], ascending: &[bool]) -> Result<Dataset, DatasetError> {
        let names: Vec<String> = self.names(by)?;
        let descending: Vec<bool> = match ascending.len() {
            0 => vec![false; names.len()],
            1 => vec![!ascending[0]; names.len()],
            n if n == names.len() => ascending.iter().map(|a| !a).collect(),
            n => {
                return Err(DatasetError::InvalidOption {
                    option: "ascending",
                    value: format!("{n} values"),
                    expected: "one value, one per column of by, or none",
                });
            }
        };
        let options: SortMultipleOptions = SortMultipleOptions {
            nulls_last: vec![true; names.len()],
            descending,
            maintain_order: true,
            ..Default::default()
        };
        // NaN sorts as a missing value, last, as in pandas (Polars puts it above every number).
        let exprs: Vec<Expr> = names.iter().map(|name| self.value(name)).collect::<Result<_, _>>()?;
        let frame: DataFrame = self.lazy().sort_by_exprs(exprs, options).collect()?;
        Ok(self.reshaped(frame, |_| false))
    }

    /// How many rows hold each value of `column`, most first, as a dataset of two columns:
    /// the values and `count` (or `proportion`, each count over the rows with a value, when
    /// `normalize`). Missing values are left out; ties keep the order the values first
    /// appear in. (pandas: `df[column].value_counts(normalize=False)`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let counts = ds.value_counts("passed", false);
    /// assert_eq!(counts.columns(), ["passed", "count"]);
    /// assert_eq!(counts.to_list("count"), [8.0, 4.0]);
    /// let shares = ds.value_counts("batch", true);
    /// assert_eq!(shares.to_list("proportion"), [1.0 / 3.0; 3]);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// counts = ds["passed"].value_counts()
    /// assert counts.tolist() == [8, 4]
    /// shares = ds["batch"].value_counts(normalize=True)
    /// assert shares.index.tolist() == ["B13", "B14", "B15"]
    /// assert shares.tolist() == [1 / 3] * 3
    /// ```
    #[track_caller]
    pub fn value_counts(&self, column: &str, normalize: bool) -> Dataset {
        self.try_value_counts(column, normalize).or_fatal()
    }

    /// [`Dataset::value_counts`], or the unknown column.
    pub fn try_value_counts(&self, column: &str, normalize: bool) -> Result<Dataset, DatasetError> {
        self.column(column)?;
        let counted: LazyFrame = (self.lazy())
            .filter(col(column).is_not_null())
            .group_by_stable([col(column)])
            .agg([len().alias("count")])
            .sort_by_exprs(
                [col("count")],
                SortMultipleOptions { descending: vec![true], maintain_order: true, ..Default::default() },
            );
        let counted: LazyFrame = match normalize {
            true => counted.select([
                col(column),
                (col("count").cast(DataType::Float64) / col("count").sum().cast(DataType::Float64)).alias("proportion"),
            ]),
            false => counted,
        };
        let frame: DataFrame = counted.collect()?;
        Ok(self.reshaped(frame, |_| false))
    }
}

#[cfg(test)]
mod tests {
    use polars::df;
    use polars::prelude::col as polars_col;

    use super::*;

    fn messy() -> Dataset {
        Dataset::new(
            df!(
                "x" => [Some(1.0), Some(f64::NAN), None, Some(4.0), Some(4.0)],
                "s" => [Some("a"), Some("b"), Some("b"), None, None],
                "n" => [1i64, 2, 2, 3, 3],
            )
            .unwrap(),
            "",
        )
    }

    #[test]
    fn nan_counts_as_missing_like_pandas() {
        let ds: Dataset = messy();
        assert_eq!(ds.dropna(&["x"]).height(), 3);
        assert_eq!(ds.dropna(&[]).height(), 1);
        let filled: Dataset = ds.fillna(&[("x", 0.0.into()), ("s", "?".into())]);
        assert_eq!(filled.to_list("x"), [1.0, 0.0, 0.0, 4.0, 4.0]);
        assert_eq!(filled.dropna(&[]).height(), 5);
    }

    #[test]
    fn unchanged_columns_keep_their_stats_and_changed_ones_compute_new_ones() {
        let ds: Dataset = messy();
        let before = Arc::clone(&ds.metadata.dataset_cache.columns["n"]);
        let filled: Dataset = ds.fillna(&[("x", 9.0.into())]);
        assert!(Arc::ptr_eq(&before, &filled.metadata.dataset_cache.columns["n"]));
        assert_eq!(filled.try_numeric_stats("x").unwrap().max, 9.0);
        let renamed: Dataset = ds.rename(&[("n", "m")]);
        assert!(Arc::ptr_eq(&before, &renamed.metadata.dataset_cache.columns["m"]));
        // Fewer rows: stats of the rows left.
        assert_eq!(ds.query(polars_col("n").lt(lit(3))).try_numeric_stats("n").unwrap().max, 2.0);
    }

    #[test]
    fn duplicates_counts_and_order_follow_pandas() {
        let ds: Dataset = messy();
        assert_eq!(ds.drop_duplicates(&["n"], "none").to_list("n"), [1.0]);
        assert_eq!(ds.drop_duplicates(&["x", "s"], "first").height(), 4); // the two (4.0, null) rows are one
        let counts: Dataset = ds.value_counts("s", false);
        assert_eq!(counts.to_list("count"), [2.0, 1.0]); // null left out; b before a
        assert_eq!(ds.sort_values(&["n", "x"], &[false]).to_list("n"), [3.0, 3.0, 2.0, 2.0, 1.0]);
        // NaN and null both last, descending too (pandas: [4, 4, 1, nan, nan]).
        let x: Vec<f64> = ds.sort_values(&["x"], &[false]).to_list("x");
        assert_eq!(x[..3], [4.0, 4.0, 1.0]);
        assert!(x[3].is_nan() && x[4].is_nan());
    }

    #[test]
    fn bad_input_says_what_is_wrong() {
        let ds: Dataset = messy();
        let error = |e: DatasetError| e.to_string();
        assert!(error(ds.try_dropna(&["q"]).unwrap_err()).contains("'q' not found"));
        assert!(error(ds.try_drop_duplicates(&[], "false").unwrap_err()).contains("keep = 'false'"));
        assert!(error(ds.try_astype(&[("n", "category")]).unwrap_err()).contains("astype = 'category'"));
        assert!(error(ds.try_sample(6, 0).unwrap_err()).contains("at most the number of rows"));
        assert!(error(ds.try_sort_values(&["n", "x"], &[true, true, true]).unwrap_err()).contains("ascending"));
        assert!(ds.try_astype(&[("s", "int64")]).is_err()); // "a" is not a number
    }
}
