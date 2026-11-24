//! More of pandas' everyday `DataFrame` methods on a [`Dataset`]: combining (`merge`,
//! `concat`), grouping and reshaping (`groupby`, `pivot_table`, `melt`, `get_dummies`),
//! missing values (`isna`, `notna`, `ffill`, `bfill`), reductions over the columns (`sum`,
//! `mean`, `median`, `min`, `max`, `std`, `var`, `quantile`, `count`, `nunique`, `corr`) and
//! picks and changes (`unique`, `nlargest`, `nsmallest`, `iloc`, `select_dtypes`, `assign`,
//! `replace`, `clip`, `round`, `shift`, `diff`). Missing values follow pandas: NaN counts as
//! missing, and a reduction skips it. Each gives a new dataset; pandas' index has no
//! counterpart, so a reduction gives one row, and `corr` names its rows in a first column.

use std::ops::Range;

use polars::prelude::{
    ChunkApply, Column, DataFrame, DataType, Expr, FillNullStrategy, IntoColumn, JoinArgs, JoinCoalesce, JoinType, LazyFrame,
    MaintainOrderJoin, NULL, QuantileMethod, RankMethod, RankOptions, SortMultipleOptions, SortOptions, UnionArgs, lit, when,
};

use crate::{
    dataset::{Dataset, clean::Scalar, error::DatasetError},
    error::OrFatal,
    plot::heat::pearson,
    utils::col,
};

/// pandas' aggregation of `v`, one value per group (or per column).
fn agg(v: Expr, func: &str) -> Result<Expr, DatasetError> {
    Ok(match func {
        "sum" => v.sum(),
        "mean" => v.mean(),
        "median" => v.median(),
        "min" => v.min(),
        "max" => v.max(),
        "std" => v.std(1),
        "var" => v.var(1),
        "count" => v.count(),
        "nunique" => v.drop_nulls().n_unique(),
        "first" => v.drop_nulls().first(),
        "last" => v.drop_nulls().last(),
        "size" => v.len(),
        _ => {
            return Err(DatasetError::InvalidOption {
                option: "aggfunc",
                value: func.into(),
                expected: "\"sum\", \"mean\", \"median\", \"min\", \"max\", \"std\", \"var\", \"count\", \"nunique\", \"first\", \"last\" or \"size\"",
            });
        }
    })
}

impl Dataset {
    /// The number and true / false columns, which pandas' `numeric_only=True` keeps.
    fn numeric(&self) -> Vec<String> {
        (self.dataframe.columns().iter())
            .filter(|c| c.dtype().is_primitive_numeric() || c.dtype().is_bool())
            .map(|c| c.name().to_string())
            .collect()
    }

    /// The number columns, without true / false (pandas' `select_dtypes("number")`).
    fn numbers(&self) -> Vec<String> {
        (self.dataframe.columns().iter())
            .filter(|c| c.dtype().is_primitive_numeric())
            .map(|c| c.name().to_string())
            .collect()
    }

    /// [`Dataset::value`], with true / false as 1 / 0 so it sums and averages.
    fn number(&self, name: &str) -> Result<Expr, DatasetError> {
        let v: Expr = self.value(name)?;
        Ok(match self.column(name)?.dtype().is_bool() {
            true => v.cast(DataType::Int64),
            false => v,
        })
    }

    /// `f` of each of `columns`, under its own name: one row for an aggregation, else every row.
    fn try_each(&self, columns: &[String], f: impl Fn(&str) -> Result<Expr, DatasetError>) -> Result<Dataset, DatasetError> {
        let exprs: Vec<Expr> = columns.iter().map(|c| Ok(f(c)?.alias(c))).collect::<Result<_, DatasetError>>()?;
        Ok(self.reshaped(self.lazy().select(exprs).collect()?, |_| false))
    }

    fn reduce(&self, func: &str) -> Dataset {
        self.try_each(&self.numeric(), |c| agg(self.number(c)?, func)).or_fatal()
    }

    /// The sorted values of `column`, without missing ones, as text (the names of the
    /// columns that `pivot_table` and `get_dummies` make).
    fn levels(&self, column: &str) -> Result<Vec<String>, DatasetError> {
        let sorted: Expr = self.value(column)?.drop_nulls().unique().sort(SortOptions::default());
        let frame: DataFrame = self.lazy().select([sorted.cast(DataType::String)]).collect()?;
        Ok(frame.columns()[0].str()?.iter().flatten().map(String::from).collect())
    }

    /// The rows of `self` and `other` joined on the columns `on` (`&[]`: the columns they
    /// share), pandas' way: `how` is `"inner"`, `"left"`, `"right"` or `"outer"`; the keys
    /// appear once; another column in both gets `_x` and `_y`; missing keys match each other;
    /// inner and left joins keep the left rows' order, a right join the right's, and an outer
    /// join sorts the keys. (pandas: `df.merge(other, on="batch", how="left")`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let tutors = Dataset::new(df!("batch" => ["B13", "B14", "B16"], "tutor" => ["Ng", "Ito", "Roy"]).unwrap(), "");
    /// let inner = ds.merge(&tutors, &["batch"], "inner");
    /// assert_eq!(inner.to_list("id"), [0.0, 1.0, 3.0, 4.0, 6.0, 7.0, 9.0, 10.0]);
    /// assert_eq!(ds.merge(&tutors, &["batch"], "left").shape(), (12, 7));
    /// let outer = ds.merge(&tutors, &["batch"], "outer");
    /// assert_eq!(outer.to_list("id")[..5], [0.0, 3.0, 6.0, 9.0, 1.0]);
    /// assert!(outer.to_list("id")[12].is_nan()); // B16 has no student
    /// let twice = ds.merge(&ds.select(&["id", "age"]), &["id"], "inner");
    /// assert_eq!(twice.columns(), ["id", "name", "age_x", "cgpa", "batch", "passed", "age_y"]);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// tutors = pd.DataFrame({"batch": ["B13", "B14", "B16"], "tutor": ["Ng", "Ito", "Roy"]})
    /// inner = ds.merge(tutors, on="batch", how="inner")
    /// assert inner["id"].tolist() == [0, 1, 3, 4, 6, 7, 9, 10]
    /// assert ds.merge(tutors, on="batch", how="left").shape == (12, 7)
    /// outer = ds.merge(tutors, on="batch", how="outer")
    /// assert outer["id"].tolist()[:5] == [0, 3, 6, 9, 1]
    /// assert pd.isna(outer["id"].tolist()[12])
    /// twice = ds.merge(ds[["id", "age"]], on="id")
    /// assert twice.columns.tolist() == ["id", "name", "age_x", "cgpa", "batch", "passed", "age_y"]
    /// ```
    #[track_caller]
    pub fn merge(&self, other: &Dataset, on: &[&str], how: &str) -> Dataset {
        self.try_merge(other, on, how).or_fatal()
    }

    /// [`Dataset::merge`], or an unknown column or `how`.
    pub fn try_merge(&self, other: &Dataset, on: &[&str], how: &str) -> Result<Dataset, DatasetError> {
        let (join, order): (JoinType, MaintainOrderJoin) = match how {
            "inner" => (JoinType::Inner, MaintainOrderJoin::Left),
            "left" => (JoinType::Left, MaintainOrderJoin::Left),
            "right" => (JoinType::Right, MaintainOrderJoin::Right),
            "outer" => (JoinType::Full, MaintainOrderJoin::LeftRight),
            _ => {
                return Err(DatasetError::InvalidOption {
                    option: "how",
                    value: how.into(),
                    expected: "\"inner\", \"left\", \"right\" or \"outer\"",
                });
            }
        };
        let theirs: Vec<String> = other.columns();
        let keys: Vec<String> = match on.is_empty() {
            true => self.columns().into_iter().filter(|c| theirs.contains(c)).collect(),
            false => on.iter().map(|c| other.column(c).and(self.column(c)).map(|_| c.to_string())).collect::<Result<_, _>>()?,
        };
        let both: Vec<String> = (self.columns().into_iter()).filter(|c| theirs.contains(c) && !keys.contains(c)).collect();
        let suffixed = |end: &str| both.iter().map(|c| format!("{c}{end}")).collect::<Vec<String>>();
        let mut args: JoinArgs = JoinArgs::new(join).with_coalesce(JoinCoalesce::CoalesceColumns);
        args.maintain_order = order;
        args.nulls_equal = true;
        let on: Vec<Expr> = keys.iter().map(col).collect();
        let mut joined: LazyFrame = (self.lazy().rename(&both, suffixed("_x"), true))
            .join(other.lazy().rename(&both, suffixed("_y"), true), &on, &on, args);
        if how == "outer" {
            let sort: SortMultipleOptions = SortMultipleOptions { nulls_last: vec![true; on.len()], maintain_order: true, ..Default::default() };
            joined = joined.sort_by_exprs(on, sort);
        }
        Ok(self.reshaped(joined.collect()?, |_| false))
    }

    /// The rows of every dataset, one after the other (pandas: `pd.concat([a, b])`). Each needs
    /// the columns of the first, in any order; a column of another type becomes the type both
    /// fit, as in pandas.
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let both = Dataset::concat(&[&ds, &ds.iloc(0..2)]);
    /// assert_eq!(both.shape(), (14, 6));
    /// assert_eq!(both.to_list("id")[11..], [11.0, 0.0, 1.0]);
    /// ```
    ///
    /// Edge case: another set of columns is refused:
    ///
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// Dataset::concat(&[&ds, &ds.select(&["id"])]); // panics: needs the same columns
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// both = pd.concat([ds, ds.iloc[0:2]])
    /// assert both.shape == (14, 6)
    /// assert both["id"].tolist()[11:] == [11, 0, 1]
    /// ```
    #[track_caller]
    pub fn concat(datasets: &[&Dataset]) -> Dataset {
        Dataset::try_concat(datasets).or_fatal()
    }

    /// [`Dataset::concat`], or no dataset, or one with other columns.
    pub fn try_concat(datasets: &[&Dataset]) -> Result<Dataset, DatasetError> {
        let first: &Dataset = datasets.first().ok_or(DatasetError::InvalidOption {
            option: "datasets",
            value: "[]".into(),
            expected: "one dataset or more",
        })?;
        let mut names: Vec<String> = first.columns();
        names.sort();
        let order: Vec<Expr> = first.columns().iter().map(col).collect();
        let mut parts: Vec<LazyFrame> = Vec::new();
        for ds in datasets {
            let mut theirs: Vec<String> = ds.columns();
            theirs.sort();
            if theirs != names {
                return Err(DatasetError::InvalidOption {
                    option: "concat",
                    value: format!("columns {theirs:?}"),
                    expected: "datasets with the same columns",
                });
            }
            parts.push(ds.lazy().select(order.clone()));
        }
        let args: UnionArgs = UnionArgs { to_supertypes: true, ..Default::default() };
        Ok(first.reshaped(polars::prelude::concat(parts, args)?.collect()?, |_| false))
    }

    /// One row per group of the columns `by`, with an aggregation of other columns: `agg`
    /// pairs a column with `"sum"`, `"mean"`, `"median"`, `"min"`, `"max"`, `"std"`, `"var"`,
    /// `"count"`, `"nunique"`, `"first"`, `"last"` or `"size"`. Each result keeps its column's
    /// name (`{column}_{agg}` for a column aggregated twice). The groups sort by their keys,
    /// and rows with a missing key are left out, as in pandas.
    /// (pandas: `df.groupby("batch").agg({"cgpa": "mean"}).reset_index()`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let per_batch = ds.groupby(&["batch"], &[("cgpa", "mean"), ("age", "max"), ("id", "count")]);
    /// assert_eq!(per_batch.columns(), ["batch", "cgpa", "age", "id"]);
    /// assert_eq!(per_batch.round(6).to_list("cgpa"), [7.925, 6.375, 7.866667]);
    /// assert_eq!(per_batch.to_list("age"), [22.0, 23.0, 24.0]);
    /// let both = ds.groupby(&["batch", "passed"], &[("cgpa", "min"), ("cgpa", "max")]);
    /// assert_eq!(both.to_list("cgpa_min")[..2], [6.7, 7.9]); // B13 failed, B13 passed
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// per_batch = ds.groupby("batch").agg({"cgpa": "mean", "age": "max", "id": "count"}).reset_index()
    /// assert per_batch.columns.tolist() == ["batch", "cgpa", "age", "id"]
    /// assert per_batch["cgpa"].round(6).tolist() == [7.925, 6.375, 7.866667]
    /// assert per_batch["age"].tolist() == [22, 23, 24]
    /// both = ds.groupby(["batch", "passed"])["cgpa"].agg(["min", "max"]).reset_index()
    /// assert both["min"].tolist()[:2] == [6.7, 7.9]
    /// ```
    #[track_caller]
    pub fn groupby(&self, by: &[&str], agg: &[(&str, &str)]) -> Dataset {
        self.try_groupby(by, agg).or_fatal()
    }

    /// [`Dataset::groupby`], or an unknown column or aggregation, or no key.
    pub fn try_groupby(&self, by: &[&str], aggs: &[(&str, &str)]) -> Result<Dataset, DatasetError> {
        if by.is_empty() {
            return Err(DatasetError::InvalidOption { option: "by", value: "[]".into(), expected: "one column or more" });
        }
        let mut keyed: Expr = lit(true);
        for key in by {
            keyed = keyed.and(self.present(key)?);
        }
        let mut exprs: Vec<Expr> = Vec::new();
        for (column, func) in aggs {
            let twice: bool = aggs.iter().filter(|(c, _)| c == column).count() > 1;
            let name: String = if twice { format!("{column}_{func}") } else { column.to_string() };
            exprs.push(agg(self.number(column)?, func)?.alias(name));
        }
        let keys: Vec<Expr> = by.iter().map(|k| col(*k)).collect();
        let frame: DataFrame = (self.lazy().filter(keyed))
            .group_by(keys.clone())
            .agg(exprs)
            .sort_by_exprs(keys, SortMultipleOptions::default())
            .collect()?;
        Ok(self.reshaped(frame, |_| false))
    }

    /// A table of `aggfunc` (see [`Dataset::groupby`]) of `values`: one row per value of
    /// `index`, one column per value of `columns`, named by the value as text (`"true"` and
    /// `"false"` where pandas has `True` and `False`). Both sort, and missing keys are left
    /// out. (pandas: `df.pivot_table(values="cgpa", index="batch", columns="passed",
    /// aggfunc="mean").reset_index()`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let table = ds.pivot_table("cgpa", "batch", "passed", "mean");
    /// assert_eq!(table.columns(), ["batch", "false", "true"]);
    /// assert_eq!(table.round(6).to_list("false"), [6.7, 5.45, 6.2]);
    /// assert_eq!(table.round(6).to_list("true"), [8.333333, 7.3, 8.7]);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// table = ds.pivot_table(values="cgpa", index="batch", columns="passed", aggfunc="mean").reset_index()
    /// assert table.columns.tolist() == ["batch", False, True]
    /// assert table[False].round(6).tolist() == [6.7, 5.45, 6.2]
    /// assert table[True].round(6).tolist() == [8.333333, 7.3, 8.7]
    /// ```
    #[track_caller]
    pub fn pivot_table(&self, values: &str, index: &str, columns: &str, aggfunc: &str) -> Dataset {
        self.try_pivot_table(values, index, columns, aggfunc).or_fatal()
    }

    /// [`Dataset::pivot_table`], or an unknown column or aggregation.
    pub fn try_pivot_table(&self, values: &str, index: &str, columns: &str, aggfunc: &str) -> Result<Dataset, DatasetError> {
        let value: Expr = self.number(values)?;
        let mut cells: Vec<Expr> = Vec::new();
        for level in self.levels(columns)? {
            let here: Expr = col(columns).cast(DataType::String).eq(lit(level.as_str()));
            // A cell with no rows is missing, as in pandas, not a sum or count of 0.
            let cell: Expr = agg(value.clone().filter(here.clone()), aggfunc)?;
            cells.push(when(here.any(true)).then(cell).otherwise(lit(NULL)).alias(level));
        }
        let frame: DataFrame = (self.lazy().filter(self.present(index)?))
            .group_by([col(index)])
            .agg(cells)
            .sort_by_exprs([col(index)], SortMultipleOptions::default())
            .collect()?;
        Ok(self.reshaped(frame, |_| false))
    }

    /// Wide to long: the columns `id_vars`, then `variable` (a column's name) and `value`, one
    /// row per row and column of `value_vars` (`&[]`: every other column), column by column.
    /// The values are numbers if every column is, else text. (pandas:
    /// `df.melt(id_vars=["id"], value_vars=["age", "cgpa"])`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let long = ds.melt(&["id"], &["age", "cgpa"]);
    /// assert_eq!(long.columns(), ["id", "variable", "value"]);
    /// assert_eq!(long.shape(), (24, 3));
    /// assert_eq!([long.to_list("value")[0], long.to_list("value")[12]], [19.0, 8.1]);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// long = ds.melt(id_vars=["id"], value_vars=["age", "cgpa"])
    /// assert long.columns.tolist() == ["id", "variable", "value"]
    /// assert long.shape == (24, 3)
    /// assert [long["value"][0], long["value"][12]] == [19.0, 8.1]
    /// ```
    #[track_caller]
    pub fn melt(&self, id_vars: &[&str], value_vars: &[&str]) -> Dataset {
        self.try_melt(id_vars, value_vars).or_fatal()
    }

    /// [`Dataset::melt`], or an unknown column, or no column to melt.
    pub fn try_melt(&self, id_vars: &[&str], value_vars: &[&str]) -> Result<Dataset, DatasetError> {
        for c in id_vars {
            self.column(c)?;
        }
        let vars: Vec<String> = match value_vars.is_empty() {
            true => self.columns().into_iter().filter(|c| !id_vars.contains(&c.as_str())).collect(),
            false => self.names(value_vars)?,
        };
        let to: DataType = match vars.iter().all(|v| self.numbers().contains(v)) {
            true => DataType::Float64,
            false => DataType::String,
        };
        let parts: Vec<LazyFrame> = (vars.iter())
            .map(|v| {
                let mut picked: Vec<Expr> = id_vars.iter().map(|c| col(*c)).collect();
                picked.extend([lit(v.as_str()).alias("variable"), col(v).cast(to.clone()).alias("value")]);
                self.lazy().select(picked)
            })
            .collect();
        Ok(self.reshaped(polars::prelude::concat(parts, UnionArgs::default())?.collect()?, |_| false))
    }

    /// Each of `columns` (`&[]`: the text columns) as one true / false column per value,
    /// named `{column}_{value}` and put last, in value order; a missing value is false in all.
    /// (pandas: `pd.get_dummies(df, columns=["batch"])`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let wide = ds.get_dummies(&["batch"]);
    /// assert_eq!(wide.columns()[5..], ["batch_B13", "batch_B14", "batch_B15"]);
    /// assert_eq!(wide.to_list("batch_B13")[..4], [1.0, 0.0, 0.0, 1.0]);
    /// assert_eq!(ds.get_dummies(&[]).width(), 4 + 11 + 3); // name (11 values) and batch
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// wide = pd.get_dummies(ds, columns=["batch"])
    /// assert wide.columns.tolist()[5:] == ["batch_B13", "batch_B14", "batch_B15"]
    /// assert wide["batch_B13"].tolist()[:4] == [True, False, False, True]
    /// assert pd.get_dummies(ds).shape[1] == 4 + 11 + 3
    /// ```
    #[track_caller]
    pub fn get_dummies(&self, columns: &[&str]) -> Dataset {
        self.try_get_dummies(columns).or_fatal()
    }

    /// [`Dataset::get_dummies`], or an unknown column.
    pub fn try_get_dummies(&self, columns: &[&str]) -> Result<Dataset, DatasetError> {
        let encoded: Vec<String> = match columns.is_empty() {
            true => (self.dataframe.columns().iter()).filter(|c| c.dtype().is_string()).map(|c| c.name().to_string()).collect(),
            false => self.names(columns)?,
        };
        let mut picked: Vec<Expr> = (self.columns().into_iter()).filter(|c| !encoded.contains(c)).map(col).collect();
        for c in &encoded {
            for level in self.levels(c)? {
                let hit: Expr = col(c).cast(DataType::String).eq(lit(level.as_str())).fill_null(lit(false));
                picked.push(hit.alias(format!("{c}_{level}")));
            }
        }
        let frame: DataFrame = self.lazy().select(picked).collect()?;
        Ok(self.reshaped(frame, |_| true))
    }

    /// Every column as true where a value is missing (null, or NaN in a float column).
    /// `ds.isna().sum()` counts them. (pandas: `df.isna()`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let missing = ds.isna().sum();
    /// assert_eq!(missing.columns(), ds.columns());
    /// assert_eq!(["id", "name", "age", "cgpa"].map(|c| missing.to_list(c)[0]), [0.0, 1.0, 1.0, 1.0]);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// missing = ds.isna().sum()
    /// assert missing.index.tolist() == ds.columns.tolist()
    /// assert [missing[c] for c in ["id", "name", "age", "cgpa"]] == [0, 1, 1, 1]
    /// ```
    pub fn isna(&self) -> Dataset {
        self.try_each(&self.columns(), |c| Ok(self.present(c)?.not())).or_fatal()
    }

    /// Every column as true where a value is there: [`Dataset::isna`] flipped. (pandas:
    /// `df.notna()`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// assert_eq!(ds.notna().to_list("age")[5..8], [1.0, 0.0, 1.0]);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// assert ds.notna()["age"].tolist()[5:8] == [True, False, True]
    /// ```
    pub fn notna(&self) -> Dataset {
        self.try_each(&self.columns(), |c| self.present(c)).or_fatal()
    }

    /// Every missing value filled with the last value above it; a column that starts missing
    /// stays missing there. (pandas: `df.ffill()`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// assert_eq!(ds.ffill().to_list("age")[5..8], [20.0, 20.0, 23.0]);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// assert ds.ffill()["age"].tolist()[5:8] == [20, 20, 23]
    /// ```
    pub fn ffill(&self) -> Dataset {
        self.try_each(&self.columns(), |c| Ok(self.value(c)?.fill_null_with_strategy(FillNullStrategy::Forward(None)))).or_fatal()
    }

    /// Every missing value filled with the next value below it. (pandas: `df.bfill()`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// assert_eq!(ds.bfill().to_list("cgpa")[4..7], [5.8, 7.9, 7.9]);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// assert ds.bfill()["cgpa"].tolist()[4:7] == [5.8, 7.9, 7.9]
    /// ```
    pub fn bfill(&self) -> Dataset {
        self.try_each(&self.columns(), |c| Ok(self.value(c)?.fill_null_with_strategy(FillNullStrategy::Backward(None)))).or_fatal()
    }

    /// The sum of each number and true / false column (true counts 1), missing values
    /// skipped, as one row. (pandas: `df.sum(numeric_only=True)`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let total = ds.sum();
    /// assert_eq!(total.columns(), ["id", "age", "cgpa", "passed"]);
    /// assert_eq!(["id", "age", "passed"].map(|c| total.to_list(c)[0]), [66.0, 230.0, 8.0]);
    /// assert!((total.to_list("cgpa")[0] - 80.8).abs() < 1e-12);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// total = ds.sum(numeric_only=True)
    /// assert total.index.tolist() == ["id", "age", "cgpa", "passed"]
    /// assert [total[c] for c in ["id", "age", "passed"]] == [66, 230, 8]
    /// assert abs(total["cgpa"] - 80.8) < 1e-12
    /// ```
    pub fn sum(&self) -> Dataset {
        self.reduce("sum")
    }

    /// The mean of each number and true / false column, missing values skipped, as one row.
    /// (pandas: `df.mean(numeric_only=True)`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let mean = ds.mean();
    /// assert_eq!(["id", "passed"].map(|c| mean.to_list(c)[0]), [5.5, 8.0 / 12.0]);
    /// assert!((mean.to_list("age")[0] - 230.0 / 11.0).abs() < 1e-12); // 11 ages
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// mean = ds.mean(numeric_only=True)
    /// assert [mean["id"], mean["passed"]] == [5.5, 8 / 12]
    /// assert abs(mean["age"] - 230 / 11) < 1e-12
    /// ```
    pub fn mean(&self) -> Dataset {
        self.reduce("mean")
    }

    /// The median of each number and true / false column, missing values skipped, as one
    /// row. (pandas: `df.median(numeric_only=True)`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let median = ds.median();
    /// assert_eq!(["id", "age", "cgpa"].map(|c| median.to_list(c)[0]), [5.5, 21.0, 7.4]);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// median = ds.median(numeric_only=True)
    /// assert [median[c] for c in ["id", "age", "cgpa"]] == [5.5, 21.0, 7.4]
    /// ```
    pub fn median(&self) -> Dataset {
        self.reduce("median")
    }

    /// The smallest value of each number and true / false column, as one row. (pandas:
    /// `df.min(numeric_only=True)`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let low = ds.min();
    /// assert_eq!(["id", "age", "cgpa", "passed"].map(|c| low.to_list(c)[0]), [0.0, 19.0, 5.1, 0.0]);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// low = ds.min(numeric_only=True)
    /// assert [low[c] for c in ["id", "age", "cgpa", "passed"]] == [0, 19, 5.1, False]
    /// ```
    pub fn min(&self) -> Dataset {
        self.reduce("min")
    }

    /// The largest value of each number and true / false column, as one row. (pandas:
    /// `df.max(numeric_only=True)`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let high = ds.max();
    /// assert_eq!(["id", "age", "cgpa", "passed"].map(|c| high.to_list(c)[0]), [11.0, 24.0, 9.0, 1.0]);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// high = ds.max(numeric_only=True)
    /// assert [high[c] for c in ["id", "age", "cgpa", "passed"]] == [11, 24, 9.0, True]
    /// ```
    pub fn max(&self) -> Dataset {
        self.reduce("max")
    }

    /// The sample standard deviation (`ddof = 1`) of each number and true / false column,
    /// missing values skipped, as one row. (pandas: `df.std(numeric_only=True)`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let std = ds.std();
    /// assert!((std.to_list("age")[0] - 1.7002673586554236).abs() < 1e-12);
    /// assert!((std.to_list("cgpa")[0] - 1.2840277538773346).abs() < 1e-12);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// std = ds.std(numeric_only=True)
    /// assert abs(std["age"] - 1.7002673586554236) < 1e-12
    /// assert abs(std["cgpa"] - 1.2840277538773346) < 1e-12
    /// ```
    pub fn std(&self) -> Dataset {
        self.reduce("std")
    }

    /// The sample variance (`ddof = 1`) of each number and true / false column, missing
    /// values skipped, as one row. (pandas: `df.var(numeric_only=True)`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// assert!((ds.var().to_list("age")[0] - 2.890909090909091).abs() < 1e-12);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// assert abs(ds.var(numeric_only=True)["age"] - 2.890909090909091) < 1e-12
    /// ```
    pub fn var(&self) -> Dataset {
        self.reduce("var")
    }

    /// The `q` quantile (0 to 1) of each number and true / false column, interpolated
    /// linearly between two values as pandas does, missing values skipped, as one row.
    /// (pandas: `df.quantile(0.25, numeric_only=True)`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let q = ds.quantile(0.25);
    /// assert_eq!(["id", "age"].map(|c| q.to_list(c)[0]), [2.75, 19.5]);
    /// assert!((q.to_list("cgpa")[0] - 6.45).abs() < 1e-12);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// q = ds.select_dtypes("number").quantile(0.25)  # pandas 3 fails on the bool column
    /// assert [q["id"], q["age"]] == [2.75, 19.5]
    /// assert abs(q["cgpa"] - 6.45) < 1e-12
    /// ```
    #[track_caller]
    pub fn quantile(&self, q: f64) -> Dataset {
        self.try_quantile(q).or_fatal()
    }

    /// [`Dataset::quantile`], or `q` outside 0 to 1.
    pub fn try_quantile(&self, q: f64) -> Result<Dataset, DatasetError> {
        if !(0.0..=1.0).contains(&q) {
            return Err(DatasetError::InvalidOption { option: "q", value: q.to_string(), expected: "a number from 0 to 1" });
        }
        self.try_each(&self.numeric(), |c| Ok(self.number(c)?.quantile(lit(q), QuantileMethod::Linear)))
    }

    /// How many values each column has (missing ones left out), as one row. (pandas:
    /// `df.count()`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let count = ds.count();
    /// assert_eq!(ds.columns().iter().map(|c| count.to_list(c)[0]).collect::<Vec<f64>>(), [12.0, 11.0, 11.0, 11.0, 12.0, 12.0]);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// assert ds.count().tolist() == [12, 11, 11, 11, 12, 12]
    /// ```
    pub fn count(&self) -> Dataset {
        self.try_each(&self.columns(), |c| agg(self.value(c)?, "count")).or_fatal()
    }

    /// How many different values each column has (missing ones left out), as one row.
    /// (pandas: `df.nunique()`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let distinct = ds.nunique();
    /// assert_eq!(ds.columns().iter().map(|c| distinct.to_list(c)[0]).collect::<Vec<f64>>(), [12.0, 11.0, 6.0, 11.0, 3.0, 2.0]);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// assert ds.nunique().tolist() == [12, 11, 6, 11, 3, 2]
    /// ```
    pub fn nunique(&self) -> Dataset {
        self.try_each(&self.columns(), |c| agg(self.value(c)?, "nunique")).or_fatal()
    }

    /// The Pearson correlation of each pair of number and true / false columns, on the rows
    /// where both have a value: a first column `column` names the rows, then one column per
    /// column. (pandas: `df.corr(numeric_only=True)`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let corr = ds.corr();
    /// assert_eq!(corr.columns(), ["column", "id", "age", "cgpa", "passed"]);
    /// assert_eq!(corr.to_list("cgpa")[2], 1.0);
    /// assert_eq!(corr.round(6).to_list("cgpa")[3], 0.861626); // passed students have a higher cgpa
    /// assert_eq!(corr.to_list("age")[0], corr.to_list("id")[1]);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// corr = ds.corr(numeric_only=True)
    /// assert corr.columns.tolist() == ["id", "age", "cgpa", "passed"]
    /// assert corr["cgpa"]["cgpa"] == 1.0
    /// assert round(corr["cgpa"]["passed"], 6) == 0.861626
    /// assert corr["age"]["id"] == corr["id"]["age"]
    /// ```
    pub fn corr(&self) -> Dataset {
        self.try_corr().or_fatal()
    }

    fn try_corr(&self) -> Result<Dataset, DatasetError> {
        let names: Vec<String> = self.numeric();
        let n: usize = names.len();
        let pairs: Vec<Expr> = (0..n)
            .flat_map(|i| (i..n).map(move |j| (i, j)))
            .map(|(i, j)| pearson(&names[i], &names[j]).alias(format!("{i},{j}")))
            .collect();
        let rs: DataFrame = self.lazy().select(pairs).collect()?;
        let r = |i: usize, j: usize| -> Result<f64, DatasetError> {
            let name: String = format!("{},{}", i.min(j), i.max(j));
            Ok(rs.column(&name)?.f64()?.get(0).unwrap_or(f64::NAN).clamp(-1.0, 1.0))
        };
        let mut columns: Vec<Column> = vec![Column::new("column".into(), names.as_slice())];
        for (j, name) in names.iter().enumerate() {
            let values: Vec<f64> = (0..n).map(|i| r(i, j)).collect::<Result<_, _>>()?;
            columns.push(Column::new(name.into(), values));
        }
        Ok(self.reshaped(DataFrame::new_infer_height(columns)?, |_| false))
    }

    /// The values of `column` in the order they first appear, a missing value once, as a
    /// dataset of that column. (pandas: `df["batch"].unique()`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// assert_eq!(ds.unique("batch").height(), 3);
    /// assert_eq!(ds.unique("age").to_list("age")[..4], [19.0, 20.0, 21.0, 22.0]);
    /// assert!(ds.unique("age").to_list("age")[4].is_nan()); // the missing age
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// assert len(ds["batch"].unique()) == 3
    /// assert ds["age"].unique().tolist()[:4] == [19, 20, 21, 22]
    /// assert pd.isna(ds["age"].unique()[4])
    /// ```
    #[track_caller]
    pub fn unique(&self, column: &str) -> Dataset {
        self.try_unique(column).or_fatal()
    }

    /// [`Dataset::unique`], or the unknown column.
    pub fn try_unique(&self, column: &str) -> Result<Dataset, DatasetError> {
        let frame: DataFrame = self.lazy().select([self.value(column)?.unique_stable()]).collect()?;
        Ok(self.reshaped(frame, |_| false))
    }

    /// The `n` rows with the largest values of `column`, largest first; ties keep their order
    /// and missing values are left out. (pandas: `df.nlargest(3, "cgpa")`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// assert_eq!(ds.nlargest(3, "cgpa").to_list("id"), [3.0, 11.0, 8.0]);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// assert ds.nlargest(3, "cgpa")["id"].tolist() == [3, 11, 8]
    /// ```
    #[track_caller]
    pub fn nlargest(&self, n: usize, column: &str) -> Dataset {
        self.try_nlargest(n, column).or_fatal()
    }

    /// [`Dataset::nlargest`], or the unknown column.
    pub fn try_nlargest(&self, n: usize, column: &str) -> Result<Dataset, DatasetError> {
        self.try_top(n, column, false)
    }

    /// The first `n` rows with a value in `column`, sorted by it.
    fn try_top(&self, n: usize, column: &str, ascending: bool) -> Result<Dataset, DatasetError> {
        Ok(self.try_query(self.present(column)?)?.try_sort_values(&[column], &[ascending])?.iloc(0..n))
    }

    /// The `n` rows with the smallest values of `column`, smallest first; ties keep their
    /// order and missing values are left out. (pandas: `df.nsmallest(3, "age")`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// assert_eq!(ds.nsmallest(3, "age").to_list("id"), [0.0, 3.0, 10.0]);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// assert ds.nsmallest(3, "age")["id"].tolist() == [0, 3, 10]
    /// ```
    #[track_caller]
    pub fn nsmallest(&self, n: usize, column: &str) -> Dataset {
        self.try_nsmallest(n, column).or_fatal()
    }

    /// [`Dataset::nsmallest`], or the unknown column.
    pub fn try_nsmallest(&self, n: usize, column: &str) -> Result<Dataset, DatasetError> {
        self.try_top(n, column, true)
    }

    /// The rows at the places `rows` (a range past the end stops at the end). (pandas:
    /// `df.iloc[2:5]`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// assert_eq!(ds.iloc(2..5).to_list("id"), [2.0, 3.0, 4.0]);
    /// assert_eq!(ds.iloc(10..20).height(), 2);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// assert ds.iloc[2:5]["id"].tolist() == [2, 3, 4]
    /// assert len(ds.iloc[10:20]) == 2
    /// ```
    pub fn iloc(&self, rows: Range<usize>) -> Dataset {
        let frame: DataFrame = self.dataframe.slice(rows.start as i64, rows.len());
        self.reshaped(frame, |_| false)
    }

    /// The columns of one kind: `"number"` (whole numbers and floats), `"bool"` or `"str"`
    /// (text; pandas 3's name, `"string"` and `"object"` too). They keep their stats. (pandas:
    /// `df.select_dtypes("number")`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// assert_eq!(ds.select_dtypes("number").columns(), ["id", "age", "cgpa"]);
    /// assert_eq!(ds.select_dtypes("str").columns(), ["name", "batch"]);
    /// assert_eq!(ds.select_dtypes("bool").columns(), ["passed"]);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// assert ds.select_dtypes("number").columns.tolist() == ["id", "age", "cgpa"]
    /// assert ds.select_dtypes("str").columns.tolist() == ["name", "batch"]
    /// assert ds.select_dtypes("bool").columns.tolist() == ["passed"]
    /// ```
    #[track_caller]
    pub fn select_dtypes(&self, include: &str) -> Dataset {
        self.try_select_dtypes(include).or_fatal()
    }

    /// [`Dataset::select_dtypes`], or an unknown kind.
    pub fn try_select_dtypes(&self, include: &str) -> Result<Dataset, DatasetError> {
        let kind: fn(&DataType) -> bool = match include {
            "number" => DataType::is_primitive_numeric,
            "bool" => DataType::is_bool,
            "object" | "str" | "string" => DataType::is_string,
            _ => {
                return Err(DatasetError::InvalidOption {
                    option: "include",
                    value: include.into(),
                    expected: "\"number\", \"bool\" or \"str\"",
                });
            }
        };
        let picked: Vec<Column> = (self.dataframe.columns().iter()).filter(|c| kind(c.dtype())).cloned().collect();
        Ok(self.reshaped(DataFrame::new(self.height(), picked)?, |_| true))
    }

    /// New or replaced columns, each from a Polars expression; the other columns keep their
    /// stats. (pandas: `df.assign(older=df["age"] + 1)`)
    ///
    /// ```
    /// use rusmlai::polars::prelude::{col, lit};
    /// use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let more = ds.assign(&[("older", col("age") + lit(1))]);
    /// assert_eq!(more.to_list("older")[..3], [20.0, 21.0, 22.0]);
    /// ```
    ///
    /// The features of Kaggle notebooks: text columns joined, a log, a rank, a set test, a
    /// range test, bins and a group mean on every row (pandas' `groupby(..).transform`).
    ///
    /// ```
    /// use rusmlai::polars::prelude::*;
    /// use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let average = RankOptions { method: RankMethod::Average, descending: false };
    /// let early: Expr = lit(Series::new("".into(), ["B13", "B14"])).implode(false);
    /// let more = ds.assign(&[
    ///     ("id_batch", concat_str([col("id").cast(DataType::String), col("batch")], "_", false)),
    ///     ("log_cgpa", col("cgpa").log1p()),
    ///     ("cgpa_rank", col("cgpa").rank(average, None)),
    ///     ("early", col("batch").is_in(early, false)),
    ///     ("twenties", col("age").is_between(lit(20), lit(29), ClosedInterval::Both)),
    ///     ("age_bin", col("age").cut(vec![20.0, 22.0], Some(["young", "mid", "old"]), false, false)),
    ///     ("batch_cgpa", col("cgpa").mean().over([col("batch")]).unwrap()),
    /// ]);
    /// assert_eq!(more.query(col("id_batch").eq(lit("0_B13"))).height(), 1);
    /// assert_eq!(more.to_list("log_cgpa")[0], 8.1_f64.ln_1p());
    /// assert_eq!(more.to_list("cgpa_rank")[..4], [8.0, 6.0, 3.0, 11.0]);
    /// assert_eq!(more.to_list("early")[..3], [1.0, 1.0, 0.0]);
    /// assert_eq!(more.to_list("twenties")[..2], [0.0, 1.0]);
    /// let _ = more.describe();
    /// assert_eq!(more.query(col("age_bin").eq(lit("young"))).height(), 5);
    /// assert!((more.to_list("batch_cgpa")[0] - 7.925).abs() < 1e-12);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import numpy as np
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// more = ds.assign(older=ds["age"] + 1)
    /// assert more["older"].tolist()[:3] == [20, 21, 22]
    /// more = ds.assign(
    ///     id_batch=ds["id"].astype(str) + "_" + ds["batch"],
    ///     log_cgpa=np.log1p(ds["cgpa"]),
    ///     cgpa_rank=ds["cgpa"].rank(),
    ///     early=ds["batch"].isin(["B13", "B14"]),
    ///     twenties=ds["age"].between(20, 29),
    ///     age_bin=pd.cut(ds["age"], [-np.inf, 20, 22, np.inf], labels=["young", "mid", "old"]),
    ///     batch_cgpa=ds.groupby("batch")["cgpa"].transform("mean"),
    /// )
    /// assert (more["id_batch"] == "0_B13").sum() == 1
    /// assert more["log_cgpa"][0] == np.log1p(8.1)
    /// assert more["cgpa_rank"].tolist()[:4] == [8.0, 6.0, 3.0, 11.0]
    /// assert more["early"].tolist()[:3] == [True, True, False]
    /// assert more["twenties"].tolist()[:2] == [False, True]
    /// assert (more["age_bin"] == "young").sum() == 5
    /// assert abs(more["batch_cgpa"][0] - 7.925) < 1e-12
    /// ```
    #[track_caller]
    pub fn assign(&self, columns: &[(&str, Expr)]) -> Dataset {
        self.try_assign(columns).or_fatal()
    }

    /// [`Dataset::assign`], or why Polars could not run an expression.
    pub fn try_assign(&self, columns: &[(&str, Expr)]) -> Result<Dataset, DatasetError> {
        let exprs: Vec<Expr> = columns.iter().map(|(name, e)| e.clone().alias(*name)).collect();
        let frame: DataFrame = self.lazy().with_columns(exprs).collect()?;
        Ok(self.reshaped(frame, |c| !columns.iter().any(|(name, _)| *name == c)))
    }

    /// The values of `column` changed, old to new, all at once (a new value is not replaced
    /// again). (pandas: `df.replace({"batch": {"B13": "first"}})`)
    ///
    /// ```
    /// use rusmlai::polars::prelude::{col, lit};
    /// use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let named = ds.replace("batch", &[("B13".into(), "first".into()), ("first".into(), "x".into())]);
    /// assert_eq!(named.query(col("batch").eq(lit("first"))).height(), 4);
    /// assert_eq!(ds.replace("age", &[(19.into(), 18.into())]).to_list("age")[0], 18.0);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// named = ds.replace({"batch": {"B13": "first", "first": "x"}})
    /// assert (named["batch"] == "first").sum() == 4
    /// assert ds.replace({"age": {19: 18}})["age"][0] == 18
    /// ```
    #[track_caller]
    pub fn replace(&self, column: &str, pairs: &[(Scalar, Scalar)]) -> Dataset {
        self.try_replace(column, pairs).or_fatal()
    }

    /// [`Dataset::replace`], or the unknown column, or values Polars can't compare with it.
    pub fn try_replace(&self, column: &str, pairs: &[(Scalar, Scalar)]) -> Result<Dataset, DatasetError> {
        self.column(column)?;
        // Built inside out, so the first pair that matches wins and every test reads the old value.
        let mut changed: Expr = col(column);
        for (old, new) in pairs.iter().rev() {
            changed = when(col(column).eq(old.lit())).then(new.lit()).otherwise(changed);
        }
        self.try_assign(&[(column, changed)])
    }

    /// Every number column held between `lower` and `upper` (`f64::NEG_INFINITY` or
    /// `f64::INFINITY` for no bound); a missing value stays missing. (pandas:
    /// `df.select_dtypes("number").clip(6, 8.5)`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let held = ds.clip(6.0, 8.5);
    /// assert_eq!(held.to_list("cgpa")[..5], [8.1, 7.4, 6.2, 8.5, 6.0]);
    /// assert_eq!(ds.clip(20.0, f64::INFINITY).to_list("age")[..3], [20.0, 20.0, 21.0]);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// held = ds.select_dtypes("number").clip(6, 8.5)
    /// assert held["cgpa"].tolist()[:5] == [8.1, 7.4, 6.2, 8.5, 6.0]
    /// assert ds["age"].clip(lower=20).tolist()[:3] == [20, 20, 21]
    /// ```
    pub fn clip(&self, lower: f64, upper: f64) -> Dataset {
        let held = |c: &str| -> Expr {
            let mut e: Expr = col(c);
            if lower > f64::NEG_INFINITY {
                e = when(col(c).lt(lit(lower))).then(lit(lower)).otherwise(e);
            }
            if upper < f64::INFINITY {
                e = when(col(c).gt(lit(upper))).then(lit(upper)).otherwise(e);
            }
            e
        };
        let names: Vec<String> = self.numbers();
        let columns: Vec<(&str, Expr)> = names.iter().map(|c| (c.as_str(), held(c))).collect();
        self.try_assign(&columns).or_fatal()
    }

    /// Every float column rounded to `decimals` places, half to even, as numpy does.
    /// (pandas: `df.round(1)`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// assert_eq!(ds.round(0).to_list("cgpa")[..5], [8.0, 7.0, 6.0, 9.0, 6.0]);
    /// assert_eq!(ds.round(0).to_list("age")[..3], [19.0, 20.0, 21.0]); // whole numbers stay
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// assert ds.round(0)["cgpa"].tolist()[:5] == [8.0, 7.0, 6.0, 9.0, 6.0]
    /// assert ds.round(0)["age"].tolist()[:3] == [19, 20, 21]
    /// ```
    pub fn round(&self, decimals: u32) -> Dataset {
        self.try_round(decimals).or_fatal()
    }

    fn try_round(&self, decimals: u32) -> Result<Dataset, DatasetError> {
        let scale: f64 = 10f64.powi(decimals as i32);
        let round = |c: &Column| -> Result<Column, DatasetError> {
            match c.dtype().is_float() {
                true => Ok(c.cast(&DataType::Float64)?.f64()?.apply_values(|v| (v * scale).round_ties_even() / scale).into_column()),
                false => Ok(c.clone()),
            }
        };
        let columns: Vec<Column> = self.dataframe.columns().iter().map(round).collect::<Result<_, _>>()?;
        let floats: Vec<&str> = (self.dataframe.columns().iter()).filter(|c| c.dtype().is_float()).map(|c| c.name().as_str()).collect();
        Ok(self.reshaped(DataFrame::new(self.height(), columns)?, |c| !floats.contains(&c)))
    }

    /// Every column moved down `periods` rows (up when negative), missing values coming in.
    /// (pandas: `df.shift(1)`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let before = ds.shift(1).to_list("id");
    /// assert!(before[0].is_nan());
    /// assert_eq!(before[1..3], [0.0, 1.0]);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// before = ds.shift(1)["id"].tolist()
    /// assert pd.isna(before[0])
    /// assert before[1:3] == [0, 1]
    /// ```
    pub fn shift(&self, periods: i64) -> Dataset {
        self.try_each(&self.columns(), |c| Ok(col(c).shift(lit(periods)))).or_fatal()
    }

    /// Each value of every number column minus the one `periods` rows above it, as floats;
    /// the other columns are left out. (pandas: `df.select_dtypes("number").diff()`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let steps = ds.diff(1);
    /// assert_eq!(steps.columns(), ["id", "age", "cgpa"]);
    /// assert_eq!(steps.to_list("age")[1..4], [1.0, 1.0, -2.0]);
    /// assert!(steps.to_list("cgpa")[5].is_nan() && steps.to_list("cgpa")[6].is_nan());
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// steps = ds.select_dtypes("number").diff()
    /// assert steps.columns.tolist() == ["id", "age", "cgpa"]
    /// assert steps["age"].tolist()[1:4] == [1, 1, -2]
    /// assert pd.isna(steps["cgpa"][5]) and pd.isna(steps["cgpa"][6])
    /// ```
    pub fn diff(&self, periods: i64) -> Dataset {
        let step = |c: &str| {
            let v: Expr = self.value(c)?.cast(DataType::Float64);
            Ok(v.clone() - v.shift(lit(periods)))
        };
        self.try_each(&self.numbers(), step).or_fatal()
    }

    /// The rank of each value of every number column within its column, as floats, smallest
    /// first (`ascending`) from 1; the other columns are left out and a missing value stays
    /// missing. Ties take `method`: `"average"` (their mean rank), `"min"`, `"max"`, `"first"`
    /// (in row order) or `"dense"` (no gaps). `pct` divides by the values present (by the
    /// greatest rank for `"dense"`). The step before averaging the predictions of several
    /// models whose scores are on different scales. (pandas:
    /// `df.select_dtypes("number").rank(method, ascending=, pct=)`)
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// // ages 19, 20, 21, 19, 22, 20, missing, 23
    /// assert_eq!(ds.rank("average", true, false).to_list("age")[..6], [2.0, 4.5, 6.5, 2.0, 8.5, 4.5]);
    /// assert_eq!(ds.rank("first", true, false).to_list("age")[..6], [1.0, 4.0, 6.0, 2.0, 8.0, 5.0]);
    /// assert_eq!(ds.rank("dense", true, false).to_list("age")[..6], [1.0, 2.0, 3.0, 1.0, 4.0, 2.0]);
    /// assert_eq!(ds.rank("average", false, false).to_list("age")[..2], [10.0, 7.5]);
    /// assert_eq!(ds.rank("average", true, true).to_list("age")[0], 2.0 / 11.0);
    /// assert!(ds.rank("min", true, false).to_list("age")[6].is_nan());
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// ds = pd.read_csv("tests/fixtures/data/students.csv").select_dtypes("number")
    /// assert ds.rank()["age"].tolist()[:6] == [2.0, 4.5, 6.5, 2.0, 8.5, 4.5]
    /// assert ds.rank(method="first")["age"].tolist()[:6] == [1.0, 4.0, 6.0, 2.0, 8.0, 5.0]
    /// assert ds.rank(method="dense")["age"].tolist()[:6] == [1.0, 2.0, 3.0, 1.0, 4.0, 2.0]
    /// assert ds.rank(ascending=False)["age"].tolist()[:2] == [10.0, 7.5]
    /// assert ds.rank(pct=True)["age"][0] == 2 / 11
    /// assert pd.isna(ds.rank(method="min")["age"][6])
    /// ```
    #[track_caller]
    pub fn rank(&self, method: &str, ascending: bool, pct: bool) -> Dataset {
        self.try_rank(method, ascending, pct).or_fatal()
    }

    /// [`Dataset::rank`], or [`DatasetError::InvalidOption`] for another `method`.
    pub fn try_rank(&self, method: &str, ascending: bool, pct: bool) -> Result<Dataset, DatasetError> {
        let method: RankMethod = match method {
            "average" => RankMethod::Average,
            "min" => RankMethod::Min,
            "max" => RankMethod::Max,
            "first" => RankMethod::Ordinal,
            "dense" => RankMethod::Dense,
            _ => {
                return Err(DatasetError::InvalidOption {
                    option: "method",
                    value: method.into(),
                    expected: "\"average\", \"min\", \"max\", \"first\" or \"dense\"",
                });
            }
        };
        let options: RankOptions = RankOptions { method, descending: !ascending };
        let dense: bool = matches!(method, RankMethod::Dense);
        self.try_each(&self.numbers(), |c| {
            let rank: Expr = self.value(c)?.rank(options, None).cast(DataType::Float64);
            Ok(match (pct, dense) {
                (false, _) => rank,
                (true, true) => rank.clone() / rank.max(),
                // `value` made every NaN null, so `count` counts the values present.
                (true, false) => rank / self.value(c)?.count().cast(DataType::Float64),
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use polars::df;

    use super::*;

    /// A NaN, a null and a true / false column, which students.csv does not have.
    fn messy() -> Dataset {
        Dataset::new(
            df!(
                "x" => [Some(1.0), Some(f64::NAN), None, Some(4.0), Some(2.5)],
                "b" => [true, false, true, true, false],
                "k" => [Some("a"), Some("b"), None, Some("a"), Some("b")],
            )
            .unwrap(),
            "",
        )
    }

    #[test]
    fn nan_is_missing_everywhere_like_pandas() {
        let ds: Dataset = messy();
        // pandas: x.sum() 7.5, x.mean() 2.5, x.count() 3, x.nunique() 3, b.sum() 3
        assert_eq!(ds.sum().to_list("x"), [7.5]);
        assert_eq!(ds.mean().to_list("x"), [2.5]);
        assert_eq!(ds.median().to_list("x"), [2.5]);
        assert_eq!(ds.sum().to_list("b"), [3.0]);
        assert_eq!(ds.count().to_list("x"), [3.0]);
        assert_eq!(ds.nunique().to_list("x"), [3.0]);
        assert_eq!(ds.isna().to_list("x"), [0.0, 1.0, 1.0, 0.0, 0.0]);
        assert_eq!(ds.ffill().to_list("x"), [1.0, 1.0, 1.0, 4.0, 2.5]);
        // groupby drops the missing key and sorts; pivot_table names columns by value.
        let g: Dataset = ds.groupby(&["k"], &[("x", "sum"), ("b", "size")]);
        assert_eq!((g.to_list("x"), g.to_list("b")), (vec![5.0, 2.5], vec![2.0, 2.0]));
        assert_eq!(ds.pivot_table("x", "k", "b", "mean").columns(), ["k", "false", "true"]);
        // pandas: a: false NaN, true 2; b: false 2, true NaN (size counts the NaN row).
        let sizes: Dataset = ds.pivot_table("x", "k", "b", "size");
        assert!(sizes.to_list("false")[0].is_nan() && sizes.to_list("true")[1].is_nan());
        assert_eq!((sizes.to_list("false")[1], sizes.to_list("true")[0]), (2.0, 2.0));
    }

    #[test]
    fn a_right_join_keeps_pandas_columns_and_the_right_order() {
        let ds: Dataset = messy();
        let other: Dataset = Dataset::new(df!("k" => ["b", "z", "a"], "t" => [1, 2, 3]).unwrap(), "");
        // pandas: ds.merge(other, on="k", how="right") gives x, b, k, t and the keys b, b, z, a, a.
        let right: Dataset = ds.merge(&other, &["k"], "right");
        assert_eq!(right.columns(), ["x", "b", "k", "t"]);
        assert_eq!(right.to_list("t"), [1.0, 1.0, 2.0, 3.0, 3.0]);
        // Missing keys match each other, as in pandas.
        assert_eq!(ds.merge(&ds.select(&["k"]), &["k"], "inner").height(), 1 + 4 + 4);
    }

    #[test]
    fn round_is_half_to_even_like_numpy() {
        let ds: Dataset = Dataset::new(df!("v" => [0.5, 1.5, 2.5, -0.5, 0.125]).unwrap(), "");
        assert_eq!(ds.round(0).to_list("v"), [0.0, 2.0, 2.0, -0.0, 0.0]);
        assert_eq!(ds.round(2).to_list("v")[4], 0.12);
    }

    #[test]
    fn bad_input_says_what_is_wrong() {
        let ds: Dataset = messy();
        let error = |e: DatasetError| e.to_string();
        assert!(error(ds.try_groupby(&["k"], &[("x", "mode")]).unwrap_err()).contains("aggfunc = 'mode'"));
        assert!(error(ds.try_groupby(&[], &[("x", "sum")]).unwrap_err()).contains("by"));
        assert!(error(ds.try_merge(&ds, &["x"], "cross").unwrap_err()).contains("how = 'cross'"));
        assert!(error(ds.try_merge(&ds, &["q"], "inner").unwrap_err()).contains("'q' not found"));
        assert!(error(Dataset::try_concat(&[]).unwrap_err()).contains("one dataset or more"));
        assert!(error(ds.try_quantile(2.0).unwrap_err()).contains("q = '2'"));
        assert!(error(ds.try_select_dtypes("category").unwrap_err()).contains("include"));
    }
}
