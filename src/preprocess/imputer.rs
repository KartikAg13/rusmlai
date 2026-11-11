//! [`SimpleImputer`].

use indexmap::IndexMap;
use polars::prelude::{
    Column, DataFrame, DataType, Expr, IntoLazy, Literal, Scalar, SortMultipleOptions, len, lit,
};

use crate::utils::col;
use crate::{
    dataset::{Dataset, error::DatasetError},
    error::OrFatal,
    preprocess::{Exprs, PreprocessError, Transform, apply, identity, one_of},
};

/// Fills nulls with a value learned from the column, like scikit-learn's `SimpleImputer`.
///
/// `strategy` is one of:
/// - `"mean"` (the default) or `"median"`, for numeric columns;
/// - `"most_frequent"`, for any column. A tie goes to the smallest value, as in scikit-learn;
/// - `"constant"`, which fills with `fill_value`: by default 0 for numeric columns and
///   `"missing_value"` for text columns.
///
/// `fit` reads the cached stats (`"most_frequent"` on a numeric column makes one pass over it) and
/// learns a fill for every column, so `transform` also fills a column that had no nulls in the
/// data `fit` saw. A column with no nulls in the data it transforms passes through unchanged, so
/// a step after the imputer still reads its cached stats.
///
/// ```
/// use polars::df;
/// use rusmlai::prelude::*;
/// let ds = Dataset::new(df!("x" => [Some(1.0), None, Some(5.0), Some(9.0)]).unwrap(), "");
/// let out = SimpleImputer::new().strategy("median").fit_transform(&ds);
/// let x: Vec<f64> = out.dataframe.column("x").unwrap().f64().unwrap().into_no_null_iter().collect();
/// assert_eq!(x, [1.0, 5.0, 5.0, 9.0]);
///
/// let ds = Dataset::new(df!("n" => [Some(3), Some(1), Some(3), Some(1), None]).unwrap(), "");
/// let out = SimpleImputer::new().strategy("most_frequent").fit_transform(&ds);
/// assert_eq!(out.dataframe.column("n").unwrap().cast(&polars::prelude::DataType::Float64).unwrap().f64().unwrap().get(4), Some(1.0));
///
/// let ds = Dataset::new(df!("c" => [Some("a"), None]).unwrap(), "");
/// let out = SimpleImputer::new().strategy("constant").fit_transform(&ds);
/// assert_eq!(out.dataframe.column("c").unwrap().str().unwrap().get(1), Some("missing_value"));
/// ```
#[derive(Debug, Clone)]
pub struct SimpleImputer {
    strategy: String,
    fill_value: Option<Expr>,
    /// Column to its fill value, learned by `fit` (scikit-learn's `statistics_`).
    pub statistics_: IndexMap<String, Expr>,
}

impl Default for SimpleImputer {
    fn default() -> Self {
        Self {
            strategy: "mean".into(),
            fill_value: None,
            statistics_: IndexMap::new(),
        }
    }
}

impl SimpleImputer {
    /// An imputer with `strategy = "mean"`.
    pub fn new() -> Self {
        Self::default()
    }

    /// `"mean"`, `"median"`, `"most_frequent"` or `"constant"`.
    #[track_caller]
    pub fn strategy(mut self, strategy: &str) -> Self {
        let allowed: [&str; 4] = ["mean", "median", "most_frequent", "constant"];
        self.strategy = one_of("strategy", strategy, &allowed).or_fatal();
        self
    }

    /// The value that `strategy("constant")` fills with, for example `0`, `-1.0` or `"none"`.
    pub fn fill_value<L: Literal>(mut self, value: L) -> Self {
        self.fill_value = Some(lit(value));
        self
    }
}

impl Transform for SimpleImputer {
    fn try_fit(&mut self, dataset: &Dataset) -> Result<(), PreprocessError> {
        self.statistics_.clear();
        for column in dataset.columns() {
            let dtype: &DataType = &dataset.try_column_info(&column)?.dtype;
            let fill: Expr = match self.strategy.as_str() {
                "mean" => lit(dataset.try_numeric_stats(&column)?.mean),
                "median" => lit(dataset.try_numeric_stats(&column)?.median),
                "most_frequent" => match dataset.try_categorical_stats(&column) {
                    // The cache names categories as text, which is the value itself only in a
                    // text column: a true/false, date or numeric column needs its own dtype.
                    Ok(stats) if dtype.is_string() && stats.unique_count > 0 => {
                        lit(stats.mode.clone())
                    }
                    Ok(_) | Err(DatasetError::ColumnNotCategorical { .. }) => {
                        mode(dataset, &column)?
                    }
                    Err(error) => return Err(error.into()),
                },
                "constant" => match &self.fill_value {
                    Some(value) => value.clone(),
                    None if dtype.is_primitive_numeric() => lit(0),
                    None => lit("missing_value"),
                },
                other => unreachable!("strategy() only takes known values, got {other}"),
            };
            self.statistics_.insert(column, fill);
        }
        Ok(())
    }

    fn exprs(&self, inputs: &Exprs) -> Result<Exprs, PreprocessError> {
        inputs
            .iter()
            .map(|(name, expr)| match self.statistics_.get(name) {
                None => Err(PreprocessError::NotFitted {
                    transformer: "SimpleImputer",
                    column: name.clone(),
                }),
                Some(fill) => Ok((name.clone(), expr.clone().fill_null(fill.clone()))),
            })
            .collect()
    }

    /// A column with no nulls in `dataset` is left as it is, so it keeps its cached stats.
    fn try_transform(&self, dataset: &Dataset) -> Result<Dataset, PreprocessError> {
        let mut outputs: Exprs = self.exprs(&identity(dataset))?;
        for (name, expr) in &mut outputs {
            if dataset.try_column_info(name)?.null_count == 0 {
                *expr = col(name.as_str());
            }
        }
        apply(dataset, &outputs)
    }
}

/// The most common value of a column, in the column's own dtype. A tie goes to the smallest
/// value, and a column with no values gives null (nothing to fill with).
// ponytail: one group_by pass, since the cache keeps counts only as text.
fn mode(dataset: &Dataset, column: &str) -> Result<Expr, PreprocessError> {
    let counts: DataFrame = dataset
        .dataframe
        .clone()
        .lazy()
        .select([col(column).drop_nulls()])
        .group_by([col(column)])
        .agg([len().alias("count")])
        .sort_by_exprs(
            [col("count"), col(column)],
            SortMultipleOptions::default().with_order_descending_multi([true, false]),
        )
        .limit(1)
        .collect()?;
    let top: &Column = counts.column(column)?;
    let dtype: DataType = top.dtype().clone();
    Ok(lit(match top.get(0) {
        Ok(value) => Scalar::new(dtype, value.into_static()),
        Err(_) => Scalar::null(dtype),
    }))
}

#[cfg(test)]
mod tests {
    use polars::df;

    use super::*;

    fn floats(dataset: &Dataset, name: &str) -> Vec<Option<f64>> {
        let column: &Column = dataset.dataframe.column(name).unwrap();
        column.f64().unwrap().iter().collect()
    }

    #[test]
    fn transform_fills_a_column_that_had_no_nulls_in_fit() {
        // scikit-learn learns `statistics_` for every column; a test set can have the first null.
        let train: Dataset = Dataset::new(df!("x" => [1.0, 2.0, 6.0]).unwrap(), "");
        let test: Dataset = Dataset::new(df!("x" => [Some(4.0), None]).unwrap(), "");
        let mut imputer: SimpleImputer = SimpleImputer::new();
        imputer.fit(&train);
        assert_eq!(
            floats(&imputer.transform(&test), "x"),
            [Some(4.0), Some(3.0)]
        );
        // Inside a ColumnTransformer the fill is part of the plan, so it must be there too.
        let mut prep = crate::preprocess::ColumnTransformer::new().add(
            "num",
            &["x"],
            crate::steps![SimpleImputer::new().strategy("median")],
        );
        prep.fit(&train);
        assert_eq!(
            floats(&prep.transform(&test), "num__x"),
            [Some(4.0), Some(2.0)]
        );
    }

    #[test]
    fn a_column_without_nulls_keeps_its_cached_stats() {
        let ds: Dataset = Dataset::new(
            df!("full" => [1.0, 2.0], "holes" => [Some(1.0), None]).unwrap(),
            "",
        );
        let out: Dataset = SimpleImputer::new().fit_transform(&ds);
        let (before, after) = (
            &ds.metadata.dataset_cache.columns,
            &out.metadata.dataset_cache.columns,
        );
        assert!(std::sync::Arc::ptr_eq(&before["full"], &after["full"]));
        assert!(!std::sync::Arc::ptr_eq(&before["holes"], &after["holes"]));
        assert_eq!(floats(&out, "holes"), [Some(1.0), Some(1.0)]);
    }

    #[test]
    fn most_frequent_keeps_the_dtype_and_breaks_ties_by_the_smallest_value() {
        let ds: Dataset = Dataset::new(
            df!(
                "flag" => [Some(true), Some(true), Some(false), None],
                "city" => [Some("b"), Some("a"), Some("b"), None],
                "code" => [Some(7), Some(3), Some(7), None],
                "tie" => [Some(2.5), Some(1.5), None, None]
            )
            .unwrap(),
            "",
        );
        let out: Dataset = SimpleImputer::new()
            .strategy("most_frequent")
            .fit_transform(&ds);
        assert_eq!(out.dtypes(), ds.dtypes());
        let frame: &DataFrame = &out.dataframe;
        assert!(frame.columns().iter().all(|c| c.null_count() == 0));
        assert_eq!(
            frame.column("flag").unwrap().bool().unwrap().get(3),
            Some(true)
        );
        assert_eq!(
            frame.column("city").unwrap().str().unwrap().get(3),
            Some("b")
        );
        assert_eq!(frame.column("code").unwrap().i32().unwrap().get(3), Some(7));
        assert_eq!(floats(&out, "tie")[2..], [Some(1.5), Some(1.5)]);
    }

    #[test]
    fn constant_fills_by_dtype_or_with_the_given_value() {
        let ds: Dataset = Dataset::new(
            df!("n" => [Some(4), None], "s" => [Some("a"), None]).unwrap(),
            "",
        );
        let out: Dataset = SimpleImputer::new().strategy("constant").fit_transform(&ds);
        assert_eq!(
            out.dataframe.column("n").unwrap().i32().unwrap().get(1),
            Some(0)
        );
        assert_eq!(
            out.dataframe.column("s").unwrap().str().unwrap().get(1),
            Some("missing_value")
        );
        let numbers: Dataset = ds.select(&["n"]);
        let out: Dataset = SimpleImputer::new()
            .strategy("constant")
            .fill_value(-1)
            .fit_transform(&numbers);
        assert_eq!(
            out.dataframe.column("n").unwrap().i32().unwrap().get(1),
            Some(-1)
        );
    }

    #[test]
    fn errors_name_the_problem() {
        let ds: Dataset = Dataset::new(df!("s" => [Some("a"), None]).unwrap(), "");
        let err: String = SimpleImputer::new().try_fit(&ds).unwrap_err().to_string();
        assert_eq!(err, "column 's' is not numeric (found str)");
        let unfitted: String = SimpleImputer::new()
            .try_transform(&ds)
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(
            unfitted.starts_with("SimpleImputer was not fitted on column 's'"),
            "{unfitted}"
        );
    }
}
