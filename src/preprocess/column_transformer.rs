//! [`ColumnTransformer`]: all preprocessing, built with `.add` only.

use std::collections::HashSet;

use indexmap::IndexMap;
use polars::prelude::Expr;

use crate::utils::col;
use crate::{
    dataset::Dataset,
    error::OrFatal,
    preprocess::{Exprs, PreprocessError, Transform, apply, identity, one_of},
};

/// Runs a list of steps on chosen columns, joins the outputs, and can run more steps on them.
///
/// - `.add(name, columns, steps![..])` runs `steps` in order on `columns`. Outputs are named
///   `{name}__{column}`, like scikit-learn's `get_feature_names_out`.
///   `verbose_feature_names_out(false)` keeps the plain names, which must then be unique.
/// - An add can read input columns or the outputs of earlier adds. `&[]` means every input
///   column in the first add, and the joined outputs of every earlier add after that.
/// - An add uses up what it reads. The result is every output that no later add reads, plus the
///   unread input columns when `remainder("passthrough")` is set (the default is `"drop"`).
/// - `fit` reads the cached stats wherever it can. `transform` is one lazy `select`, so adds that
///   don't depend on each other run in parallel inside Polars.
///
/// ```
/// use rusmlai::{prelude::*, steps};
/// let ds = read_csv("tests/fixtures/data/students.csv");
/// let mut prep = ColumnTransformer::new()
///     .add("num", &["age", "cgpa"], steps![SimpleImputer::new().strategy("median"), StandardScaler::new()])
///     .add("cat", &["batch"], steps![OneHotEncoder::new()]);
/// let out = prep.fit_transform(&ds);
/// assert_eq!(
///     out.columns(),
///     ["num__age", "num__cgpa", "cat__batch_B13", "cat__batch_B14", "cat__batch_B15"],
/// );
/// ```
///
/// Python:
///
/// ```python
/// import pandas as pd
/// ds = pd.read_csv("tests/fixtures/data/students.csv")
/// from sklearn.compose import ColumnTransformer
/// from sklearn.impute import SimpleImputer
/// from sklearn.pipeline import make_pipeline
/// from sklearn.preprocessing import OneHotEncoder, StandardScaler
/// prep = ColumnTransformer([
///     ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), ["age", "cgpa"]),
///     ("cat", OneHotEncoder(), ["batch"]),
/// ])
/// prep.fit(ds)
/// assert prep.get_feature_names_out().tolist() == [
///     "num__age", "num__cgpa", "cat__batch_B13", "cat__batch_B14", "cat__batch_B15",
/// ]
/// ```
// ponytail: not Clone, because the steps are `Box<dyn Transform>`, so a `Pipeline` can't be copied.
// Before `cross_val_score`, add a `clone_box` supertrait with a blanket impl for every
// `Transform + Clone` (the dyn-clone trick, no crate), then derive Clone here (decided 2026-11-11).
#[derive(Debug, Default)]
pub struct ColumnTransformer {
    adds: Vec<Add>,
    passthrough: bool,
    plain_names: bool,
    input_columns: Vec<String>,
}

#[derive(Debug)]
struct Add {
    name: String,
    columns: Vec<String>,
    steps: Vec<Box<dyn Transform>>,
}

impl ColumnTransformer {
    /// An empty column transformer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Runs `steps` in order on `columns`; see [`ColumnTransformer`] for the rules.
    pub fn add(mut self, name: &str, columns: &[&str], steps: Vec<Box<dyn Transform>>) -> Self {
        self.adds.push(Add {
            name: name.into(),
            columns: columns.iter().map(|c| c.to_string()).collect(),
            steps,
        });
        self
    }

    /// `"drop"` (the default) or `"passthrough"`: what happens to input columns no add reads.
    #[track_caller]
    pub fn remainder(mut self, remainder: &str) -> Self {
        self.passthrough =
            one_of("remainder", remainder, &["drop", "passthrough"]).or_fatal() == "passthrough";
        self
    }

    /// `true` (the default) names outputs `{add}__{column}`; `false` keeps `{column}`, and a
    /// name that comes out twice fails `fit`. Later adds read outputs by the same names.
    pub fn verbose_feature_names_out(mut self, verbose: bool) -> Self {
        self.plain_names = !verbose;
        self
    }

    /// The output column names, in order. Empty before `fit`.
    pub fn get_feature_names_out(&self) -> Vec<String> {
        let inputs: Exprs = self
            .input_columns
            .iter()
            .map(|c| (c.clone(), col(c.as_str())))
            .collect();
        self.plan(&inputs)
            .map(|outputs| outputs.into_keys().collect())
            .unwrap_or_default()
    }

    /// The expression of every output column, built from the `inputs` expressions.
    fn plan(&self, inputs: &Exprs) -> Result<Exprs, PreprocessError> {
        let mut produced: Exprs = IndexMap::new();
        // What later adds read: earlier outputs leave the result, inputs leave the remainder.
        let (mut read_outputs, mut read_inputs): (HashSet<String>, HashSet<String>) =
            (HashSet::new(), HashSet::new());
        for add in &self.adds {
            let x: Exprs = reads(add, inputs, &produced)?;
            for name in x.keys() {
                if produced.contains_key(name) {
                    read_outputs.insert(name.clone());
                } else {
                    read_inputs.insert(name.clone());
                }
            }
            for (name, expr) in run(add, x, self.plain_names)? {
                // With plain names, a step that rewrites an earlier output keeps its name.
                if read_outputs.remove(&name) {
                    produced.shift_remove(&name);
                }
                self.unique(&produced, &name)?;
                produced.insert(name, expr);
            }
        }
        let mut outputs: Exprs = produced
            .into_iter()
            .filter(|(name, _)| !read_outputs.contains(name))
            .collect();
        if self.passthrough {
            for (name, expr) in inputs {
                if !read_inputs.contains(name) {
                    self.unique(&outputs, name)?;
                    outputs.insert(name.clone(), expr.clone());
                }
            }
        }
        Ok(outputs)
    }

    /// With plain names, two outputs can get one name, which scikit-learn refuses too.
    fn unique(&self, outputs: &Exprs, name: &str) -> Result<(), PreprocessError> {
        if self.plain_names && outputs.contains_key(name) {
            return Err(PreprocessError::InvalidOption {
                option: "verbose_feature_names_out",
                value: "false".into(),
                expected: format!(
                    "true, or adds whose outputs have unique names: '{name}' comes out twice"
                ),
            });
        }
        Ok(())
    }
}

impl Transform for ColumnTransformer {
    fn try_fit(&mut self, dataset: &Dataset) -> Result<(), PreprocessError> {
        let inputs: Exprs = identity(dataset);
        let mut produced: Exprs = IndexMap::new();
        for add in &mut self.adds {
            let x: Exprs = reads(add, &inputs, &produced)?;
            let mut data: Dataset = if x.keys().any(|name| produced.contains_key(name)) {
                // Earlier outputs: computed here, their stats on first use.
                apply(dataset, &x)?
            } else {
                // Input columns only: `select` shares the cached stats, so no pass over the data.
                dataset.try_select(&x.keys().map(String::as_str).collect::<Vec<&str>>())?
            };
            let last: usize = add.steps.len().saturating_sub(1);
            for (i, step) in add.steps.iter_mut().enumerate() {
                // Columns the step before changed have no stats yet: one pass for all of them,
                // not one pass per column as each `try_*_stats` call would make.
                data.try_fill_stats()?;
                step.try_fit(&data)?;
                if i < last {
                    // The next step learns from this step's output.
                    data = step.try_transform(&data)?;
                }
            }
            produced.extend(run(add, x, self.plain_names)?);
        }
        self.input_columns = dataset.columns();
        self.plan(&inputs)?; // names that clash fail here, not at transform
        Ok(())
    }

    fn exprs(&self, inputs: &Exprs) -> Result<Exprs, PreprocessError> {
        if self.input_columns.is_empty() {
            return self.plan(inputs); // not fitted: the steps say so
        }
        // The columns fit saw, like scikit-learn: `&[]` and the remainder never pick up a column
        // that only the new data has, for example the target in a test set. A column fit saw
        // may be missing only if nothing reads it.
        let all_inputs: bool =
            self.passthrough || self.adds.first().is_some_and(|a| a.columns.is_empty());
        let mut known: Exprs = IndexMap::new();
        for name in &self.input_columns {
            match inputs.get(name) {
                Some(expr) => {
                    known.insert(name.clone(), expr.clone());
                }
                None if all_inputs || self.adds.iter().any(|a| a.columns.contains(name)) => {
                    return Err(PreprocessError::MissingColumn {
                        column: name.clone(),
                    });
                }
                None => {}
            }
        }
        self.plan(&known)
    }
}

/// The columns `add` reads, each as an input column or an earlier output.
fn reads(add: &Add, inputs: &Exprs, produced: &Exprs) -> Result<Exprs, PreprocessError> {
    if add.columns.is_empty() {
        return Ok(if produced.is_empty() {
            inputs
        } else {
            produced
        }
        .clone());
    }
    add.columns
        .iter()
        .map(|name| {
            let expr: &Expr = produced
                .get(name)
                .or_else(|| inputs.get(name))
                .ok_or_else(|| PreprocessError::UnknownColumn {
                    add: add.name.clone(),
                    column: name.clone(),
                    available: inputs
                        .keys()
                        .chain(produced.keys())
                        .cloned()
                        .collect::<Vec<String>>()
                        .join(", "),
                })?;
            Ok((name.clone(), expr.clone()))
        })
        .collect()
}

/// Chains the steps of `add` on `x` and prefixes the outputs with the add's name, unless
/// `plain_names`.
fn run(add: &Add, x: Exprs, plain_names: bool) -> Result<Exprs, PreprocessError> {
    let mut out: Exprs = x;
    for step in &add.steps {
        out = step.exprs(&out)?;
    }
    if plain_names {
        return Ok(out);
    }
    Ok(out
        .into_iter()
        .map(|(name, expr)| (format!("{}__{name}", add.name), expr))
        .collect())
}

#[cfg(test)]
mod tests {
    use polars::prelude::{Column, DataFrame};

    use super::*;
    use crate::{
        dataset::read_csv,
        preprocess::{MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler},
        steps,
    };

    fn values(frame: &DataFrame, name: &str) -> Vec<Option<f64>> {
        let column: &Column = frame.column(name).unwrap();
        column.f64().unwrap().iter().collect()
    }

    #[test]
    fn branches_impute_scale_and_encode_from_the_cached_stats() {
        let ds: Dataset = read_csv("tests/fixtures/data/students.csv");
        let mut prep: ColumnTransformer = ColumnTransformer::new()
            .add(
                "num",
                &["cgpa"],
                steps![
                    SimpleImputer::new().strategy("median"),
                    StandardScaler::new()
                ],
            )
            .add("cat", &["batch"], steps![OneHotEncoder::new()]);
        let out: Dataset = prep.fit_transform(&ds);

        let cgpa: Vec<f64> = values(&out.dataframe, "num__cgpa")
            .into_iter()
            .map(Option::unwrap)
            .collect();
        let n: f64 = cgpa.len() as f64;
        let mean: f64 = cgpa.iter().sum::<f64>() / n;
        let std: f64 = (cgpa.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
        assert!(
            mean.abs() < 1e-12 && (std - 1.0).abs() < 1e-12,
            "mean {mean}, std {std}"
        );

        assert_eq!(
            values(&out.dataframe, "cat__batch_B13")[..3],
            [Some(1.0), Some(0.0), Some(0.0)],
        );
        assert_eq!(prep.get_feature_names_out(), out.columns());
    }

    #[test]
    fn a_later_add_reads_and_uses_up_an_earlier_output() {
        let ds: Dataset = read_csv("tests/fixtures/data/students.csv");
        let mut prep: ColumnTransformer = ColumnTransformer::new()
            .add("num", &["cgpa"], steps![SimpleImputer::new()])
            .add("cat", &["batch"], steps![OneHotEncoder::new()])
            .add("scale", &["num__cgpa"], steps![StandardScaler::new()])
            .remainder("passthrough");
        let out: Dataset = prep.fit_transform(&ds);
        assert_eq!(
            out.columns(),
            [
                "cat__batch_B13",
                "cat__batch_B14",
                "cat__batch_B15",
                "scale__num__cgpa",
                "id",
                "name",
                "age",
                "passed"
            ],
        );
        // The passthrough columns share the parent's stats.
        assert!(std::sync::Arc::ptr_eq(
            &ds.metadata.dataset_cache.columns["age"],
            &out.metadata.dataset_cache.columns["age"],
        ));
    }

    #[test]
    fn transform_uses_only_what_fit_learned() {
        let ds: Dataset = read_csv("tests/fixtures/data/students.csv");
        let mut prep: ColumnTransformer =
            ColumnTransformer::new().add("num", &["age"], steps![StandardScaler::new()]);
        prep.fit(&ds);
        let (mean, scale): (f64, f64) = {
            let stats = ds.try_numeric_stats("age").unwrap();
            let n: f64 =
                (ds.metadata.height - ds.try_column_info("age").unwrap().null_count) as f64;
            (stats.mean, (stats.variance * (n - 1.0) / n).sqrt())
        };
        let test: Dataset = ds.try_select(&["age"]).unwrap();
        let out: Dataset = prep.transform(&test);
        let first: f64 = values(&out.dataframe, "num__age")[0].unwrap();
        assert!((first - (19.0 - mean) / scale).abs() < 1e-12);
    }

    #[test]
    fn transform_reads_only_the_columns_fit_saw() {
        let ds: Dataset = read_csv("tests/fixtures/data/students.csv");
        let mut prep: ColumnTransformer = ColumnTransformer::new()
            .add("num", &["age"], steps![StandardScaler::new()])
            .remainder("passthrough");
        prep.fit(&ds.drop(&["passed"]));
        // `passed` is only in the new data, so the remainder leaves it out.
        let out: Dataset = prep.transform(&ds);
        assert!(!out.columns().contains(&"passed".to_string()));
        assert_eq!(out.columns(), prep.get_feature_names_out());

        let err: String = prep
            .try_transform(&ds.drop(&["name"])) // the remainder passes `name` through
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(err.starts_with("transform needs column 'name'"), "{err}");
    }

    #[test]
    fn plain_names_drop_the_prefix_and_must_be_unique() {
        let ds: Dataset = read_csv("tests/fixtures/data/students.csv");
        let mut prep: ColumnTransformer = ColumnTransformer::new()
            .add("num", &["age"], steps![StandardScaler::new()])
            .add("cat", &["batch"], steps![OneHotEncoder::new()])
            .verbose_feature_names_out(false);
        assert_eq!(
            prep.fit_transform(&ds).columns(),
            ["age", "batch_B13", "batch_B14", "batch_B15"]
        );
        // A rewrite of an earlier output keeps the name and replaces it.
        let mut chained: ColumnTransformer = ColumnTransformer::new()
            .add("a", &["age"], steps![StandardScaler::new()])
            .add("b", &["age"], steps![MinMaxScaler::new()])
            .verbose_feature_names_out(false);
        assert_eq!(chained.fit_transform(&ds).columns(), ["age"]);
        // An encoder output and a passed-through input with one name clash.
        let two: Dataset = Dataset::new(
            polars::df!("c" => ["x", "y"], "c_x" => [1.0, 2.0]).unwrap(),
            "",
        );
        let mut twice: ColumnTransformer = ColumnTransformer::new()
            .add("cat", &["c"], steps![OneHotEncoder::new()])
            .remainder("passthrough")
            .verbose_feature_names_out(false);
        let err: String = twice.try_fit(&two).unwrap_err().to_string();
        assert!(err.contains("'c_x' comes out twice"), "{err}");
    }

    #[test]
    fn errors_name_the_problem() {
        let ds: Dataset = read_csv("tests/fixtures/data/students.csv");
        let mut prep: ColumnTransformer =
            ColumnTransformer::new().add("num", &["grade"], steps![StandardScaler::new()]);
        let err: String = prep.try_fit(&ds).unwrap_err().to_string();
        assert!(err.starts_with("add 'num' reads column 'grade'"), "{err}");

        let mut prep: ColumnTransformer =
            ColumnTransformer::new().add("num", &["batch"], steps![StandardScaler::new()]);
        assert!(prep.try_fit(&ds).is_err());

        let prep: ColumnTransformer =
            ColumnTransformer::new().add("num", &["age"], steps![StandardScaler::new()]);
        let err: String = prep.try_transform(&ds).map(|_| ()).unwrap_err().to_string();
        assert!(
            err.starts_with("StandardScaler was not fitted on column 'age'"),
            "{err}"
        );
    }
}
