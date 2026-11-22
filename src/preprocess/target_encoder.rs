//! [`TargetEncoder`]: each category becomes the mean of the target over its rows, shrunk
//! toward the overall mean, like scikit-learn's `TargetEncoder`.

use indexmap::IndexMap;
use polars::prelude::{Column, DataFrame, DataType, IdxSize, NamedFrom, PlHashMap, Series};
use polars_core::runtime::THREAD_POOL;
use rayon::prelude::*;

use polars::prelude::{DataType as Dtype, Expr, IntoColumn, lit, when};

use crate::{
    dataset::Dataset,
    error::OrFatal,
    preprocess::{Exprs, PreprocessError, Transform, encoder::lookup},
    models::{
        ModelError, check,
        folds::{KFold, StratifiedKFold},
        label_target,
    },
};
use crate::models::{at_least_zero};

/// Encodes categories by the target, scikit-learn's `TargetEncoder`: each category of a
/// column becomes the mean of the target over its rows, pulled toward the mean of all rows
/// when the category has few rows or a noisy target. A binary target is encoded as the share
/// of its second class (one column per feature), a target with more classes as one column per
/// class (`{column}_{class}`), a continuous target as its mean.
///
/// `smooth` sets the pull. By default (scikit-learn's `"auto"`) it is the empirical Bayes
/// weight λ = n·σ² / (n·σ² + σ²ᶜ) of a category with n rows, target variance σ²ᶜ inside
/// it and σ² over all rows: encoding = λ·(mean of the category) + (1 − λ)·(mean of all rows). A
/// number m gives (Σy + m·mean) / (n + m).
///
/// [`fit_transform`](Self::fit_transform) is not `fit` then `transform`: to keep a row's own
/// target out of its encoding, it cuts the rows into `cv` folds (shuffled, stratified for
/// classes) and encodes each fold with what the other folds learned ("cross fitting"). The
/// folds are scikit-learn's, so with the same `random_state` the output is scikit-learn's.
/// [`transform`](Self::transform) of new rows uses what `fit` learned on every row. Missing
/// values are a category of their own; a category `fit` never saw gets the mean of all rows.
///
/// It encodes the text and boolean columns, or the ones given to [`columns`](Self::columns);
/// the other columns, the target among them, pass through unchanged.
///
/// In a [`ColumnTransformer`](crate::preprocess::ColumnTransformer) inside a
/// [`Pipeline`](crate::pipeline::Pipeline), it is a step like the others: the pipeline passes
/// it the target, its training rows get the cross-fitted encodings and new rows `transform`'s,
/// as in scikit-learn. It must be the last step of its add, because the steps after it would
/// learn from the training rows' encodings.
///
/// Not yet: a `categories` list of your own; `cv` as a splitter.
///
/// ```
/// use rusmlai::{prelude::*, steps};
/// let ds = Dataset::new(
///     df!(
///         "city" => ["a", "a", "a", "b", "b", "c", "c", "c", "c", "a"],
///         "y" => [1.0, 2.0, 3.0, 10.0, 12.0, 5.0, 6.0, 7.0, 6.0, 2.0],
///     )
///     .unwrap(),
///     "",
/// );
/// // Whole numbers count as classes, so say the target is continuous.
/// let mut encoder = TargetEncoder::new().target_type("continuous").smooth(0.0);
/// encoder.fit(&ds, "y");
/// assert_eq!(encoder.encodings_["city"][0], [2.0, 11.0, 6.0]); // a, b and c
/// let new = Dataset::new(df!("city" => ["b", "z"]).unwrap(), "");
/// assert_eq!(encoder.transform(&new).to_list("city"), [11.0, 5.4]); // z: the mean of all rows
///
/// let crossed = TargetEncoder::new().target_type("continuous").cv(2).fit_transform(&ds, "y");
/// assert_eq!(crossed.columns(), ["city", "y"]);
///
/// // In a pipeline: the training rows get cross-fitted encodings, new rows fit's.
/// let prep = ColumnTransformer::new()
///     .add("te", &["city"], steps![TargetEncoder::new().target_type("continuous").cv(2)]);
/// let mut pipe = Pipeline::new(prep, LinearRegression::new());
/// pipe.fit(&ds, "y");
/// assert_eq!(pipe.predict(&new).len(), 2);
/// ```
///
/// Python:
///
/// ```python
/// import numpy as np
/// import pandas as pd
/// from sklearn.compose import ColumnTransformer
/// from sklearn.linear_model import LinearRegression
/// from sklearn.pipeline import Pipeline
/// from sklearn.preprocessing import TargetEncoder
/// ds = pd.DataFrame({
///     "city": ["a", "a", "a", "b", "b", "c", "c", "c", "c", "a"],
///     "y": [1.0, 2.0, 3.0, 10.0, 12.0, 5.0, 6.0, 7.0, 6.0, 2.0],
/// })
/// encoder = TargetEncoder(smooth=0.0, target_type="continuous").fit(ds[["city"]], ds["y"])
/// assert encoder.encodings_[0].tolist() == [2.0, 11.0, 6.0]
/// new = pd.DataFrame({"city": ["b", "z"]})
/// assert np.allclose(encoder.transform(new)[:, 0], [11.0, 5.4])
/// crossed = TargetEncoder(target_type="continuous", cv=2, random_state=42).fit_transform(ds[["city"]], ds["y"])
/// assert crossed.shape == (10, 1)
///
/// prep = ColumnTransformer([("te", TargetEncoder(target_type="continuous", cv=2, random_state=42), ["city"])])
/// pipe = Pipeline([("prep", prep), ("model", LinearRegression())]).fit(ds[["city"]], ds["y"])
/// assert len(pipe.predict(new)) == 2
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TargetEncoder {
    target_type: String,
    smooth: Option<f64>,
    cv: usize,
    shuffle: bool,
    random_state: u64,
    columns: Vec<String>,
    /// The encoded columns, learned by `fit`.
    pub feature_names_in_: Vec<String>,
    /// Each encoded column's categories, sorted as text; a missing value is last, as `None`.
    pub categories_: IndexMap<String, Vec<Option<String>>>,
    /// Each column's encoding per class (one list for a binary or continuous target), in the
    /// order of `categories_`, learned on every row.
    pub encodings_: IndexMap<String, Vec<Vec<f64>>>,
    /// `"binary"`, `"multiclass"` or `"continuous"`.
    pub target_type_: String,
    /// The target's classes, sorted; empty for a continuous target.
    pub classes_: Vec<String>,
    /// The mean of the (encoded) target over every row, per class.
    pub target_mean_: Vec<f64>,
}

impl Default for TargetEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// A target as one f64 column per encoded output: the value, or each class's 0 / 1.
struct Target {
    kind: &'static str,
    classes: Vec<String>,
    columns: Vec<Vec<f64>>,
    /// The class code of each row, for the stratified folds.
    labels: Option<Series>,
}

/// The categories of a column and each row's place in them.
struct Codes {
    categories: Vec<Option<String>>,
    codes: Vec<u32>,
}

impl TargetEncoder {
    /// `smooth` "auto", 5 shuffled folds, `random_state` 42.
    pub fn new() -> Self {
        Self {
            target_type: "auto".into(),
            smooth: None,
            cv: 5,
            shuffle: true,
            random_state: 42,
            columns: Vec::new(),
            feature_names_in_: Vec::new(),
            categories_: IndexMap::new(),
            encodings_: IndexMap::new(),
            target_type_: String::new(),
            classes_: Vec::new(),
            target_mean_: Vec::new(),
        }
    }

    /// What the target is: `"auto"` (the default: scikit-learn's `type_of_target`, where whole
    /// numbers, even as floats, are classes), `"continuous"`, `"binary"` or `"multiclass"`.
    #[track_caller]
    pub fn target_type(mut self, target_type: &str) -> Self {
        let allowed = ["auto", "continuous", "binary", "multiclass"];
        self.target_type = crate::preprocess::one_of("target_type", target_type, &allowed).or_fatal();
        self
    }

    /// The pull toward the overall mean, a number of 0 or more (0: the plain category mean).
    /// Default: scikit-learn's `"auto"`, the empirical Bayes weight.
    #[track_caller]
    pub fn smooth(mut self, smooth: f64) -> Self {
        at_least_zero("smooth", smooth);
        self.smooth = Some(smooth);
        self
    }

    /// The folds of `fit_transform`, 2 or more. Default 5.
    #[track_caller]
    pub fn cv(mut self, cv: usize) -> Self {
        check(cv >= 2, "cv", cv, "2 folds or more");
        self.cv = cv;
        self
    }

    /// Shuffle the rows before the folds of `fit_transform`. Default `true`, as in
    /// scikit-learn.
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// The seed of the shuffle. Default 42.
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = random_state;
        self
    }

    /// The columns to encode, of any type. Default: every text and boolean column.
    pub fn columns(mut self, columns: &[&str]) -> Self {
        self.columns = columns.iter().map(|c| c.to_string()).collect();
        self
    }

    /// Learns every category's encoding from all the rows.
    #[track_caller]
    pub fn fit(&mut self, dataset: &Dataset, target: &str) {
        self.try_fit(dataset, target).or_fatal()
    }

    /// [`Self::fit`] that returns the error: a missing column, or a target with nulls or one
    /// class.
    pub fn try_fit(&mut self, dataset: &Dataset, target: &str) -> Result<(), ModelError> {
        let (y, codes) = self.prepare(dataset, target)?;
        let all: Vec<u32> = (0..dataset.height() as u32).collect();
        self.target_mean_ = y.columns.iter().map(|c| mean(c, &all)).collect();
        self.encodings_ = (self.feature_names_in_.iter().zip(&codes))
            .map(|(name, c)| {
                let per_class: Vec<Vec<f64>> = (y.columns.iter().zip(&self.target_mean_))
                    .map(|(column, m)| self.encode(&c.codes, c.categories.len(), column, &all, *m))
                    .collect();
                (name.clone(), per_class)
            })
            .collect();
        self.categories_ = (self.feature_names_in_.iter().zip(codes))
            .map(|(name, c)| (name.clone(), c.categories))
            .collect();
        (self.target_type_, self.classes_) = (y.kind.into(), y.classes);
        Ok(())
    }

    /// The encoded columns of new rows, from what `fit` learned; the other columns pass
    /// through.
    #[track_caller]
    pub fn transform(&self, dataset: &Dataset) -> Dataset {
        self.try_transform(dataset).or_fatal()
    }

    /// [`Self::transform`] that returns the error: not fitted, or a column `fit` saw is
    /// missing.
    pub fn try_transform(&self, dataset: &Dataset) -> Result<Dataset, ModelError> {
        if self.encodings_.is_empty() {
            return Err(ModelError::NotFitted { model: "TargetEncoder" });
        }
        let encoded: Vec<Vec<Vec<f64>>> = THREAD_POOL.install(|| {
            (self.feature_names_in_.par_iter())
                .map(|name| {
                    let place: PlHashMap<Option<&str>, usize> = (self.categories_[name].iter().enumerate())
                        .map(|(i, c)| (c.as_deref(), i))
                        .collect();
                    let text = dataset.column(name)?.cast(&DataType::String)?;
                    let text = text.str()?;
                    Ok((self.encodings_[name].iter().zip(&self.target_mean_))
                        .map(|(encoding, m)| {
                            text.iter().map(|v| place.get(&v).map_or(*m, |i| encoding[*i])).collect()
                        })
                        .collect())
                })
                .collect::<Result<_, ModelError>>()
        })?;
        self.output(dataset, encoded)
    }

    /// Cross-fitted encodings of the training rows: each fold encoded by the other folds,
    /// with scikit-learn's folds. `fit` runs too, so `transform` then encodes new rows.
    #[track_caller]
    pub fn fit_transform(&mut self, dataset: &Dataset, target: &str) -> Dataset {
        self.try_fit_transform(dataset, target).or_fatal()
    }

    /// [`Self::fit_transform`] that returns the error.
    pub fn try_fit_transform(&mut self, dataset: &Dataset, target: &str) -> Result<Dataset, ModelError> {
        self.try_fit(dataset, target)?;
        let (y, codes) = self.prepare(dataset, target)?;
        let n: usize = dataset.height();
        let tests: Vec<Vec<IdxSize>> = match &y.labels {
            Some(labels) => {
                let frame: DataFrame = DataFrame::new(n, vec![Column::from(labels.clone())])?;
                let splitter = StratifiedKFold::new().n_splits(self.cv).shuffle(self.shuffle).random_state(self.random_state);
                splitter.try_split(&Dataset::new(frame, ""), labels.name())?.into_iter().map(|f| f.1).collect()
            }
            None => KFold::new().n_splits(self.cv).shuffle(self.shuffle).random_state(self.random_state).test_rows(n)?,
        };
        let mut fold_of: Vec<usize> = vec![0; n];
        for (i, rows) in tests.iter().enumerate() {
            rows.iter().for_each(|r| fold_of[*r as usize] = i);
        }
        let trains: Vec<Vec<u32>> = (0..tests.len())
            .map(|i| (0..n as u32).filter(|r| fold_of[*r as usize] != i).collect())
            .collect();
        let encoded: Vec<Vec<Vec<f64>>> = THREAD_POOL.install(|| {
            codes
                .par_iter()
                .map(|c| {
                    y.columns
                        .iter()
                        .map(|column| {
                            let mut out: Vec<f64> = vec![0.0; n];
                            for (train, test) in trains.iter().zip(&tests) {
                                let m: f64 = mean(column, train);
                                let encoding: Vec<f64> = self.encode(&c.codes, c.categories.len(), column, train, m);
                                for row in test {
                                    out[*row as usize] = encoding[c.codes[*row as usize] as usize];
                                }
                            }
                            out
                        })
                        .collect()
                })
                .collect()
        });
        self.output(dataset, encoded)
    }

    /// The encoded target and every encoded column's codes; sets `feature_names_in_`.
    fn prepare(&mut self, dataset: &Dataset, target: &str) -> Result<(Target, Vec<Codes>), ModelError> {
        let names: Vec<String> = if self.columns.is_empty() {
            (dataset.columns().into_iter())
                .filter(|c| c != target)
                .filter(|c| dataset.column(c).is_ok_and(|s| matches!(s.dtype(), DataType::String | DataType::Boolean | DataType::Categorical(..) | DataType::Enum(..))))
                .collect()
        } else {
            self.columns.clone()
        };
        if names.is_empty() {
            return Err(ModelError::NoFeatures { target: target.into() });
        }
        let y: Target = target_columns(dataset, target, &self.target_type)?;
        let codes: Vec<Codes> = THREAD_POOL.install(|| {
            names.par_iter().map(|name| codes(dataset, name)).collect::<Result<_, ModelError>>()
        })?;
        self.feature_names_in_ = names;
        Ok((y, codes))
    }

    /// The encoding of each category from the rows `rows`, scikit-learn's
    /// `_fit_encoding_fast` and `_fit_encoding_fast_auto_smooth`.
    fn encode(&self, codes: &[u32], k: usize, y: &[f64], rows: &[u32], mean: f64) -> Vec<f64> {
        let (mut sums, mut counts): (Vec<f64>, Vec<f64>) = (vec![0.0; k], vec![0.0; k]);
        for row in rows {
            let (c, v) = (codes[*row as usize] as usize, y[*row as usize]);
            sums[c] += v;
            counts[c] += 1.0;
        }
        match self.smooth {
            Some(m) => (0..k).map(|c| (sums[c] + m * mean) / (counts[c] + m)).collect(),
            None => {
                let means: Vec<f64> = (0..k).map(|c| sums[c] / counts[c]).collect();
                let mut squares: Vec<f64> = vec![0.0; k];
                let mut variance: f64 = 0.0;
                for row in rows {
                    let (c, v) = (codes[*row as usize] as usize, y[*row as usize]);
                    squares[c] += (v - means[c]) * (v - means[c]);
                    variance += (v - mean) * (v - mean);
                }
                variance /= rows.len() as f64;
                (0..k)
                    .map(|c| {
                        let lambda: f64 = variance * counts[c] / (variance * counts[c] + squares[c] / counts[c]);
                        if lambda.is_nan() { mean } else { lambda * means[c] + (1.0 - lambda) * mean }
                    })
                    .collect()
            }
        }
    }

    /// `dataset` with each encoded column replaced by its output columns, in place.
    fn output(&self, dataset: &Dataset, encoded: Vec<Vec<Vec<f64>>>) -> Result<Dataset, ModelError> {
        let mut by_name: IndexMap<&str, Vec<Vec<f64>>> =
            self.feature_names_in_.iter().map(String::as_str).zip(encoded).collect();
        let mut columns: Vec<Column> = Vec::new();
        for column in dataset.dataframe().columns() {
            match by_name.swap_remove(column.name().as_str()) {
                Some(outputs) if outputs.len() == 1 => {
                    let values: Vec<f64> = outputs.into_iter().next().unwrap_or_default();
                    columns.push(Series::new(column.name().clone(), values).into());
                }
                Some(outputs) => {
                    for (class, values) in self.classes_.iter().zip(outputs) {
                        columns.push(Series::new(format!("{}_{class}", column.name()).into(), values).into());
                    }
                }
                None => columns.push(column.clone()),
            }
        }
        if let Some(name) = by_name.keys().next() {
            return Err(crate::preprocess::PreprocessError::MissingColumn { column: name.to_string() }.into());
        }
        let frame: DataFrame = DataFrame::new(dataset.height(), columns)?;
        Ok(Dataset::new(frame, ""))
    }
}

/// The name the target takes next to the features while a pipeline fits the encoder.
const TARGET: &str = "__rusmlai_target__";

/// A step of a `ColumnTransformer`: it fits only with the target, which a `Pipeline` passes.
impl Transform for TargetEncoder {
    fn try_fit(&mut self, _dataset: &Dataset) -> Result<(), PreprocessError> {
        Err(PreprocessError::NeedsTarget { step: "TargetEncoder" })
    }

    /// Fits on every row, and gives the cross-fitted encodings of the training rows.
    fn try_fit_with_target(&mut self, dataset: &Dataset, y: &Series) -> Result<Option<Dataset>, PreprocessError> {
        let mut frame: DataFrame = dataset.dataframe().clone();
        frame.with_column(y.clone().with_name(TARGET.into()).into_column())?;
        let encoded: Dataset = TargetEncoder::try_fit_transform(self, &Dataset::new(frame, ""), TARGET)
            .map_err(|e| PreprocessError::Step(e.to_string()))?;
        Ok(Some(encoded.try_drop(&[TARGET])?))
    }

    /// Each encoded column as a lookup of its categories (an unknown one: the mean of all
    /// rows; a missing value: its own encoding, if `fit` saw one); the others unchanged.
    fn exprs(&self, inputs: &Exprs) -> Result<Exprs, PreprocessError> {
        if self.encodings_.is_empty() {
            let column: String = inputs.keys().next().cloned().unwrap_or_default();
            return Err(PreprocessError::NotFitted { transformer: "TargetEncoder", column });
        }
        if let Some(column) = self.feature_names_in_.iter().find(|name| !inputs.contains_key(*name)) {
            return Err(PreprocessError::MissingColumn { column: column.clone() });
        }
        let mut out: Exprs = Exprs::new();
        for (name, expr) in inputs {
            let Some(per_class) = self.encodings_.get(name) else {
                out.insert(name.clone(), expr.clone());
                continue;
            };
            let categories: &[Option<String>] = &self.categories_[name];
            // The categories are sorted with a missing value last.
            let known: Vec<String> = categories.iter().flatten().cloned().collect();
            let text: Expr = expr.clone().cast(Dtype::String);
            for (i, (encoding, mean)) in per_class.iter().zip(&self.target_mean_).enumerate() {
                let missing: f64 = encoding.get(known.len()).copied().unwrap_or(*mean);
                let mapped: Expr = lookup(text.clone(), &known, encoding[..known.len()].to_vec(), Some(*mean));
                let named: String = match per_class.len() {
                    1 => name.clone(),
                    _ => format!("{name}_{}", self.classes_[i]),
                };
                out.insert(named, when(text.clone().is_null()).then(lit(missing)).otherwise(mapped));
            }
        }
        Ok(out)
    }
}

/// The mean of `y` over `rows`.
fn mean(y: &[f64], rows: &[u32]) -> f64 {
    rows.iter().map(|r| y[*r as usize]).sum::<f64>() / rows.len() as f64
}

/// The target as scikit-learn's `type_of_target` sees it: whole numbers, text or booleans with
/// two classes are binary, with more multiclass; a float with a fraction is continuous.
fn target_columns(dataset: &Dataset, target: &str, target_type: &str) -> Result<Target, ModelError> {
    let labels = if target_type == "continuous" {
        Err(ModelError::ContinuousTarget { column: target.into(), value: f64::NAN })
    } else {
        label_target(dataset, target)
    };
    if let Ok((classes, _)) = &labels {
        let two: bool = classes.len() == 2;
        if (target_type == "binary" && !two) || (target_type == "multiclass" && two) {
            return Err(ModelError::InvalidSetting {
                setting: "target_type",
                value: format!("{target_type} with {} classes", classes.len()),
                expected: "\"binary\" for 2 classes, \"multiclass\" for more",
            });
        }
    }
    match labels {
        Ok((classes, codes)) => {
            let codes: Vec<f64> = codes.f64()?.into_no_null_iter().collect();
            let labels: Series = Series::new("\0class".into(), codes.iter().map(|c| *c as i64).collect::<Vec<i64>>());
            let one = |k: f64| codes.iter().map(|c| f64::from(u8::from(*c == k))).collect::<Vec<f64>>();
            let (kind, columns) = if classes.len() == 2 {
                ("binary", vec![one(1.0)])
            } else {
                ("multiclass", (0..classes.len()).map(|k| one(k as f64)).collect())
            };
            Ok(Target { kind, classes, columns, labels: Some(labels) })
        }
        Err(ModelError::ContinuousTarget { .. }) => {
            let y: Series = dataset.column(target)?.cast(&DataType::Float64)?.as_materialized_series().clone();
            Ok(Target {
                kind: "continuous",
                classes: Vec::new(),
                columns: vec![y.f64()?.into_no_null_iter().collect()],
                labels: None,
            })
        }
        Err(error) => Err(error),
    }
}

/// A column's categories, sorted as text with a missing value last, and each row's code.
fn codes(dataset: &Dataset, name: &str) -> Result<Codes, ModelError> {
    let text = dataset.column(name)?.cast(&DataType::String)?;
    let text = text.str()?;
    let mut seen: PlHashMap<Option<&str>, u32> = PlHashMap::default();
    let first: Vec<u32> = text
        .iter()
        .map(|v| {
            let next: u32 = seen.len() as u32;
            *seen.entry(v).or_insert(next)
        })
        .collect();
    let mut categories: Vec<Option<&str>> = seen.keys().copied().collect();
    categories.sort_by(|a, b| match (a, b) {
        (None, None) => std::cmp::Ordering::Equal,
        (None, _) => std::cmp::Ordering::Greater,
        (_, None) => std::cmp::Ordering::Less,
        (Some(a), Some(b)) => a.cmp(b),
    });
    let mut sorted: Vec<u32> = vec![0; categories.len()];
    for (place, category) in categories.iter().enumerate() {
        sorted[seen[category] as usize] = place as u32;
    }
    Ok(Codes {
        categories: categories.into_iter().map(|c| c.map(str::to_string)).collect(),
        codes: first.into_iter().map(|c| sorted[c as usize]).collect(),
    })
}

#[cfg(test)]
mod tests {
    use polars::df;

    use super::*;

    /// The rows of `analysis/2026-11-18/new-features/target-encoder-parity.py`.
    fn rows() -> Dataset {
        let n: usize = 300;
        let city: Vec<Option<String>> = (0..n).map(|k| (k % 31 != 0).then(|| format!("c{}", k * 7 % 9))).collect();
        let tier: Vec<String> = (0..n).map(|k| format!("t{}", k * 5 % 4)).collect();
        let cont: Vec<f64> = (0..n).map(|k| (k * 13 % 17) as f64 + 0.5 * (k * 7 % 9) as f64 + 0.25).collect();
        let bin: Vec<&str> = (0..n).map(|k| if (k * 11 % 7) + (k * 7 % 9) > 7 { "yes" } else { "no" }).collect();
        let multi: Vec<&str> = (0..n).map(|k| ["a", "b", "c"][(k * 13 % 17 + k * 7 % 9) % 3]).collect();
        Dataset::new(df!("city" => city, "tier" => tier, "cont" => cont, "bin" => bin, "multi" => multi).unwrap(), "")
    }

    fn close(got: &[f64], want: &[f64]) {
        assert!(got.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-10), "{got:?}\nwant {want:?}");
    }

    fn check(target: &str, encoder: TargetEncoder, encodings: &[&[f64]], first: &[f64], sums: &[f64]) {
        let ds: Dataset = rows().select(&["city", "tier", target]);
        let mut encoder = encoder;
        let out: Dataset = encoder.try_fit_transform(&ds, target).unwrap();
        let names: Vec<String> = out.columns().into_iter().filter(|c| c != target).collect();
        let firsts: Vec<f64> = names.iter().map(|c| out.to_list(c)[0]).collect();
        let totals: Vec<f64> = names.iter().map(|c| out.to_list(c).iter().sum()).collect();
        close(&firsts, first);
        assert!(totals.iter().zip(sums).all(|(g, w)| (g - w).abs() < 1e-8), "{totals:?}\nwant {sums:?}");
        let learned: Vec<&Vec<f64>> = encoder.encodings_.values().flatten().collect();
        for (got, want) in learned.iter().zip(encodings) {
            close(&got[..4], want);
        }
    }

    #[test]
    fn cross_fitted_encodings_are_scikit_learn_s() {
        check(
            "cont",
            TargetEncoder::new(),
            &[&[8.609061746517781, 8.823341242812491, 9.431049967869521, 9.97667961708376], &[9.730265603407542, 10.335224660483275, 10.65798989864421, 10.24992743136355]],
            &[12.118493071840197, 9.71293350836293],
            &[3072.966048649826, 3079.7200675223103],
        );
        check(
            "bin",
            TargetEncoder::new(),
            &[&[0.0, 0.0, 0.16101162672975225, 0.316048190921367], &[0.4269271623372987, 0.4532453805464398, 0.4532453805464398, 0.4532453805464398]],
            &[0.44458325622772554, 0.3843143403006731],
            &[132.68199919855613, 134.03339010111574],
        );
        check(
            "multi",
            TargetEncoder::new(),
            &[&[0.31291961159709397, 0.31291961159709397, 0.31291961159709397, 0.3734424311079521], &[0.37367052706216025, 0.3131109662894721, 0.3131109662894721, 0.3434297123419359]],
            &[0.29078221696478995, 0.2912371134020618, 0.41624958379798216, 0.3215626940645338, 0.35671154448271236, 0.32170373240578876],
            &[97.75855822468031, 99.41702012474659, 102.64440895644647, 97.71775439019356, 100.41343254820217, 101.85916090769216],
        );
        check(
            "cont",
            TargetEncoder::new().smooth(3.0).random_state(7).cv(3),
            &[&[8.70657142857143, 8.906571428571429, 9.478, 9.992285714285714], &[9.743333333333334, 10.333076923076923, 10.647179487179487, 10.24974358974359]],
            &[11.7365625, 10.094086538461537],
            &[3062.349150257548, 3062.9474695615054],
        );
    }

    #[test]
    fn new_rows_take_the_full_fit_and_unknowns_the_mean() {
        let ds: Dataset = rows().select(&["city", "tier", "bin"]);
        let mut encoder = TargetEncoder::new();
        encoder.try_fit(&ds, "bin").unwrap();
        assert_eq!(encoder.target_type_, "binary");
        let new: Dataset = Dataset::new(df!("city" => [Some("c1"), Some("zz"), None], "tier" => ["t0", "t1", "t9"]).unwrap(), "");
        let out: Dataset = encoder.try_transform(&new).unwrap();
        close(&out.to_list("city"), &[0.0, 0.44666666666666666, 0.40413047041468614]);
        close(&out.to_list("tier"), &[0.4269271623372987, 0.4532453805464398, 0.44666666666666666]);
        assert!(TargetEncoder::new().try_transform(&new).is_err());
    }

    /// Against `analysis/2026-11-20/target-pipeline/parity.py`: the coefficients come from the
    /// cross-fitted encodings of the training rows, the predictions of new rows from `fit`'s.
    #[test]
    fn a_pipeline_passes_the_target_and_matches_scikit_learn() {
        use crate::{models::{LinearRegression, Model, Regressor}, pipeline::Pipeline, preprocess::ColumnTransformer, steps};
        let n: usize = 60;
        let city: Vec<&str> = (0..n).map(|i| ["a", "b", "c", "d"][i * 7 % 4]).collect();
        let shop: Vec<&str> = (0..n).map(|i| ["x", "y", "z"][i * 5 % 3]).collect();
        let hours: Vec<f64> = (0..n).map(|i| (i * 13 % 17) as f64).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let c: f64 = match city[i] { "a" => 3.0, "b" => -2.0, "c" => 1.0, _ => 0.5 };
                hours[i] * 0.5 + c + (i * 11 % 7) as f64 * 0.3 + if shop[i] == "y" { 1.5 } else { 0.0 }
            })
            .collect();
        let ds = Dataset::new(polars::df!("city" => city, "shop" => shop, "hours" => hours, "y" => y).unwrap(), "");
        let prep = ColumnTransformer::new()
            .add("te", &["city", "shop"], steps![TargetEncoder::new().target_type("continuous")])
            .remainder("passthrough");
        let mut pipe = Pipeline::new(prep, LinearRegression::new());
        pipe.fit(&ds, "y");
        let close = |got: &[f64], want: &[f64]| got.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-9);
        let coef: Vec<f64> = pipe.model.coef_.clone();
        assert!(close(&coef, &[1.1647879279007598, 0.7398228159441206, 0.5762015579359514]), "{coef:?}");
        assert!((pipe.model.intercept_ - -9.745049009831849).abs() < 1e-9);
        let train: Vec<f64> = pipe.predict(&ds).f64().unwrap().into_no_null_iter().take(4).collect();
        assert!(close(&train, &[3.2474099098117044, 8.381789109018158, 8.041315238251858, 1.5771277714674081]), "{train:?}");
        let new = Dataset::new(polars::df!("city" => ["a", "d", "q"], "shop" => ["z", "y", "x"], "hours" => [1.0, 5.0, 2.0]).unwrap(), "");
        let predicted: Vec<f64> = pipe.predict(&new).f64().unwrap().into_no_null_iter().collect();
        assert!(close(&predicted, &[3.8709633658787297, 4.955593602702381, 2.2493783225432473]), "{predicted:?}");
    }

    #[test]
    fn without_the_target_or_before_another_step_it_says_why() {
        use crate::{preprocess::{ColumnTransformer, StandardScaler, Transform}, steps};
        let ds = Dataset::new(polars::df!("city" => ["a", "b", "a", "b"], "y" => [1.0, 2.0, 3.0, 4.0]).unwrap(), "");
        let mut alone = ColumnTransformer::new().add("te", &["city"], steps![TargetEncoder::new()]);
        let error: String = alone.try_fit(&ds).unwrap_err().to_string();
        assert!(error.contains("learns from the target"), "{error}");
        let mut before = ColumnTransformer::new().add("te", &["city"], steps![TargetEncoder::new().target_type("continuous").cv(2), StandardScaler::new()]);
        let y: Series = ds.dataframe().column("y").unwrap().as_materialized_series().clone();
        let error: String = before.try_fit_with_target(&ds.drop(&["y"]), &y).unwrap_err().to_string();
        assert!(error.contains("last step of its add"), "{error}");
    }
}
