//! [`KFold`], [`StratifiedKFold`] and [`cross_val_predict`]: scikit-learn's splitters, and the
//! out-of-fold predictions that stacking is built on.
//!
//! With `shuffle(true)` the rows are shuffled by numpy's own generator (`RandomState`, the
//! Mersenne Twister), as scikit-learn does, so `KFold::new().shuffle(true)` with the default
//! `random_state` 42 gives the folds of scikit-learn's `KFold(5, shuffle=True, random_state=42)`
//! row for row. Out-of-fold files made in Python and here then line up.

use indexmap::IndexMap;
use polars::prelude::{Column, DataFrame, DataType, IdxCa, IdxSize, PlSmallStr};
use polars_core::runtime::THREAD_POOL;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    dataset::{Dataset, split::take},
    error::{OrFatal, warn},
    models::{Model, ModelError, check, trainable},
};

/// numpy's `RandomState`: MT19937 with its legacy integer seeding, and its `shuffle`.
pub(crate) struct Mt19937 {
    state: [u32; 624],
    next: usize,
}

impl Mt19937 {
    /// `np.random.RandomState(seed)`: the low 32 bits of the seed, as numpy takes them.
    pub(crate) fn new(seed: u64) -> Self {
        let mut state: [u32; 624] = [0; 624];
        state[0] = seed as u32;
        for i in 1..624 {
            let last: u32 = state[i - 1];
            state[i] = 1_812_433_253u32
                .wrapping_mul(last ^ (last >> 30))
                .wrapping_add(i as u32);
        }
        Self { state, next: 624 }
    }

    fn next_u32(&mut self) -> u32 {
        if self.next == 624 {
            for k in 0..624 {
                let y: u32 = (self.state[k] & 0x8000_0000) | (self.state[(k + 1) % 624] & 0x7fff_ffff);
                let odd: u32 = if y & 1 == 1 { 0x9908_b0df } else { 0 };
                self.state[k] = self.state[(k + 397) % 624] ^ (y >> 1) ^ odd;
            }
            self.next = 0;
        }
        let mut y: u32 = self.state[self.next];
        self.next += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// A uniform integer in `0..=max`, numpy's `random_interval`: masked draws, redrawn while
    /// above `max`.
    pub(crate) fn interval(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        let mask: u64 = u64::MAX >> max.leading_zeros();
        loop {
            let value: u64 = if max <= u64::from(u32::MAX) {
                u64::from(self.next_u32()) & mask
            } else {
                ((u64::from(self.next_u32()) << 32) | u64::from(self.next_u32())) & mask
            };
            if value <= max {
                return value;
            }
        }
    }

    /// `RandomState.shuffle` of a 1-d array: Fisher-Yates from the end.
    pub(crate) fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j: usize = self.interval(i as u64) as usize;
            items.swap(i, j);
        }
    }
}

/// Splits the rows into `n_splits` folds of consecutive rows, like scikit-learn's `KFold`:
/// each fold is the test rows once, and the other folds train. The first `rows % n_splits`
/// folds have one row more. With `shuffle(true)` the rows are shuffled first, by numpy's
/// generator seeded with `random_state`, so the folds are scikit-learn's.
///
/// Give it to [`cross_val_predict`], [`cross_val_score`](crate::models::cross_val_score) or
/// [`GridSearchCV::cv`](crate::models::GridSearchCV::cv) in place of a fold count.
///
/// ```
/// use rusmlai::prelude::*;
/// let ds = Dataset::new(df!("x" => [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap(), "");
/// let folds = KFold::new().n_splits(3).split(&ds);
/// assert_eq!(folds[0].1, [0, 1, 2, 3]); // (train rows, test rows)
/// assert_eq!(folds[2].0, [0, 1, 2, 3, 4, 5, 6]);
/// let shuffled = KFold::new().n_splits(5).shuffle(true).split(&ds);
/// assert_eq!(shuffled[0].1, [1, 8]); // scikit-learn's, with random_state 42
/// ```
///
/// Python:
///
/// ```python
/// import numpy as np
/// from sklearn.model_selection import KFold
/// x = np.arange(10).reshape(-1, 1)
/// folds = list(KFold(n_splits=3).split(x))
/// assert folds[0][1].tolist() == [0, 1, 2, 3]
/// assert folds[2][0].tolist() == [0, 1, 2, 3, 4, 5, 6]
/// shuffled = list(KFold(n_splits=5, shuffle=True, random_state=42).split(x))
/// assert shuffled[0][1].tolist() == [1, 8]
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KFold {
    n_splits: usize,
    shuffle: bool,
    random_state: u64,
}

impl Default for KFold {
    fn default() -> Self {
        Self::new()
    }
}

/// One fold: the train rows and the test rows, each in row order.
pub type Fold = (Vec<IdxSize>, Vec<IdxSize>);

impl KFold {
    /// 5 folds, not shuffled, `random_state` 42 (used only with `shuffle(true)`).
    pub fn new() -> Self {
        Self {
            n_splits: 5,
            shuffle: false,
            random_state: 42,
        }
    }

    /// The number of folds, 2 or more. Default 5.
    #[track_caller]
    pub fn n_splits(mut self, n_splits: usize) -> Self {
        check(n_splits >= 2, "n_splits", n_splits, "2 folds or more");
        self.n_splits = n_splits;
        self
    }

    /// Shuffle the rows before they are cut into folds. Default `false`.
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// The seed of the shuffle. Default 42.
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = random_state;
        self
    }

    /// The number of folds.
    pub fn get_n_splits(&self) -> usize {
        self.n_splits
    }

    /// The train and test rows of each fold.
    #[track_caller]
    pub fn split(&self, dataset: &Dataset) -> Vec<Fold> {
        self.try_split(dataset).or_fatal()
    }

    /// [`KFold::split`] that returns the error: more folds than rows.
    pub fn try_split(&self, dataset: &Dataset) -> Result<Vec<Fold>, ModelError> {
        Ok(train_and_test(self.test_rows(dataset.height())?, dataset.height()))
    }

    pub(crate) fn test_rows(&self, n: usize) -> Result<Vec<Vec<IdxSize>>, ModelError> {
        let k: usize = self.n_splits;
        too_many_folds(k, n)?;
        let mut order: Vec<IdxSize> = (0..n as IdxSize).collect();
        if self.shuffle {
            Mt19937::new(self.random_state).shuffle(&mut order);
        }
        let mut folds: Vec<Vec<IdxSize>> = Vec::with_capacity(k);
        let mut start: usize = 0;
        for i in 0..k {
            let size: usize = n / k + usize::from(i < n % k);
            let mut fold: Vec<IdxSize> = order[start..start + size].to_vec();
            fold.sort_unstable();
            folds.push(fold);
            start += size;
        }
        Ok(folds)
    }
}

/// Splits the rows into `n_splits` folds that each hold about the same share of each class of
/// the target, like scikit-learn's `StratifiedKFold`. Without shuffle a class's rows fill the
/// folds in row order; with `shuffle(true)` they are dealt to the folds by numpy's generator
/// seeded with `random_state`, so the folds are scikit-learn's.
///
/// ```
/// use rusmlai::prelude::*;
/// let ds = Dataset::new(
///     df!("y" => ["a", "a", "a", "a", "b", "b", "a", "a", "b", "a"]).unwrap(),
///     "",
/// );
/// let folds = StratifiedKFold::new().n_splits(2).split(&ds, "y");
/// assert_eq!(folds[0].1, [0, 1, 2, 3, 4]); // 4 of the 7 "a" rows, 1 of the 3 "b" rows
/// let shuffled = StratifiedKFold::new().n_splits(2).shuffle(true).split(&ds, "y");
/// assert_eq!(shuffled[0].1, [0, 1, 3, 7, 8]); // scikit-learn's, with random_state 42
/// ```
///
/// Python:
///
/// ```python
/// import numpy as np
/// from sklearn.model_selection import StratifiedKFold
/// y = np.array(["a", "a", "a", "a", "b", "b", "a", "a", "b", "a"])
/// x = np.zeros((10, 1))
/// folds = list(StratifiedKFold(n_splits=2).split(x, y))
/// assert folds[0][1].tolist() == [0, 1, 2, 3, 4]
/// shuffled = list(StratifiedKFold(n_splits=2, shuffle=True, random_state=42).split(x, y))
/// assert shuffled[0][1].tolist() == [0, 1, 3, 7, 8]
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StratifiedKFold {
    n_splits: usize,
    shuffle: bool,
    random_state: u64,
}

impl Default for StratifiedKFold {
    fn default() -> Self {
        Self::new()
    }
}

impl StratifiedKFold {
    /// 5 folds, not shuffled, `random_state` 42 (used only with `shuffle(true)`).
    pub fn new() -> Self {
        Self {
            n_splits: 5,
            shuffle: false,
            random_state: 42,
        }
    }

    /// The number of folds, 2 or more. Default 5.
    #[track_caller]
    pub fn n_splits(mut self, n_splits: usize) -> Self {
        check(n_splits >= 2, "n_splits", n_splits, "2 folds or more");
        self.n_splits = n_splits;
        self
    }

    /// Deal each class's rows to the folds at random. Default `false`.
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// The seed of the shuffle. Default 42.
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = random_state;
        self
    }

    /// The number of folds.
    pub fn get_n_splits(&self) -> usize {
        self.n_splits
    }

    /// The train and test rows of each fold, stratified on the column `target`.
    #[track_caller]
    pub fn split(&self, dataset: &Dataset, target: &str) -> Vec<Fold> {
        self.try_split(dataset, target).or_fatal()
    }

    /// [`StratifiedKFold::split`] that returns the error: a missing target, more folds than
    /// rows, or more folds than the rows of the largest class.
    pub fn try_split(&self, dataset: &Dataset, target: &str) -> Result<Vec<Fold>, ModelError> {
        Ok(train_and_test(self.test_rows(dataset, target)?, dataset.height()))
    }

    /// scikit-learn's `_make_test_folds`.
    fn test_rows(&self, dataset: &Dataset, target: &str) -> Result<Vec<Vec<IdxSize>>, ModelError> {
        let (n, k): (usize, usize) = (dataset.height(), self.n_splits);
        too_many_folds(k, n)?;
        // Each class in the order it first appears.
        let keys = dataset.column(target)?.cast(&DataType::String)?;
        let mut classes: IndexMap<Option<String>, usize> = IndexMap::new();
        let codes: Vec<usize> = (keys.str()?.iter())
            .map(|key| {
                let next: usize = classes.len();
                *classes.entry(key.map(str::to_string)).or_insert(next)
            })
            .collect();
        let mut counts: Vec<usize> = vec![0; classes.len()];
        codes.iter().for_each(|code| counts[*code] += 1);
        if counts.iter().all(|count| k > *count) {
            return Err(ModelError::InvalidSetting {
                setting: "n_splits",
                value: k.to_string(),
                expected: "at most the rows of the largest class: each fold needs a row of it",
            });
        }
        if let Some(least) = counts.iter().min().filter(|least| k > **least) {
            warn(format_args!(
                "the smallest class of '{target}' has {least} rows, fewer than the {k} folds"
            ));
        }
        // How many rows of each class each fold gets: deal the rows, sorted by class, to the
        // folds in turn.
        let mut share: Vec<Vec<usize>> = vec![vec![0; classes.len()]; k];
        let mut place: usize = 0;
        for (code, count) in counts.iter().enumerate() {
            for _ in 0..*count {
                share[place % k][code] += 1;
                place += 1;
            }
        }
        // Each class's fold labels in fold order, shuffled one class after the other from one
        // generator, then given to its rows in row order.
        let mut rng: Mt19937 = Mt19937::new(self.random_state);
        let mut labels: Vec<std::vec::IntoIter<usize>> = (0..classes.len())
            .map(|code| {
                let mut folds: Vec<usize> = (0..k)
                    .flat_map(|fold| std::iter::repeat_n(fold, share[fold][code]))
                    .collect();
                if self.shuffle {
                    rng.shuffle(&mut folds);
                }
                folds.into_iter()
            })
            .collect();
        let mut folds: Vec<Vec<IdxSize>> = vec![Vec::new(); k];
        for (row, code) in codes.iter().enumerate() {
            let fold: usize = labels[*code].next().unwrap_or(0);
            folds[fold].push(row as IdxSize);
        }
        Ok(folds)
    }
}

fn too_many_folds(k: usize, n: usize) -> Result<(), ModelError> {
    if k > n {
        return Err(ModelError::InvalidSetting {
            setting: "n_splits",
            value: format!("{k} folds of {n} rows"),
            expected: "at most one fold per row",
        });
    }
    Ok(())
}

/// The train rows of each fold: every row not in its test rows.
fn train_and_test(tests: Vec<Vec<IdxSize>>, n: usize) -> Vec<Fold> {
    let mut fold_of: Vec<usize> = vec![0; n];
    for (i, rows) in tests.iter().enumerate() {
        rows.iter().for_each(|row| fold_of[*row as usize] = i);
    }
    tests
        .into_iter()
        .enumerate()
        .map(|(i, test)| {
            let train: Vec<IdxSize> = (0..n as IdxSize).filter(|row| fold_of[*row as usize] != i).collect();
            (train, test)
        })
        .collect()
}

/// How the rows are cut into folds: a number of folds, or a splitter. An integer `cv` is
/// scikit-learn's: `StratifiedKFold(cv)` for a classifier, `KFold(cv)` for a regressor, not
/// shuffled. Every function that takes `cv` takes any of them:
/// `cross_val_predict(&model, &ds, "y", 5, "predict")` or
/// `cross_val_predict(&model, &ds, "y", KFold::new().shuffle(true), "predict")`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Cv {
    /// A number of folds, 2 or more.
    Folds(usize),
    /// A splitter.
    Splitter(Splitter),
}

/// A splitter that [`Cv`] holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Splitter {
    /// [`KFold`].
    KFold(KFold),
    /// [`StratifiedKFold`].
    StratifiedKFold(StratifiedKFold),
}

impl From<usize> for Cv {
    fn from(folds: usize) -> Self {
        Cv::Folds(folds)
    }
}

/// So that a literal such as `5` is a `Cv`: Rust reads an untyped integer as `i32`. A negative
/// count is refused where the folds are made.
impl From<i32> for Cv {
    fn from(folds: i32) -> Self {
        Cv::Folds(usize::try_from(folds).unwrap_or(0))
    }
}

impl From<KFold> for Cv {
    fn from(splitter: KFold) -> Self {
        Cv::Splitter(Splitter::KFold(splitter))
    }
}

impl From<StratifiedKFold> for Cv {
    fn from(splitter: StratifiedKFold) -> Self {
        Cv::Splitter(Splitter::StratifiedKFold(splitter))
    }
}

impl Cv {
    /// The number of folds.
    pub fn n_splits(&self) -> usize {
        match self {
            Cv::Folds(k) => *k,
            Cv::Splitter(Splitter::KFold(s)) => s.n_splits,
            Cv::Splitter(Splitter::StratifiedKFold(s)) => s.n_splits,
        }
    }

    pub(crate) fn check(&self) -> Result<(), ModelError> {
        if self.n_splits() < 2 {
            return Err(ModelError::InvalidSetting {
                setting: "cv",
                value: self.n_splits().to_string(),
                expected: "2 folds or more",
            });
        }
        Ok(())
    }

    /// The test rows of each fold, in row order.
    pub(crate) fn test_rows(
        &self,
        dataset: &Dataset,
        target: &str,
        classifier: bool,
    ) -> Result<Vec<Vec<IdxSize>>, ModelError> {
        self.check()?;
        match self {
            Cv::Folds(k) if classifier => StratifiedKFold::new().n_splits(*k).test_rows(dataset, target),
            Cv::Folds(k) => KFold::new().n_splits(*k).test_rows(dataset.height()),
            Cv::Splitter(Splitter::KFold(s)) => s.test_rows(dataset.height()),
            Cv::Splitter(Splitter::StratifiedKFold(s)) => s.test_rows(dataset, target),
        }
    }

    /// The train and test rows of each fold, as datasets of their own.
    pub(crate) fn datasets(
        &self,
        dataset: &Dataset,
        target: &str,
        classifier: bool,
    ) -> Result<Vec<(Dataset, Dataset)>, ModelError> {
        let n: usize = dataset.height();
        train_and_test(self.test_rows(dataset, target, classifier)?, n)
            .into_iter()
            .map(|(train, test)| Ok((take(dataset, train)?, take(dataset, test)?)))
            .collect()
    }
}

/// The out-of-fold predictions of `model`: each row is predicted by a copy of the model that
/// trained on the other folds, like scikit-learn's `cross_val_predict`. These are the columns
/// a stacking model learns from, and what Kaggle solutions save as "OOF" files.
///
/// `method` is `"predict"` (one column, named after the target) or, for a classifier,
/// `"predict_proba"` (one column per class). `cv` is a fold count or a splitter ([`Cv`]). The
/// rows come back in the order of `dataset`. The folds train in parallel on the Polars pool,
/// outside the model cache.
///
/// ```
/// use rusmlai::prelude::*;
/// let ds = Dataset::new(
///     df!(
///         "x" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
///         "y" => [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
///     )
///     .unwrap(),
///     "",
/// );
/// let oof = cross_val_predict(&LinearRegression::new(), &ds, "y", 4, "predict");
/// let predicted: Vec<f64> = oof.to_list("y");
/// assert!(predicted.iter().zip(1..).all(|(p, i)| (p - 2.0 * i as f64).abs() < 1e-9));
///
/// let labels = Dataset::new(
///     df!(
///         "x" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
///         "y" => ["a", "a", "b", "a", "b", "b", "a", "b"],
///     )
///     .unwrap(),
///     "",
/// );
/// let cv = StratifiedKFold::new().n_splits(2).shuffle(true);
/// let proba = cross_val_predict(&LogisticRegression::new(), &labels, "y", cv, "predict_proba");
/// assert_eq!(proba.columns(), ["a", "b"]);
/// assert_eq!(proba.height(), 8);
/// ```
///
/// Python:
///
/// ```python
/// import numpy as np
/// import pandas as pd
/// from sklearn.linear_model import LinearRegression, LogisticRegression
/// from sklearn.model_selection import StratifiedKFold, cross_val_predict
/// ds = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
/// ds["y"] = 2 * ds["x"]
/// oof = cross_val_predict(LinearRegression(), ds[["x"]], ds["y"], cv=4)
/// assert np.allclose(oof, ds["y"])
/// labels = ds.assign(y=["a", "a", "b", "a", "b", "b", "a", "b"])
/// cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
/// proba = cross_val_predict(LogisticRegression(), labels[["x"]], labels["y"], cv=cv, method="predict_proba")
/// assert proba.shape == (8, 2)
/// ```
#[track_caller]
pub fn cross_val_predict<M: Model + Clone + Send + Sync>(
    model: &M,
    dataset: &Dataset,
    target: &str,
    cv: impl Into<Cv>,
    method: &str,
) -> Dataset {
    try_cross_val_predict(model, dataset, target, cv, method).or_fatal()
}

/// [`cross_val_predict`] that returns the error.
///
/// # Errors
///
/// A bad `cv` or `method`, and any error of the model's fit or predict.
pub fn try_cross_val_predict<M: Model + Clone + Send + Sync>(
    model: &M,
    dataset: &Dataset,
    target: &str,
    cv: impl Into<Cv>,
    method: &str,
) -> Result<Dataset, ModelError> {
    let reduced: Option<Dataset> = trainable(dataset, target)?;
    oof(model, reduced.as_ref().unwrap_or(dataset), target, cv.into(), method)
}

/// [`try_cross_val_predict`] on columns already chosen by `trainable`.
pub(crate) fn oof<M: Model + Clone + Send + Sync>(
    model: &M,
    dataset: &Dataset,
    target: &str,
    cv: Cv,
    method: &str,
) -> Result<Dataset, ModelError> {
    let tests: Vec<Vec<IdxSize>> = cv.test_rows(dataset, target, model.is_classifier())?;
    let n: usize = dataset.height();
    let folds: Vec<Fold> = train_and_test(tests, n);
    let params: Value = model.params();
    let parts: Vec<(Vec<IdxSize>, DataFrame)> = THREAD_POOL.install(|| {
        folds
            .into_par_iter()
            .map(|(train, test)| {
                let mut copy: M = model.clone();
                copy.set_run(None);
                copy.try_set_params(params.clone())?;
                copy.try_train(&take(dataset, train)?, target)?;
                let predicted: Dataset = copy.try_predict_method(&take(dataset, test.clone())?, method)?;
                Ok((test, predicted.into_dataframe()))
            })
            .collect::<Result<_, ModelError>>()
    })?;
    oof_frame(parts, n)
}

/// Puts the predictions of each fold back in row order. A class that a fold never saw in
/// training has no column there: it gets 0, as scikit-learn fills it.
pub(crate) fn oof_frame(parts: Vec<(Vec<IdxSize>, DataFrame)>, n: usize) -> Result<Dataset, ModelError> {
    let mut names: Vec<PlSmallStr> = Vec::new();
    for (_, frame) in &parts {
        for name in frame.get_column_names() {
            if !names.contains(name) {
                names.push(name.clone());
            }
        }
    }
    let float: bool = parts.iter().all(|(_, frame)| {
        frame.columns().iter().all(|c| c.dtype().is_primitive_numeric() || c.dtype().is_bool())
    }) && names.len() > 1;
    let mut stacked: Vec<DataFrame> = Vec::with_capacity(parts.len());
    let mut rows: Vec<IdxSize> = Vec::with_capacity(n);
    for (test, frame) in parts {
        let height: usize = frame.height();
        let columns: Vec<Column> = names
            .iter()
            .map(|name| match frame.column(name) {
                Ok(column) if float => column.cast(&DataType::Float64),
                Ok(column) => Ok(column.clone()),
                Err(_) => Ok(Column::full_null(name.clone(), height, &DataType::Float64)
                    .fill_null(polars::prelude::FillNullStrategy::Zero)?),
            })
            .collect::<Result<_, _>>()?;
        stacked.push(DataFrame::new(height, columns)?);
        rows.extend(test);
    }
    let mut all: DataFrame = stacked.remove(0);
    for frame in &stacked {
        all.vstack_mut(frame)?;
    }
    // `rows[i]` is the dataset row of stacked row i; invert it.
    let mut place: Vec<IdxSize> = vec![0; n];
    for (i, row) in rows.iter().enumerate() {
        place[*row as usize] = i as IdxSize;
    }
    let mut ordered: DataFrame = all.take(&IdxCa::from_vec(PlSmallStr::EMPTY, place))?;
    ordered.rechunk_mut_par();
    Ok(Dataset::new(ordered, ""))
}

#[cfg(test)]
mod tests {
    use polars::df;

    use super::*;

    #[test]
    fn the_generator_is_numpys() {
        // np.random.RandomState(42).shuffle(np.arange(10))
        let mut items: Vec<u32> = (0..10).collect();
        Mt19937::new(42).shuffle(&mut items);
        assert_eq!(items, [8, 1, 5, 0, 7, 2, 9, 4, 3, 6]);
    }

    #[test]
    fn bad_folds_are_refused() {
        let ds: Dataset = Dataset::new(df!("y" => [1, 1, 2]).unwrap(), "");
        assert!(KFold::new().n_splits(4).try_split(&ds).is_err());
        assert!(StratifiedKFold::new().n_splits(3).try_split(&ds, "y").is_err());
        assert!(Cv::from(1).check().is_err());
        assert!(Cv::from(-3).check().is_err());
    }

    #[test]
    fn cv_keeps_a_plain_count_as_a_number_in_json() {
        assert_eq!(serde_json::to_value(Cv::from(5)).unwrap(), serde_json::json!(5));
        let shuffled: Cv = KFold::new().shuffle(true).into();
        let back: Cv = serde_json::from_value(serde_json::to_value(shuffled).unwrap()).unwrap();
        assert_eq!(back, shuffled);
    }
    /// scikit-learn 1.9 on `table()`: analysis/2026-11-18/new-features/folds-parity.py.
    #[test]
    fn shuffled_folds_and_oof_predictions_are_scikit_learn_s() {
        use crate::models::{LinearRegression, LogisticRegression, tree::tests::table};
        let ds: Dataset = table();
        let firsts = |folds: Vec<Fold>| -> Vec<Vec<IdxSize>> {
            folds.into_iter().map(|(_, test)| test[..6].to_vec()).collect()
        };
        let kf: KFold = KFold::new().shuffle(true);
        assert_eq!(
            firsts(kf.split(&ds)),
            [[0, 5, 9, 15, 22, 25], [3, 6, 7, 10, 16, 17], [2, 11, 23, 26, 37, 38], [4, 8, 12, 14, 27, 28], [1, 13, 20, 21, 34, 43]]
        );
        let skf: StratifiedKFold = StratifiedKFold::new().shuffle(true).random_state(7);
        assert_eq!(
            firsts(skf.split(&ds, "class")),
            [[6, 8, 13, 16, 20, 24], [12, 19, 23, 39, 45, 49], [7, 9, 10, 15, 21, 22], [0, 1, 2, 3, 4, 5], [18, 33, 34, 35, 37, 38]]
        );
        let oof: Vec<f64> = cross_val_predict(&LinearRegression::new(), &ds.drop(&["class"]), "y", kf, "predict").to_list("y");
        let want = [1.4818751429223305, 7.270430606855839, 21.69133582532276, 2.2164355483609572];
        assert!(oof.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-9), "{oof:?}");
        assert!((oof.iter().sum::<f64>() - 6492.040540994298).abs() < 1e-7);
        let cv: StratifiedKFold = StratifiedKFold::new().shuffle(true);
        let proba: Dataset = cross_val_predict(&LogisticRegression::new(), &ds.drop(&["y"]), "class", cv, "predict_proba");
        assert_eq!(proba.columns(), ["0", "1", "2"]);
        let first: Vec<f64> = proba.to_list("0");
        assert!((first[0] - 0.6713291198876534).abs() < 1e-6, "{}", first[0]);
        assert!((first.iter().sum::<f64>() - 114.581955820067).abs() < 1e-4);
    }

}
