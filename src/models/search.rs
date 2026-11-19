//! [`GridSearchCV`] and [`cross_val_score`]: scores of a model on folds of the rows, and the
//! search for the settings that score best.
//!
//! The folds are scikit-learn's: with an integer `cv`, `StratifiedKFold` for a classifier and
//! `KFold` for a regressor, not shuffled; or the splitter given ([`Cv`]). So a search on the same rows sees the same folds as
//! scikit-learn's. Each fold is taken once; every candidate trains on the same copies, and a
//! tree model makes the buckets of a fold once for all candidates.

use std::time::Instant;

use indexmap::IndexMap;
use polars::prelude::{Column, DataFrame, NamedFrom, Series};
use polars_core::runtime::THREAD_POOL;
use rayon::prelude::*;
use serde_json::{Map, Value, json};

use crate::{
    dataset::Dataset,
    error::OrFatal,
    metadata::models::CachedRun,
    models::{Classifier, Model, ModelError, Regressor, bad_learned, check, folds::Cv, trainable},
};

/// Every scorer name a search takes.
const SCORERS: [&str; 9] = [
    "",
    "accuracy",
    "balanced_accuracy",
    "roc_auc",
    "neg_log_loss",
    "r2",
    "neg_mean_squared_error",
    "neg_root_mean_squared_error",
    "neg_mean_absolute_error",
];

/// `f(0)` to `f(n - 1)`, in that order: all at once on the Polars pool, or with `limit` at most
/// that many at once, each on the pool (so its own parallel work spreads over every thread).
/// The first error stops the jobs not yet started.
fn run_jobs<T: Send>(
    n: usize,
    limit: Option<usize>,
    f: impl Fn(usize) -> Result<T, ModelError> + Sync,
) -> Result<Vec<T>, ModelError> {
    let Some(limit) = limit.filter(|l| *l < n) else {
        return THREAD_POOL.install(|| (0..n).into_par_iter().map(&f).collect());
    };
    let next = std::sync::atomic::AtomicUsize::new(0);
    let failed = std::sync::atomic::AtomicBool::new(false);
    let done: Vec<std::sync::Mutex<Option<Result<T, ModelError>>>> = (0..n).map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|scope| {
        for _ in 0..limit {
            scope.spawn(|| {
                loop {
                    let i: usize = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= n || failed.load(std::sync::atomic::Ordering::Relaxed) {
                        break;
                    }
                    let result: Result<T, ModelError> = THREAD_POOL.install(|| f(i));
                    if result.is_err() {
                        failed.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                    *done[i].lock().unwrap_or_else(|e| e.into_inner()) = Some(result);
                }
            });
        }
    });
    // A job not started (`None`) only follows a failure, so an error is among the slots.
    let results: Vec<Result<T, ModelError>> = (done.into_iter())
        .filter_map(|slot| slot.into_inner().unwrap_or_else(|e| e.into_inner()))
        .collect();
    results.into_iter().collect()
}

/// What one fit of one candidate on one fold gave.
struct Trial {
    score: f64,
    fit_seconds: f64,
    score_seconds: f64,
}

/// Trains a fresh copy of `model` with `params` on the train rows of `fold`, and scores it
/// on the test rows. Nothing goes into the model cache.
fn trial<M: Model + Clone>(
    model: &M,
    params: &Value,
    (train, test): &(Dataset, Dataset),
    target: &str,
    scoring: &str,
) -> Result<Trial, ModelError> {
    let mut copy: M = model.clone();
    copy.set_run(None);
    copy.try_set_params(params.clone())?;
    let start: Instant = Instant::now();
    copy.try_train(train, target)?;
    let fit_seconds: f64 = start.elapsed().as_secs_f64();
    let start: Instant = Instant::now();
    let score: f64 = copy.try_evaluate(test, target, scoring)?;
    Ok(Trial {
        score,
        fit_seconds,
        score_seconds: start.elapsed().as_secs_f64(),
    })
}

/// The score of `model` on each of `cv` folds of `dataset`, like scikit-learn's
/// `cross_val_score`: for each fold, a fresh copy of the model trains on the other folds and
/// is scored on that one.
///
/// The folds are scikit-learn's: for a classifier each fold has the same share of each class
/// (`StratifiedKFold`), for a regressor the folds are runs of rows in order (`KFold`); neither
/// shuffles, so sort or shuffle the rows first if their order means something. `scoring` is a
/// scorer name ([`Model::try_evaluate`]); `""` is the model's own score. The fits run on the
/// Polars pool and record nothing in the model cache.
///
/// ```
/// use rusmlai::prelude::*;
/// let ds = Dataset::new(
///     df!(
///         "x" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
///         "y" => [2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1],
///     )
///     .unwrap(),
///     "",
/// );
/// let scores: Vec<f64> = cross_val_score(&LinearRegression::new(), &ds, "y", 4, "r2");
/// assert_eq!(scores.len(), 4);
/// assert!(scores.iter().all(|r2| *r2 > 0.9));
/// let errors = cross_val_score(&LinearRegression::new(), &ds, "y", 4, "neg_mean_absolute_error");
/// assert!(errors.iter().all(|e| *e < 0.0 && *e > -0.5)); // negated, so higher is better
/// ```
///
/// Python:
///
/// ```python
/// import pandas as pd
/// from sklearn.linear_model import LinearRegression
/// from sklearn.model_selection import cross_val_score
/// ds = pd.DataFrame({
///     "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
///     "y": [2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1],
/// })
/// scores = cross_val_score(LinearRegression(), ds[["x"]], ds["y"], cv=4, scoring="r2")
/// assert len(scores) == 4
/// assert (scores > 0.9).all()
/// errors = cross_val_score(LinearRegression(), ds[["x"]], ds["y"], cv=4, scoring="neg_mean_absolute_error")
/// assert ((errors < 0.0) & (errors > -0.5)).all()
/// ```
#[track_caller]
pub fn cross_val_score<M: Model + Clone + Send + Sync>(
    model: &M,
    dataset: &Dataset,
    target: &str,
    cv: impl Into<Cv>,
    scoring: &str,
) -> Vec<f64> {
    try_cross_val_score(model, dataset, target, cv, scoring).or_fatal()
}

/// [`cross_val_score`] that returns the error.
pub fn try_cross_val_score<M: Model + Clone + Send + Sync>(
    model: &M,
    dataset: &Dataset,
    target: &str,
    cv: impl Into<Cv>,
    scoring: &str,
) -> Result<Vec<f64>, ModelError> {
    check_scoring(scoring)?;
    let cv: Cv = cv.into();
    let reduced: Option<Dataset> = trainable(dataset, target)?;
    let dataset: &Dataset = reduced.as_ref().unwrap_or(dataset);
    let folds: Vec<(Dataset, Dataset)> = cv.datasets(dataset, target, model.is_classifier())?;
    let params: Value = model.params();
    THREAD_POOL.install(|| {
        folds
            .par_iter()
            .map(|fold| Ok(trial(model, &params, fold, target, scoring)?.score))
            .collect()
    })
}

fn check_scoring(scoring: &str) -> Result<(), ModelError> {
    if SCORERS.contains(&scoring) {
        return Ok(());
    }
    Err(ModelError::InvalidSetting {
        setting: "scoring",
        value: scoring.into(),
        expected: "\"\" (the model's own score), \"accuracy\", \"balanced_accuracy\", \
                   \"roc_auc\", \"neg_log_loss\", \"r2\", \
                   \"neg_mean_squared_error\", \"neg_root_mean_squared_error\" or \
                   \"neg_mean_absolute_error\"",
    })
}

/// Where setting `name` is in `params`: a name of the model's settings, or
/// `estimator__name` for a setting of the model inside a bag or AdaBoost, like scikit-learn's
/// paths. A JSON pointer.
fn pointer(name: &str) -> String {
    match name.split_once("__") {
        Some((outer, inner)) => format!("/{outer}/params/{inner}"),
        None => format!("/{name}"),
    }
}

/// The names `params` takes, for an error message.
fn names(params: &Value) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for (name, value) in params.as_object().into_iter().flatten() {
        match value.get("params").and_then(Value::as_object) {
            Some(inner) => names.extend(inner.keys().map(|key| format!("{name}__{key}"))),
            None => names.push(name.clone()),
        }
    }
    names
}

/// Tries every combination of the settings in a grid, each one on the folds of the rows, and
/// keeps the one with the best mean score, like scikit-learn's `GridSearchCV`.
///
/// `new` takes the model with its other settings; each [`GridSearchCV::param_grid`] call adds
/// one setting and the values to try, by its scikit-learn name (an entry of scikit-learn's
/// `param_grid`). A setting of the model inside a bag or AdaBoost is `estimator__max_depth`;
/// a `Pipeline`'s model settings keep their own names. The combinations come in
/// scikit-learn's order: the settings sorted by name, the last one changing fastest.
///
/// `fit` trains a fresh copy of the model for each combination on each fold (scikit-learn's
/// `clone`), on the Polars pool, and scores it on the fold's test rows: `cv_results_` has one
/// row per combination. The best one, by the mean test score, is `best_params_`. Then the
/// model with those settings trains on all the rows (`refit`), through the model cache, and
/// `predict` and `score` use it.
///
/// The folds are those of [`cross_val_score`]: stratified for a classifier, in row order, not
/// shuffled. Each fold is copied once for all combinations, so a tree model makes the
/// buckets of a fold once.
///
/// A value goes into the model as it is: a search does not run the setters' checks, so a value
/// a setter refuses can give an odd model rather than an error.
///
/// [`n_jobs`](Self::n_jobs) caps how many fits run at once (each one still uses the whole
/// Polars pool for its own work, such as a forest's trees): a big model times every fit at once
/// can be more than the memory.
///
/// scikit-learn settings with no setter: `verbose`, `pre_dispatch`. Not yet: a list of grids, several scorers,
/// `return_train_score`, `error_score` (a fit that fails stops the search).
///
/// ```
/// use rusmlai::prelude::*;
/// let ds = Dataset::new(
///     df!(
///         "hours" => [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
///         "sleep" => [4.0, 5.0, 4.5, 5.5, 4.0, 8.0, 7.0, 4.0, 7.5, 8.5, 7.0, 8.0],
///         "passed" => ["no", "no", "no", "no", "no", "yes", "yes", "no", "yes", "yes", "yes", "yes"],
///     )
///     .unwrap(),
///     "",
/// );
/// let mut search = GridSearchCV::new(DecisionTreeClassifier::new())
///     .param_grid("max_depth", [1, 2, 3])
///     .param_grid("criterion", ["gini", "entropy"])
///     .cv(3);
/// search.fit(&ds, "passed");
/// assert_eq!(search.cv_results_.height(), 6); // 2 criteria × 3 depths
/// let ranks: Vec<f64> = search.cv_results_.to_list("rank_test_score");
/// assert!(ranks.contains(&1.0));
/// assert!(search.best_score_ > 0.5);
/// let best: &DecisionTreeClassifier = search.best_estimator_.as_ref().unwrap();
/// assert_eq!(best.params()["max_depth"], search.best_params_["max_depth"]);
/// assert_eq!(search.predict(&ds).len(), 12); // the best model, trained on every row
/// ```
///
/// Python:
///
/// ```python
/// import pandas as pd
/// from sklearn.model_selection import GridSearchCV
/// from sklearn.tree import DecisionTreeClassifier
/// ds = pd.DataFrame({
///     "hours": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
///     "sleep": [4.0, 5.0, 4.5, 5.5, 4.0, 8.0, 7.0, 4.0, 7.5, 8.5, 7.0, 8.0],
///     "passed": ["no", "no", "no", "no", "no", "yes", "yes", "no", "yes", "yes", "yes", "yes"],
/// })
/// x, y = ds[["hours", "sleep"]], ds["passed"]
/// search = GridSearchCV(
///     DecisionTreeClassifier(random_state=42),
///     {"max_depth": [1, 2, 3], "criterion": ["gini", "entropy"]},
///     cv=3,
/// ).fit(x, y)
/// assert len(search.cv_results_["params"]) == 6
/// assert 1 in search.cv_results_["rank_test_score"]
/// assert search.best_score_ > 0.5
/// assert search.best_estimator_.get_params()["max_depth"] == search.best_params_["max_depth"]
/// assert len(search.predict(x)) == 12
/// ```
#[derive(Debug)]
pub struct GridSearchCV<M> {
    estimator: M,
    grid: Vec<(String, Vec<Value>)>,
    cv: Cv,
    scoring: String,
    refit: bool,
    n_jobs: Option<usize>,
    /// One row per combination, in scikit-learn's columns: `mean_fit_time`, `std_fit_time`,
    /// `mean_score_time`, `std_score_time`, `param_{name}` for each setting, `params` (the
    /// combination as JSON), `split{i}_test_score` for each fold, `mean_test_score`,
    /// `std_test_score` and `rank_test_score` (1 is the best; ties share the best rank).
    /// Empty before `fit`.
    pub cv_results_: Dataset,
    /// The settings of the best combination, by name.
    pub best_params_: Value,
    /// The mean test score of the best combination.
    pub best_score_: f64,
    /// The row of the best combination in `cv_results_`.
    pub best_index_: usize,
    /// The model with the best settings, trained on all the rows; `None` before `fit` or
    /// with `refit(false)`.
    pub best_estimator_: Option<M>,
    /// The folds each combination was scored on.
    pub n_splits_: usize,
}

impl<M: Model + Clone + Send + Sync> GridSearchCV<M> {
    /// A search over the settings of `estimator`, with 5 folds and the model's own score.
    pub fn new(estimator: M) -> Self {
        Self {
            estimator,
            grid: Vec::new(),
            cv: Cv::Folds(5),
            scoring: String::new(),
            refit: true,
            n_jobs: None,
            cv_results_: Dataset::new(DataFrame::empty(), ""),
            best_params_: Value::Null,
            best_score_: f64::NAN,
            best_index_: 0,
            best_estimator_: None,
            n_splits_: 0,
        }
    }

    /// The values to try for the setting `name`, a name of the model's settings:
    /// `.param_grid("max_depth", [2, 4, 8])`, `.param_grid("max_features", ["sqrt", "log2"])`,
    /// `.param_grid("max_depth", [Some(4), None])`. A setting given twice keeps the last
    /// values.
    #[track_caller]
    pub fn param_grid<V: Into<Value>>(
        mut self,
        name: &str,
        values: impl IntoIterator<Item = V>,
    ) -> Self {
        let values: Vec<Value> = values.into_iter().map(Into::into).collect();
        check(!values.is_empty(), "param_grid", name, "one value or more for each setting");
        let params: Value = self.estimator.params();
        if params.pointer(&pointer(name)).is_none() {
            let valid: String = names(&params).join(", ");
            Err::<(), ModelError>(ModelError::InvalidSetting {
                setting: "param_grid",
                value: format!("{name} (the settings of {} are {valid})", self.estimator.name()),
                expected: "a setting of the model",
            })
            .or_fatal();
        }
        self.grid.retain(|(kept, _)| kept != name);
        self.grid.push((name.into(), values));
        self.grid.sort_by(|a, b| a.0.cmp(&b.0));
        self
    }

    /// The folds: a count, 2 or more (default 5), or a splitter such as
    /// `KFold::new().shuffle(true)` ([`Cv`]).
    #[track_caller]
    pub fn cv(mut self, cv: impl Into<Cv>) -> Self {
        let cv: Cv = cv.into();
        cv.check().or_fatal();
        self.cv = cv;
        self
    }

    /// The score that ranks the combinations, a scikit-learn scorer name: `"accuracy"`,
    /// `"balanced_accuracy"`, `"roc_auc"` (a binary target), `"neg_log_loss"`, `"r2"`, `"neg_mean_squared_error"`, `"neg_root_mean_squared_error"`
    /// or `"neg_mean_absolute_error"`. Higher is always better, hence the `neg_` errors.
    /// Default `""`: the model's own score, R² or the accuracy.
    #[track_caller]
    pub fn scoring(mut self, scoring: &str) -> Self {
        check_scoring(scoring).or_fatal();
        self.scoring = scoring.into();
        self
    }

    /// `true` (the default) trains the best combination on all the rows after the search.
    pub fn refit(mut self, refit: bool) -> Self {
        self.refit = refit;
        self
    }

    /// At most this many fits at once, 1 or more. Default: as many as the Polars pool runs (one
    /// per thread). Each fit still spreads its own work over every thread, so `n_jobs(1)` is
    /// scikit-learn's search with `n_jobs=1` around a forest with `n_jobs=-1`: one model in
    /// memory at a time, every core busy. Lower it when the models are big (forests of many deep
    /// trees): the memory of a search is about that many models.
    #[track_caller]
    pub fn n_jobs(mut self, n_jobs: usize) -> Self {
        check(n_jobs >= 1, "n_jobs", n_jobs, "1 fit or more");
        self.n_jobs = Some(n_jobs);
        self
    }

    /// Every combination of the grid, as the model's whole settings, in scikit-learn's order.
    fn candidates(&self) -> Vec<(Value, Value)> {
        let base: Value = self.estimator.params();
        let mut out: Vec<(Value, Value)> = vec![(base, json!({}))];
        for (name, values) in &self.grid {
            out = out
                .into_iter()
                .flat_map(|(params, chosen)| {
                    values.iter().map(move |value| {
                        let (mut params, mut chosen) = (params.clone(), chosen.clone());
                        if let Some(slot) = params.pointer_mut(&pointer(name)) {
                            *slot = value.clone();
                        }
                        chosen[name] = value.clone();
                        (params, chosen)
                    })
                })
                .collect();
        }
        out
    }

    /// The fitted model, or the error of a search that was not refit.
    fn best(&self) -> Result<&M, ModelError> {
        self.best_estimator_.as_ref().ok_or(ModelError::NotFitted {
            model: "GridSearchCV (with refit)",
        })
    }
}

/// The mean and the population standard deviation, like numpy's.
fn mean_std(values: &[f64]) -> (f64, f64) {
    let n: f64 = values.len() as f64;
    let mean: f64 = values.iter().sum::<f64>() / n;
    let var: f64 = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    (mean, var.sqrt())
}

/// The column `param_{name}` of `cv_results_`: whole numbers, numbers, booleans or text, as
/// the values allow; JSON text for anything else.
fn param_column(name: &str, values: Vec<&Value>) -> Column {
    let name: String = format!("param_{name}");
    if values.iter().all(|v| v.is_i64()) {
        let ints: Vec<i64> = values.iter().filter_map(|v| v.as_i64()).collect();
        return Series::new(name.into(), ints).into();
    }
    if values.iter().all(|v| v.is_number()) {
        let floats: Vec<f64> = values.iter().filter_map(|v| v.as_f64()).collect();
        return Series::new(name.into(), floats).into();
    }
    if values.iter().all(|v| v.is_boolean()) {
        let bools: Vec<bool> = values.iter().filter_map(|v| v.as_bool()).collect();
        return Series::new(name.into(), bools).into();
    }
    let text: Vec<Option<String>> = (values.iter())
        .map(|v| match v {
            Value::Null => None,
            Value::String(s) => Some(s.clone()),
            other => Some(other.to_string()),
        })
        .collect();
    Series::new(name.into(), text).into()
}

impl<M: Model + Clone + Send + Sync> Model for GridSearchCV<M> {
    /// Runs the search, then (with `refit`) trains the best combination on every row.
    fn try_train(&mut self, dataset: &Dataset, target: &str) -> Result<(), ModelError> {
        self.cv.check()?;
        check_scoring(&self.scoring)?;
        // Left out once here, so the folds and the refit train on the same columns.
        let reduced: Option<Dataset> = trainable(dataset, target)?;
        let dataset: &Dataset = reduced.as_ref().unwrap_or(dataset);
        let folds: Vec<(Dataset, Dataset)> =
            self.cv.datasets(dataset, target, self.estimator.is_classifier())?;
        let candidates: Vec<(Value, Value)> = self.candidates();
        let jobs: Vec<(usize, usize)> = (0..candidates.len())
            .flat_map(|c| (0..folds.len()).map(move |f| (c, f)))
            .collect();
        let (estimator, scoring) = (&self.estimator, self.scoring.as_str());
        let trials: Vec<Trial> = run_jobs(jobs.len(), self.n_jobs, |i| {
            let (c, f) = jobs[i];
            trial(estimator, &candidates[c].0, &folds[f], target, scoring)
        })?;
        let k: usize = folds.len();
        let by_candidate: Vec<&[Trial]> = trials.chunks(k).collect();
        let pick = |get: fn(&Trial) -> f64| -> Vec<(f64, f64)> {
            (by_candidate.iter())
                .map(|trials| mean_std(&trials.iter().map(get).collect::<Vec<f64>>()))
                .collect()
        };
        let (scores, fits, times) = (pick(|t| t.score), pick(|t| t.fit_seconds), pick(|t| t.score_seconds));
        // scikit-learn's rank: 1 for the best mean, and ties share the best rank of their run.
        let ranks: Vec<i64> = (scores.iter())
            .map(|(mean, _)| 1 + scores.iter().filter(|(other, _)| other > mean).count() as i64)
            .collect();
        let best: usize = (0..ranks.len()).find(|c| ranks[*c] == 1).unwrap_or(0);

        let mut columns: Vec<Column> = Vec::new();
        let mut number = |name: &str, values: Vec<f64>| {
            columns.push(Series::new(name.into(), values).into());
        };
        number("mean_fit_time", fits.iter().map(|f| f.0).collect());
        number("std_fit_time", fits.iter().map(|f| f.1).collect());
        number("mean_score_time", times.iter().map(|t| t.0).collect());
        number("std_score_time", times.iter().map(|t| t.1).collect());
        for (name, _) in &self.grid {
            let values: Vec<&Value> = candidates.iter().map(|(_, chosen)| &chosen[name]).collect();
            columns.push(param_column(name, values));
        }
        let params: Vec<String> = candidates.iter().map(|(_, chosen)| chosen.to_string()).collect();
        columns.push(Series::new("params".into(), params).into());
        for split in 0..k {
            let fold: Vec<f64> = by_candidate.iter().map(|trials| trials[split].score).collect();
            columns.push(Series::new(format!("split{split}_test_score").into(), fold).into());
        }
        let mut number = |name: &str, values: Vec<f64>| {
            columns.push(Series::new(name.into(), values).into());
        };
        number("mean_test_score", scores.iter().map(|s| s.0).collect());
        number("std_test_score", scores.iter().map(|s| s.1).collect());
        columns.push(Series::new("rank_test_score".into(), ranks).into());
        self.cv_results_ = Dataset::new(DataFrame::new_infer_height(columns)?, "");
        (self.best_index_, self.best_score_, self.n_splits_) = (best, scores[best].0, k);
        self.best_params_ = candidates[best].1.clone();
        self.best_estimator_ = None;
        if self.refit {
            let mut model: M = self.estimator.clone();
            model.set_run(None);
            model.try_set_params(candidates[best].0.clone())?;
            model.try_fit(dataset, target)?;
            self.best_estimator_ = Some(model);
        }
        Ok(())
    }

    /// The search itself is not kept in the model cache: it only trains. Its best model, the
    /// refit, goes through the cache and into the journal.
    fn try_fit(&mut self, dataset: &Dataset, target: &str) -> Result<(), ModelError> {
        self.try_train(dataset, target)
    }

    fn params(&self) -> Value {
        let grid: Map<String, Value> = (self.grid.iter())
            .map(|(name, values)| (name.clone(), Value::from(values.clone())))
            .collect();
        json!({
            "estimator": { "model": self.estimator.name(), "params": self.estimator.params() },
            "param_grid": grid,
            "cv": self.cv,
            "scoring": self.scoring,
            "refit": self.refit,
            "n_jobs": self.n_jobs,
        })
    }

    fn learned(&self) -> Value {
        json!({
            "best_params_": self.best_params_,
            "best_score_": self.best_score_,
            "best_index_": self.best_index_,
        })
    }

    fn try_load(&mut self, _: Value, _: &[u8], _: &str) -> Result<(), ModelError> {
        Err(bad_learned(
            "a search is not kept in the model cache; load its best model's run".into(),
        ))
    }

    fn try_set_params(&mut self, params: Value) -> Result<(), ModelError> {
        let grid: IndexMap<String, Vec<Value>> =
            serde_json::from_value(params["param_grid"].clone())?;
        let cv: Cv = serde_json::from_value(params["cv"].clone())?;
        let scoring: String = serde_json::from_value(params["scoring"].clone())?;
        let refit: bool = serde_json::from_value(params["refit"].clone())?;
        check_scoring(&scoring)?;
        self.estimator
            .try_set_params(params["estimator"]["params"].clone())?;
        self.grid = grid.into_iter().collect();
        self.grid.sort_by(|a, b| a.0.cmp(&b.0));
        (self.cv, self.scoring, self.refit) = (cv, scoring, refit);
        self.n_jobs = serde_json::from_value(params.get("n_jobs").cloned().unwrap_or_default())?;
        Ok(())
    }

    fn try_evaluate(&self, dataset: &Dataset, target: &str, scoring: &str) -> Result<f64, ModelError> {
        self.best()?.try_evaluate(dataset, target, scoring)
    }

    fn try_predict_method(&self, dataset: &Dataset, method: &str) -> Result<Dataset, ModelError> {
        self.best()?.try_predict_method(dataset, method)
    }

    fn is_classifier(&self) -> bool {
        self.estimator.is_classifier()
    }

    fn run(&self) -> Option<&CachedRun> {
        self.best_estimator_.as_ref().and_then(Model::run)
    }

    fn set_run(&mut self, run: Option<CachedRun>) {
        if let Some(best) = &mut self.best_estimator_ {
            best.set_run(run);
        }
    }

    fn name(&self) -> &'static str {
        "GridSearchCV"
    }
}

impl<M: Regressor + Clone + Send + Sync> Regressor for GridSearchCV<M> {
    /// The best model's predictions.
    fn try_predict(&self, dataset: &Dataset) -> Result<Series, ModelError> {
        self.best()?.try_predict(dataset)
    }

    /// The best model's score by `scoring`, like scikit-learn's: its own `score` (R², recorded
    /// in the cache) when `scoring` is `""`.
    fn try_score(&self, dataset: &Dataset, target: &str) -> Result<f64, ModelError> {
        match self.scoring.as_str() {
            "" => self.best()?.try_score(dataset, target),
            scoring => self.best()?.try_evaluate(dataset, target, scoring),
        }
    }
}

impl<M: Classifier + Clone + Send + Sync> Classifier for GridSearchCV<M> {
    /// The best model's classes.
    fn try_predict(&self, dataset: &Dataset) -> Result<Series, ModelError> {
        self.best()?.try_predict(dataset)
    }

    /// The best model's probabilities.
    fn try_predict_proba(&self, dataset: &Dataset) -> Result<Dataset, ModelError> {
        self.best()?.try_predict_proba(dataset)
    }

    /// The best model's score by `scoring`, like scikit-learn's: its own `score` (the
    /// accuracy, recorded in the cache) when `scoring` is `""`.
    fn try_score(&self, dataset: &Dataset, target: &str) -> Result<f64, ModelError> {
        match self.scoring.as_str() {
            "" => self.best()?.try_score(dataset, target),
            scoring => self.best()?.try_evaluate(dataset, target, scoring),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{DecisionTreeClassifier, DecisionTreeRegressor, ElasticNet, tree::tests::table};

    fn assert_close(got: &[f64], want: &[f64], eps: f64) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < eps, "{got:?}, want {want:?}");
        }
    }

    #[test]
    fn the_folds_are_scikit_learn_s() {
        let ds: Dataset = table();
        // KFold(4) of 10 rows: 3, 3, 2, 2.
        let small: Dataset = Dataset::new(ds.dataframe().slice(0, 10), "");
        let plain: Vec<Vec<polars::prelude::IdxSize>> = Cv::from(4).test_rows(&small, "y", false).unwrap();
        assert_eq!(plain, [vec![0, 1, 2], vec![3, 4, 5], vec![6, 7], vec![8, 9]]);
        // StratifiedKFold(5): each fold has 1/5 of each class, and the classes keep their
        // order of rows.
        let folds: Vec<Vec<polars::prelude::IdxSize>> = Cv::from(5).test_rows(&ds, "class", true).unwrap();
        let class: Vec<f64> = ds.to_list("class");
        for fold in &folds {
            let counts: Vec<usize> = (0..3)
                .map(|c| fold.iter().filter(|r| class[**r as usize] == f64::from(c)).count())
                .collect();
            assert!(counts.iter().zip([114, 159, 127]).all(|(n, all)| n * 5 >= all - 4 && n * 5 <= all + 4), "{counts:?}");
            assert!(fold.windows(2).all(|w| w[0] < w[1]));
        }
        let all: usize = folds.iter().map(Vec::len).sum();
        assert_eq!(all, 400);
    }

    #[test]
    fn cross_val_score_matches_scikit_learn() {
        // scikit-learn 1.9.1 (analysis/2026-11-17/boosting/parity.py).
        let ds: Dataset = table();
        let tree = DecisionTreeClassifier::new().max_depth(3);
        let scores: Vec<f64> = cross_val_score(&tree, &ds.drop(&["y"]), "class", 5, "");
        assert_close(&scores, &[0.675, 0.675, 0.7125, 0.675, 0.7125], 1e-12);
        let tree = DecisionTreeRegressor::new().max_depth(3);
        let scores: Vec<f64> = cross_val_score(&tree, &ds.drop(&["class"]), "y", 4, "r2");
        let want = [0.799044245944894, 0.7920710902940862, 0.7543263436561034, 0.8071541539866278];
        assert_close(&scores, &want, 1e-9);
    }

    #[test]
    fn a_grid_search_ranks_like_scikit_learn() {
        let ds: Dataset = table().drop(&["class"]);
        let mut search = GridSearchCV::new(ElasticNet::new())
            .param_grid("l1_ratio", [0.2, 0.8])
            .param_grid("alpha", [0.01, 0.1, 1.0]);
        search.fit(&ds, "y");
        let results: &Dataset = &search.cv_results_;
        // The settings by name, the last one changing fastest: alpha, then l1_ratio.
        assert_eq!(results.to_list("param_alpha"), [0.01, 0.01, 0.1, 0.1, 1.0, 1.0]);
        assert_eq!(results.to_list("param_l1_ratio"), [0.2, 0.8, 0.2, 0.8, 0.2, 0.8]);
        let want = [
            0.9879443332933526, 0.9879443714862834, 0.9878985553471263, 0.9879062440977485,
            0.9842764387290763, 0.9841011489902453,
        ];
        assert_close(&results.to_list("mean_test_score"), &want, 1e-9);
        assert_eq!(results.to_list("rank_test_score"), [2.0, 1.0, 4.0, 3.0, 5.0, 6.0]);
        assert_eq!((search.best_index_, search.n_splits_), (1, 5));
        assert_eq!(search.best_params_, json!({"alpha": 0.01, "l1_ratio": 0.8}));
        assert!((search.best_score_ - 0.9879443714862834).abs() < 1e-9);
        let best: &ElasticNet = search.best_estimator_.as_ref().unwrap();
        let mut by_hand: ElasticNet = ElasticNet::new().alpha(0.01).l1_ratio(0.8);
        by_hand.fit(&ds, "y");
        assert_eq!(best.coef_, by_hand.coef_, "the refit trains on every row");
        assert_eq!(search.predict(&ds), by_hand.predict(&ds));

        // Another scorer, another fold count.
        let mut rmse = GridSearchCV::new(ElasticNet::new())
            .param_grid("alpha", [0.01, 1.0])
            .cv(3)
            .scoring("neg_root_mean_squared_error");
        rmse.fit(&ds, "y");
        assert_close(
            &rmse.cv_results_.to_list("mean_test_score"),
            &[-0.9459890855313616, -1.0851047046863698],
            1e-9,
        );
        assert_eq!(rmse.best_params_["alpha"], 0.01);
        assert!(rmse.score(&ds, "y") < 0.0, "score uses the scorer: a negated error");
    }

    #[test]
    fn a_search_reaches_the_settings_of_a_model_inside_another() {
        let ds: Dataset = table().drop(&["y"]);
        let mut search = GridSearchCV::new(crate::models::AdaBoostClassifier::new().n_estimators(3))
            .param_grid("estimator__max_depth", [1, 3])
            .cv(3)
            .scoring("neg_log_loss");
        search.fit(&ds, "class");
        assert_eq!(search.cv_results_.to_list("param_estimator__max_depth"), [1.0, 3.0]);
        let best = search.best_estimator_.as_ref().unwrap();
        let depth = &search.best_params_["estimator__max_depth"];
        assert_eq!(&best.params()["estimator"]["params"]["max_depth"], depth);
        assert!(search.best_score_ < 0.0);
    }

    #[test]
    fn a_setting_the_model_does_not_have_is_named_with_the_ones_it_has() {
        let err = std::panic::catch_unwind(|| {
            GridSearchCV::new(DecisionTreeClassifier::new()).param_grid("max_dept", [2, 3])
        })
        .unwrap_err();
        let message: &String = err.downcast_ref::<String>().unwrap();
        let listed: &str = "max_dept (the settings of DecisionTreeClassifier are criterion, max_bins, max_depth";
        assert!(message.contains(listed), "{message}");
        let ds: Dataset = table().drop(&["class"]);
        let wrong = GridSearchCV::new(DecisionTreeRegressor::new()).scoring("accuracy");
        let mut wrong = wrong;
        let err: String = wrong.try_fit(&ds, "y").unwrap_err().to_string();
        assert!(err.contains("scoring = accuracy is not valid"), "{err}");
    }
    #[test]
    fn a_cap_on_the_fits_at_once_changes_nothing_but_the_memory() {
        let ds: Dataset = table().drop(&["y"]);
        let scores = |search: GridSearchCV<DecisionTreeClassifier>| {
            let mut search = search.param_grid("max_depth", [1, 2, 3, 4]).param_grid("criterion", ["gini", "entropy"]);
            search.try_train(&ds, "class").unwrap();
            (search.cv_results_.to_list("mean_test_score"), search.best_params_)
        };
        let all = scores(GridSearchCV::new(DecisionTreeClassifier::new()));
        assert_eq!(scores(GridSearchCV::new(DecisionTreeClassifier::new()).n_jobs(1)), all);
        assert_eq!(scores(GridSearchCV::new(DecisionTreeClassifier::new()).n_jobs(2)), all);
        let params: Value = GridSearchCV::new(DecisionTreeClassifier::new()).n_jobs(2).params();
        assert_eq!(params["n_jobs"], 2);
    }

}
