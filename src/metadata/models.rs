//! [`ModelCache`]: every fit and score on one CSV, in `{csv dir}/metadata/{stem}.models.jsonl`.
//!
//! A fit is keyed by the model, its settings, the target, the features and a hash of the data
//! it saw. The same fit again loads the learned values instead of training, and the journal
//! (`Dataset::journal`) draws every record. The file has no fingerprint line: when the CSV
//! changes, old records stay for the journal, and new fits just never match them.

use std::{
    fs::{self, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use indexmap::IndexMap;
use polars::prelude::{Column, DataType};
use polars_core::runtime::THREAD_POOL;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    dataset::Dataset,
    error::warn,
    metadata::{
        constants::MODELS_CACHE,
        journal,
        utils::{cache_path, print_saved},
    },
    models::{Model, ModelError},
};

/// One `fit` that trained.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitRecord {
    /// The key: model, settings, target, features and data.
    pub id: String,
    /// The model, scikit-learn's class name.
    pub model: String,
    /// Its settings, by scikit-learn name.
    pub params: Value,
    /// The target column.
    pub target: String,
    /// The feature columns.
    pub features: Vec<String>,
    /// Rows it trained on.
    pub rows: usize,
    /// The hash of the data it trained on, in hex.
    pub data: String,
    /// What it learned (`coef_`, `intercept_`, ...).
    pub learned: Value,
    /// The training loss per round, for models that train in rounds.
    pub loss_curve: Option<Vec<f64>>,
    /// The validation loss per round, for models that hold rows out.
    pub validation_loss_curve: Option<Vec<f64>>,
    /// How long the training took.
    pub fit_seconds: f64,
    /// When, in seconds since 1970.
    pub time: u64,
}

/// One `score` of a fitted model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreRecord {
    /// The [`FitRecord::id`] of the model.
    pub id: String,
    /// Every metric, by scikit-learn name: `r2`, `mse`, `rmse`, `mae`.
    pub metrics: IndexMap<String, f64>,
    /// Rows scored.
    pub rows: usize,
    /// The hash of the data scored, in hex: the fit's own hash means a training score.
    pub data: String,
    /// When, in seconds since 1970.
    pub time: u64,
}

/// One line of the cache file.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Entry {
    /// A training run.
    Fit(FitRecord),
    /// A score of a training run.
    Score(ScoreRecord),
}

/// The model cache of one CSV.
///
/// ```
/// use std::path::Path;
/// use rusmlai::{metadata::models::ModelCache, prelude::*};
/// let ds = read_csv("tests/fixtures/data/students.csv");
/// let cache = ModelCache::of(&ds).unwrap();
/// assert_eq!(cache.path, Path::new("tests/fixtures/data/metadata/students.models.jsonl"));
/// assert!(ModelCache::of(&Dataset::new(ds.dataframe.clone(), "")).is_none());
/// ```
#[derive(Debug, Clone)]
pub struct ModelCache {
    /// `{csv dir}/metadata/{stem}.models.jsonl`.
    pub path: PathBuf,
    /// The CSV it belongs to; the journal goes next to it.
    pub csv: PathBuf,
}

impl ModelCache {
    /// The cache of the CSV that `dataset` came from; `None` for a dataset made in memory.
    pub fn of(dataset: &Dataset) -> Option<Self> {
        let csv: &Path = Path::new(dataset.metadata.filepath.as_deref()?);
        (!csv.as_os_str().is_empty()).then(|| Self {
            path: cache_path(csv, MODELS_CACHE),
            csv: csv.to_path_buf(),
        })
    }

    /// Every record, oldest first. A missing file is empty; a line that doesn't parse (from an
    /// older version, or cut by a crash) is skipped.
    pub fn entries(&self) -> Vec<Entry> {
        fs::read_to_string(&self.path)
            .map(|text| {
                text.lines()
                    .filter_map(|line| serde_json::from_str(line).ok())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The fit with this id, if one trained before. Only a line that holds the id is parsed, so
    /// a lookup reads the file but does not decode what every other run learned.
    fn find(&self, id: &str) -> Option<FitRecord> {
        let text: String = fs::read_to_string(&self.path).ok()?;
        text.lines()
            .rev()
            .filter(|line| line.contains(id))
            .find_map(|line| match serde_json::from_str(line).ok()? {
                Entry::Fit(fit) if fit.id == id => Some(fit),
                _ => None,
            })
    }

    /// Appends one line, creating the file, then rewrites the journal so it shows the new
    /// line. A failure is only a warning: the model is fitted either way.
    fn append(&self, entry: &Entry) {
        let written: io::Result<()> = serde_json::to_string(entry)
            .map_err(io::Error::from)
            .and_then(|line| {
                fs::create_dir_all(self.path.parent().unwrap_or(Path::new(".")))?;
                let mut file = OpenOptions::new()
                    .create(true)
                    .read(true)
                    .append(true)
                    .open(&self.path)?;
                // A line cut by a crash has no newline: start a fresh one, so this record is
                // not glued to the broken one and lost with it.
                let mut last: [u8; 1] = *b"\n";
                if file.metadata()?.len() > 0 {
                    file.seek(SeekFrom::End(-1))?;
                    file.read_exact(&mut last)?;
                }
                let start: &str = if last[0] == b'\n' { "" } else { "\n" };
                file.write_all(format!("{start}{line}\n").as_bytes())
            });
        if let Err(e) = written {
            return warn(format_args!(
                "could not update {}: {e}",
                self.path.display()
            ));
        }
        print_saved(&self.path);
        if let Err(e) = journal::write(self) {
            warn(format_args!("the journal was not updated: {e}"));
        }
    }
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// One step of FxHash: fast, and the same in every process and on every machine.
fn mix(hash: u64, word: u64) -> u64 {
    (hash.rotate_left(5) ^ word).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95)
}

/// FxHash of `values` in 8 lanes (value `i` goes to lane `i % 8`), then of the lanes and the
/// rest in order. The lanes don't wait for each other's multiply, and the slice is read
/// without Polars' per-value iterator.
fn hash_values(hash: u64, values: &[f64]) -> u64 {
    // +0.0 and -0.0 are the same number for a model.
    let bits = |v: f64| (v + 0.0).to_bits();
    let mut lanes: [u64; 8] = [hash; 8];
    let (chunks, rest): (&[[f64; 8]], &[f64]) = values.as_chunks::<8>();
    for chunk in chunks {
        for (lane, value) in lanes.iter_mut().zip(chunk) {
            *lane = mix(*lane, bits(*value));
        }
    }
    let hash: u64 = lanes.iter().fold(hash, |h, lane| mix(h, *lane));
    rest.iter().fold(hash, |h, v| mix(h, bits(*v)))
}

/// A hash of every value of `column`, in row order. A null hashes apart from every value.
fn column_hash(column: &Column) -> Result<u64, ModelError> {
    const NULL: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut hash: u64 = mix(0, column.len() as u64);
    let dtype: &DataType = column.dtype();
    if dtype.is_primitive_numeric() || dtype.is_bool() {
        let values = column
            .as_materialized_series()
            .cast(&DataType::Float64)?
            .rechunk();
        let values = values.f64()?;
        match values.cont_slice() {
            Ok(slice) => hash = hash_values(hash, slice),
            // Nulls: one step per value.
            Err(_) => {
                for value in values.iter() {
                    hash = mix(hash, value.map_or(NULL, |v| (v + 0.0).to_bits()));
                }
            }
        }
    } else {
        let values = column.cast(&DataType::String)?;
        for value in values.str()?.iter() {
            let Some(text) = value else {
                hash = mix(hash, NULL);
                continue;
            };
            let (words, rest) = text.as_bytes().as_chunks::<8>();
            for word in words {
                hash = mix(hash, u64::from_le_bytes(*word));
            }
            let mut last: [u8; 8] = [0; 8];
            last[..rest.len()].copy_from_slice(rest);
            hash = mix(mix(hash, u64::from_le_bytes(last)), text.len() as u64);
        }
    }
    Ok(hash)
}

/// A hash of the whole table: column names and values, in order. One column per task on the
/// Polars pool.
// ponytail: still one extra pass over the data (`benches/models.rs`, `cache_miss` against the
// plain fit). Fold it into `DesignMatrix`'s checking pass if that bench shows it matters.
pub(crate) fn data_hash(dataset: &Dataset) -> Result<String, ModelError> {
    let columns: &[Column] = dataset.dataframe.columns();
    let hashes: Vec<u64> = THREAD_POOL.install(|| {
        columns
            .par_iter()
            .map(column_hash)
            .collect::<Result<_, _>>()
    })?;
    let mut hash: u64 = 0;
    for (column, value) in columns.iter().zip(hashes) {
        for byte in column.name().as_bytes() {
            hash = mix(hash, u64::from(*byte));
        }
        hash = mix(hash, value);
    }
    Ok(format!("{hash:016x}"))
}

/// The key of a fit: the rusmlai version, the model, its settings, target, features and data.
/// The version is in it so that a release with a fixed solver never loads an older answer.
fn run_id(model: &str, params: &Value, target: &str, features: &[String], data: &str) -> String {
    let key: String = format!(
        "{}\0{model}\0{params}\0{target}\0{}\0{data}",
        env!("CARGO_PKG_VERSION"),
        features.join("\0")
    );
    let hash: u64 = key.bytes().fold(0, |h, b| mix(h, u64::from(b)));
    format!("{hash:016x}")
}

/// `fit` through the cache: a fit seen before loads its learned values, any other trains and
/// is recorded. A dataset made in memory has no cache and always trains.
pub(crate) fn fit<R: Model + ?Sized>(
    model: &mut R,
    dataset: &Dataset,
    target: &str,
) -> Result<(), ModelError> {
    let Some(cache) = ModelCache::of(dataset) else {
        model.set_run(None);
        return model.try_train(dataset, target);
    };
    let features: Vec<String> = dataset
        .columns()
        .into_iter()
        .filter(|c| c != target)
        .collect();
    let data: String = data_hash(dataset)?;
    let (name, params) = (model.name(), model.params());
    let id: String = run_id(name, &params, target, &features, &data);
    if let Some(record) = cache.find(&id)
        && model.try_load(record.learned, target).is_ok()
    {
        println!("[rusmlai] {name} loaded from {}", cache.path.display());
        model.set_run(Some(id));
        return Ok(());
    }
    let start: Instant = Instant::now();
    model.try_train(dataset, target)?;
    let fit_seconds: f64 = start.elapsed().as_secs_f64();
    cache.append(&Entry::Fit(FitRecord {
        id: id.clone(),
        model: name.into(),
        params,
        target: target.into(),
        features,
        rows: dataset.height(),
        data,
        learned: model.learned(),
        loss_curve: model.loss_curve(),
        validation_loss_curve: model.validation_loss_curve(),
        fit_seconds,
        time: now(),
    }));
    model.set_run(Some(id));
    Ok(())
}

/// The fit records of the cache, oldest first: run `r01` is the first.
pub(crate) fn fits(entries: Vec<Entry>) -> Vec<FitRecord> {
    entries
        .into_iter()
        .filter_map(|e| match e {
            Entry::Fit(fit) => Some(fit),
            Entry::Score(_) => None,
        })
        .collect()
}

/// The fit record of `run` (`"r03"` or an id) in the cache of `dataset`'s CSV.
pub(crate) fn find_run(dataset: &Dataset, run: &str) -> Result<FitRecord, ModelError> {
    let cache: ModelCache = ModelCache::of(dataset).ok_or(ModelError::NoCache)?;
    let fits: Vec<FitRecord> = fits(cache.entries());
    let runs: usize = fits.len();
    let place: Option<usize> = run
        .strip_prefix('r')
        .and_then(|n| n.parse::<usize>().ok())
        .filter(|n| (1..=runs).contains(n))
        .map(|n| n - 1);
    let found: Option<FitRecord> = match place {
        Some(i) => fits.into_iter().nth(i),
        None => fits.into_iter().rev().find(|f| f.id == run),
    };
    found.ok_or_else(|| ModelError::RunNotFound {
        run: run.into(),
        runs,
    })
}

/// Loads the settings and the learned values of `run` into `model`, after checking that it is
/// the same kind of model and that `dataset` has the run's features.
pub(crate) fn load_run<R: Model + ?Sized>(
    model: &mut R,
    dataset: &Dataset,
    run: &str,
) -> Result<(), ModelError> {
    let record: FitRecord = find_run(dataset, run)?;
    let mismatch = |reason: String| ModelError::RunMismatch {
        run: run.into(),
        reason,
    };
    if record.model != model.name() {
        return Err(mismatch(format!(
            "it is a {}, not a {}",
            record.model,
            model.name()
        )));
    }
    let columns: Vec<String> = dataset.columns();
    let missing: Vec<&String> = record
        .features
        .iter()
        .filter(|f| !columns.contains(f))
        .collect();
    if !missing.is_empty() {
        return Err(mismatch(format!(
            "it trained on columns this dataset doesn't have: {missing:?}"
        )));
    }
    model.try_set_params(record.params)?;
    model.try_load(record.learned, &record.target)?;
    model.set_run(Some(record.id));
    Ok(())
}

/// Records a score of a cached fit; does nothing for a model or dataset without a cache.
pub(crate) fn score<R: Model + ?Sized>(
    model: &R,
    dataset: &Dataset,
    metrics: IndexMap<String, f64>,
) {
    let (Some(cache), Some(id)) = (ModelCache::of(dataset), model.run()) else {
        return;
    };
    match data_hash(dataset) {
        Ok(data) => cache.append(&Entry::Score(ScoreRecord {
            id: id.into(),
            metrics,
            rows: dataset.height(),
            data,
            time: now(),
        })),
        Err(e) => warn(format_args!("the score was not recorded: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use polars::df;
    use serde_json::json;

    use super::*;
    use crate::{
        dataset::read_csv,
        models::{Classifier, LinearRegression, LogisticRegression, Regressor},
        pipeline::{ModelStacker, Pipeline},
        preprocess::{ColumnTransformer, StandardScaler},
        steps,
    };

    /// A small CSV (`age`, `cgpa`) in its own folder under `target/tmp/`, with no cache yet.
    fn csv(name: &str) -> PathBuf {
        let dir: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target/tmp/model_cache")
            .join(name);
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path: PathBuf = dir.join("students.csv");
        let rows: &str = "age,cgpa\n19,8.1\n20,7.4\n21,6.9\n19,8.8\n22,7.0\n20,7.7\n";
        fs::write(&path, rows).unwrap();
        path
    }

    /// A model that counts how often it trains.
    #[derive(Debug, Default)]
    struct Counting {
        trained: Cell<u32>,
        setting: u32,
        learned: f64,
        run: Option<String>,
    }

    impl Regressor for Counting {
        fn try_predict(&self, dataset: &Dataset) -> Result<polars::prelude::Series, ModelError> {
            Ok(dataset.column("cgpa")?.as_materialized_series().clone())
        }
    }

    impl Model for Counting {
        fn try_train(&mut self, _: &Dataset, _: &str) -> Result<(), ModelError> {
            self.trained.set(self.trained.get() + 1);
            self.learned = 7.0;
            Ok(())
        }
        fn params(&self) -> Value {
            json!({ "setting": self.setting })
        }
        fn learned(&self) -> Value {
            json!({ "learned": self.learned })
        }
        fn try_load(&mut self, learned: Value, _: &str) -> Result<(), ModelError> {
            self.learned = learned["learned"].as_f64().unwrap_or_default();
            Ok(())
        }
        fn try_set_params(&mut self, params: Value) -> Result<(), ModelError> {
            self.setting = params["setting"].as_u64().unwrap_or_default() as u32;
            Ok(())
        }
        fn run(&self) -> Option<&str> {
            self.run.as_deref()
        }
        fn set_run(&mut self, run: Option<String>) {
            self.run = run;
        }
    }

    #[test]
    fn a_fit_seen_before_loads_and_any_change_trains() {
        let ds: Dataset = read_csv(csv("hit"));
        let mut model: Counting = Counting::default();
        model.fit(&ds, "cgpa");
        model.learned = 0.0;
        model.fit(&ds, "cgpa");
        assert_eq!(
            (model.trained.get(), model.learned),
            (1, 7.0),
            "the second fit loads"
        );
        assert_eq!(model.name(), "Counting");

        model.setting = 1; // another setting
        model.fit(&ds, "cgpa");
        let fewer_rows: Dataset = Dataset {
            dataframe: ds.dataframe.slice(0, 3),
            metadata: ds.metadata.clone(),
        };
        model.fit(&fewer_rows, "cgpa");
        model.fit(&ds, "age"); // another target
        assert_eq!(model.trained.get(), 4);

        model.score(&ds, "cgpa");
        let entries: Vec<Entry> = ModelCache::of(&ds).unwrap().entries();
        let fits: usize = entries
            .iter()
            .filter(|e| matches!(e, Entry::Fit(_)))
            .count();
        assert_eq!((fits, entries.len()), (4, 5));
        let Some(Entry::Score(score)) = entries.last() else {
            panic!("{entries:?}")
        };
        assert_eq!(score.metrics["r2"], 1.0);
        assert_eq!(
            score.metrics.keys().collect::<Vec<_>>(),
            ["r2", "mse", "rmse", "mae"]
        );
    }

    #[test]
    fn a_real_model_round_trips_and_a_broken_line_is_skipped() {
        let path: PathBuf = csv("round");
        let ds: Dataset = read_csv(&path);
        let mut trained: LinearRegression = LinearRegression::new().alpha(0.1).l1_ratio(0.5);
        trained.fit(&ds, "cgpa");
        let cache: ModelCache = ModelCache::of(&ds).unwrap();
        let mut file = OpenOptions::new().append(true).open(&cache.path).unwrap();
        file.write_all(b"{\"kind\": \"fit\", \"cut").unwrap();
        let mut loaded: LinearRegression = LinearRegression::new().alpha(0.1).l1_ratio(0.5);
        loaded.fit(&ds, "cgpa");
        assert_eq!(loaded.coef_, trained.coef_);
        assert_eq!(loaded.intercept_, trained.intercept_);
        assert_eq!(loaded.n_iter_, trained.n_iter_);
        assert_eq!(loaded.predict(&ds), trained.predict(&ds));
        assert_eq!(cache.entries().len(), 1);
        drop(file);

        // Back to an older run: its settings and weights, whatever the model was built with.
        let mut plain: LinearRegression = LinearRegression::new();
        plain.fit(&ds, "cgpa");
        let mut back: LinearRegression = LinearRegression::new();
        back.load_run(&ds, "r01");
        assert_eq!(
            (back.coef_.clone(), back.n_iter_),
            (trained.coef_.clone(), trained.n_iter_)
        );
        assert_eq!(back.predict(&ds), trained.predict(&ds));
        let missing: String = back.try_load_run(&ds, "r09").unwrap_err().to_string();
        assert!(missing.contains("has 2 runs"), "{missing}");
        let mut dummy: Counting = Counting::default();
        let wrong: String = dummy.try_load_run(&ds, "r01").unwrap_err().to_string();
        assert!(
            wrong.contains("it is a LinearRegression, not a Counting"),
            "{wrong}"
        );

        // A dataset made in memory has no cache: it trains and writes nothing.
        let memory: Dataset = Dataset::new(ds.dataframe.clone(), "");
        let mut model: Counting = Counting::default();
        model.fit(&memory, "cgpa");
        model.fit(&memory, "cgpa");
        assert_eq!((model.trained.get(), model.run()), (2, None));
    }

    #[test]
    fn the_hash_sees_values_order_names_and_nulls() {
        let hash = |frame| data_hash(&Dataset::new(frame, "")).unwrap();
        let base: String = hash(df!("x" => [1.0, 2.0], "s" => ["a", "b"]).unwrap());
        assert_eq!(
            base,
            hash(df!("x" => [1.0, 2.0], "s" => ["a", "b"]).unwrap())
        );
        assert_eq!(base, hash(df!("x" => [1, 2], "s" => ["a", "b"]).unwrap()));
        assert_ne!(
            base,
            hash(df!("x" => [2.0, 1.0], "s" => ["a", "b"]).unwrap())
        );
        assert_ne!(
            base,
            hash(df!("y" => [1.0, 2.0], "s" => ["a", "b"]).unwrap())
        );
        assert_ne!(
            base,
            hash(df!("x" => [1.0, 2.0], "s" => ["a", "bb"]).unwrap())
        );
        assert_ne!(
            hash(df!("x" => [Some(0.0)]).unwrap()),
            hash(df!("x" => [None::<f64>]).unwrap())
        );
        // Past 8 values the hash runs in lanes: a swap inside a lane and across lanes shows.
        let long: Vec<f64> = (0..20).map(f64::from).collect();
        let swapped = |i: usize, j: usize| {
            let mut v: Vec<f64> = long.clone();
            v.swap(i, j);
            hash(df!("x" => v).unwrap())
        };
        let base: String = hash(df!("x" => long.clone()).unwrap());
        assert_ne!(base, swapped(0, 8), "same lane");
        assert_ne!(base, swapped(0, 1), "two lanes");
        assert_ne!(base, swapped(15, 17), "a lane and the rest");
        let ints: Vec<i64> = (0..20).collect();
        assert_eq!(
            base,
            hash(df!("x" => ints).unwrap()),
            "ints hash as their floats"
        );
    }

    #[test]
    fn a_classifier_round_trips_with_c_inf_and_its_labels() {
        let dir: PathBuf = csv("classifier").parent().unwrap().to_path_buf();
        let path: PathBuf = dir.join("passed.csv");
        let rows: &str = "hours,passed\n0.5,no\n1.0,no\n1.5,yes\n2.0,no\n2.5,yes\n3.0,yes\n";
        fs::write(&path, rows).unwrap();
        let ds: Dataset = read_csv(&path);
        let mut trained = LogisticRegression::new().C(f64::INFINITY);
        trained.fit(&ds, "passed");
        let mut loaded = LogisticRegression::new().C(f64::INFINITY);
        loaded.fit(&ds, "passed");
        assert_eq!(
            fits(ModelCache::of(&ds).unwrap().entries()).len(),
            1,
            "loaded, not trained"
        );
        assert_eq!(
            (loaded.coef_.clone(), loaded.classes_.clone()),
            (trained.coef_.clone(), trained.classes_.clone())
        );
        assert_eq!(loaded.predict(&ds), trained.predict(&ds));
        let mut back: LogisticRegression = LogisticRegression::new();
        back.load_run(&ds, "r01");
        assert_eq!(back.params()["C"], "inf", "C = inf survives JSON");
        assert_eq!(back.intercept_, trained.intercept_);
    }

    #[test]
    fn a_pipeline_loads_its_model_from_the_cache() {
        let ds: Dataset = read_csv(csv("pipeline"));
        let fit = || {
            let prep: ColumnTransformer =
                ColumnTransformer::new().add("num", &[], steps![StandardScaler::new()]);
            let mut pipe = Pipeline::new(prep, ModelStacker::new(LinearRegression::new()));
            pipe.fit(&ds, "cgpa");
            pipe
        };
        let (first, second) = (fit(), fit());
        let fits: usize = fits(ModelCache::of(&ds).unwrap().entries()).len();
        assert_eq!(fits, 1, "the second pipeline loads the first one's model");
        assert_eq!(second.model.model.coef_, first.model.model.coef_);
        assert_eq!(second.run(), first.run());
        assert!(second.run().is_some());
    }
}
