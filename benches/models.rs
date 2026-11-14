//! `src/models`: `LinearRegression`, plain and as Ridge, on the S6E1 design matrix (630,000 rows, 30
//! features after `StandardScaler` and `OneHotEncoder`), and `LogisticRegression` on S6E2 (630,000
//! rows, 13 scaled features): the matrices that `bench-python/models.py` and
//! `bench-python/linear.py` give scikit-learn. The `cache_*` benches time the model cache.

mod common;

use std::{fs, hint::black_box, path::PathBuf, time::Duration};

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use rusmlai::{metadata::models::ModelCache, prelude::*, steps};

const NUMERIC: [&str; 4] = ["age", "study_hours", "class_attendance", "sleep_hours"];
const TEXT: [&str; 7] = [
    "gender",
    "course",
    "internet_access",
    "sleep_quality",
    "study_method",
    "facility_rating",
    "exam_difficulty",
];
const TARGET: &str = "exam_score";

/// scikit-learn's matrix in `bench-python/models.py`: scaled numbers and one-hot text.
fn preprocessing() -> ColumnTransformer {
    ColumnTransformer::new()
        .add("num", &NUMERIC, steps![StandardScaler::new()])
        .add("cat", &TEXT, steps![OneHotEncoder::new()])
}

fn linear(c: &mut Criterion) {
    let csv: PathBuf = common::scratch_csv("models");
    let ds: Dataset = read_csv(&csv).drop(&["id"]);
    // The 30 features with the target passed through: the input a model's fit gets. It keeps
    // the CSV's path, so its fits go through the model cache. `memory` is the same table with
    // no CSV, so no cache: the training alone, like scikit-learn's fit.
    let features: Dataset = preprocessing().remainder("passthrough").fit_transform(&ds);
    assert_eq!((features.height(), features.width()), (630_000, 31));
    let memory: Dataset = Dataset::new(features.dataframe.clone(), "");
    let ds_memory: Dataset = Dataset::new(ds.dataframe.clone(), "");
    let cache: PathBuf = ModelCache::of(&features).unwrap().path;
    let no_cache = || {
        let _ = fs::remove_file(&cache);
    };

    let mut g = c.benchmark_group("linear");
    g.sample_size(20).measurement_time(Duration::from_secs(10));
    g.bench_function("fit/LinearRegression", |b| {
        b.iter(|| {
            let mut model: LinearRegression = LinearRegression::new();
            model.fit(&memory, TARGET);
            black_box(model)
        })
    });
    // A miss: the data hash, the training and one appended line.
    g.bench_function("fit/LinearRegression/cache_miss", |b| {
        b.iter_batched(
            no_cache,
            |()| {
                let mut model: LinearRegression = LinearRegression::new();
                model.fit(&features, TARGET);
                black_box(model)
            },
            BatchSize::PerIteration,
        )
    });
    // A hit: the data hash, reading the file and loading what the miss learned.
    g.bench_function("fit/LinearRegression/cache_hit", |b| {
        b.iter(|| {
            let mut model: LinearRegression = LinearRegression::new();
            model.fit(&features, TARGET);
            black_box(model)
        })
    });
    g.bench_function("fit/Ridge", |b| {
        b.iter(|| {
            // Ridge(alpha=1) on 630,000 rows.
            let mut model: LinearRegression = LinearRegression::new().alpha(1.0 / 630_000.0);
            model.fit(&memory, TARGET);
            black_box(model)
        })
    });
    let mut fitted: LinearRegression = LinearRegression::new();
    fitted.fit(&memory, TARGET);
    g.bench_function("predict/LinearRegression", |b| {
        b.iter(|| black_box(fitted.predict(&memory)))
    });
    g.bench_function("score/LinearRegression", |b| {
        b.iter(|| black_box(fitted.score(&memory, TARGET)))
    });
    // The score plus the data hash, one appended line and the journal rewrite. Each iteration
    // starts from the cache file with the one fit line, so the journal it writes is the same.
    let mut cached: LinearRegression = LinearRegression::new();
    cached.fit(&features, TARGET);
    let fit_line: String = fs::read_to_string(&cache).unwrap();
    g.bench_function("score/LinearRegression/cached", |b| {
        b.iter_batched(
            || fs::write(&cache, &fit_line).unwrap(),
            |()| black_box(cached.score(&features, TARGET)),
            BatchSize::PerIteration,
        )
    });
    // From the loaded CSV: the preprocessing fits from the cached stats, then the model.
    g.bench_function("pipeline_fit/LinearRegression", |b| {
        b.iter(|| {
            let mut pipe =
                Pipeline::new(preprocessing(), ModelStacker::new(LinearRegression::new()));
            pipe.fit(&ds_memory, TARGET);
            black_box(pipe)
        })
    });
    // The journal with 50 runs, each fitted and scored once: what every fit and score rewrites
    // after a search of 50 settings.
    no_cache();
    for k in 0..50 {
        let mut model: LinearRegression = LinearRegression::new().alpha(f64::from(k) * 1e-6);
        model.fit(&features, TARGET);
        model.score(&features, TARGET);
    }
    g.bench_function("journal/50_runs", |b| {
        b.iter(|| black_box(features.try_journal().unwrap()))
    });
    g.bench_function("pipeline_fit/LinearRegression/cache_miss", |b| {
        b.iter_batched(
            no_cache,
            |()| {
                let mut pipe =
                    Pipeline::new(preprocessing(), ModelStacker::new(LinearRegression::new()));
                pipe.fit(&ds, TARGET);
                black_box(pipe)
            },
            BatchSize::PerIteration,
        )
    });
    g.finish();
}

/// S6E2's 13 features, all numeric.
const S6E2: [&str; 13] = [
    "Age",
    "Sex",
    "Chest pain type",
    "BP",
    "Cholesterol",
    "FBS over 120",
    "EKG results",
    "Max HR",
    "Exercise angina",
    "ST depression",
    "Slope of ST",
    "Number of vessels fluro",
    "Thallium",
];

fn logistic(c: &mut Criterion) {
    // Made in memory, so nothing writes into the fixture's caches.
    let csv: PathBuf = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/s6e2/train.csv"
    ));
    let raw: Dataset = Dataset::new(common::load(&csv), "").drop(&["id"]);
    let features: Dataset = ColumnTransformer::new()
        .add("num", &S6E2, steps![StandardScaler::new()])
        .remainder("passthrough")
        .fit_transform(&raw);
    assert_eq!((features.height(), features.width()), (630_000, 14));

    let mut g = c.benchmark_group("logistic");
    g.sample_size(20).measurement_time(Duration::from_secs(10));
    g.bench_function("fit/LogisticRegression", |b| {
        b.iter(|| {
            let mut model: LogisticRegression = LogisticRegression::new();
            model.fit(&features, "Heart Disease");
            black_box(model)
        })
    });
    let mut fitted: LogisticRegression = LogisticRegression::new();
    fitted.fit(&features, "Heart Disease");
    g.bench_function("predict_proba/LogisticRegression", |b| {
        b.iter(|| black_box(fitted.predict_proba(&features)))
    });
    g.bench_function("predict/LogisticRegression", |b| {
        b.iter(|| black_box(fitted.predict(&features)))
    });
    // predict, predict_proba, the accuracy and the log loss, on a text target.
    g.bench_function("score/LogisticRegression", |b| {
        b.iter(|| black_box(fitted.score(&features, "Heart Disease")))
    });
    g.finish();
}

criterion_group!(benches, linear, logistic);
criterion_main!(benches);
