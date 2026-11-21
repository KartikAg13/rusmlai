//! `CatBoostRegressor` / `CatBoostClassifier` against `bench-python/catboosting.py` (CatBoost
//! 1.2 on the same rows and settings): 200 rounds of depth 6, every other setting CatBoost's
//! default (MVS or Bayesian bootstrap, noise on the scores, text columns as categories). S6E1
//! (630,000 rows, 4 numbers, 7 text columns, a number target) and S6E2 (630,000 rows, 13
//! numbers, a text target); the rows whose `id` ends in 0 or 5 are held out and scored.

mod common;

use std::{hint::black_box, path::PathBuf, time::Duration};

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use rusmlai::polars::prelude::*;
use rusmlai::prelude::*;

const ROUNDS: usize = 200;

/// The training rows and the held-out rows of a Kaggle file, without `id`.
fn split(name: &str) -> (DataFrame, Dataset) {
    let csv: String = format!("{}/tests/fixtures/{name}/train.csv", env!("CARGO_MANIFEST_DIR"));
    let frame: DataFrame = common::load(&PathBuf::from(csv));
    let part = |held: bool| {
        let keep = (col("id") % lit(5)).eq(lit(0)).eq(lit(held));
        frame.clone().lazy().filter(keep).drop(Selector::ByName { names: ["id".into()].into(), strict: true }).collect().unwrap()
    };
    (part(false), Dataset::new(part(true), ""))
}

fn seconds() -> Duration {
    let seconds: Option<u64> = std::env::var("RUSMLAI_BENCH_SECONDS").ok().and_then(|s| s.parse().ok());
    Duration::from_secs(seconds.unwrap_or(60))
}

fn catboost(c: &mut Criterion) {
    let (train1, test1): (DataFrame, Dataset) = split("s6e1");
    let (train2, test2): (DataFrame, Dataset) = split("s6e2");
    // One timed fit each: `--list` stops after them, so a one-core run needs no 10 samples.
    let start = std::time::Instant::now();
    let mut regressor = CatBoostRegressor::new().iterations(ROUNDS);
    regressor.fit(&Dataset::new(train1.clone(), ""), "exam_score");
    let middle = std::time::Instant::now();
    let mut classifier = CatBoostClassifier::new().iterations(ROUNDS);
    classifier.fit(&Dataset::new(train2.clone(), ""), "Heart Disease");
    eprintln!(
        "one fit: CatBoostRegressor {:.2} s, CatBoostClassifier {:.2} s",
        (middle - start).as_secs_f64(),
        middle.elapsed().as_secs_f64()
    );
    let y: Series = test2.dataframe().column("Heart Disease").unwrap().as_materialized_series().clone();
    let p: Series = classifier.predict_proba(&test2).dataframe().column("Presence").unwrap().as_materialized_series().clone();
    eprintln!(
        "CatBoostRegressor: held-out r2 {:.6}, rate {}; CatBoostClassifier: held-out accuracy {:.6}, roc_auc {:.6}, rate {}",
        regressor.score(&test1, "exam_score"),
        regressor.learning_rate_(),
        classifier.score(&test2, "Heart Disease"),
        roc_auc_score(&y, &p),
        classifier.learning_rate_(),
    );

    let mut g = c.benchmark_group("catboost");
    g.sample_size(10).measurement_time(seconds());
    g.bench_function("fit/CatBoostRegressor", |b| {
        b.iter_batched(
            || Dataset::new(train1.clone(), ""),
            |ds| {
                let mut model = CatBoostRegressor::new().iterations(ROUNDS);
                model.fit(&ds, "exam_score");
                black_box(model)
            },
            BatchSize::PerIteration,
        )
    });
    g.bench_function("fit/CatBoostClassifier", |b| {
        b.iter_batched(
            || Dataset::new(train2.clone(), ""),
            |ds| {
                let mut model = CatBoostClassifier::new().iterations(ROUNDS);
                model.fit(&ds, "Heart Disease");
                black_box(model)
            },
            BatchSize::PerIteration,
        )
    });
    g.bench_function("predict/CatBoostRegressor", |b| b.iter(|| black_box(regressor.predict(&test1))));
    g.bench_function("predict_proba/CatBoostClassifier", |b| b.iter(|| black_box(classifier.predict_proba(&test2))));
    g.finish();
}

criterion_group!(benches, catboost);
criterion_main!(benches);
