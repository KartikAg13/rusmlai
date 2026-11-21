//! `Model::save` / `Model::load` against `bench-python/persist.py` (`joblib.dump` /
//! `joblib.load`): a random forest of 50 full trees on S6E1 (text as ordinal codes) and
//! XGBoost-style boosting of 300 rounds, written to and read from the scratch folder.

mod common;

use std::{hint::black_box, path::PathBuf, time::Duration};

use criterion::{Criterion, criterion_group, criterion_main};
use rusmlai::{prelude::*, steps};

const TEXT: [&str; 7] = ["gender", "course", "internet_access", "sleep_quality", "study_method", "facility_rating", "exam_difficulty"];

fn seconds() -> Duration {
    let seconds: Option<u64> = std::env::var("RUSMLAI_BENCH_SECONDS").ok().and_then(|s| s.parse().ok());
    Duration::from_secs(seconds.unwrap_or(10))
}

fn persist(c: &mut Criterion) {
    let csv: String = format!("{}/tests/fixtures/s6e1/train.csv", env!("CARGO_MANIFEST_DIR"));
    let ds: Dataset = Dataset::new(common::load(&PathBuf::from(csv)), "").drop(&["id"]);
    let codes: Dataset = ColumnTransformer::new()
        .add("cat", &TEXT, steps![OrdinalEncoder::new()])
        .remainder("passthrough")
        .fit_transform(&ds);
    let codes: Dataset = Dataset::new(codes.dataframe().clone(), "");
    let mut forest = RandomForestRegressor::new().n_estimators(50);
    forest.try_train(&codes, "exam_score").unwrap();
    let mut boosted = XGBRegressor::new().n_estimators(300);
    boosted.try_train(&codes, "exam_score").unwrap();
    let dir: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
    std::fs::create_dir_all(&dir).unwrap();
    let (forest_file, boosted_file) = (dir.join("forest.model"), dir.join("boosted.model"));
    forest.save(&forest_file);
    boosted.save(&boosted_file);
    eprintln!(
        "forest: {} nodes, {} MB; boosting: {} MB",
        forest.estimators_.iter().map(|t| t.node_count()).sum::<usize>(),
        std::fs::metadata(&forest_file).unwrap().len() / 1_000_000,
        std::fs::metadata(&boosted_file).unwrap().len() / 1_000_000,
    );
    let mut g = c.benchmark_group("persist");
    g.sample_size(10).measurement_time(seconds());
    g.bench_function("save/RandomForestRegressor", |b| b.iter(|| forest.try_save(&forest_file).unwrap()));
    g.bench_function("load/RandomForestRegressor", |b| b.iter(|| black_box(RandomForestRegressor::load(&forest_file))));
    g.bench_function("save/XGBRegressor", |b| b.iter(|| boosted.try_save(&boosted_file).unwrap()));
    g.bench_function("load/XGBRegressor", |b| b.iter(|| black_box(XGBRegressor::load(&boosted_file))));
    g.finish();
}

criterion_group!(benches, persist);
criterion_main!(benches);
