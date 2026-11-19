//! The rusmlai side of the grid search showdown (analysis/2026-11-18/showdown/):
//! `GridSearchCV` over `RandomForestRegressor` on Kaggle S6E1 (local file only).
//!
//!     RUSMLAI_CACHE=off cargo run --release --example showdown -- grid.json out.json
//!
//! `grid.json` holds the grid (shared with showdown.py); `out.json` gets the timings and results.
//! An optional third argument caps the fits that run at once (`GridSearchCV::n_jobs`).

use std::time::Instant;

use rusmlai::{prelude::*, steps};
use serde_json::{Value, json};

const TEXT: [&str; 7] = [
    "gender",
    "course",
    "internet_access",
    "sleep_quality",
    "study_method",
    "facility_rating",
    "exam_difficulty",
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let grid: Value = serde_json::from_str(&std::fs::read_to_string(&args[1]).unwrap()).unwrap();
    let start: Instant = Instant::now();
    let raw: Dataset = read_csv(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/s6e1/train.csv")).drop(&["id"]);
    // The text columns as ordinal codes, as the Python side has them: outside the timing.
    let ds: Dataset = ColumnTransformer::new()
        .add("cat", &TEXT, steps![OrdinalEncoder::new()])
        .remainder("passthrough")
        .verbose_feature_names_out(false)
        .fit_transform(&raw);
    let load: f64 = start.elapsed().as_secs_f64();
    eprintln!("loaded {:?} in {load:.1} s", ds.shape());

    let mut search = GridSearchCV::new(RandomForestRegressor::new())
        .cv(KFold::new().n_splits(5).shuffle(true))
        .scoring("neg_root_mean_squared_error");
    if let Some(jobs) = args.get(3) {
        search = search.n_jobs(jobs.parse().unwrap());
    }
    for (name, values) in grid.as_object().unwrap() {
        search = search.param_grid(name, values.as_array().unwrap().clone());
    }
    let start: Instant = Instant::now();
    search.fit(&ds, "exam_score");
    let seconds: f64 = start.elapsed().as_secs_f64();
    let results: &Dataset = &search.cv_results_;
    let fit_total: f64 = results.to_list("mean_fit_time").iter().sum::<f64>() * search.n_splits_ as f64;
    eprintln!("search + refit: {seconds:.1} s; best {} = {:.6}", search.best_params_, search.best_score_);
    let out: Value = json!({
        "side": "rusmlai",
        "seconds": seconds,
        "load_seconds": load,
        "candidates": results.height(),
        "folds": search.n_splits_,
        "sum_of_fit_seconds": fit_total,
        "best_params": search.best_params_,
        "best_score": search.best_score_,
        "mean_test_score": results.to_list("mean_test_score"),
        "params": results.dataframe().column("params").unwrap().str().unwrap().iter().flatten().collect::<Vec<&str>>(),
        "n_jobs": args.get(3),
        "threads": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
    });
    std::fs::write(&args[2], serde_json::to_string_pretty(&out).unwrap()).unwrap();
}
