//! The rusmlai side of the boosting showdown (analysis/2026-11-27/showdown-s6e3/):
//! `GridSearchCV` over `HistGradientBoostingClassifier` on Kaggle S6E3 (telco churn, local file
//! only), text columns as categories, scored by ROC AUC.
//!
//!     RUSMLAI_CACHE=off cargo run --profile bench --example showdown_hgb -- grid.json out.json
//!
//! `grid.json` holds the grid (shared with showdown.py); `out.json` gets the timings and results.

use std::time::Instant;

use rusmlai::prelude::*;
use serde_json::{Value, json};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let grid: Value = serde_json::from_str(&std::fs::read_to_string(&args[1]).unwrap()).unwrap();
    let start: Instant = Instant::now();
    let ds: Dataset = read_csv(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/s6e3/train.csv")).drop(&["id"]);
    let load: f64 = start.elapsed().as_secs_f64();
    eprintln!("loaded {:?} in {load:.1} s", ds.shape());

    // Fixed rounds, so both sides do the same work; one fit at a time, each on every core.
    let mut search = GridSearchCV::new(HistGradientBoostingClassifier::new().early_stopping(false))
        .cv(StratifiedKFold::new().n_splits(5).shuffle(true))
        .scoring("roc_auc")
        .n_jobs(1);
    for (name, values) in grid.as_object().unwrap() {
        search = search.param_grid(name, values.as_array().unwrap().clone());
    }
    let start: Instant = Instant::now();
    search.fit(&ds, "Churn");
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
        "threads": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
    });
    std::fs::write(&args[2], serde_json::to_string_pretty(&out).unwrap()).unwrap();
}
