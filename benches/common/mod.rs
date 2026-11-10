#![allow(dead_code)]

use std::{
    fs,
    path::{Path, PathBuf},
};

use polars::frame::DataFrame;
use rusmlai::{
    dataset::reader::Reader,
    metadata::{
        constants::{DATASET_CACHE, PLOTS_CACHE},
        utils::cache_path,
    },
};

/// CSV every bench runs on: the Kaggle S6E1 file (630,000 rows x 13 columns, local only).
/// Override with `RUSMLAI_BENCH_CSV=/path/to/file.csv`; the benches name its columns.
pub fn source_csv() -> PathBuf {
    std::env::var_os("RUSMLAI_BENCH_CSV")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/s6e1/train.csv"
            ))
        })
}

/// Copy of the CSV under `target/tmp/<name>/`, so cache files never touch the real fixture.
pub fn scratch_csv(name: &str) -> PathBuf {
    let dir: PathBuf = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let csv: PathBuf = dir.join("train.csv");
    fs::copy(source_csv(), &csv).expect("bench CSV missing: set RUSMLAI_BENCH_CSV");
    csv
}

pub fn load(csv: &Path) -> DataFrame {
    Reader::new(csv).finish().unwrap()
}

pub fn remove_caches(csv: &Path) {
    let _ = fs::remove_file(cache_path(csv, DATASET_CACHE));
    let _ = fs::remove_file(cache_path(csv, PLOTS_CACHE));
}
