//! `read_csv` and its cache files, on copies of the tracked fixture.

mod common;

use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::PathBuf,
    time::{Duration, SystemTime},
};

use common::{cache_file, scratch_csv};
use rusmlai::prelude::*;

fn modified(path: &PathBuf) -> SystemTime {
    fs::metadata(path).unwrap().modified().unwrap()
}

#[test]
fn a_cold_read_writes_the_cache_and_a_warm_read_leaves_it_alone() {
    let csv: PathBuf = scratch_csv("read_cold_warm");
    let (stats, plots): (PathBuf, PathBuf) =
        (cache_file(&csv, ".json"), cache_file(&csv, ".plots.jsonl"));
    let cold: Dataset = read_csv(&csv);
    assert!(stats.is_file() && plots.is_file());
    let (stats_time, plots_time): (SystemTime, SystemTime) = (modified(&stats), modified(&plots));

    let warm: Dataset = read_csv(&csv);
    assert_eq!(
        (modified(&stats), modified(&plots)),
        (stats_time, plots_time)
    );
    assert_eq!(warm.shape(), cold.shape());
    assert_eq!(
        warm.try_numeric_stats("cgpa").unwrap().std,
        cold.try_numeric_stats("cgpa").unwrap().std
    );
    assert_eq!(warm.try_categorical_stats("batch").unwrap().mode, "B13");
}

#[test]
fn a_changed_csv_makes_the_cache_stale() {
    let csv: PathBuf = scratch_csv("read_stale");
    assert_eq!(read_csv(&csv).height(), 12);
    let mut text: String = fs::read_to_string(&csv).unwrap();
    text.push_str("12,New Student,30,9.9,B16,true\n");
    fs::write(&csv, text).unwrap();
    let ds: Dataset = read_csv(&csv);
    assert_eq!(ds.height(), 13);
    assert_eq!(ds.try_numeric_stats("age").unwrap().max, 30.0);
    assert_eq!(ds.try_categorical_stats("batch").unwrap().unique_count, 4);
}

#[test]
fn a_rewrite_of_the_same_size_makes_the_cache_stale() {
    // The size is equal and the write comes within the same second: only a fine mtime tells.
    let csv: PathBuf = scratch_csv("read_same_size");
    assert_eq!(read_csv(&csv).try_numeric_stats("age").unwrap().max, 24.0);
    let text: String = fs::read_to_string(&csv).unwrap();
    let changed: String = text.replacen(",24,", ",99,", 1);
    assert!(changed.len() == text.len() && changed != text);
    fs::write(&csv, changed).unwrap();
    assert_eq!(read_csv(&csv).try_numeric_stats("age").unwrap().max, 99.0);
}

#[test]
fn a_file_dated_before_1970_reads_and_caches() {
    let csv: PathBuf = scratch_csv("read_before_1970");
    let old: SystemTime = SystemTime::UNIX_EPOCH - Duration::from_secs(86_400);
    let file: fs::File = fs::File::options().write(true).open(&csv).unwrap();
    file.set_modified(old).unwrap();
    drop(file);
    assert_eq!(read_csv(&csv).shape(), (12, 6));
    let stats: PathBuf = cache_file(&csv, ".json");
    let written: SystemTime = modified(&stats);
    assert_eq!(read_csv(&csv).shape(), (12, 6));
    assert_eq!(modified(&stats), written); // the second read was a cache hit
}

#[test]
fn a_broken_cache_is_computed_again() {
    let csv: PathBuf = scratch_csv("read_broken");
    read_csv(&csv);
    fs::write(cache_file(&csv, ".json"), "{ not json").unwrap();
    fs::write(cache_file(&csv, ".plots.jsonl"), "garbage\n").unwrap();
    let ds: Dataset = read_csv(&csv);
    assert_eq!(ds.try_numeric_stats("age").unwrap().median, 21.0);
    let fixed: String = fs::read_to_string(cache_file(&csv, ".json")).unwrap();
    assert!(fixed.contains("\"fingerprint\""));
}

#[test]
fn a_read_only_folder_still_gives_a_dataset() {
    let csv: PathBuf = scratch_csv("read_only");
    let dir: PathBuf = csv.parent().unwrap().to_path_buf();
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o555)).unwrap();
    let ds: Result<Dataset, DatasetError> = try_read_csv(&csv);
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();
    assert_eq!(ds.unwrap().shape(), (12, 6));
    assert!(!cache_file(&csv, ".json").exists());
}

#[test]
fn ragged_rows_are_filled_with_nulls() {
    let csv: PathBuf = common::scratch_copy("read_ragged", "ragged.csv");
    let ds: Dataset = read_csv(&csv);
    assert_eq!(ds.columns(), ["id", "city", "temp"]);
    assert_eq!(ds.try_column_info("city").unwrap().null_count, 1);
    assert_eq!(
        ds.try_numeric_stats("temp").unwrap().mean,
        (4.5 + 19.0 + 15.5) / 3.0
    );
}

#[test]
fn a_header_that_looks_like_a_polars_selector_is_a_plain_name() {
    // Polars' `col` reads "*" as every column and "^...$" as a regex over the names.
    let dir: PathBuf = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("read_selector_names");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let csv: PathBuf = dir.join("odd.csv");
    fs::write(&csv, "*,^a.*$,a1\n1,10,x\n2,20,y\n3,60,x\n").unwrap();

    let ds: Dataset = read_csv(&csv);
    assert_eq!(ds.columns(), ["*", "^a.*$", "a1"]);
    assert_eq!(ds.try_numeric_stats("*").unwrap().mean, 2.0);
    assert_eq!(ds.try_numeric_stats("^a.*$").unwrap().max, 60.0);
    assert_eq!(ds.information_gain("a1").len(), 2);
    assert!(ds.hist("*", "a1", 3).svg().starts_with("<svg"));
    assert!(ds.bar("a1", "^a.*$", "", "mean").svg().starts_with("<svg"));
    assert!(ds.plot("*", "^a.*$", "").svg().starts_with("<svg"));
    assert!(ds.qqplot("*").svg().starts_with("<svg"));
    let scaled: Dataset = StandardScaler::new().fit_transform(&ds.select(&["*", "^a.*$"]));
    assert_eq!(scaled.columns(), ["*", "^a.*$"]);
    assert_eq!(scaled.try_numeric_stats("*").unwrap().median, 0.0);
}
