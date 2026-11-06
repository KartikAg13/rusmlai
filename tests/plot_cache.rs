//! The plot cache: one JSONL line per new chart, shared by `select`, empty after a transform.

mod common;

use std::path::PathBuf;

use common::{cache_file, lines, scratch_csv};
use rusmlai::prelude::*;

#[test]
fn a_miss_appends_one_line_and_a_new_read_hits() {
    let csv: PathBuf = scratch_csv("plots_hit");
    let plots: PathBuf = cache_file(&csv, ".plots.jsonl");
    let ds: Dataset = read_csv(&csv);
    assert_eq!(lines(&plots), 1); // the fingerprint
    let first: String = ds.hist("cgpa", "", 10).svg();
    assert_eq!(lines(&plots), 2);
    ds.hist("cgpa", "", 10).svg();
    assert_eq!(lines(&plots), 2); // a hit in memory

    let again: Dataset = read_csv(&csv);
    assert_eq!(again.hist("cgpa", "", 10).svg(), first);
    assert_eq!(lines(&plots), 2); // a hit from the file
    again.count("batch", "passed").svg();
    assert_eq!(lines(&plots), 3);
}

#[test]
fn select_shares_the_plot_cache() {
    let csv: PathBuf = scratch_csv("plots_select");
    let plots: PathBuf = cache_file(&csv, ".plots.jsonl");
    let ds: Dataset = read_csv(&csv);
    ds.boxplot("batch", "cgpa", "").svg();
    let n: usize = lines(&plots);
    ds.select(&["batch", "cgpa"])
        .boxplot("batch", "cgpa", "")
        .svg();
    assert_eq!(lines(&plots), n);
}

#[test]
fn a_transformed_dataset_has_its_own_empty_cache_and_writes_no_files() {
    let csv: PathBuf = scratch_csv("plots_transform");
    let plots: PathBuf = cache_file(&csv, ".plots.jsonl");
    let ds: Dataset = read_csv(&csv);
    ds.hist("cgpa", "", 10).svg();
    let n: usize = lines(&plots);

    let metadata = ds
        .metadata
        .dataset_cache
        .after_transform(&ds, vec!["cgpa".into()], false);
    assert!(metadata.filepath.is_none());
    assert!(metadata.plot_cache.entries.read().unwrap().is_empty());
    let new: Dataset = Dataset {
        dataframe: ds.dataframe.clone(),
        metadata,
    };
    new.hist("cgpa", "", 10).svg();
    assert_eq!(lines(&plots), n);
}
