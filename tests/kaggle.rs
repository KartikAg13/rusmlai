//! Smoke test on the local Kaggle file (not tracked): `cargo test -- --ignored`.

use std::{fs, path::PathBuf};

use rusmlai::prelude::*;

#[test]
#[ignore = "needs tests/fixtures/s6e1/train.csv, which only exists locally"]
fn kaggle_s6e1_reads_and_draws() {
    let source: PathBuf =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/s6e1/train.csv");
    if !source.is_file() {
        eprintln!("skipped: {} is missing", source.display());
        return;
    }
    let dir: PathBuf = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("kaggle");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let csv: PathBuf = dir.join("train.csv");
    fs::copy(&source, &csv).unwrap();

    let ds: Dataset = read_csv(&csv);
    assert!(ds.height() > 100_000);
    assert_eq!(read_csv(&csv).shape(), ds.shape()); // the cached read
    let numeric: Vec<String> = ds
        .columns()
        .into_iter()
        .filter(|c| ds.try_column_info(c).unwrap().dtype.is_numeric())
        .collect();
    assert!(ds.hist(&numeric[1], "", 30).svg().starts_with("<svg"));
    let target: String = ds.columns().last().unwrap().clone();
    assert!(
        !ds.drop(&[ds.columns()[0].as_str()])
            .information_gain(&target)
            .is_empty()
    );
}
