//! Column roles (backlog #14) and `Dataset::to_csv` (backlog #2), through the public API.

use std::{fs, path::PathBuf, time::SystemTime};

use rusmlai::{
    metadata::models::ModelCache,
    polars::prelude::{Column, DataFrame, DataType, df},
    prelude::*,
};

/// 100 rows: a row number, a constant, a feature and the target.
fn table() -> Dataset {
    let x: Vec<f64> = (0..100).map(|i| f64::from((i * 37) % 23)).collect();
    let y: Vec<f64> = x.iter().map(|v| 2.0 * v + 1.0).collect();
    Dataset::new(
        df!("row" => (0..100i64).collect::<Vec<_>>(), "same" => [7i64; 100], "x" => x, "y" => y).unwrap(),
        "",
    )
}

/// A fresh folder of its own under the system temp folder.
fn folder(name: &str) -> PathBuf {
    let dir: PathBuf = std::env::temp_dir().join(format!("rusmlai-roles-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn fit_leaves_out_identifiers_and_constants_unless_told_otherwise() {
    let mut ds: Dataset = table();
    assert_eq!(ds.role("row"), ColumnRole::Identifier);
    assert_eq!(ds.role("same"), ColumnRole::Constant);
    assert_eq!(ds.role("x"), ColumnRole::Feature);

    let mut model: LinearRegression = LinearRegression::new();
    model.fit(&ds, "y");
    assert_eq!(model.feature_names_in_, ["x"]);
    // predict reads its features by name: the left-out columns may stay in the rows.
    assert!((model.score(&ds, "y") - 1.0).abs() < 1e-12);

    // The two sides of a split keep the roles, so a fit on train leaves the same columns out.
    let (train, _) = train_test_split(&ds, 0.2, 42, "");
    assert_eq!(train.role("row"), ColumnRole::Identifier);
    let mut on_train: LinearRegression = LinearRegression::new();
    on_train.fit(&train, "y");
    assert_eq!(on_train.feature_names_in_, ["x"]);

    // Searches and cross-validation train on the same columns as fit.
    let scores: Vec<f64> = cross_val_score(&LinearRegression::new(), &ds, "y", 4, "r2");
    assert!(scores.iter().all(|s| (s - 1.0).abs() < 1e-9), "{scores:?}");
    let mut search = GridSearchCV::new(Ridge::new()).param_grid("alpha", [0.1, 1.0]);
    search.fit(&ds, "y");
    assert_eq!(search.best_estimator_.as_ref().unwrap().feature_names_in_, ["x"]);

    // A role set by hand puts the column back.
    ds.set_role("row", ColumnRole::Feature);
    let mut with_row: LinearRegression = LinearRegression::new();
    with_row.fit(&ds, "y");
    assert_eq!(with_row.feature_names_in_, ["row", "x"]);
    // The target is never left out, whatever its role.
    ds.set_role("y", ColumnRole::Identifier);
    with_row.fit(&ds, "y");
    assert_eq!(with_row.feature_names_in_, ["row", "x"]);
}

#[test]
fn to_csv_writes_the_stats_so_the_next_read_computes_nothing() {
    let dir: PathBuf = folder("to-csv");
    let mut ds: Dataset = read_csv("tests/fixtures/data/students.csv").drop(&["name"]);
    ds.set_role("age", ColumnRole::Identifier);
    let path: PathBuf = dir.join("clean.csv");
    ds.to_csv(&path);

    let stats: PathBuf = dir.join("metadata/clean.json");
    assert!(stats.is_file() && dir.join("metadata/clean.plots.jsonl").is_file());
    assert_eq!(ds.metadata.filepath.as_deref(), Some(path.as_os_str()));
    // Models of the dataset now cache next to the new file.
    assert_eq!(ModelCache::of(&ds).unwrap().path, dir.join("metadata/clean.models.jsonl"));

    let written: SystemTime = fs::metadata(&stats).unwrap().modified().unwrap();
    let back: Dataset = read_csv(&path);
    assert_eq!(fs::metadata(&stats).unwrap().modified().unwrap(), written, "read, not computed again");
    assert_eq!(back.shape(), ds.shape());
    assert_eq!(back.columns(), ds.columns());
    assert_eq!(back.try_numeric_stats("cgpa").unwrap().median, 7.4);
    assert_eq!(back.role("age"), ColumnRole::Identifier, "a role set by hand is kept");
    assert_eq!(back.role("id"), ColumnRole::Identifier);

    // A chart of the saved dataset goes into its new chart cache.
    ds.hist("cgpa", "", 10).svg();
    assert_eq!(fs::read_to_string(dir.join("metadata/clean.plots.jsonl")).unwrap().lines().count(), 2);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn to_csv_keeps_no_stats_for_a_column_that_reads_back_as_another_type() {
    let dir: PathBuf = folder("types");
    let day: Column = Column::new("day".into(), [19_000i32, 19_001, 19_000]).cast(&DataType::Date).unwrap();
    let frame: DataFrame = DataFrame::new(3, vec![day, Column::new("n".into(), [1i64, 2, 3])]).unwrap();
    let mut ds: Dataset = Dataset::new(frame, "");
    assert_eq!(ds.role("day"), ColumnRole::Datetime);
    let path: PathBuf = dir.join("days.csv");
    ds.to_csv(&path);
    let json: String = fs::read_to_string(dir.join("metadata/days.json")).unwrap();
    let dto: serde_json::Value = serde_json::from_str(&json).unwrap();
    // A date is written as text and read_csv reads it back as text: that column gets its
    // stats on the next use, and keeps its role.
    assert_eq!(dto["columns"]["day"]["stats"], serde_json::Value::Null);
    assert_eq!(dto["columns"]["day"]["dtype"], "String");
    assert!(!dto["columns"]["n"]["stats"].is_null());
    let back: Dataset = read_csv(&path);
    assert_eq!(back.try_categorical_stats("day").unwrap().mode, "2022-01-08");
    assert_eq!(back.role("day"), ColumnRole::Datetime);
    fs::remove_dir_all(&dir).unwrap();
}
