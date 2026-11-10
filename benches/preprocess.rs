//! `src/preprocess`: `fit` (reads the cached stats), `transform` (the pass over the data) and the
//! `ColumnTransformer` of `examples/s6e1.rs`, plus the same on data with nulls.

mod common;

use std::{hint::black_box, path::PathBuf, time::Duration};

use criterion::{Criterion, criterion_group, criterion_main};
use polars::prelude::*;
use rusmlai::{prelude::*, steps};

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

/// The preprocessing of `examples/s6e1.rs`.
fn example() -> ColumnTransformer {
    ColumnTransformer::new()
        .add(
            "num",
            &NUMERIC,
            steps![
                SimpleImputer::new().strategy("median"),
                StandardScaler::new()
            ],
        )
        .add("cat", &TEXT, steps![OneHotEncoder::new()])
}

/// One bench for `fit` and one for `transform` of a transformer on `data`.
fn fit_and_transform<T: Transform + Clone>(
    g: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    name: &str,
    new: T,
    data: &Dataset,
) {
    g.bench_function(format!("fit/{name}"), |b| {
        b.iter(|| {
            let mut step: T = new.clone();
            step.fit(data);
            black_box(step)
        })
    });
    let mut fitted: T = new.clone();
    fitted.fit(data);
    g.bench_function(format!("transform/{name}"), |b| {
        b.iter(|| black_box(fitted.transform(data)))
    });
}

fn transformers(c: &mut Criterion) {
    let csv: PathBuf = common::scratch_csv("preprocess");
    let ds: Dataset = read_csv(&csv);
    let numeric: Dataset = ds.select(&NUMERIC);
    let text: Dataset = ds.select(&TEXT);
    let target: Dataset = ds.select(&["exam_difficulty"]);

    let mut g = c.benchmark_group("preprocess");
    g.sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    fit_and_transform(&mut g, "standard_scaler", StandardScaler::new(), &numeric);
    fit_and_transform(&mut g, "minmax_scaler", MinMaxScaler::new(), &numeric);
    fit_and_transform(&mut g, "maxabs_scaler", MaxAbsScaler::new(), &numeric);
    fit_and_transform(&mut g, "robust_scaler", RobustScaler::new(), &numeric);
    fit_and_transform(&mut g, "imputer_mean", SimpleImputer::new(), &numeric);
    fit_and_transform(
        &mut g,
        "imputer_most_frequent_numeric",
        SimpleImputer::new().strategy("most_frequent"),
        &numeric,
    );
    fit_and_transform(
        &mut g,
        "imputer_most_frequent_text",
        SimpleImputer::new().strategy("most_frequent"),
        &text,
    );
    fit_and_transform(&mut g, "onehot_encoder", OneHotEncoder::new(), &text);
    fit_and_transform(&mut g, "ordinal_encoder", OrdinalEncoder::new(), &text);
    fit_and_transform(&mut g, "label_encoder", LabelEncoder::new(), &target);

    // The whole `ColumnTransformer` of the example: 4 numeric and 7 text columns in.
    let x: Dataset = ds.drop(&["id", "exam_score"]);
    g.bench_function("fit/column_transformer", |b| {
        b.iter(|| {
            let mut prep: ColumnTransformer = example();
            prep.fit(&x);
            black_box(prep)
        })
    });
    let mut prep: ColumnTransformer = example();
    prep.fit(&x);
    g.bench_function("transform/column_transformer", |b| {
        b.iter(|| black_box(prep.transform(&x)))
    });
    g.finish();
}

/// Every tenth value of the numeric columns nulled, so the imputer really fills and the scaler
/// after it learns from columns whose stats are not cached.
fn with_nulls(c: &mut Criterion) {
    let df: DataFrame = common::load(&common::source_csv());
    let holes: Vec<Expr> = NUMERIC
        .iter()
        .map(|name| {
            when((col("id") % lit(10)).eq(lit(0)))
                .then(lit(NULL))
                .otherwise(col(*name))
                .alias(*name)
        })
        .collect();
    let frame: DataFrame = df.lazy().with_columns(holes).collect().unwrap();
    let x: Dataset = Dataset::new(frame, "").drop(&["id", "exam_score"]);

    let mut g = c.benchmark_group("preprocess/nulls");
    g.sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    g.bench_function("fit/column_transformer", |b| {
        b.iter(|| {
            let mut prep: ColumnTransformer = example();
            prep.fit(&x);
            black_box(prep)
        })
    });
    let mut prep: ColumnTransformer = example();
    prep.fit(&x);
    g.bench_function("transform/column_transformer", |b| {
        b.iter(|| black_box(prep.transform(&x)))
    });
    g.finish();
}

criterion_group!(benches, transformers, with_nulls);
criterion_main!(benches);
