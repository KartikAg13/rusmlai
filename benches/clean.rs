//! pandas' cleaning methods on a `Dataset` against `bench-python/clean.py` (pandas on the same
//! rows and calls): S6E1, 630,000 rows, in memory.

mod common;

use std::{hint::black_box, path::PathBuf, time::Duration};

use criterion::{Criterion, criterion_group, criterion_main};
use rusmlai::polars::prelude::{col, lit};
use rusmlai::prelude::*;

fn seconds() -> Duration {
    let seconds: Option<u64> = std::env::var("RUSMLAI_BENCH_SECONDS").ok().and_then(|s| s.parse().ok());
    Duration::from_secs(seconds.unwrap_or(5))
}

fn clean(c: &mut Criterion) {
    let csv: String = format!("{}/tests/fixtures/s6e1/train.csv", env!("CARGO_MANIFEST_DIR"));
    let ds: Dataset = Dataset::new(common::load(&PathBuf::from(csv)), "");
    eprintln!(
        "query rows {}, drop_duplicates rows {}, value_counts {:?}",
        ds.query(col("study_hours").gt(lit(5.0))).height(),
        ds.drop_duplicates(&["course", "gender"], "first").height(),
        ds.value_counts("course", false).to_list("count"),
    );
    let mut g = c.benchmark_group("clean");
    g.sample_size(20).measurement_time(seconds());
    g.bench_function("dropna", |b| b.iter(|| black_box(ds.dropna(&[]))));
    g.bench_function("fillna", |b| b.iter(|| black_box(ds.fillna(&[("study_hours", 0.0.into())]))));
    g.bench_function("astype", |b| b.iter(|| black_box(ds.astype(&[("age", "float64")]))));
    g.bench_function("rename", |b| b.iter(|| black_box(ds.rename(&[("age", "years")]))));
    g.bench_function("query", |b| b.iter(|| black_box(ds.query(col("study_hours").gt(lit(5.0))))));
    g.bench_function("drop_duplicates", |b| b.iter(|| black_box(ds.drop_duplicates(&["course", "gender"], "first"))));
    g.bench_function("sample", |b| b.iter(|| black_box(ds.sample(100_000, 42))));
    g.bench_function("sort_values", |b| b.iter(|| black_box(ds.sort_values(&["exam_score"], &[false]))));
    g.bench_function("value_counts", |b| b.iter(|| black_box(ds.value_counts("course", false))));
    g.finish();
}

criterion_group!(benches, clean);
criterion_main!(benches);
