//! pandas' cleaning methods and the other pandas methods of `dataset/ops.rs` on a `Dataset`
//! against `bench-python/clean.py` (pandas on the same rows and calls): S6E1, 630,000 rows, in
//! memory.

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
    let means: Dataset = ds.groupby(&["course"], &[("exam_score", "mean")]);
    g.bench_function("merge", |b| b.iter(|| black_box(ds.merge(&means, &["course"], "left"))));
    g.bench_function("concat", |b| b.iter(|| black_box(Dataset::concat(&[&ds, &ds]))));
    g.bench_function("groupby", |b| {
        b.iter(|| black_box(ds.groupby(&["course", "gender"], &[("exam_score", "mean"), ("study_hours", "max"), ("id", "count")])))
    });
    g.bench_function("pivot_table", |b| b.iter(|| black_box(ds.pivot_table("exam_score", "course", "gender", "mean"))));
    g.bench_function("melt", |b| b.iter(|| black_box(ds.melt(&["id"], &["study_hours", "sleep_hours"]))));
    g.bench_function("get_dummies", |b| b.iter(|| black_box(ds.get_dummies(&["course"]))));
    g.bench_function("isna", |b| b.iter(|| black_box(ds.isna())));
    g.bench_function("ffill", |b| b.iter(|| black_box(ds.ffill())));
    g.bench_function("sum", |b| b.iter(|| black_box(ds.sum())));
    g.bench_function("mean", |b| b.iter(|| black_box(ds.mean())));
    g.bench_function("median", |b| b.iter(|| black_box(ds.median())));
    g.bench_function("std", |b| b.iter(|| black_box(ds.std())));
    g.bench_function("quantile", |b| b.iter(|| black_box(ds.quantile(0.25))));
    g.bench_function("count", |b| b.iter(|| black_box(ds.count())));
    g.bench_function("nunique", |b| b.iter(|| black_box(ds.nunique())));
    g.bench_function("corr", |b| b.iter(|| black_box(ds.corr())));
    g.bench_function("unique", |b| b.iter(|| black_box(ds.unique("course"))));
    g.bench_function("nlargest", |b| b.iter(|| black_box(ds.nlargest(100, "exam_score"))));
    g.bench_function("select_dtypes", |b| b.iter(|| black_box(ds.select_dtypes("number"))));
    g.bench_function("assign", |b| b.iter(|| black_box(ds.assign(&[("older", col("age") + lit(1))]))));
    g.bench_function("replace", |b| b.iter(|| black_box(ds.replace("course", &[("b.sc".into(), "bsc".into())]))));
    g.bench_function("clip", |b| b.iter(|| black_box(ds.clip(0.0, 50.0))));
    g.bench_function("round", |b| b.iter(|| black_box(ds.round(1))));
    g.bench_function("shift", |b| b.iter(|| black_box(ds.shift(1))));
    g.bench_function("diff", |b| b.iter(|| black_box(ds.diff(1))));
    g.finish();
}

criterion_group!(benches, clean);
criterion_main!(benches);
