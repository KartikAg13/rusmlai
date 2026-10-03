//! `src/dataset`: CSV parsing, `read_csv` end to end, and `Dataset` operations.

mod common;

use std::{hint::black_box, path::PathBuf, time::Duration};

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use rusmlai::dataset::{Dataset, read_csv, reader::Reader};

fn read(c: &mut Criterion) {
    let csv: PathBuf = common::scratch_csv("dataset");
    let bytes: u64 = std::fs::metadata(&csv).unwrap().len();

    let mut g = c.benchmark_group("dataset");
    g.sample_size(10).measurement_time(Duration::from_secs(10));
    g.throughput(Throughput::Bytes(bytes));

    // Polars CSV parse only, no metadata.
    g.bench_function("csv_parse", |b| {
        b.iter(|| black_box(Reader::new(&csv).finish().unwrap()))
    });
    // Parse + cache miss: compute every column's stats + write both cache files.
    g.bench_function("read_csv/cold", |b| {
        b.iter_batched(
            || common::remove_caches(&csv),
            |_| black_box(read_csv(&csv)),
            BatchSize::PerIteration,
        )
    });
    // Parse + cache hit: read both cache files.
    read_csv(&csv);
    g.bench_function("read_csv/warm", |b| b.iter(|| black_box(read_csv(&csv))));
    g.finish();
}

fn ops(c: &mut Criterion) {
    let csv: PathBuf = common::scratch_csv("dataset_ops");
    let ds: Dataset = read_csv(&csv);

    let mut g = c.benchmark_group("dataset");
    g.bench_function("select/2_cols", |b| {
        b.iter(|| black_box(ds.select(&["gender", "exam_score"])))
    });
    g.bench_function("drop/1_col", |b| b.iter(|| black_box(ds.drop(&["id"]))));
    g.bench_function("columns", |b| b.iter(|| black_box(ds.columns())));
    g.bench_function("dtypes", |b| b.iter(|| black_box(ds.dtypes())));
    g.bench_function("try_numeric_stats", |b| {
        b.iter(|| black_box(ds.try_numeric_stats("exam_score").unwrap()))
    });
    g.bench_function("try_categorical_stats", |b| {
        b.iter(|| black_box(ds.try_categorical_stats("course").unwrap()))
    });
    g.finish();
}

criterion_group!(benches, read, ops);
criterion_main!(benches);
