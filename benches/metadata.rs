//! `src/metadata`: per-column stats, building `Metadata`, loading it, and deriving it.

mod common;

use std::{hint::black_box, path::PathBuf};

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use polars::{
    frame::DataFrame,
    prelude::{DataType, Expr, IntoLazy, LazyFrame, QuantileMethod, col, len, lit},
};
use rusmlai::{
    dataset::Dataset,
    metadata::{
        Metadata,
        dataset::{dto::ColumnInfoDTO, info::ColumnInfo, stats::ColumnStats},
    },
};

/// One bench per column, so a slow column (or dtype) stands out.
fn stats(c: &mut Criterion) {
    let df: DataFrame = common::load(&common::source_csv());

    let mut g = c.benchmark_group("stats");
    g.sample_size(10)
        .throughput(Throughput::Elements(df.height() as u64));
    for column in df.columns() {
        let name: String = column.name().to_string();
        let kind: &str = if column.dtype().is_numeric() {
            "numeric"
        } else {
            "categorical"
        };
        let one: DataFrame = df.select([name.as_str()]).unwrap();
        g.bench_with_input(BenchmarkId::new(kind, &name), &one, |b, one| {
            b.iter(|| black_box(ColumnStats::compute_stats(one).unwrap()))
        });
    }
    g.finish();
}

/// Polars' cost for each numeric expression alone on one column, and for a string group-by.
/// Reference numbers: `compute_stats` does median/q25/q75 in Rust instead of these Polars sorts.
fn stat_exprs(c: &mut Criterion) {
    let df: DataFrame = common::load(&common::source_csv());
    let numeric: LazyFrame = df.select(["exam_score"]).unwrap().lazy();
    let x = || col("exam_score").cast(DataType::Float64);
    let exprs: Vec<(&str, Vec<Expr>)> = vec![
        ("mean", vec![x().mean()]),
        ("median", vec![x().median()]),
        ("std", vec![x().std(1)]),
        ("var", vec![x().var(1)]),
        ("min", vec![x().min()]),
        ("max", vec![x().max()]),
        ("q25", vec![x().quantile(lit(0.25), QuantileMethod::Linear)]),
        (
            "median+q25+q75",
            vec![
                x().median().alias("a"),
                x().quantile(lit(0.25), QuantileMethod::Linear).alias("b"),
                x().quantile(lit(0.75), QuantileMethod::Linear).alias("c"),
            ],
        ),
        ("sort", vec![x().sort(Default::default())]),
    ];

    let mut g = c.benchmark_group("stats/expr");
    g.sample_size(10);
    for (name, e) in exprs {
        g.bench_function(name, |b| {
            b.iter(|| black_box(numeric.clone().select(e.clone()).collect().unwrap()))
        });
    }
    let categorical: LazyFrame = df.select(["study_method"]).unwrap().lazy();
    g.bench_function("group_by_len", |b| {
        b.iter(|| {
            black_box(
                categorical
                    .clone()
                    .drop_nulls(None)
                    .group_by_stable([col("study_method")])
                    .agg([len().alias("count")])
                    .collect()
                    .unwrap(),
            )
        })
    });
    for parallel in [false, true] {
        let e: Expr = col("study_method")
            .drop_nulls()
            .value_counts(false, parallel, "count", false)
            .implode(false);
        g.bench_function(format!("value_counts/parallel={parallel}"), |b| {
            b.iter(|| black_box(categorical.clone().select([e.clone()]).collect().unwrap()))
        });
    }
    g.finish();
}

/// `Metadata::new` computes every column: scaling by row count.
fn build(c: &mut Criterion) {
    let csv: PathBuf = common::source_csv();
    let df: DataFrame = common::load(&csv);

    let mut g = c.benchmark_group("metadata");
    g.sample_size(10);
    for rows in [10_000, 100_000, df.height()] {
        let part: DataFrame = df.head(Some(rows));
        g.throughput(Throughput::Elements(rows as u64));
        g.bench_with_input(BenchmarkId::new("new", rows), &part, |b, part| {
            b.iter(|| black_box(Metadata::new(part, &csv)))
        });
    }
    g.finish();
}

fn to_dtos(meta: &Metadata) -> Vec<(String, ColumnInfoDTO)> {
    meta.dataset_cache
        .columns
        .iter()
        .map(|(n, i)| (n.clone(), ColumnInfoDTO::from(i.as_ref())))
        .collect()
}

/// Loading from the cache files, and deriving new metadata from existing metadata.
fn load_and_derive(c: &mut Criterion) {
    let csv: PathBuf = common::scratch_csv("metadata");
    let df: DataFrame = common::load(&csv);

    let mut g = c.benchmark_group("metadata");
    g.sample_size(10);
    g.bench_function("load/cold", |b| {
        b.iter_batched(
            || common::remove_caches(&csv),
            |_| black_box(Metadata::load(&df, &csv)),
            BatchSize::PerIteration,
        )
    });
    let meta: Metadata = Metadata::load(&df, &csv);
    g.bench_function("load/warm", |b| {
        b.iter(|| black_box(Metadata::load(&df, &csv)))
    });

    g.bench_function("select/2_cols", |b| {
        b.iter(|| black_box(meta.select(&["gender", "exam_score"])))
    });
    let ds: Dataset = Dataset {
        dataframe: df.clone(),
        metadata: meta.clone(),
    };
    g.bench_function("after_transform/1_col", |b| {
        b.iter(|| {
            black_box(meta.dataset_cache.after_transform(
                &ds,
                vec!["exam_score".to_string()],
                false,
            ))
        })
    });
    g.bench_function("after_transform/all_cols", |b| {
        b.iter(|| black_box(meta.dataset_cache.after_transform(&ds, vec![], true)))
    });

    g.bench_function("dto/to", |b| b.iter(|| black_box(to_dtos(&meta))));
    g.bench_function("dto/from", |b| {
        b.iter_batched(
            || to_dtos(&meta),
            |dtos| {
                let infos: Vec<(String, ColumnInfo)> = dtos
                    .into_iter()
                    .map(|(n, d)| (n, ColumnInfo::from(d)))
                    .collect();
                black_box(infos)
            },
            BatchSize::SmallInput,
        )
    });
    g.finish();
}

criterion_group!(benches, stats, stat_exprs, build, load_and_derive);
criterion_main!(benches);
