//! Cache file I/O (`src/metadata/utils.rs`) and the plot cache (`Metadata::plot`).

mod common;

use std::{fs, hint::black_box, path::PathBuf};

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use polars::frame::DataFrame;
use rusmlai::{
    metadata::{
        Metadata,
        constants::{DATASET_CACHE, PLOTS_CACHE},
        dataset::dto::{ColumnInfoDTO, DatasetCacheDTO},
        plot::PlotCache,
        utils::{append_jsonl, cache_path, file_fingerprint, read_jsonl, write_jsonl},
    },
    plot::{result::PlotResult, spec::PlotSpec},
};

fn spec(bins: usize) -> PlotSpec {
    PlotSpec::Hist {
        column: "exam_score".to_string(),
        hue: None,
        bins,
    }
}

/// About the size of a real bar chart result (8 categories).
fn result() -> PlotResult {
    PlotResult::Series {
        labels: (0..8).map(|i| format!("category_{i}")).collect(),
        values: (0..8).map(|i| i as f64 * 1.5).collect(),
    }
}

fn files(c: &mut Criterion) {
    let csv: PathBuf = common::scratch_csv("cache");
    let df: DataFrame = common::load(&csv);
    let meta: Metadata = Metadata::load(&df, &csv);
    let fingerprint: (u64, u64) = file_fingerprint(&csv).unwrap();
    let columns_path: PathBuf = cache_path(&csv, DATASET_CACHE);
    let dto: DatasetCacheDTO = DatasetCacheDTO {
        fingerprint,
        columns: meta
            .dataset_cache
            .columns
            .iter()
            .map(|(n, i)| (n.clone(), ColumnInfoDTO::from(i.as_ref())))
            .collect(),
    };

    let mut g = c.benchmark_group("cache");
    g.bench_function("file_fingerprint", |b| {
        b.iter(|| black_box(file_fingerprint(&csv).unwrap()))
    });
    g.bench_function("cache_path", |b| {
        b.iter(|| black_box(cache_path(&csv, PLOTS_CACHE)))
    });
    g.bench_function("write_columns", |b| {
        b.iter(|| fs::write(&columns_path, serde_json::to_vec_pretty(&dto).unwrap()).unwrap())
    });
    g.bench_function("read_columns", |b| {
        b.iter(|| {
            black_box(
                serde_json::from_slice::<DatasetCacheDTO>(&fs::read(&columns_path).unwrap())
                    .unwrap(),
            )
        })
    });

    // Loading cost as the plots file grows.
    for n in [10usize, 1_000] {
        let path: PathBuf = cache_path(&csv, &format!(".plots_{n}.jsonl"));
        let entries: Vec<(PlotSpec, PlotResult)> = (0..n).map(|i| (spec(i), result())).collect();
        write_jsonl(&path, fingerprint, &entries).unwrap();
        g.bench_with_input(BenchmarkId::new("read_plots", n), &path, |b, p| {
            b.iter(|| black_box(read_jsonl::<(PlotSpec, PlotResult)>(p, fingerprint).unwrap()))
        });
    }
    let plots_path: PathBuf = cache_path(&csv, PLOTS_CACHE);
    g.bench_function("append_plot", |b| {
        b.iter(|| append_jsonl(&plots_path, &(spec(0), result())).unwrap())
    });
    g.finish();
}

fn plot(c: &mut Criterion) {
    let csv: PathBuf = common::scratch_csv("plot");
    let df: DataFrame = common::load(&csv);
    let meta: Metadata = Metadata::load(&df, &csv);
    let fingerprint: (u64, u64) = file_fingerprint(&csv).unwrap();
    let plots_path: PathBuf = cache_path(&csv, PLOTS_CACHE);
    let cached: PlotResult = result();
    meta.plot(spec(0), || Ok::<_, ()>(cached.clone())).unwrap();

    let mut g = c.benchmark_group("plot");
    // Lookup only: read lock + clone of the stored result.
    g.bench_function("hit", |b| {
        b.iter(|| black_box(meta.plot(spec(0), || Ok::<_, ()>(cached.clone())).unwrap()))
    });
    // Miss with a trivial compute: lookup + append a line + write lock + insert.
    // Each batch gets an empty cache and a header-only file so neither grows without bound.
    g.bench_function("miss", |b| {
        b.iter_batched(
            || {
                let mut m: Metadata = meta.clone();
                m.plot_cache = PlotCache::default();
                write_jsonl::<()>(&plots_path, fingerprint, &[]).unwrap();
                m
            },
            |m| {
                let r: PlotResult = m.plot(spec(1), || Ok::<_, ()>(cached.clone())).unwrap();
                (m, r) // returned, so dropping them isn't timed
            },
            BatchSize::SmallInput,
        )
    });
    g.finish();
}

criterion_group!(benches, files, plot);
criterion_main!(benches);
