//! `src/plot`: every chart, three ways. `cold` is the first draw (query + render), `warm` is a
//! redraw (cached numbers where the chart caches them + render), `render` is the SVG alone.
//! `png` is what a program's auto-save costs on top.

mod common;

use std::{hint::black_box, path::PathBuf, time::Duration};

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use polars::frame::DataFrame;
use rusmlai::{
    dataset::Dataset,
    display::Figure,
    metadata::{Metadata, plot::PlotCache},
    plot::error::PlotError,
};

type Chart = (&'static str, fn(&Dataset) -> Result<Figure, PlotError>);

const CHARTS: &[Chart] = &[
    ("bar", |d| d.try_bar("course", "exam_score", "", "mean")),
    ("bar_hue", |d| {
        d.try_bar("course", "exam_score", "gender", "mean")
    }),
    ("count", |d| d.try_bar("course", "", "", "")),
    ("count_hue", |d| d.try_bar("course", "", "gender", "")),
    ("describe_bar", |d| d.try_describe_bar("", "")),
    ("missing", |d| d.try_missing()),
    ("pie", |d| d.try_pie("course")),
    ("hist", |d| d.try_hist("exam_score", "", 30)),
    ("hist_hue", |d| d.try_hist("exam_score", "gender", 30)),
    ("boxplot", |d| d.try_boxplot("course", "exam_score", "")),
    ("stripplot", |d| d.try_stripplot("course", "exam_score", "")),
    ("kdeplot", |d| d.try_kdeplot("exam_score", "gender")),
    ("violinplot", |d| {
        d.try_violinplot("course", "exam_score", "")
    }),
    ("qqplot", |d| d.try_qqplot("exam_score")),
    ("heatmap", |d| {
        d.try_heatmap("course", "study_method", "exam_score", "mean")
    }),
    ("missing_matrix", |d| d.try_missing_matrix()),
    ("corrplot", |d| d.try_corrplot()),
    ("scatter", |d| {
        d.try_scatter("study_hours", "exam_score", "")
    }),
    ("regplot", |d| {
        d.try_regplot("study_hours", "exam_score", "")
    }),
    ("line", |d| d.try_plot("study_hours", "exam_score", "")),
    ("pairplot", |d| {
        d.try_pairplot(
            &["age", "study_hours", "sleep_hours", "exam_score"],
            "gender",
        )
    }),
];

fn charts(c: &mut Criterion) {
    let df: DataFrame = common::load(&common::source_csv());
    // In memory: no cache files, so nothing here touches the fixture's `metadata/` folder.
    let meta: Metadata = Metadata::new(&df, "");
    let fresh = || Dataset {
        dataframe: df.clone(),
        metadata: Metadata {
            plot_cache: PlotCache::default(),
            ..meta.clone()
        },
    };
    let drawn: Dataset = fresh();

    let mut g = c.benchmark_group("plot");
    g.sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    for (name, chart) in CHARTS {
        g.bench_function(format!("cold/{name}"), |b| {
            b.iter_batched(
                fresh,
                |ds| {
                    let svg: String = chart(&ds).unwrap().svg();
                    (ds, svg) // returned, so dropping them isn't timed
                },
                BatchSize::SmallInput,
            )
        });
        let figure: Figure = chart(&drawn).unwrap();
        g.bench_function(format!("warm/{name}"), |b| {
            b.iter(|| black_box(chart(&drawn).unwrap().svg()))
        });
        g.bench_function(format!("render/{name}"), |b| {
            b.iter(|| black_box(figure.svg()))
        });
    }
    g.finish();
}

/// SVG to PNG: what the auto-save of a program adds to each chart.
fn png(c: &mut Criterion) {
    let df: DataFrame = common::load(&common::source_csv());
    let ds: Dataset = Dataset::new(df, "");
    let dir: PathBuf = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("plot_png");

    let mut g = c.benchmark_group("plot/png");
    g.sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    for name in ["hist", "scatter", "corrplot"] {
        let chart = CHARTS.iter().find(|(n, _)| *n == name).unwrap().1;
        let figure: Figure = chart(&ds).unwrap();
        let path: PathBuf = dir.join(format!("{name}.png"));
        g.bench_function(name, |b| b.iter(|| figure.save(&path)));
    }
    g.finish();
}

criterion_group!(benches, charts, png);
criterion_main!(benches);
