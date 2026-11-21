//! `HillClimbing` against `bench-python/hill.py` (Chris Deotte's hill climbing loop of
//! Kaggle S6E3 on numpy, his vectorized rank AUC on the CPU): 630,000 rows, 5 models of a 0/1
//! target made by the same formulas on both sides, weights −0.5 to 0.5 by 0.01.

use std::{hint::black_box, time::Duration};

use criterion::{Criterion, criterion_group, criterion_main};
use rusmlai::polars::prelude::*;
use rusmlai::prelude::*;

const ROWS: usize = 630_000;

/// The formulas of `bench-python/hill.py`.
fn models() -> Dataset {
    let i = |r: usize| r as f64;
    let y: Vec<i64> = (0..ROWS).map(|r| i64::from((i(r) * 0.37).sin() + (i(r) * 0.011).cos() > 0.3)).collect();
    let m = |a: f64, b: f64| -> Column {
        let values: Vec<f64> = (0..ROWS).map(|r| y[r] as f64 * a + ((i(r) * b).sin() * 0.5 + 0.5) * (1.0 - a)).collect();
        Column::new(format!("m{a}").into(), values)
    };
    let columns: Vec<Column> = vec![m(0.3, 1.7), m(0.25, 2.3), m(0.2, 0.9), m(0.15, 3.1), m(0.1, 0.7), Column::new("y".into(), y.clone())];
    Dataset::new(DataFrame::new(ROWS, columns).unwrap(), "")
}

fn seconds() -> Duration {
    let seconds: Option<u64> = std::env::var("RUSMLAI_BENCH_SECONDS").ok().and_then(|s| s.parse().ok());
    Duration::from_secs(seconds.unwrap_or(30))
}

fn hill(c: &mut Criterion) {
    let ds: Dataset = models();
    // One timed fit each: `--list` stops here, so a one-core run needs no 10 samples.
    for scoring in ["roc_auc", "neg_root_mean_squared_error"] {
        let start = std::time::Instant::now();
        let mut blend = HillClimbing::new().scoring(scoring);
        blend.fit(&ds, "y");
        eprintln!(
            "{scoring}: one fit {:.1} s, score {:.10}, {} steps, weights {:?}",
            start.elapsed().as_secs_f64(),
            blend.score_,
            blend.history_.len(),
            blend.weights_
        );
    }
    let mut g = c.benchmark_group("hill");
    g.sample_size(10).measurement_time(seconds());
    for scoring in ["roc_auc", "neg_root_mean_squared_error"] {
        g.bench_function(format!("fit/HillClimbing/{scoring}"), |b| {
            b.iter(|| {
                let mut blend = HillClimbing::new().scoring(scoring);
                blend.fit(&ds, "y");
                black_box(blend)
            })
        });
    }
    g.finish();
}

criterion_group!(benches, hill);
criterion_main!(benches);
