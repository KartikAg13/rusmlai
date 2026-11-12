//! `src/models`: `LinearRegression` and `Ridge` on the S6E1 design matrix (630,000 rows, 30
//! features after `StandardScaler` and `OneHotEncoder`), the same matrix that
//! `bench-python/models.py` and `bench-python/linear.py` give scikit-learn.

mod common;

use std::{hint::black_box, path::PathBuf, time::Duration};

use criterion::{Criterion, criterion_group, criterion_main};
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
const TARGET: &str = "exam_score";

/// scikit-learn's matrix in `bench-python/models.py`: scaled numbers and one-hot text.
fn preprocessing() -> ColumnTransformer {
    ColumnTransformer::new()
        .add("num", &NUMERIC, steps![StandardScaler::new()])
        .add("cat", &TEXT, steps![OneHotEncoder::new()])
}

fn linear(c: &mut Criterion) {
    let csv: PathBuf = common::scratch_csv("models");
    let ds: Dataset = read_csv(&csv).drop(&["id"]);
    // The 30 features with the target passed through: the input a model's fit gets.
    let features: Dataset = preprocessing().remainder("passthrough").fit_transform(&ds);
    assert_eq!((features.height(), features.width()), (630_000, 31));

    let mut g = c.benchmark_group("linear");
    g.sample_size(20).measurement_time(Duration::from_secs(10));
    g.bench_function("fit/LinearRegression", |b| {
        b.iter(|| {
            let mut model: LinearRegression = LinearRegression::new();
            model.fit(&features, TARGET);
            black_box(model)
        })
    });
    g.bench_function("fit/Ridge", |b| {
        b.iter(|| {
            let mut model: Ridge = Ridge::new();
            model.fit(&features, TARGET);
            black_box(model)
        })
    });
    let mut fitted: LinearRegression = LinearRegression::new();
    fitted.fit(&features, TARGET);
    g.bench_function("predict/LinearRegression", |b| {
        b.iter(|| black_box(fitted.predict(&features)))
    });
    // From the loaded CSV: the preprocessing fits from the cached stats, then the model.
    g.bench_function("pipeline_fit/LinearRegression", |b| {
        b.iter(|| {
            let mut pipe =
                Pipeline::new(preprocessing(), ModelStacker::new(LinearRegression::new()));
            pipe.fit(&ds, TARGET);
            black_box(pipe)
        })
    });
    g.finish();
}

criterion_group!(benches, linear);
criterion_main!(benches);
