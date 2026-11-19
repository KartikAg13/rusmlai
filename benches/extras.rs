//! The pieces added on 2026-11-18 from the Kaggle S6 winners' notebooks, against
//! `bench-python/extras.py` (same rows, settings and calls): the ROC metrics, the shuffled
//! splitters, out-of-fold predictions, `RidgeCV`, stacking, `TargetEncoder`,
//! `KBinsDiscretizer`, `QuantileTransformer`, `CountEncoder`, `compute_sample_weight` and Extra
//! Trees. S6E1 (630,000 rows, 4 numbers and 7 text columns, a number target) for the
//! regressors and encoders, S6E2 (630,000 rows, 13 numbers, a binary text target) for the
//! classifiers. Every table is in memory, so nothing goes into a model cache.

mod common;

use std::{hint::black_box, path::PathBuf, time::Duration};

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use rusmlai::polars::prelude::{DataFrame, Series};
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
const HEART: &str = "Heart Disease";

fn source(name: &str) -> Dataset {
    let csv: String = format!("{}/tests/fixtures/{name}/train.csv", env!("CARGO_MANIFEST_DIR"));
    Dataset::new(common::load(&PathBuf::from(csv)), "").drop(&["id"])
}

fn seconds() -> Duration {
    let seconds: Option<u64> = std::env::var("RUSMLAI_BENCH_SECONDS").ok().and_then(|s| s.parse().ok());
    Duration::from_secs(seconds.unwrap_or(10))
}

fn extras(c: &mut Criterion) {
    let s6e1: Dataset = source("s6e1");
    let s6e2: Dataset = source("s6e2");
    assert_eq!((s6e1.shape(), s6e2.shape()), ((630_000, 12), (630_000, 14)));
    // S6E2's numbers scaled, the target passed through: what the logistic models read.
    let scaled: Dataset = ColumnTransformer::new()
        .add("num", &[], steps![StandardScaler::new()])
        .verbose_feature_names_out(false)
        .fit_transform(&s6e2.drop(&[HEART]))
        .dataframe()
        .clone()
        .hstack(&[s6e2.dataframe().column(HEART).unwrap().clone()])
        .map(|frame: DataFrame| Dataset::new(frame, ""))
        .unwrap();
    // S6E1's design matrix: the numbers scaled and the text one-hot, the target passed through.
    let design: Dataset = ColumnTransformer::new()
        .add("num", &NUMERIC, steps![StandardScaler::new()])
        .add("cat", &TEXT, steps![OneHotEncoder::new()])
        .remainder("passthrough")
        .fit_transform(&s6e1);
    let design: Dataset = Dataset::new(design.dataframe().clone(), "");
    assert_eq!(design.width(), 31);

    let y: Series = s6e2.dataframe().column(HEART).unwrap().as_materialized_series().clone();
    let mut logistic: LogisticRegression = LogisticRegression::new();
    logistic.fit(&scaled, HEART);
    let score: Series = logistic.predict_proba(&scaled).dataframe().column("Presence").unwrap().as_materialized_series().clone();
    let predicted: Series = logistic.predict(&scaled);
    eprintln!("roc_auc {:.6}, balanced accuracy {:.6}", roc_auc_score(&y, &score), balanced_accuracy_score(&y, &predicted));

    let mut g = c.benchmark_group("extras");
    g.sample_size(10).measurement_time(seconds());
    g.bench_function("metric/roc_auc_score", |b| b.iter(|| black_box(roc_auc_score(&y, &score))));
    g.bench_function("metric/roc_curve", |b| b.iter(|| black_box(roc_curve(&y, &score, ""))));
    g.bench_function("metric/balanced_accuracy_score", |b| {
        b.iter(|| black_box(balanced_accuracy_score(&y, &predicted)))
    });
    g.bench_function("metric/compute_sample_weight", |b| {
        b.iter(|| black_box(compute_sample_weight("balanced", &y)))
    });
    let kfold: KFold = KFold::new().shuffle(true);
    let stratified: StratifiedKFold = StratifiedKFold::new().shuffle(true);
    g.bench_function("split/KFold", |b| b.iter(|| black_box(kfold.split(&s6e2))));
    g.bench_function("split/StratifiedKFold", |b| b.iter(|| black_box(stratified.split(&s6e2, HEART))));
    g.bench_function("oof/LogisticRegression", |b| {
        b.iter(|| black_box(cross_val_predict(&LogisticRegression::new(), &scaled, HEART, stratified, "predict_proba")))
    });
    g.bench_function("fit/RidgeCV", |b| {
        b.iter(|| {
            let mut model: RidgeCV = RidgeCV::new();
            model.fit(&design, "exam_score");
            black_box(model)
        })
    });
    let stack = || {
        StackingClassifier::new()
            .add("logistic", LogisticRegression::new())
            .add("tree", DecisionTreeClassifier::new().max_depth(8))
    };
    let mut fitted = stack();
    fitted.fit(&scaled, HEART);
    eprintln!("stack train accuracy {:.6}", fitted.score(&scaled, HEART));
    g.bench_function("fit/StackingClassifier", |b| {
        b.iter(|| {
            let mut model = stack();
            model.fit(&scaled, HEART);
            black_box(model)
        })
    });

    let text: Dataset = s6e1.select(&[&TEXT[..], &["exam_score"]].concat());
    let numbers: Dataset = s6e1.select(&NUMERIC);
    g.bench_function("fit_transform/TargetEncoder", |b| {
        b.iter(|| black_box(TargetEncoder::new().target_type("continuous").fit_transform(&text, "exam_score")))
    });
    g.bench_function("fit_transform/KBinsDiscretizer", |b| {
        b.iter(|| black_box(KBinsDiscretizer::new().fit_transform(&numbers)))
    });
    g.bench_function("fit_transform/QuantileTransformer", |b| {
        b.iter(|| black_box(QuantileTransformer::new().fit_transform(&numbers)))
    });
    // From the stats `read_csv` already cached, and with them computed in the timing, as
    // pandas' value_counts computes its counts.
    let categories: Dataset = s6e1.select(&TEXT);
    g.bench_function("fit_transform/CountEncoder", |b| {
        b.iter(|| black_box(CountEncoder::new().fit_transform(&categories)))
    });
    let frame: DataFrame = categories.dataframe().clone();
    g.bench_function("fit_transform/CountEncoder/with-stats", |b| {
        b.iter(|| black_box(CountEncoder::new().fit_transform(&Dataset::new(frame.clone(), ""))))
    });

    // Extra Trees: 10 full trees, like the forests of `bench-python/forests.py`, on the
    // tables of the `tree` group of `benches/models.rs`.
    let codes: Dataset = ColumnTransformer::new()
        .add("cat", &TEXT, steps![OrdinalEncoder::new()])
        .remainder("passthrough")
        .fit_transform(&s6e1);
    let (codes, numbers2): (DataFrame, DataFrame) = (codes.dataframe().clone(), s6e2.dataframe().clone());
    let mut regressor = ExtraTreesRegressor::new().n_estimators(10);
    regressor.fit(&Dataset::new(codes.clone(), ""), "exam_score");
    let mut classifier = ExtraTreesClassifier::new().n_estimators(10);
    classifier.fit(&Dataset::new(numbers2.clone(), ""), HEART);
    let nodes = |trees: &[rusmlai::models::tree::Tree]| trees.iter().map(|t| t.node_count()).sum::<usize>();
    eprintln!(
        "ExtraTreesRegressor: nodes {} train r2 {:.6}; ExtraTreesClassifier: nodes {} train accuracy {:.6}",
        nodes(&regressor.estimators_),
        regressor.score(&Dataset::new(codes.clone(), ""), "exam_score"),
        nodes(&classifier.estimators_),
        classifier.score(&Dataset::new(numbers2.clone(), ""), HEART),
    );
    g.bench_function("fit/ExtraTreesRegressor", |b| {
        b.iter_batched(
            || Dataset::new(codes.clone(), ""),
            |ds| {
                let mut model = ExtraTreesRegressor::new().n_estimators(10);
                model.fit(&ds, "exam_score");
                black_box(model)
            },
            BatchSize::PerIteration,
        )
    });
    g.bench_function("fit/ExtraTreesClassifier", |b| {
        b.iter_batched(
            || Dataset::new(numbers2.clone(), ""),
            |ds| {
                let mut model = ExtraTreesClassifier::new().n_estimators(10);
                model.fit(&ds, HEART);
                black_box(model)
            },
            BatchSize::PerIteration,
        )
    });
    g.finish();
}

criterion_group!(benches, extras);
criterion_main!(benches);
