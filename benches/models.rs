//! `src/models`: `LinearRegression`, plain and as Ridge, on the S6E1 design matrix (630,000 rows, 30
//! features after `StandardScaler` and `OneHotEncoder`), and `LogisticRegression` on S6E2 (630,000
//! rows, 13 scaled features): the matrices that `bench-python/models.py` and
//! `bench-python/linear.py` give scikit-learn. The `cache_*` benches time the model cache. The
//! trees (`tree`) run against `bench-python/trees.py`, the forests and bags (`forest`) against
//! `bench-python/forests.py`, boosting, AdaBoost and the grid search (`boost`) against
//! `bench-python/boosting.py`.

mod common;

use std::{fs, hint::black_box, path::PathBuf, time::Duration};

use criterion::{
    BatchSize, BenchmarkGroup, Criterion, criterion_group, criterion_main, measurement::WallTime,
};
use rusmlai::{models::tree::Tree, polars::prelude::DataFrame};
use rusmlai::{metadata::models::ModelCache, prelude::*, steps};

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
    // The 30 features with the target passed through: the input a model's fit gets. It keeps
    // the CSV's path, so its fits go through the model cache. `memory` is the same table with
    // no CSV, so no cache: the training alone, like scikit-learn's fit.
    let features: Dataset = preprocessing().remainder("passthrough").fit_transform(&ds);
    assert_eq!((features.height(), features.width()), (630_000, 31));
    let memory: Dataset = Dataset::new(features.dataframe().clone(), "");
    let ds_memory: Dataset = Dataset::new(ds.dataframe().clone(), "");
    let cache: PathBuf = ModelCache::of(&features).unwrap().path;
    let no_cache = || {
        let _ = fs::remove_file(&cache);
    };

    let mut g = c.benchmark_group("linear");
    g.sample_size(20).measurement_time(Duration::from_secs(10));
    g.bench_function("fit/LinearRegression", |b| {
        b.iter(|| {
            let mut model: LinearRegression = LinearRegression::new();
            model.fit(&memory, TARGET);
            black_box(model)
        })
    });
    // A miss: the data hash, the training and one appended line.
    g.bench_function("fit/LinearRegression/cache_miss", |b| {
        b.iter_batched(
            no_cache,
            |()| {
                let mut model: LinearRegression = LinearRegression::new();
                model.fit(&features, TARGET);
                black_box(model)
            },
            BatchSize::PerIteration,
        )
    });
    // A hit: the data hash, reading the file and loading what the miss learned.
    g.bench_function("fit/LinearRegression/cache_hit", |b| {
        b.iter(|| {
            let mut model: LinearRegression = LinearRegression::new();
            model.fit(&features, TARGET);
            black_box(model)
        })
    });
    g.bench_function("fit/Ridge", |b| {
        b.iter(|| {
            let mut model: Ridge = Ridge::new();
            model.fit(&memory, TARGET);
            black_box(model)
        })
    });
    let mut fitted: LinearRegression = LinearRegression::new();
    fitted.fit(&memory, TARGET);
    g.bench_function("predict/LinearRegression", |b| {
        b.iter(|| black_box(fitted.predict(&memory)))
    });
    g.bench_function("score/LinearRegression", |b| {
        b.iter(|| black_box(fitted.score(&memory, TARGET)))
    });
    // The score plus the data hash, one appended line and the journal rewrite. Each iteration
    // starts from the cache file with the one fit line, so the journal it writes is the same.
    let mut cached: LinearRegression = LinearRegression::new();
    cached.fit(&features, TARGET);
    let fit_line: String = fs::read_to_string(&cache).unwrap();
    g.bench_function("score/LinearRegression/cached", |b| {
        b.iter_batched(
            || fs::write(&cache, &fit_line).unwrap(),
            |()| black_box(cached.score(&features, TARGET)),
            BatchSize::PerIteration,
        )
    });
    // From the loaded CSV: the preprocessing fits from the cached stats, then the model.
    g.bench_function("pipeline_fit/LinearRegression", |b| {
        b.iter(|| {
            let mut pipe = Pipeline::new(preprocessing(), LinearRegression::new());
            pipe.fit(&ds_memory, TARGET);
            black_box(pipe)
        })
    });
    // The journal with 50 runs, each fitted and scored once: what every fit and score rewrites
    // after a search of 50 settings.
    no_cache();
    for k in 0..50 {
        let mut model: Ridge = Ridge::new().alpha(f64::from(k));
        model.fit(&features, TARGET);
        model.score(&features, TARGET);
    }
    g.bench_function("journal/50_runs", |b| {
        b.iter(|| black_box(features.try_journal().unwrap()))
    });
    g.bench_function("pipeline_fit/LinearRegression/cache_miss", |b| {
        b.iter_batched(
            no_cache,
            |()| {
                let mut pipe = Pipeline::new(preprocessing(), LinearRegression::new());
                pipe.fit(&ds, TARGET);
                black_box(pipe)
            },
            BatchSize::PerIteration,
        )
    });
    g.finish();
}

/// S6E2's 13 features, all numeric.
const S6E2: [&str; 13] = [
    "Age",
    "Sex",
    "Chest pain type",
    "BP",
    "Cholesterol",
    "FBS over 120",
    "EKG results",
    "Max HR",
    "Exercise angina",
    "ST depression",
    "Slope of ST",
    "Number of vessels fluro",
    "Thallium",
];

fn logistic(c: &mut Criterion) {
    // Made in memory, so nothing writes into the fixture's caches.
    let csv: PathBuf = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/s6e2/train.csv"
    ));
    let raw: Dataset = Dataset::new(common::load(&csv), "").drop(&["id"]);
    let features: Dataset = ColumnTransformer::new()
        .add("num", &S6E2, steps![StandardScaler::new()])
        .remainder("passthrough")
        .fit_transform(&raw);
    assert_eq!((features.height(), features.width()), (630_000, 14));

    let mut g = c.benchmark_group("logistic");
    g.sample_size(20).measurement_time(Duration::from_secs(10));
    g.bench_function("fit/LogisticRegression", |b| {
        b.iter(|| {
            let mut model: LogisticRegression = LogisticRegression::new();
            model.fit(&features, "Heart Disease");
            black_box(model)
        })
    });
    let mut fitted: LogisticRegression = LogisticRegression::new();
    fitted.fit(&features, "Heart Disease");
    g.bench_function("predict_proba/LogisticRegression", |b| {
        b.iter(|| black_box(fitted.predict_proba(&features)))
    });
    g.bench_function("predict/LogisticRegression", |b| {
        b.iter(|| black_box(fitted.predict(&features)))
    });
    // predict, predict_proba, the accuracy and the log loss, on a text target.
    g.bench_function("score/LogisticRegression", |b| {
        b.iter(|| black_box(fitted.score(&features, "Heart Disease")))
    });
    g.finish();
}

/// The time criterion measures each tree bench for: 20 s, or `RUSMLAI_BENCH_SECONDS` for a
/// quick look while tuning.
fn seconds() -> Duration {
    let seconds: Option<u64> = std::env::var("RUSMLAI_BENCH_SECONDS")
        .ok()
        .and_then(|s| s.parse().ok());
    Duration::from_secs(seconds.unwrap_or(20))
}

/// With `RUSMLAI_FIXED=name:n`, runs the fit `name` n times and exits, for a count of CPU
/// cycles that a hot CPU does not change: `perf stat -e cycles:u` on the bench binary, minus
/// the same with n = 0.
fn fixed(name: &str, mut fit: impl FnMut()) {
    let Ok(asked) = std::env::var("RUSMLAI_FIXED") else {
        return;
    };
    if let Some((wanted, times)) = asked.split_once(':')
        && wanted == name
    {
        let start = std::time::Instant::now();
        let times: u32 = times.parse().unwrap_or(1);
        (0..times).for_each(|_| fit());
        eprintln!("{name}: {times} fits in {:?}", start.elapsed());
        std::process::exit(0);
    }
}

/// Times a whole `fit` of `make()` on a fresh `Dataset` of `frame` each time, so that no fit
/// takes the buckets an earlier fit kept: scikit-learn's side makes its own on every fit. The
/// copy is made outside the timing. `RUSMLAI_FIXED` runs it the same way.
fn bench_fit<M: Model>(
    g: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    frame: &DataFrame,
    target: &str,
    make: impl Fn() -> M,
) {
    let fresh = || Dataset::new(frame.clone(), "");
    fixed(name, || make().fit(&fresh(), target));
    g.bench_function(name, |b| {
        b.iter_batched(
            fresh,
            |dataset| {
                let mut model: M = make();
                model.fit(&dataset, target);
                black_box(model)
            },
            BatchSize::PerIteration,
        )
    });
}

/// The S6E1 table with the 7 text columns as `OrdinalEncoder` codes, then the 4 numbers and
/// the target (630,000 rows), and the S6E2 table with its 13 numbers and its text target.
fn tree_tables() -> (DataFrame, DataFrame) {
    let mut ordinal = ColumnTransformer::new()
        .add("cat", &TEXT, steps![OrdinalEncoder::new()])
        .remainder("passthrough");
    let source = |name: &str| {
        let csv: String = format!("{}/tests/fixtures/{name}/train.csv", env!("CARGO_MANIFEST_DIR"));
        Dataset::new(common::load(&PathBuf::from(csv)), "").drop(&["id"])
    };
    let s6e1: DataFrame = ordinal.fit_transform(&source("s6e1")).dataframe().clone();
    let s6e2: DataFrame = source("s6e2").dataframe().clone();
    assert_eq!((s6e1.shape(), s6e2.shape()), ((630_000, 12), (630_000, 14)));
    (s6e1, s6e2)
}

/// The nodes of all `trees`.
fn nodes(trees: &[Tree]) -> usize {
    trees.iter().map(Tree::node_count).sum()
}

const HEART: &str = "Heart Disease";

/// `DecisionTreeRegressor` on S6E1 and `DecisionTreeClassifier` on S6E2 ([`tree_tables`]):
/// the matrices and settings of `bench-python/trees.py`. A fit is the whole `fit` call on a
/// dataset in memory: the buckets, the labels and the tree.
fn tree(c: &mut Criterion) {
    let (s6e1, s6e2): (DataFrame, DataFrame) = tree_tables();
    let csv: PathBuf = common::scratch_csv("trees");
    let cached_table = || {
        ColumnTransformer::new()
            .add("cat", &TEXT, steps![OrdinalEncoder::new()])
            .remainder("passthrough")
            .fit_transform(&read_csv(&csv).drop(&["id"]))
    };
    let (ds1, ds2): (Dataset, Dataset) = (Dataset::new(s6e1.clone(), ""), Dataset::new(s6e2.clone(), ""));
    let mut g = c.benchmark_group("tree");
    g.sample_size(10).measurement_time(seconds());
    let regressors = [
        ("fit/DecisionTreeRegressor", DecisionTreeRegressor::new()),
        ("fit/DecisionTreeRegressor/max_depth_8", DecisionTreeRegressor::new().max_depth(8)),
    ];
    for (name, settings) in &regressors {
        let mut model: DecisionTreeRegressor = settings.clone();
        model.fit(&ds1, TARGET);
        // What the timed fit builds, to set next to scikit-learn's tree.
        eprintln!(
            "{name}: nodes {} leaves {} depth {} train r2 {:.6}",
            model.tree_.node_count(),
            model.get_n_leaves(),
            model.get_depth(),
            model.score(&ds1, TARGET)
        );
        bench_fit(&mut g, name, &s6e1, TARGET, || settings.clone());
    }
    // The same fit on a dataset that already holds its buckets: a refit, a search, a bag.
    g.bench_function("fit/DecisionTreeRegressor/max_depth_8/cached_buckets", |b| {
        b.iter(|| {
            let mut model: DecisionTreeRegressor = DecisionTreeRegressor::new().max_depth(8);
            model.fit(&ds1, TARGET);
            black_box(model)
        })
    });
    let mut full: DecisionTreeRegressor = DecisionTreeRegressor::new();
    full.fit(&ds1, TARGET);
    g.bench_function("predict/DecisionTreeRegressor", |b| {
        b.iter(|| black_box(full.predict(&ds1)))
    });
    // A miss: the buckets, the data hash, the training, the tree written to its file and one
    // index line. A hit: the data hash, and the tree read back.
    let cached: Dataset = cached_table();
    let index: PathBuf = ModelCache::of(&cached).unwrap().path;
    for (name, settings) in &regressors {
        g.bench_function(format!("{name}/cache_miss"), |b| {
            b.iter_batched(
                || {
                    let _ = fs::remove_file(&index);
                    cached_table()
                },
                |cached| {
                    let mut model: DecisionTreeRegressor = settings.clone();
                    model.fit(&cached, TARGET);
                    black_box(model)
                },
                BatchSize::PerIteration,
            )
        });
        g.bench_function(format!("{name}/cache_hit"), |b| {
            b.iter(|| {
                let mut model: DecisionTreeRegressor = settings.clone();
                model.fit(&cached, TARGET);
                black_box(model)
            })
        });
    }

    let classifiers = [
        ("fit/DecisionTreeClassifier", DecisionTreeClassifier::new()),
        ("fit/DecisionTreeClassifier/max_depth_8", DecisionTreeClassifier::new().max_depth(8)),
    ];
    for (name, settings) in &classifiers {
        let mut model: DecisionTreeClassifier = settings.clone();
        model.fit(&ds2, HEART);
        eprintln!(
            "{name}: nodes {} leaves {} depth {} train accuracy {:.6}",
            model.tree_.node_count(),
            model.get_n_leaves(),
            model.get_depth(),
            model.score(&ds2, HEART)
        );
        bench_fit(&mut g, name, &s6e2, HEART, || settings.clone());
    }
    let mut full: DecisionTreeClassifier = DecisionTreeClassifier::new();
    full.fit(&ds2, HEART);
    g.bench_function("predict/DecisionTreeClassifier", |b| {
        b.iter(|| black_box(full.predict(&ds2)))
    });
    g.bench_function("predict_proba/DecisionTreeClassifier", |b| {
        b.iter(|| black_box(full.predict_proba(&ds2)))
    });
    g.finish();
}

/// `RandomForest*` and `Bagging*` of 10 full trees on the matrices of [`tree`]: the settings
/// of `bench-python/forests.py`. 10 trees, not scikit-learn's 100, so that its side ends.
fn forest(c: &mut Criterion) {
    let (s6e1, s6e2): (DataFrame, DataFrame) = tree_tables();
    let (ds1, ds2): (Dataset, Dataset) = (Dataset::new(s6e1.clone(), ""), Dataset::new(s6e2.clone(), ""));
    let mut g = c.benchmark_group("forest");
    g.sample_size(10).measurement_time(seconds());

    let regressor = || RandomForestRegressor::new().n_estimators(10);
    let mut fitted: RandomForestRegressor = regressor().oob_score(true);
    fitted.fit(&ds1, TARGET);
    // What the timed fit builds, to set next to scikit-learn's forest.
    eprintln!(
        "fit/RandomForestRegressor: nodes {} train r2 {:.6} oob r2 {:.6}",
        nodes(&fitted.estimators_),
        fitted.score(&ds1, TARGET),
        fitted.oob_score_.unwrap()
    );
    bench_fit(&mut g, "fit/RandomForestRegressor", &s6e1, TARGET, regressor);
    g.bench_function("predict/RandomForestRegressor", |b| {
        b.iter(|| black_box(fitted.predict(&ds1)))
    });

    let classifier = || RandomForestClassifier::new().n_estimators(10);
    let mut fitted: RandomForestClassifier = classifier().oob_score(true);
    fitted.fit(&ds2, HEART);
    eprintln!(
        "fit/RandomForestClassifier: nodes {} train accuracy {:.6} oob accuracy {:.6}",
        nodes(&fitted.estimators_),
        fitted.score(&ds2, HEART),
        fitted.oob_score_.unwrap()
    );
    bench_fit(&mut g, "fit/RandomForestClassifier", &s6e2, HEART, classifier);
    g.bench_function("predict/RandomForestClassifier", |b| {
        b.iter(|| black_box(fitted.predict(&ds2)))
    });
    g.bench_function("predict_proba/RandomForestClassifier", |b| {
        b.iter(|| black_box(fitted.predict_proba(&ds2)))
    });
    // `class_weight="balanced"`: a weight per row on top of the bootstrap counts.
    let mut fitted: RandomForestClassifier = classifier().class_weight("balanced").oob_score(true);
    fitted.fit(&ds2, HEART);
    eprintln!(
        "fit/RandomForestClassifier/balanced: nodes {} train accuracy {:.6} oob accuracy {:.6}",
        nodes(&fitted.estimators_),
        fitted.score(&ds2, HEART),
        fitted.oob_score_.unwrap()
    );
    bench_fit(&mut g, "fit/RandomForestClassifier/balanced", &s6e2, HEART, || {
        classifier().class_weight("balanced")
    });

    // A bag of trees: each of the 10 copies takes its sample as row weights. The copies share
    // the buckets, which the first one makes.
    let bag = || BaggingClassifier::new(DecisionTreeClassifier::new());
    let mut fitted = bag();
    fitted.fit(&ds2, HEART);
    let trees: Vec<Tree> = fitted.estimators_.iter().map(|copy| copy.tree_.clone()).collect();
    eprintln!(
        "fit/BaggingClassifier: nodes {} train accuracy {:.6}",
        nodes(&trees),
        fitted.score(&ds2, HEART)
    );
    bench_fit(&mut g, "fit/BaggingClassifier", &s6e2, HEART, bag);
    let bag = || BaggingRegressor::new(DecisionTreeRegressor::new());
    let mut fitted = bag();
    fitted.fit(&ds1, TARGET);
    let trees: Vec<Tree> = fitted.estimators_.iter().map(|copy| copy.tree_.clone()).collect();
    eprintln!(
        "fit/BaggingRegressor: nodes {} train r2 {:.6}",
        nodes(&trees),
        fitted.score(&ds1, TARGET)
    );
    bench_fit(&mut g, "fit/BaggingRegressor", &s6e1, TARGET, bag);
    g.finish();
}

/// `GradientBoosting*` (histogram boosting), `AdaBoostClassifier` and `GridSearchCV` on the
/// matrices of [`tree`]: the settings of `bench-python/boosting.py`. Boosting runs its 100
/// rounds without early stopping, so both sides grow the same number of trees; one bench
/// times the default, with early stopping, as a user gets it.
fn boost(c: &mut Criterion) {
    let (s6e1, s6e2): (DataFrame, DataFrame) = tree_tables();
    let (ds1, ds2): (Dataset, Dataset) = (Dataset::new(s6e1.clone(), ""), Dataset::new(s6e2.clone(), ""));
    let mut g = c.benchmark_group("boost");
    g.sample_size(10).measurement_time(seconds());
    let leaves = |trees: &[Tree]| -> usize { trees.iter().map(Tree::n_leaves).sum() };

    let regressor = || GradientBoostingRegressor::new().early_stopping(false);
    let mut fitted: GradientBoostingRegressor = regressor();
    fitted.fit(&ds1, TARGET);
    eprintln!(
        "fit/GradientBoostingRegressor: rounds {} leaves {} train r2 {:.6}",
        fitted.n_iter_,
        leaves(&fitted.estimators_),
        fitted.score(&ds1, TARGET)
    );
    bench_fit(&mut g, "fit/GradientBoostingRegressor", &s6e1, TARGET, regressor);
    g.bench_function("predict/GradientBoostingRegressor", |b| {
        b.iter(|| black_box(fitted.predict(&ds1)))
    });

    // The 7 text columns as categories, not codes: a split sends a set of categories left.
    // scikit-learn's side gives them the pandas category dtype.
    let csv: String = format!("{}/tests/fixtures/s6e1/train.csv", env!("CARGO_MANIFEST_DIR"));
    let raw: DataFrame = Dataset::new(common::load(&PathBuf::from(csv)), "").drop(&["id"]).dataframe().clone();
    let mut fitted: GradientBoostingRegressor = regressor();
    fitted.fit(&Dataset::new(raw.clone(), ""), TARGET);
    eprintln!(
        "fit/GradientBoostingRegressor/categorical: rounds {} leaves {} train r2 {:.6}",
        fitted.n_iter_,
        leaves(&fitted.estimators_),
        fitted.score(&Dataset::new(raw.clone(), ""), TARGET)
    );
    bench_fit(&mut g, "fit/GradientBoostingRegressor/categorical", &raw, TARGET, regressor);

    let classifier = || GradientBoostingClassifier::new().early_stopping(false);
    let mut fitted: GradientBoostingClassifier = classifier();
    fitted.fit(&ds2, HEART);
    eprintln!(
        "fit/GradientBoostingClassifier: rounds {} leaves {} train accuracy {:.6}",
        fitted.n_iter_,
        leaves(&fitted.estimators_),
        fitted.score(&ds2, HEART)
    );
    bench_fit(&mut g, "fit/GradientBoostingClassifier", &s6e2, HEART, classifier);
    g.bench_function("predict_proba/GradientBoostingClassifier", |b| {
        b.iter(|| black_box(fitted.predict_proba(&ds2)))
    });
    let mut early: GradientBoostingClassifier = GradientBoostingClassifier::new();
    early.fit(&ds2, HEART);
    eprintln!(
        "fit/GradientBoostingClassifier/early_stopping: rounds {} leaves {} train accuracy {:.6}",
        early.n_iter_,
        leaves(&early.estimators_),
        early.score(&ds2, HEART)
    );
    bench_fit(&mut g, "fit/GradientBoostingClassifier/early_stopping", &s6e2, HEART, GradientBoostingClassifier::new);

    let ada = || AdaBoostClassifier::new();
    let mut fitted: AdaBoostClassifier = ada();
    fitted.fit(&ds2, HEART);
    eprintln!(
        "fit/AdaBoostClassifier: stumps {} train accuracy {:.6}",
        fitted.estimators_.len(),
        fitted.score(&ds2, HEART)
    );
    bench_fit(&mut g, "fit/AdaBoostClassifier", &s6e2, HEART, ada);

    // 6 trees × 5 folds, then the refit on every row.
    let search = || {
        GridSearchCV::new(DecisionTreeClassifier::new())
            .param_grid("max_depth", [4, 8, 12])
            .param_grid("min_samples_leaf", [1, 50])
    };
    let mut fitted = search();
    fitted.fit(&ds2, HEART);
    eprintln!(
        "fit/GridSearchCV: best {} mean {:?}",
        fitted.best_params_,
        fitted.cv_results_.to_list("mean_test_score")
    );
    bench_fit(&mut g, "fit/GridSearchCV", &s6e2, HEART, search);
    g.finish();
}

criterion_group!(benches, linear, logistic, tree, forest, boost);
criterion_main!(benches);
