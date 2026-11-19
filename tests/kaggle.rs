//! Smoke test on the local Kaggle file (not tracked): `cargo test -- --ignored`.

use std::{fs, path::PathBuf};

use rusmlai::prelude::*;

#[test]
#[ignore = "needs tests/fixtures/s6e1/train.csv, which only exists locally"]
fn kaggle_s6e1_reads_and_draws() {
    let source: PathBuf =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/s6e1/train.csv");
    if !source.is_file() {
        eprintln!("skipped: {} is missing", source.display());
        return;
    }
    let dir: PathBuf = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("kaggle");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let csv: PathBuf = dir.join("train.csv");
    fs::copy(&source, &csv).unwrap();

    let ds: Dataset = read_csv(&csv);
    assert!(ds.height() > 100_000);
    assert_eq!(read_csv(&csv).shape(), ds.shape()); // the cached read
    let numeric: Vec<String> = ds
        .columns()
        .into_iter()
        .filter(|c| ds.try_column_info(c).unwrap().dtype.is_numeric())
        .collect();
    assert!(ds.hist(&numeric[1], "", 30).svg().starts_with("<svg"));
    let target: String = ds.columns().last().unwrap().clone();
    assert!(
        !ds.drop(&[ds.columns()[0].as_str()])
            .information_gain(&target)
            .is_empty()
    );
}

/// The same split, preprocessing and models as `analysis/2026-11-11/linear-bench/parity.py`,
/// which gave these scikit-learn 1.9.1 values: (r2, mse, rmse, mae, intercept).
const SKLEARN: [(&str, [f64; 5]); 2] = [
    (
        "LinearRegression",
        [
            0.780338007014,
            79.118311450163,
            8.894847466380,
            7.107754790594,
            62.519760569949,
        ],
    ),
    (
        "Ridge",
        [
            0.780338003432,
            79.118312740233,
            8.894847538898,
            7.107755635474,
            62.519760662643,
        ],
    ),
];

#[test]
#[ignore = "needs tests/fixtures/s6e1/train.csv, which only exists locally"]
fn linear_models_match_scikit_learn_on_s6e1() {
    let source: PathBuf =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/s6e1/train.csv");
    if !source.is_file() {
        eprintln!("skipped: {} is missing", source.display());
        return;
    }
    let all: Dataset = Dataset::new(read_csv(&source).drop(&["id"]).into_dataframe(), "");
    // A fixed split, so both libraries see the same rows: first 504,000 train, last 126,000 test.
    let train: Dataset = Dataset::new(all.dataframe().slice(0, 504_000), "");
    let test: Dataset = Dataset::new(all.dataframe().slice(504_000, 126_000), "");
    let numeric: [&str; 4] = ["age", "study_hours", "class_attendance", "sleep_hours"];
    let text: [&str; 7] = [
        "gender",
        "course",
        "internet_access",
        "sleep_quality",
        "study_method",
        "facility_rating",
        "exam_difficulty",
    ];
    let prep = || {
        ColumnTransformer::new()
            .add("num", &numeric, rusmlai::steps![StandardScaler::new()])
            .add("cat", &text, rusmlai::steps![OneHotEncoder::new()])
    };
    let y = test
        .dataframe()
        .column("exam_score")
        .unwrap()
        .as_materialized_series()
        .clone();
    for (name, want) in SKLEARN {
        let (predicted, intercept) = if name == "Ridge" {
            let mut pipe = Pipeline::new(prep(), Ridge::new());
            pipe.fit(&train, "exam_score");
            (pipe.predict(&test), pipe.model.intercept_)
        } else {
            let mut pipe = Pipeline::new(prep(), LinearRegression::new());
            pipe.fit(&train, "exam_score");
            (pipe.predict(&test), pipe.model.intercept_)
        };
        let got: [f64; 5] = [
            r2_score(&y, &predicted),
            mean_squared_error(&y, &predicted),
            root_mean_squared_error(&y, &predicted),
            mean_absolute_error(&y, &predicted),
            intercept,
        ];
        eprintln!("{name}: rusmlai {got:?}\n{name}: sklearn {want:?}");
        for (label, (g, w)) in ["r2", "mse", "rmse", "mae", "intercept"]
            .iter()
            .zip(got.iter().zip(want))
        {
            assert!(
                (g - w).abs() <= 1e-8 * w.abs().max(1.0),
                "{name} {label}: {g} against {w}"
            );
        }
    }
}

/// scikit-learn 1.9.1 `LogisticRegression(solver="newton-cholesky")` on the same split and
/// preprocessing (`analysis/2026-11-14/logistic-parity.py`): test accuracy, test log loss,
/// the intercept and the Newton steps. Its default L-BFGS stops at a different point of the
/// same flat optimum (intercept -0.2258, log loss 0.276158887682), so Newton is the reference.
#[test]
#[ignore = "needs tests/fixtures/s6e2/train.csv, which only exists locally"]
fn logistic_regression_matches_scikit_learn_on_s6e2() {
    let source: PathBuf =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/s6e2/train.csv");
    if !source.is_file() {
        eprintln!("skipped: {} is missing", source.display());
        return;
    }
    let all: Dataset = Dataset::new(read_csv(&source).drop(&["id"]).into_dataframe(), "");
    let train: Dataset = Dataset::new(all.dataframe().slice(0, 504_000), "");
    let test: Dataset = Dataset::new(all.dataframe().slice(504_000, 126_000), "");
    let prep: ColumnTransformer = ColumnTransformer::new()
        .add(
            "num",
            &["Age", "BP", "Cholesterol", "Max HR", "ST depression"],
            rusmlai::steps![StandardScaler::new()],
        )
        .add(
            "code",
            &[
                "Chest pain type",
                "EKG results",
                "Slope of ST",
                "Number of vessels fluro",
                "Thallium",
            ],
            rusmlai::steps![OneHotEncoder::new()],
        )
        .remainder("passthrough");
    let mut pipe = Pipeline::new(prep, LogisticRegression::new());
    pipe.fit(&train, "Heart Disease");
    let y = test
        .dataframe()
        .column("Heart Disease")
        .unwrap()
        .as_materialized_series()
        .clone();
    let accuracy: f64 = accuracy_score(&y, &pipe.predict(&test));
    let loss: f64 = log_loss(&y, &pipe.predict_proba(&test));
    let model: &LogisticRegression = &pipe.model;
    eprintln!(
        "rusmlai: accuracy {accuracy:.12} log_loss {loss:.12} intercept {:.12} n_iter {}",
        model.intercept_[0], model.n_iter_
    );
    assert_eq!(model.classes_, ["Absence", "Presence"]);
    assert_eq!((model.coef_[0].len(), model.n_iter_), (25, 5));
    assert!(
        (accuracy - 0.885682539683).abs() < 1e-11,
        "accuracy {accuracy}"
    );
    assert!((loss - 0.276141205869).abs() < 1e-10, "log loss {loss}");
    assert!(
        (model.intercept_[0] + 0.653501039672).abs() < 1e-8,
        "intercept {}",
        model.intercept_[0]
    );
}

/// scikit-learn 1.9.1 trees on the same fixed splits
/// (`analysis/2026-11-16/tree-bench/kaggle-parity.py`, three seeds each).
///
/// S6E2's features have at most 150 distinct values, so the buckets are exact and a tree
/// without ties is scikit-learn's: depth 8 agrees in every digit. Deeper trees have ties,
/// which scikit-learn breaks by seed (12,085 or 12,087 nodes with `min_samples_leaf = 50`;
/// 143,239 to 143,295 for a full tree), so there the sizes and scores are close, not equal.
///
/// S6E1's `study_hours` and `class_attendance` have 792 and 617 distinct values, more than
/// 255 buckets: depth 8 scores as scikit-learn's exact tree does to 3 digits, and a full tree
/// stops at the buckets (scikit-learn: 980,957 nodes, test R² 0.543).
#[test]
#[ignore = "needs tests/fixtures/s6e1 and s6e2, which only exist locally"]
fn trees_match_scikit_learn_on_the_kaggle_files() {
    let root: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let (s6e1, s6e2) = (root.join("s6e1/train.csv"), root.join("s6e2/train.csv"));
    if !s6e1.is_file() || !s6e2.is_file() {
        eprintln!("skipped: the Kaggle files are missing");
        return;
    }
    let split = |csv: &PathBuf| {
        let all: Dataset = Dataset::new(read_csv(csv).drop(&["id"]).into_dataframe(), "");
        let train: Dataset = Dataset::new(all.dataframe().slice(0, 504_000), "");
        let test: Dataset = Dataset::new(all.dataframe().slice(504_000, 126_000), "");
        (train, test)
    };

    let (train, test) = split(&s6e2);
    let target: &str = "Heart Disease";
    let y = test.dataframe().column(target).unwrap().as_materialized_series().clone();
    let fit = |name: &str, model: DecisionTreeClassifier| {
        let mut model: DecisionTreeClassifier = model;
        model.fit(&train, target);
        let scores: [f64; 3] = [
            model.score(&train, target),
            accuracy_score(&y, &model.predict(&test)),
            log_loss(&y, &model.predict_proba(&test)),
        ];
        eprintln!(
            "s6e2 classifier {name}: nodes {} leaves {} depth {} train {:.12} test {:.12} log_loss {:.12}",
            model.tree_.node_count(),
            model.get_n_leaves(),
            model.get_depth(),
            scores[0],
            scores[1],
            scores[2]
        );
        (model.tree_.node_count(), scores)
    };
    let (nodes, scores) = fit("depth 8", DecisionTreeClassifier::new().max_depth(8));
    assert_eq!(nodes, 511);
    for (got, want) in scores.iter().zip([0.879827380952, 0.877841269841, 0.292600507163]) {
        assert!((got - want).abs() < 1e-11, "depth 8: {got} against {want}");
    }
    let (nodes, scores) = fit("leaf 50", DecisionTreeClassifier::new().min_samples_leaf(50));
    assert!((12_075..=12_095).contains(&nodes), "{nodes} nodes");
    assert!((scores[0] - 0.889589285714).abs() < 1e-11, "train {}", scores[0]);
    assert!((scores[1] - 0.8814).abs() < 5e-4, "test {}", scores[1]);
    let entropy = DecisionTreeClassifier::new().criterion("entropy").max_depth(12);
    let (nodes, scores) = fit("entropy depth 12", entropy);
    assert!((6_245..=6_265).contains(&nodes), "{nodes} nodes");
    assert!((scores[0] - 0.89128).abs() < 2e-5, "train {}", scores[0]);
    let (nodes, scores) = fit("full", DecisionTreeClassifier::new());
    assert!((143_000..=143_500).contains(&nodes), "{nodes} nodes");
    assert_eq!(scores[0], 1.0);
    assert!((scores[1] - 0.824).abs() < 2e-3, "test {}", scores[1]);

    let (train, test) = split(&s6e1);
    let text: [&str; 7] = [
        "gender",
        "course",
        "internet_access",
        "sleep_quality",
        "study_method",
        "facility_rating",
        "exam_difficulty",
    ];
    let fit = |name: &str, model: DecisionTreeRegressor| {
        let prep: ColumnTransformer = ColumnTransformer::new()
            .add("cat", &text, rusmlai::steps![OrdinalEncoder::new()])
            .remainder("passthrough");
        let mut pipe = Pipeline::new(prep, model);
        pipe.fit(&train, "exam_score");
        let scores: [f64; 2] = [pipe.score(&train, "exam_score"), pipe.score(&test, "exam_score")];
        eprintln!(
            "s6e1 regressor {name}: nodes {} leaves {} depth {} train {:.12} test {:.12}",
            pipe.model.tree_.node_count(),
            pipe.model.get_n_leaves(),
            pipe.model.get_depth(),
            scores[0],
            scores[1]
        );
        (pipe.model.tree_.node_count(), scores)
    };
    let (nodes, scores) = fit("depth 8", DecisionTreeRegressor::new().max_depth(8));
    assert_eq!(nodes, 511);
    assert!((scores[0] - 0.734989549140).abs() < 1e-3, "train {}", scores[0]);
    assert!((scores[1] - 0.734191950729).abs() < 1e-3, "test {}", scores[1]);
    let (_, scores) = fit("leaf 50", DecisionTreeRegressor::new().min_samples_leaf(50));
    assert!((scores[1] - 0.766992839836).abs() < 3e-3, "test {}", scores[1]);
    let (_, scores) = fit("full", DecisionTreeRegressor::new());
    assert!(scores[1] > 0.5, "test {}", scores[1]);
}

/// scikit-learn 1.9.1 forests and bags on the same fixed splits
/// (`analysis/2026-11-16/tree-bench/forest-parity.py`, three seeds each, in the comments).
/// A forest draws its rows and features at random, so no two fits grow the same trees: the
/// scores must be as close to scikit-learn's as its own seeds are to each other.
#[test]
#[ignore = "needs tests/fixtures/s6e1 and s6e2, which only exist locally"]
fn forests_score_like_scikit_learn_on_the_kaggle_files() {
    let root: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let (s6e1, s6e2) = (root.join("s6e1/train.csv"), root.join("s6e2/train.csv"));
    if !s6e1.is_file() || !s6e2.is_file() {
        eprintln!("skipped: the Kaggle files are missing");
        return;
    }
    let split = |csv: &PathBuf| {
        let all: Dataset = Dataset::new(read_csv(csv).drop(&["id"]).into_dataframe(), "");
        let train: Dataset = Dataset::new(all.dataframe().slice(0, 504_000), "");
        let test: Dataset = Dataset::new(all.dataframe().slice(504_000, 126_000), "");
        (train, test)
    };
    let near = |name: &str, what: &str, got: f64, want: f64, by: f64| {
        assert!((got - want).abs() <= by, "{name} {what}: {got} against {want}");
    };

    let (train, test) = split(&s6e2);
    let target: &str = "Heart Disease";
    let y = test.dataframe().column(target).unwrap().as_materialized_series().clone();
    // (test accuracy, log loss) of the fitted classifier.
    let scores = |name: &str, seed: u64, model: &dyn Classifier, nodes: usize, oob: Option<f64>| {
        let accuracy: f64 = accuracy_score(&y, &model.predict(&test));
        let loss: f64 = log_loss(&y, &model.predict_proba(&test));
        eprintln!(
            "s6e2 {name} seed {seed}: nodes {nodes} train {:.6} test {accuracy:.6} log_loss {loss:.6} oob {:.6}",
            model.score(&train, target),
            oob.unwrap_or(f64::NAN)
        );
        (accuracy, loss)
    };
    let nodes = |trees: &[rusmlai::models::tree::Tree]| -> usize {
        trees.iter().map(|tree| tree.node_count()).sum()
    };
    for seed in 0..3 {
        // scikit-learn: test 0.876952 to 0.877468, log loss 0.640 to 0.649, oob 0.8684 to 0.8689.
        let mut model = RandomForestClassifier::new().n_estimators(20).oob_score(true).random_state(seed);
        model.fit(&train, target);
        let name: &str = "forest 20 trees";
        let (accuracy, loss) = scores(name, seed, &model, nodes(&model.estimators_), model.oob_score_);
        near(name, "accuracy", accuracy, 0.8771, 0.002);
        near(name, "log loss", loss, 0.645, 0.03);
        near(name, "oob", model.oob_score_.unwrap(), 0.8686, 0.002);

        // scikit-learn: test 0.883421 to 0.884063, log loss 0.2939 to 0.2975, oob 0.8814 to 0.8816.
        let mut model = RandomForestClassifier::new()
            .n_estimators(20)
            .min_samples_leaf(5)
            .oob_score(true)
            .random_state(seed);
        model.fit(&train, target);
        let name: &str = "forest 20 trees leaf 5";
        let (accuracy, loss) = scores(name, seed, &model, nodes(&model.estimators_), model.oob_score_);
        near(name, "accuracy", accuracy, 0.8838, 0.0015);
        near(name, "log loss", loss, 0.2955, 0.01);
        near(name, "oob", model.oob_score_.unwrap(), 0.8815, 0.0015);

        // scikit-learn: test 0.878175 to 0.878754, log loss 0.477 to 0.485.
        let tree = DecisionTreeClassifier::new().min_samples_leaf(5);
        let mut model = BaggingClassifier::new(tree).random_state(seed);
        model.fit(&train, target);
        let name: &str = "bag 10 trees leaf 5";
        let trees: Vec<_> = model.estimators_.iter().map(|copy| copy.tree_.clone()).collect();
        let (accuracy, loss) = scores(name, seed, &model, nodes(&trees), None);
        near(name, "accuracy", accuracy, 0.8785, 0.002);
        near(name, "log loss", loss, 0.48, 0.03);
    }

    let (train, test) = split(&s6e1);
    let text: [&str; 7] = [
        "gender",
        "course",
        "internet_access",
        "sleep_quality",
        "study_method",
        "facility_rating",
        "exam_difficulty",
    ];
    let mut prep: ColumnTransformer = ColumnTransformer::new()
        .add("cat", &text, rusmlai::steps![OrdinalEncoder::new()])
        .remainder("passthrough");
    let train: Dataset = prep.fit_transform(&train);
    let test: Dataset = prep.transform(&test);
    let y = test.dataframe().column("exam_score").unwrap().as_materialized_series().clone();
    // (test R², RMSE) of the fitted regressor.
    let scores = |name: &str, seed: u64, model: &dyn Regressor, nodes: usize, oob: Option<f64>| {
        let predicted = model.predict(&test);
        let (r2, rmse): (f64, f64) = (r2_score(&y, &predicted), root_mean_squared_error(&y, &predicted));
        eprintln!(
            "s6e1 {name} seed {seed}: nodes {nodes} train {:.6} test {r2:.6} rmse {rmse:.6} oob {:.6}",
            model.score(&train, "exam_score"),
            oob.unwrap_or(f64::NAN)
        );
        (r2, rmse)
    };
    for seed in 0..3 {
        // scikit-learn: test R² 0.770848 to 0.771215, RMSE 9.0777 to 9.0850, oob 0.7563 to 0.7567.
        let forest = || RandomForestRegressor::new().n_estimators(20).min_samples_leaf(5).oob_score(true);
        let mut model = forest().random_state(seed);
        model.fit(&train, "exam_score");
        let name: &str = "forest 20 trees leaf 5";
        let (r2, rmse) = scores(name, seed, &model, nodes(&model.estimators_), model.oob_score_);
        near(name, "r2", r2, 0.7710, 0.002);
        near(name, "rmse", rmse, 9.082, 0.04);
        near(name, "oob", model.oob_score_.unwrap(), 0.7565, 0.002);

        // scikit-learn: test R² 0.775862 to 0.776644, RMSE 8.9693 to 8.9850, oob 0.7623 to 0.7632.
        let mut model = forest().max_features(0.33).random_state(seed);
        model.fit(&train, "exam_score");
        let name: &str = "forest 20 trees leaf 5 third";
        let (r2, rmse) = scores(name, seed, &model, nodes(&model.estimators_), model.oob_score_);
        near(name, "r2", r2, 0.7764, 0.002);
        near(name, "rmse", rmse, 8.975, 0.04);
        near(name, "oob", model.oob_score_.unwrap(), 0.7628, 0.002);

        // scikit-learn: test R² 0.765405 to 0.765688, RMSE 9.1867 to 9.1922.
        let tree = DecisionTreeRegressor::new().min_samples_leaf(5);
        let mut model = BaggingRegressor::new(tree).random_state(seed);
        model.fit(&train, "exam_score");
        let name: &str = "bag 10 trees leaf 5";
        let trees: Vec<_> = model.estimators_.iter().map(|copy| copy.tree_.clone()).collect();
        let (r2, rmse) = scores(name, seed, &model, nodes(&trees), None);
        near(name, "r2", r2, 0.7656, 0.002);
        near(name, "rmse", rmse, 9.189, 0.04);
    }
}

/// scikit-learn 1.9.1 histogram boosting, AdaBoost and a grid search on the same fixed splits
/// (`analysis/2026-11-17/boosting/kaggle-parity.py`, its numbers in the comments).
///
/// Boosting's first rounds have exact ties between splits on classes (every row of a class
/// has the same gradient), which scikit-learn breaks by rounding and rusmlai by the first
/// feature, and S6E1's two wide features get other quantile buckets: so the scores agree
/// closely, not in every digit. The rows held out for early stopping are drawn by each
/// library's own generator.
#[test]
#[ignore = "needs tests/fixtures/s6e1 and s6e2, which only exist locally"]
fn boosting_adaboost_and_search_score_like_scikit_learn_on_the_kaggle_files() {
    let root: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let (s6e1, s6e2) = (root.join("s6e1/train.csv"), root.join("s6e2/train.csv"));
    if !s6e1.is_file() || !s6e2.is_file() {
        eprintln!("skipped: the Kaggle files are missing");
        return;
    }
    let split = |csv: &PathBuf| {
        let all: Dataset = Dataset::new(read_csv(csv).drop(&["id"]).into_dataframe(), "");
        let train: Dataset = Dataset::new(all.dataframe().slice(0, 504_000), "");
        let test: Dataset = Dataset::new(all.dataframe().slice(504_000, 126_000), "");
        (train, test)
    };
    let near = |what: &str, got: f64, want: f64, by: f64| {
        assert!((got - want).abs() <= by, "{what}: {got} against {want}");
    };

    let (train, test) = split(&s6e2);
    let target: &str = "Heart Disease";
    let y = test.dataframe().column(target).unwrap().as_materialized_series().clone();
    let scores = |name: &str, model: &dyn Classifier| {
        let accuracy: f64 = accuracy_score(&y, &model.predict(&test));
        let loss: f64 = log_loss(&y, &model.predict_proba(&test));
        eprintln!("s6e2 {name}: test accuracy {accuracy:.6} log_loss {loss:.6}");
        (accuracy, loss)
    };
    // scikit-learn: 100 rounds, accuracy 0.887913, log loss 0.270137.
    let mut model = GradientBoostingClassifier::new().early_stopping(false);
    model.fit(&train, target);
    let (accuracy, loss) = scores("boosting 100 rounds", &model);
    near("boosting accuracy", accuracy, 0.887913, 0.0015);
    near("boosting log loss", loss, 0.270137, 0.002);
    // scikit-learn, seeds 0 to 2: 100 rounds each, accuracy 0.887556 to 0.887952, log loss
    // 0.270375 to 0.270478.
    let mut model = GradientBoostingClassifier::new();
    model.fit(&train, target);
    eprintln!("s6e2 boosting early stopping: rounds {}", model.n_iter_);
    let (accuracy, loss) = scores("boosting early stopping", &model);
    near("early stopping accuracy", accuracy, 0.8878, 0.0015);
    near("early stopping log loss", loss, 0.2704, 0.002);
    assert!(model.n_iter_ >= 80, "{} rounds", model.n_iter_);
    // scikit-learn: accuracy 0.885794, log loss 0.485818, first votes 1.418146, 1.055934, 0.73213.
    let mut model = AdaBoostClassifier::new();
    model.fit(&train, target);
    eprintln!("s6e2 adaboost first votes {:?}", &model.estimator_weights_[..3]);
    let (accuracy, loss) = scores("adaboost 50 stumps", &model);
    near("adaboost accuracy", accuracy, 0.885794, 0.001);
    near("adaboost log loss", loss, 0.485818, 0.002);
    for (got, want) in model.estimator_weights_.iter().zip([1.418146, 1.055934, 0.73213]) {
        near("adaboost vote", *got, want, 1e-4);
    }
    // scikit-learn: means 0.854099, 0.854099, 0.878514, 0.878542, 0.87925, 0.881556, ranks
    // 5, 5, 4, 3, 2, 1, best max_depth 12 with min_samples_leaf 50, test accuracy 0.882183.
    let mut search = GridSearchCV::new(DecisionTreeClassifier::new())
        .param_grid("max_depth", [4, 8, 12])
        .param_grid("min_samples_leaf", [1, 50]);
    search.fit(&train, target);
    let means: Vec<f64> = search.cv_results_.to_list("mean_test_score");
    eprintln!("s6e2 search: means {means:?} best {}", search.best_params_);
    let want = [0.854099, 0.854099, 0.878514, 0.878542, 0.87925, 0.881556];
    for (got, want) in means.iter().zip(want) {
        near("search mean", *got, want, 0.0015);
    }
    assert_eq!(search.best_params_["max_depth"], 12);
    assert_eq!(search.best_params_["min_samples_leaf"], 50);
    let (accuracy, _) = scores("search best", &search);
    near("search test accuracy", accuracy, 0.882183, 0.0015);

    let (train, test) = split(&s6e1);
    let text: [&str; 7] = [
        "gender",
        "course",
        "internet_access",
        "sleep_quality",
        "study_method",
        "facility_rating",
        "exam_difficulty",
    ];
    // The text columns as they are: categorical features, a set of categories per split.
    // scikit-learn with the pandas category dtype, seeds 0 to 2 (kaggle-categorical.py):
    // R² 0.784208 to 0.784448, RMSE 8.8112 to 8.8161.
    let y = test.dataframe().column("exam_score").unwrap().as_materialized_series().clone();
    let mut model = GradientBoostingRegressor::new().early_stopping(false);
    model.fit(&train, "exam_score");
    let predicted = model.predict(&test);
    let (r2, rmse) = (r2_score(&y, &predicted), root_mean_squared_error(&y, &predicted));
    let categorical: usize = model.is_categorical_.iter().filter(|c| **c).count();
    eprintln!("s6e1 boosting categorical 100 rounds: {categorical} categorical, test r2 {r2:.6} rmse {rmse:.6}");
    assert_eq!(categorical, 7);
    near("categorical boosting r2", r2, 0.78433, 0.0015);
    near("categorical boosting rmse", rmse, 8.8137, 0.03);

    let mut prep: ColumnTransformer = ColumnTransformer::new()
        .add("cat", &text, rusmlai::steps![OrdinalEncoder::new()])
        .remainder("passthrough");
    let train: Dataset = prep.fit_transform(&train);
    let test: Dataset = prep.transform(&test);
    let y = test.dataframe().column("exam_score").unwrap().as_materialized_series().clone();
    // scikit-learn, seeds 0 to 2: R² 0.783925 to 0.784106 (100 rounds), 0.783908 to 0.784160
    // (early stopping, 100 rounds each); RMSE 8.8171 to 8.8223.
    for (name, model) in [
        ("100 rounds", GradientBoostingRegressor::new().early_stopping(false)),
        ("early stopping", GradientBoostingRegressor::new()),
    ] {
        let mut model: GradientBoostingRegressor = model;
        model.fit(&train, "exam_score");
        let predicted = model.predict(&test);
        let (r2, rmse) = (r2_score(&y, &predicted), root_mean_squared_error(&y, &predicted));
        eprintln!("s6e1 boosting {name}: rounds {} test r2 {r2:.6} rmse {rmse:.6}", model.n_iter_);
        near("boosting r2", r2, 0.78403, 0.002);
        near("boosting rmse", rmse, 8.8199, 0.04);
    }
}

#[test]
#[ignore = "needs tests/fixtures/s6e2/train.csv, which only exists locally"]
fn class_weights_score_like_scikit_learn_on_the_kaggle_file() {
    let csv: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/s6e2/train.csv");
    if !csv.is_file() {
        eprintln!("skipped: the Kaggle file is missing");
        return;
    }
    let all: Dataset = Dataset::new(read_csv(&csv).drop(&["id"]).into_dataframe(), "");
    let train: Dataset = Dataset::new(all.dataframe().slice(0, 504_000), "");
    let test: Dataset = Dataset::new(all.dataframe().slice(504_000, 126_000), "");
    let target: &str = "Heart Disease";
    let y = test.dataframe().column(target).unwrap().as_materialized_series().clone();
    let near = |what: &str, got: f64, want: f64, by: f64| {
        assert!((got - want).abs() <= by, "{what}: {got} against {want}");
    };
    // One tree on every training row, no draws: scikit-learn grows 68,721 to 68,747 nodes
    // without weights (seeds 0 to 2). With "balanced" it grows 71,937 (seed 0), of which
    // 1,621 splits are of nodes that are already pure: its class sums and its node weight
    // round apart, so the node's impurity is up to 1.25e-13, above its ε. Those splits change
    // no prediction, and rusmlai makes none: 71,937 - 2 × 1,621 = 68,695 nodes.
    for (weight, want) in [("", 68_735.0), ("balanced", 68_695.0)] {
        let mut tree = DecisionTreeClassifier::new().min_samples_leaf(5);
        if !weight.is_empty() {
            tree = tree.class_weight(weight);
        }
        tree.fit(&train, target);
        let nodes: f64 = tree.tree_.node_count() as f64;
        eprintln!("s6e2 tree leaf 5 {weight}: nodes {nodes}");
        near("tree nodes", nodes, want, 0.005 * want);
    }
    // analysis/2026-11-17/boosting/balanced-parity.py, seeds 0 to 2: (test accuracy, log loss,
    // recall of Presence). No weights: 0.88342 to 0.88406, 0.2939 to 0.2975, 0.8632 to 0.8638;
    // balanced: 0.88316 to 0.88354, 0.2959 to 0.2983, 0.8790 to 0.8798; balanced_subsample:
    // 0.88352 to 0.88384, 0.2943 to 0.2979, 0.8769 to 0.8777. The weights move the recall
    // by 0.015, so a forest that ignored them fails here.
    let weights = [("", (0.8838, 0.2954, 0.8635)), ("balanced", (0.8834, 0.2971, 0.8793)), ("balanced_subsample", (0.8837, 0.2961, 0.8773))];
    for (weight, want) in weights {
        for seed in 0..3 {
            let mut model = RandomForestClassifier::new().n_estimators(20).min_samples_leaf(5).random_state(seed);
            if !weight.is_empty() {
                model = model.class_weight(weight);
            }
            model.fit(&train, target);
            let predicted = model.predict(&test);
            let accuracy: f64 = accuracy_score(&y, &predicted);
            let loss: f64 = log_loss(&y, &model.predict_proba(&test));
            let pairs = y.str().unwrap().iter().zip(predicted.str().unwrap().iter());
            let (hits, presence) = pairs
                .filter(|(truth, _)| *truth == Some("Presence"))
                .fold((0, 0), |(hits, all), (_, guess)| (hits + usize::from(guess == Some("Presence")), all + 1));
            let recall: f64 = hits as f64 / presence as f64;
            let nodes: usize = model.estimators_.iter().map(|tree| tree.node_count()).sum();
            eprintln!(
                "s6e2 forest 20 trees leaf 5 {weight} seed {seed}: nodes {nodes} test {accuracy:.6} log_loss {loss:.6} presence recall {recall:.6}"
            );
            near("accuracy", accuracy, want.0, 0.0015);
            near("log loss", loss, want.1, 0.01);
            near("recall", recall, want.2, 0.002);
        }
    }
}

/// The column roles on every Season 6 train file there is: the `id` column is the only
/// identifier, and no real feature is taken for one.
#[test]
#[ignore = "needs tests/fixtures/s6e*/train.csv, which only exist locally"]
fn every_s6_file_has_one_identifier_its_id() {
    let root: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    for n in 1..=11 {
        let path: PathBuf = root.join(format!("s6e{n}/train.csv"));
        if !path.is_file() {
            continue;
        }
        let ds: Dataset = read_csv(&path);
        let roles: Vec<(String, ColumnRole)> = (ds.columns().into_iter())
            .map(|c| {
                let role: ColumnRole = ds.role(&c);
                (c, role)
            })
            .filter(|(_, role)| *role != ColumnRole::Feature)
            .collect();
        eprintln!("s6e{n}: {roles:?}");
        assert_eq!(roles, [("id".to_string(), ColumnRole::Identifier)], "s6e{n}");
    }
}
