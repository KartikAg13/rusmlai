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
    let all: Dataset = Dataset::new(read_csv(&source).drop(&["id"]).dataframe, "");
    // A fixed split, so both libraries see the same rows: first 504,000 train, last 126,000 test.
    let train: Dataset = Dataset::new(all.dataframe.slice(0, 504_000), "");
    let test: Dataset = Dataset::new(all.dataframe.slice(504_000, 126_000), "");
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
        .dataframe
        .column("exam_score")
        .unwrap()
        .as_materialized_series()
        .clone();
    for (name, want) in SKLEARN {
        // Ridge(alpha=1) is alpha = 1 / rows on ElasticNet's scale.
        let alpha: f64 = if name == "Ridge" {
            1.0 / train.height() as f64
        } else {
            0.0
        };
        let mut pipe = Pipeline::new(
            prep(),
            ModelStacker::new(LinearRegression::new().alpha(alpha)),
        );
        pipe.fit(&train, "exam_score");
        let (predicted, intercept) = (pipe.predict(&test), pipe.model.model.intercept_);
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
    let all: Dataset = Dataset::new(read_csv(&source).drop(&["id"]).dataframe, "");
    let train: Dataset = Dataset::new(all.dataframe.slice(0, 504_000), "");
    let test: Dataset = Dataset::new(all.dataframe.slice(504_000, 126_000), "");
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
    let mut pipe = Pipeline::new(prep, ModelStacker::new(LogisticRegression::new()));
    pipe.fit(&train, "Heart Disease");
    let y = test
        .dataframe
        .column("Heart Disease")
        .unwrap()
        .as_materialized_series()
        .clone();
    let accuracy: f64 = accuracy_score(&y, &pipe.predict(&test));
    let loss: f64 = log_loss(&y, &pipe.predict_proba(&test));
    let model: &LogisticRegression = &pipe.model.model;
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
