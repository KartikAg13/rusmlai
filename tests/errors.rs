//! The error variants no other test reaches, and the hints in the messages: each error says
//! what went wrong, with the values, and what to do.

use rusmlai::metrics::try_log_loss;
use rusmlai::polars::prelude::{NamedFrom, Series};
use rusmlai::{prelude::*, steps};

fn students() -> Dataset {
    read_csv("tests/fixtures/data/students.csv")
}

#[test]
fn an_unknown_column_suggests_the_closest_name() {
    let error: String = students().try_select(&["nmae"]).unwrap_err().to_string();
    assert!(error.contains("Did you mean 'name'?"), "{error}");
    let error: String = students().try_select(&["grade"]).unwrap_err().to_string();
    assert!(!error.contains("Did you mean"), "{error}");
}

#[test]
fn a_column_transformer_add_reading_a_missing_column_suggests_one() {
    let mut prep = ColumnTransformer::new().add("num", &["cgpaa"], steps![StandardScaler::new()]);
    let error: String = prep.try_fit(&students().select(&["age", "cgpa"])).unwrap_err().to_string();
    assert!(error.contains("add 'num' reads column 'cgpaa'") && error.contains("Did you mean 'cgpa'?"), "{error}");
}

#[test]
fn to_csv_into_a_folder_that_does_not_exist_says_where() {
    let mut ds = students();
    let path = std::env::temp_dir().join("rusmlai-no-such-folder").join("deeper").join("out.csv");
    let error = ds.try_to_csv(&path).unwrap_err();
    assert!(matches!(error, DatasetError::Write { .. } | DatasetError::Polars(_)), "{error}");
    assert!(error.to_string().contains("out.csv") || error.to_string().contains("No such file"), "{error}");
}

#[test]
fn a_label_a_classifier_never_saw_is_named() {
    let y_true = Series::new("y".into(), ["a", "b", "c"]);
    let proba = Dataset::new(df!("a" => [0.5, 0.5, 0.5], "b" => [0.5, 0.5, 0.5]).unwrap(), "");
    let error: String = try_log_loss(&y_true, &proba).unwrap_err().to_string();
    assert!(error.contains("'c'"), "{error}");
}

#[test]
fn warnings_print_once_and_are_collected() {
    set_warnings(false);
    let _ = take_warnings();
    let ds = Dataset::new(df!("x" => [1.0, 1.0, 1.0]).unwrap(), "");
    for _ in 0..3 {
        let _ = KBinsDiscretizer::new().strategy("uniform").fit_transform(&ds);
    }
    let seen: Vec<String> = take_warnings();
    assert_eq!(seen.iter().filter(|w| w.contains("constant")).count(), 1, "{seen:?}");
    set_warnings(true);
}
