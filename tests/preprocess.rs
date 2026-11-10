//! Transformers end to end: learn on a training part, apply to a test part, and the invariants
//! that scikit-learn's versions keep.

use polars::prelude::*;
use proptest::{collection::vec, option, prelude::*};
use rusmlai::{prelude::*, steps};

fn floats(dataset: &Dataset, name: &str) -> Vec<Option<f64>> {
    let column: &Column = dataset.dataframe.column(name).unwrap();
    column.f64().unwrap().iter().collect()
}

/// The students fixture in memory, as (first 8 rows, last 4 rows).
fn train_test() -> (Dataset, Dataset) {
    let frame: DataFrame = try_read_csv("tests/fixtures/data/students.csv")
        .unwrap()
        .dataframe;
    (
        Dataset::new(frame.head(Some(8)), ""),
        Dataset::new(frame.tail(Some(4)), ""),
    )
}

#[test]
fn a_column_transformer_applies_what_the_training_part_taught_it() {
    let (train, test) = train_test();
    let mut prep: ColumnTransformer = ColumnTransformer::new()
        .add(
            "num",
            &["age", "cgpa"],
            steps![
                SimpleImputer::new().strategy("median"),
                StandardScaler::new()
            ],
        )
        .add(
            "cat",
            &["batch"],
            steps![OneHotEncoder::new().handle_unknown("ignore")],
        )
        .add("flag", &["passed"], steps![OrdinalEncoder::new()]);
    prep.fit(&train);
    let out: Dataset = prep.transform(&test);
    assert_eq!(
        out.columns(),
        [
            "num__age",
            "num__cgpa",
            "cat__batch_B13",
            "cat__batch_B14",
            "cat__batch_B15",
            "flag__passed"
        ]
    );
    assert_eq!(out.columns(), prep.get_feature_names_out());
    assert_eq!(out.height(), 4);

    // The training part: ages 19 20 21 19 22 20 (null) 23, median 20 fills the null.
    let ages: [f64; 8] = [19.0, 20.0, 21.0, 19.0, 22.0, 20.0, 20.0, 23.0];
    let mean: f64 = ages.iter().sum::<f64>() / 8.0;
    let scale: f64 = (ages.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / 8.0).sqrt();
    let expected: Vec<Option<f64>> = [21.0, 22.0, 19.0, 24.0]
        .iter()
        .map(|a| Some((a - mean) / scale))
        .collect();
    for (got, want) in floats(&out, "num__age").iter().zip(&expected) {
        assert!(
            (got.unwrap() - want.unwrap()).abs() < 1e-12,
            "{got:?} vs {want:?}"
        );
    }
    // The test part: batches B15 B13 B14 B15, passed true false true true.
    assert_eq!(
        floats(&out, "cat__batch_B15"),
        [Some(1.0), Some(0.0), Some(0.0), Some(1.0)]
    );
    assert_eq!(
        floats(&out, "flag__passed"),
        [Some(1.0), Some(0.0), Some(1.0), Some(1.0)]
    );
    assert!(out.dataframe.columns().iter().all(|c| c.null_count() == 0));
}

#[test]
fn a_transformed_dataset_still_describes_and_draws() {
    let (train, _) = train_test();
    let scaled: Dataset = StandardScaler::new().fit_transform(&train.select(&["cgpa"]));
    // Stats of a transformed column are computed on first use.
    let stats = scaled.try_numeric_stats("cgpa").unwrap();
    assert!(stats.mean.abs() < 1e-12, "{stats:?}");
    assert!(scaled.hist("cgpa", "", 5).svg().starts_with("<svg"));
    scaled.describe();
    assert!(scaled.metadata.try_describe().to_string().contains("cgpa"));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn scalers_keep_their_promises(values in vec(option::of(-1e6f64..1e6), 2..200)) {
        let present: Vec<f64> = values.iter().flatten().copied().collect();
        prop_assume!(present.len() >= 2);
        let spread: bool = present.iter().any(|v| *v != present[0]);
        let ds: Dataset = Dataset::new(df!("x" => &values).unwrap(), "");
        let column = |out: &Dataset| -> Vec<f64> { floats(out, "x").into_iter().flatten().collect() };

        let standard: Vec<f64> = column(&StandardScaler::new().fit_transform(&ds));
        prop_assert_eq!(standard.len(), present.len()); // nulls stay null
        let n: f64 = standard.len() as f64;
        let mean: f64 = standard.iter().sum::<f64>() / n;
        let var: f64 = standard.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        prop_assert!(mean.abs() < 1e-6, "mean {}", mean);
        if spread {
            prop_assert!((var - 1.0).abs() < 1e-6, "variance {}", var);
        }

        let minmax: Vec<f64> = column(&MinMaxScaler::new().feature_range(-2.0, 3.0).fit_transform(&ds));
        let (lo, hi): (f64, f64) = minmax.iter().fold((f64::MAX, f64::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
        prop_assert!((lo + 2.0).abs() < 1e-9, "min {}", lo);
        if spread {
            prop_assert!((hi - 3.0).abs() < 1e-9, "max {}", hi);
        }

        let maxabs: Vec<f64> = column(&MaxAbsScaler::new().fit_transform(&ds));
        let largest: f64 = maxabs.iter().fold(0.0, |m: f64, v| m.max(v.abs()));
        prop_assert!(largest <= 1.0 + 1e-12);
        if present.iter().any(|v| *v != 0.0) {
            prop_assert!((largest - 1.0).abs() < 1e-12, "largest {}", largest);
        }

        let robust: Vec<f64> = column(&RobustScaler::new().fit_transform(&ds));
        let mut sorted: Vec<f64> = robust.clone();
        sorted.sort_by(f64::total_cmp);
        let mid: usize = sorted.len() / 2;
        let median: f64 = if sorted.len() % 2 == 1 { sorted[mid] } else { (sorted[mid - 1] + sorted[mid]) / 2.0 };
        prop_assert!(median.abs() < 1e-6, "median {}", median);
    }

    #[test]
    fn encoders_keep_their_promises(
        values in vec(option::of(prop::sample::select(vec!["a", "b", "c", "ë", "long name"])), 1..120),
    ) {
        let distinct: usize = {
            let mut seen: Vec<&str> = values.iter().flatten().copied().collect();
            seen.sort_unstable();
            seen.dedup();
            seen.len()
        };
        prop_assume!(distinct > 0);
        let ds: Dataset = Dataset::new(df!("c" => &values).unwrap(), "");

        // One-hot: one column per category, and each row with a value has exactly one 1.
        let hot: Dataset = OneHotEncoder::new().fit_transform(&ds);
        prop_assert_eq!(hot.width(), distinct);
        for (row, value) in values.iter().enumerate() {
            let cells: Vec<Option<f64>> = hot.columns().iter().map(|c| floats(&hot, c)[row]).collect();
            match value {
                Some(_) => prop_assert_eq!(cells.iter().flatten().sum::<f64>(), 1.0),
                None => prop_assert!(cells.iter().all(Option::is_none)),
            }
        }

        // Ordinal: codes 0..k in sorted order, so a smaller name never gets a larger code.
        let codes: Vec<Option<f64>> = floats(&OrdinalEncoder::new().fit_transform(&ds), "c");
        for (i, a) in values.iter().enumerate() {
            for (j, b) in values.iter().enumerate() {
                if let (Some(a), Some(b)) = (a, b) {
                    prop_assert_eq!(a.cmp(b), codes[i].unwrap().total_cmp(&codes[j].unwrap()));
                }
            }
        }

        // Label: codes and back gives the column again, nulls included.
        let mut label: LabelEncoder = LabelEncoder::new();
        let coded: Dataset = label.fit_transform(&ds);
        let back: Dataset = label.inverse_transform(&coded);
        let names: Vec<Option<&str>> = back.dataframe.column("c").unwrap().str().unwrap().iter().collect();
        prop_assert_eq!(names, values);
    }
}
