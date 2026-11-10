//! The numbers behind the cached charts, checked against Polars and the column stats. The SVG
//! tests only say that a chart draws; these say that it draws the right values.

use polars::prelude::*;
use proptest::{collection::vec, option, prelude::*};
use rusmlai::{
    plot::{result::PlotResult, spec::PlotSpec},
    prelude::*,
};

/// The cached result of the one chart drawn on `ds`.
fn only_result(ds: &Dataset) -> (PlotSpec, PlotResult) {
    let entries = ds.metadata.plot_cache.entries.read().unwrap();
    assert_eq!(entries.len(), 1, "{entries:?}");
    let (spec, result) = entries.first().unwrap();
    (spec.clone(), result.clone())
}

fn students() -> Dataset {
    let frame: DataFrame = try_read_csv("tests/fixtures/data/students.csv")
        .unwrap()
        .dataframe;
    Dataset::new(frame, "") // in memory: an empty plot cache and no files
}

fn close(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
}

#[test]
fn bar_values_are_the_group_aggregates() {
    for estimator in ["mean", "median", "min", "max", "std", "sum", "count"] {
        let ds: Dataset = students();
        ds.bar("batch", "cgpa", "", estimator).svg();
        let (_, result) = only_result(&ds);
        let PlotResult::Series { labels, values } = result else {
            panic!("{estimator}: expected a Series");
        };
        assert_eq!(labels, ["B13", "B14", "B15"]);
        let x = || col("cgpa");
        let agg: Expr = match estimator {
            "mean" => x().mean(),
            "median" => x().median(),
            "min" => x().min(),
            "max" => x().max(),
            "std" => x().std(1),
            "sum" => x().sum(),
            _ => x().count(),
        };
        let expected: DataFrame = ds
            .dataframe
            .clone()
            .lazy()
            .group_by([col("batch")])
            .agg([agg.cast(DataType::Float64).alias("v")])
            .sort(["batch"], SortMultipleOptions::default())
            .collect()
            .unwrap();
        let expected: Vec<f64> = expected
            .column("v")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        for (got, want) in values.iter().zip(&expected) {
            assert!(
                close(*got, *want),
                "{estimator}: {values:?} vs {expected:?}"
            );
        }
    }
}

#[test]
fn count_with_hue_has_nan_for_a_missing_group_and_adds_up() {
    let frame: DataFrame = df!(
        "x" => ["a", "a", "b", "b", "b"],
        "h" => ["u", "v", "u", "u", "u"]
    )
    .unwrap();
    let ds: Dataset = Dataset::new(frame, "");
    ds.countplot("x", "h").svg();
    let (_, result) = only_result(&ds);
    let PlotResult::Clustered {
        x_labels,
        hue_labels,
        grid,
    } = result
    else {
        panic!("expected a Clustered result");
    };
    assert_eq!(
        (x_labels, hue_labels),
        (
            vec!["b".to_string(), "a".into()],
            vec!["u".to_string(), "v".into()]
        )
    );
    assert_eq!(grid[0][0], 3.0);
    assert!(grid[0][1].is_nan(), "no (b, v) rows is NaN, never 0");
    assert_eq!(grid[1], [1.0, 1.0]);
}

#[test]
fn boxes_hold_the_cached_quartiles_and_every_row_is_in_a_box_or_an_outlier() {
    let ds: Dataset = students();
    ds.boxplot("", "cgpa", "").svg();
    let (_, result) = only_result(&ds);
    let PlotResult::Boxes {
        boxes, outliers, ..
    } = result
    else {
        panic!("expected Boxes");
    };
    let stats = ds.try_numeric_stats("cgpa").unwrap();
    let five: [f64; 5] = boxes[0][0];
    assert!(
        close(five[1], stats.quantile_25)
            && close(five[2], stats.median)
            && close(five[3], stats.quantile_75),
        "{five:?} vs {stats:?}"
    );
    // No cgpa is past 1.5 x IQR, so the fences are the min and the max.
    assert_eq!((five[0], five[4]), (stats.min, stats.max));
    assert!(outliers[0][0].is_empty());

    let spread: DataFrame = df!("y" => [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]).unwrap();
    let ds: Dataset = Dataset::new(spread, "");
    ds.boxplot("", "y", "").svg();
    let (
        _,
        PlotResult::Boxes {
            boxes, outliers, ..
        },
    ) = only_result(&ds)
    else {
        panic!("expected Boxes");
    };
    assert_eq!(outliers[0][0], [100.0]);
    assert_eq!((boxes[0][0][0], boxes[0][0][4]), (1.0, 5.0));
}

#[test]
fn heatmap_counts_add_up_to_the_rows_and_corrplot_matches_polars() {
    let ds: Dataset = students();
    ds.heatmap("batch", "passed", "", "").svg();
    let (_, PlotResult::Matrix { grid, .. }) = only_result(&ds) else {
        panic!("expected a Matrix");
    };
    let total: f64 = grid.iter().flatten().filter(|v| v.is_finite()).sum();
    assert_eq!(total, 12.0);

    let ds: Dataset = students();
    ds.corrplot().svg();
    let (spec, PlotResult::Matrix { x_labels, grid, .. }) = only_result(&ds) else {
        panic!("expected a Matrix");
    };
    assert!(matches!(spec, PlotSpec::Corrplot { .. }));
    assert_eq!(x_labels, ["id", "age", "cgpa"]);
    // pandas: df[["age", "cgpa"]].corr() drops the rows where either is null.
    let both: DataFrame = ds
        .dataframe
        .clone()
        .lazy()
        .select([col("age").cast(DataType::Float64), col("cgpa")])
        .drop_nulls(None)
        .collect()
        .unwrap();
    let floats = |name: &str| -> Vec<f64> {
        let column: &Column = both.column(name).unwrap();
        column.f64().unwrap().into_no_null_iter().collect()
    };
    let (a, b): (Vec<f64>, Vec<f64>) = (floats("age"), floats("cgpa"));
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (ma, mb): (f64, f64) = (mean(&a), mean(&b));
    let sum = |f: &dyn Fn(usize) -> f64| (0..a.len()).map(f).sum::<f64>();
    let r: f64 = sum(&|i| (a[i] - ma) * (b[i] - mb))
        / (sum(&|i| (a[i] - ma).powi(2)) * sum(&|i| (b[i] - mb).powi(2))).sqrt();
    assert_eq!(a.len(), 10);
    assert!(close(grid[1][2], r), "{} vs {r}", grid[1][2]);
    assert_eq!(grid[1][2], grid[2][1]);
    assert!(close(grid[0][0], 1.0) && close(grid[2][2], 1.0));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// Every row with a value lands in exactly one bin, whatever the bin count or the hue.
    #[test]
    fn hist_counts_add_up_to_the_rows_with_values(
        values in vec(option::of(-1e3f64..1e3), 1..300),
        bins in 1usize..60,
        with_hue in any::<bool>(),
    ) {
        let hue: Vec<&str> = (0..values.len()).map(|i| ["p", "q", "r"][i % 3]).collect();
        let present: usize = values.iter().flatten().count();
        let ds: Dataset = Dataset::new(df!("v" => &values, "h" => hue).unwrap(), "");
        match ds.try_hist("v", if with_hue { "h" } else { "" }, bins) {
            Ok(figure) => {
                figure.svg();
                let (_, PlotResult::Hist { lo, hi, counts, .. }) = only_result(&ds) else {
                    panic!("expected a Hist");
                };
                let stats = ds.try_numeric_stats("v").unwrap();
                prop_assert_eq!((lo, hi), (stats.min, stats.max));
                prop_assert_eq!(counts.iter().flatten().sum::<f64>(), present as f64);
                // Fine bins: a multiple of `bins`, so regrouping is exact.
                prop_assert!(counts.iter().all(|c| c.len() % bins == 0));
            }
            Err(error) => prop_assert_eq!(present, 0, "{}", error),
        }
    }
}
