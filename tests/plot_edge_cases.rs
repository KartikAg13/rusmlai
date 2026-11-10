//! Every chart on hostile data: empty, null, constant, NaN and ±inf, huge and tiny numbers,
//! odd names, 5,000 categories. Each call must give a clean SVG or a clear `PlotError`, never a
//! panic or a raw Polars error. Run memory-capped: an endless tick loop once filled 15 GB here.

use std::panic::{self, AssertUnwindSafe};

use polars::prelude::*;
use rusmlai::{dataset::Dataset, display::Figure, plot::error::PlotError};

type Chart = (&'static str, fn(&Dataset) -> Result<Figure, PlotError>);
type Call<'a> = Box<dyn Fn() -> String + 'a>;

const CHARTS: &[Chart] = &[
    ("bar", |d| d.try_bar("x", "y", "", "mean")),
    ("bar sum hue", |d| d.try_bar("x", "y", "h", "sum")),
    ("bar numeric x", |d| d.try_bar("n", "y", "", "median")),
    ("bar std", |d| d.try_bar("x", "y", "", "std")),
    ("countplot", |d| d.try_bar("x", "", "", "")),
    ("count hue", |d| d.try_bar("x", "", "h", "")),
    ("count hue = x", |d| d.try_bar("x", "", "x", "")),
    ("count numeric", |d| d.try_bar("n", "", "", "")),
    ("describe_bar", |d| d.try_describe_bar("", "")),
    ("describe_bar y", |d| d.try_describe_bar("y", "")),
    ("describe_bar sum", |d| d.try_describe_bar("y", "sum")),
    ("missing", |d| d.try_missing()),
    ("pie", |d| d.try_pie("x")),
    ("pie numeric", |d| d.try_pie("n")),
    ("hist", |d| d.try_hist("y", "", 10)),
    ("hist hue", |d| d.try_hist("y", "h", 10)),
    ("boxplot", |d| d.try_boxplot("x", "y", "")),
    ("boxplot no x", |d| d.try_boxplot("", "y", "")),
    ("boxplot hue", |d| d.try_boxplot("x", "y", "h")),
    ("stripplot", |d| d.try_stripplot("x", "y", "h")),
    ("kdeplot", |d| d.try_kdeplot("y", "")),
    ("kdeplot hue", |d| d.try_kdeplot("y", "h")),
    ("violinplot", |d| d.try_violinplot("x", "y", "h")),
    ("qqplot", |d| d.try_qqplot("y")),
    ("heatmap", |d| d.try_heatmap("x", "h", "y", "mean")),
    ("heatmap x = y", |d| d.try_heatmap("x", "x", "", "")),
    ("missing_matrix", |d| d.try_missing_matrix()),
    ("corrplot", |d| d.try_corrplot()),
    ("scatter", |d| d.try_scatter("y", "z", "h")),
    ("scatter x = y", |d| d.try_scatter("y", "y", "")),
    ("regplot", |d| d.try_regplot("y", "z", "")),
    ("regplot hue", |d| d.try_regplot("y", "z", "h")),
    ("regplot x = y", |d| d.try_regplot("y", "y", "")),
    ("plot", |d| d.try_plot("n", "y", "")),
    ("line hue", |d| d.try_plot("z", "y", "h")),
    ("pairplot", |d| d.try_pairplot(&[], "")),
    ("pairplot hue, repeated column", |d| {
        d.try_pairplot(&["y", "z", "y"], "h")
    }),
];

fn base() -> DataFrame {
    df!(
        "x" => ["a", "b", "c", "a", "b", "c", "a", "b"],
        "y" => [1.5, -2.0, 3.0, -4.5, 5.0, 0.0, 7.25, -8.0],
        "h" => ["p", "q", "p", "q", "p", "q", "p", "q"],
        "n" => [1i64, 2, 3, 4, 5, 6, 7, 8],
        "z" => [0.5, 1.0, -1.5, 2.0, 2.5, -3.0, 3.5, 4.0]
    )
    .unwrap()
}

/// Each frame has the columns x (text), y and z (numbers), h (hue) and n (integers).
fn hostile() -> Vec<(&'static str, DataFrame)> {
    let cycle = |a: &'static str, b: &'static str| [a, b, a, b, a, b, a, b];
    let xs: [&str; 8] = ["a", "b", "c", "a", "b", "c", "a", "b"];
    let n: [i64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let frame = |y: [f64; 8], z: [f64; 8]| {
        df!("x" => xs, "y" => y, "h" => cycle("p", "q"), "n" => n, "z" => z).unwrap()
    };
    let (none_f, none_s, none_i) = (
        vec![None::<f64>; 8],
        vec![None::<&str>; 8],
        vec![None::<i64>; 8],
    );
    let long: String = "x".repeat(300);
    let many: Vec<String> = (0..5000).map(|i| format!("cat{i}")).collect();
    let (inf, nan, max) = (f64::INFINITY, f64::NAN, f64::MAX);
    vec![
        ("normal", base()),
        ("empty", base().head(Some(0))),
        ("one row", base().head(Some(1))),
        ("all null", df!("x" => &none_s, "y" => &none_f, "h" => cycle("p", "q"), "n" => &none_i, "z" => &none_f).unwrap()),
        ("constant", df!("x" => ["a"; 8], "y" => [5.0; 8], "h" => ["p"; 8], "n" => [1i64; 8], "z" => [5.0; 8]).unwrap()),
        ("NaN and inf", frame([nan, inf, -inf, 1.0, 2.0, nan, 3.0, -1.0], [nan, 1.0, 2.0, inf, 3.0, 4.0, 5.0, 6.0])),
        ("±1e300 and 1e-12 spread", frame([1e300, -1e300, 1e-300, 0.0, 1e300, -1e300, 5.0, 1e299], [1.0, 1.0 + 1e-12, 1.0, 1.0 + 2e-12, 1.0, 1.0, 1.0 + 3e-12, 1.0])),
        ("near f64::MAX", frame([1e308, -1e308, 0.0, 1.0, max, -max, 2.0, 3.0], [max, -max, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])),
        // 1e18 and the next float: the axis start divided by the tick step passes 2^53.
        ("one float step apart", frame([1e18, 1e18 + 128.0, 1e18, 1e18 + 128.0, 1e18, 1e18, 1e18 + 128.0, 1e18], [1e18; 8])),
        ("constant 1e20", df!("x" => ["a"; 8], "y" => [1e20; 8], "h" => ["p"; 8], "n" => [i64::MAX; 8], "z" => [-1e20; 8]).unwrap()),
        (
            "odd names",
            df!("x" => ["", " ", "<b>&\"'", "😀", long.as_str(), "1e5", "NaN", "-0"], "y" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "h" => ["中文", "<", "&amp;", "", "中文", "<", "&amp;", ""], "n" => n, "z" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .unwrap(),
        ),
        (
            "5,000 categories",
            df!("x" => &many, "y" => (0..5000).map(f64::from).collect::<Vec<f64>>(),
                "h" => (0..5000).map(|i| if i % 2 == 0 { "p" } else { "q" }).collect::<Vec<&str>>(),
                "n" => (0..5000i64).collect::<Vec<i64>>(), "z" => (0..5000).map(|i| f64::from(i % 17)).collect::<Vec<f64>>())
            .unwrap(),
        ),
        (
            "bool and int columns",
            df!("x" => [true, false, true, true, false, true, true, false], "y" => [1i64, -2, 3, -4, 5, 0, 7, -8],
                "h" => [1i32, 2, 1, 2, 1, 2, 1, 2], "n" => [1i32, 2, 3, 4, 5, 6, 7, 8], "z" => [0.5f32, 1.0, -1.5, 2.0, 2.5, -3.0, 3.5, 4.0])
            .unwrap(),
        ),
        (
            "nulls here and there",
            df!("x" => [Some("a"), None, Some("c"), Some("a"), None, Some("c"), Some("a"), Some("b")],
                "y" => [Some(1.5), Some(-2.0), None, Some(-4.5), Some(5.0), None, Some(7.25), Some(-8.0)],
                "h" => [Some("p"), Some("q"), None, Some("q"), Some("p"), Some("q"), None, Some("q")],
                "n" => [Some(1i64), None, Some(3), Some(4), None, Some(6), Some(7), Some(8)],
                "z" => [Some(0.5), Some(1.0), Some(-1.5), None, Some(2.5), Some(-3.0), Some(3.5), None])
            .unwrap(),
        ),
        ("one-row groups", df!("x" => ["a", "b", "b", "b"], "y" => [1.0, 2.0, 2.0, 3.0], "h" => ["p", "p", "q", "q"], "n" => [1i64, 1, 1, 2], "z" => [1.0; 4]).unwrap()),
    ]
}

/// What is wrong with an SVG: it doesn't parse, or a tag holds NaN, inf or a negative size.
fn svg_problem(svg: &str) -> Option<String> {
    if let Err(e) = resvg::usvg::Tree::from_str(svg, &resvg::usvg::Options::default()) {
        return Some(format!("does not parse: {e}"));
    }
    let tags: String = svg
        .split('<')
        .filter_map(|piece| piece.split('>').next())
        .collect::<Vec<&str>>()
        .join("<");
    ["NaN", "inf", "width=\"-", "height=\"-", " r=\"-"]
        .iter()
        .find(|bad| tags.contains(*bad))
        .map(|bad| format!("a tag holds {bad}"))
}

fn message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

#[test]
fn every_chart_on_hostile_data_gives_an_svg_or_a_clear_error() {
    let mut failures: Vec<String> = Vec::new();
    for (data, frame) in hostile() {
        let ds: Dataset = Dataset::new(frame, "");
        for (chart, call) in CHARTS {
            match panic::catch_unwind(AssertUnwindSafe(|| call(&ds).map(|figure| figure.svg()))) {
                Ok(Ok(svg)) => {
                    failures.extend(svg_problem(&svg).map(|p| format!("{data} / {chart}: {p}")))
                }
                Ok(Err(PlotError::Polars(e))) => {
                    failures.push(format!("{data} / {chart}: raw Polars error: {e}"))
                }
                Ok(Err(_)) => {}
                Err(payload) => {
                    failures.push(format!("{data} / {chart}: panicked: {}", message(payload)))
                }
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn options_reject_impossible_values_and_draw_odd_ones() {
    let ds: Dataset = Dataset::new(base(), "");
    let bar = || ds.bar("x", "y", "h", "sum");
    let rejected: Vec<(&str, Call)> = vec![
        (".ylim", Box::new(|| bar().ylim(-10.0, -20.0).svg())),
        (".ylim", Box::new(|| bar().ylim(-1e308, 1e308).svg())), // the width overflows
        (
            ".xlim",
            Box::new(|| ds.hist("y", "", 10).xlim(f64::NAN, 1.0).svg()),
        ),
        (".size", Box::new(|| bar().size(-100.0, -100.0).svg())),
        (".top", Box::new(|| bar().top(0).svg())),
        (
            ".sample",
            Box::new(|| ds.scatter("y", "z", "").sample(0).svg()),
        ),
        (
            ".sample",
            Box::new(|| ds.scatter("y", "z", "").sample(60_000).svg()),
        ),
        (".hline", Box::new(|| bar().hline(f64::NAN, "").svg())),
        (".order", Box::new(|| bar().order(&["zzz"]).svg())),
        (
            ".hue_order",
            Box::new(|| ds.bar("x", "y", "", "mean").hue_order(&["p"]).svg()),
        ),
        (".colors", Box::new(|| bar().colors(&[]).svg())),
        (".color", Box::new(|| bar().color("C99").svg())),
        (".percent", Box::new(|| ds.pie("x").percent().svg())),
        (".kde", Box::new(|| bar().kde().svg())),
        (".legend", Box::new(|| bar().legend("left").svg())),
    ];
    for (option, call) in &rejected {
        let text: String = match panic::catch_unwind(AssertUnwindSafe(call)) {
            Ok(_) => panic!("{option} accepted an impossible value"),
            Err(payload) => message(payload),
        };
        assert!(
            text.starts_with(option) && text.len() < 200,
            "{option}: {text}"
        );
    }
    let drawn: Vec<String> = vec![
        bar().ylim(100.0, 200.0).svg(), // every bar outside the range
        bar().ylim(0.0, 1e-300).svg(),
        bar().hline(-1e300, "far").horizontal().svg(),
        bar().size(200.0, 150.0).legend("bottom").svg(),
        bar()
            .title(&"T".repeat(400))
            .subtitle(&"s".repeat(400))
            .svg(),
        ds.countplot("x", "h").percent().stacked().svg(),
        ds.hist("y", "", 10).xlim(100.0, 200.0).kde().svg(), // every row outside
        ds.scatter("y", "z", "").vline(1e300, "far").sample(1).svg(),
        ds.heatmap("x", "h", "y", "sum").cmap("rdbu").svg(),
        ds.pie("x").top(1).svg(),
        Dataset::new(df!("v" => [3.0; 10]).unwrap(), "")
            .hist("v", "", 10)
            .kde()
            .svg(),
        Dataset::new(
            df!("x" => ["a", "b", "c"], "y" => [Some(1.0), None, Some(f64::NAN)]).unwrap(),
            "",
        )
        .bar("x", "y", "", "mean")
        .sort("value")
        .svg(),
    ];
    for (i, svg) in drawn.iter().enumerate() {
        assert_eq!(svg_problem(svg), None, "drawn case {i}");
    }
}

#[test]
fn files_keep_their_name_and_a_bounded_size() {
    let dir: String = format!("{}/plot_edge_cases", env!("CARGO_TARGET_TMPDIR"));
    let _ = std::fs::remove_dir_all(&dir);
    let ds: Dataset = Dataset::new(base(), format!("{dir}/data.csv"));
    ds.bar("x", "y", "", "mean")
        .size(20000.0, 20000.0)
        .save(format!("{dir}/huge.png"));
    let png: Vec<u8> = std::fs::read(format!("{dir}/huge.png")).unwrap();
    let side = |at: usize| u64::from(u32::from_be_bytes(png[at..at + 4].try_into().unwrap()));
    assert!(
        side(16) * side(20) <= 40_100_000,
        "the PNG is capped at 40 M pixels (give or take rounding)"
    );
    drop(ds.bar("x", "y", "", "mean").title("???")); // no letters left: named after the chart
    drop(ds.bar("x", "y", "", "mean").title("../../a/b")); // no path, no leading dots
    for name in ["bar.png", "ab.png"] {
        assert!(
            std::path::Path::new(&format!("{dir}/metadata/plots/{name}")).exists(),
            "{name}"
        );
    }
    let saved = panic::catch_unwind(AssertUnwindSafe(|| {
        ds.bar("x", "y", "", "mean")
            .save(format!("{dir}/chart.jpg"))
    }));
    assert!(saved.is_err(), "only .png and .svg are written");
}
