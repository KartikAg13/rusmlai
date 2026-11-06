//! Property tests: stats against Polars' own expressions, category invariants, the JSONL cache
//! format, and `select` / `drop` on random column subsets.

use std::{fs, path::PathBuf};

use polars::prelude::*;
use proptest::{collection::vec, option, prelude::*, sample::subsequence};
use rusmlai::{
    metadata::{
        dataset::stats::{CategoricalStats, ColumnStats, NumericStats},
        utils::{read_jsonl, unwrap_f64, write_jsonl},
    },
    prelude::*,
};

fn close(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
}

fn numeric(values: &[Option<f64>]) -> NumericStats {
    let frame: DataFrame = df!("x" => values).unwrap();
    match ColumnStats::compute_stats(&frame).unwrap().remove(0) {
        ColumnStats::Numeric(s) => s,
        other => panic!("{other:?}"),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn numeric_stats_match_polars(values in vec(option::of(-1e6f64..1e6), 0..300)) {
        let ours: NumericStats = numeric(&values);
        let frame: DataFrame = df!("x" => &values).unwrap();
        let q = |p: f64| col("x").quantile(lit(p), QuantileMethod::Linear);
        let polars: DataFrame = frame
            .lazy()
            .select([
                col("x").mean().alias("mean"),
                col("x").min().alias("min"),
                col("x").max().alias("max"),
                col("x").std(1).alias("std"),
                q(0.5).alias("median"),
                q(0.25).alias("q25"),
                q(0.75).alias("q75"),
            ])
            .collect()
            .unwrap();
        let theirs = |name: &str| unwrap_f64(&polars, name).unwrap();
        for (name, value) in [
            ("mean", ours.mean),
            ("min", ours.min),
            ("max", ours.max),
            ("std", ours.std),
            ("median", ours.median),
            ("q25", ours.quantile_25),
            ("q75", ours.quantile_75),
        ] {
            prop_assert!(close(value, theirs(name)), "{name}: {value} vs {}", theirs(name));
        }
    }

    #[test]
    fn categories_add_up_and_sort_with_the_mode_first(
        values in vec(option::of(prop::sample::select(vec!["a", "b", "c", "d", "ë"])), 0..200),
    ) {
        let frame: DataFrame = df!("s" => &values).unwrap();
        let c: CategoricalStats = match ColumnStats::compute_stats(&frame).unwrap().remove(0) {
            ColumnStats::Categorical(c) => c,
            other => panic!("{other:?}"),
        };
        let total: u64 = c.categories.iter().map(|(_, n)| n).sum();
        prop_assert_eq!(total as usize, values.iter().flatten().count());
        prop_assert_eq!(c.unique_count, c.categories.len());
        for pair in c.categories.windows(2) {
            prop_assert!(pair[0].1 > pair[1].1 || (pair[0].1 == pair[1].1 && pair[0].0 < pair[1].0));
        }
        prop_assert_eq!(&c.mode, &c.categories.first().map(|(n, _)| n.clone()).unwrap_or_default());
    }

    #[test]
    fn select_and_drop_split_the_columns(
        picked in subsequence(vec!["id", "name", "age", "cgpa", "batch", "passed"], 1..6),
    ) {
        let ds: Dataset = read_csv("tests/fixtures/data/students.csv");
        prop_assert_eq!(ds.select(&picked).columns(), picked.clone());
        let rest: Vec<String> = ds.columns().into_iter().filter(|c| !picked.contains(&c.as_str())).collect();
        prop_assert_eq!(ds.drop(&picked).columns(), rest); // the others, in their order
    }
}

proptest! {
    #[test]
    fn jsonl_round_trips_and_a_cut_keeps_earlier_lines(
        entries in vec(vec(any::<u16>(), 0..5), 0..10),
        cut in any::<prop::sample::Index>(),
    ) {
        let dir: PathBuf = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("properties");
        fs::create_dir_all(&dir).unwrap();
        let path: PathBuf = dir.join("cache.jsonl");
        write_jsonl(&path, (7, 8), &entries).unwrap();
        prop_assert_eq!(read_jsonl::<Vec<u16>>(&path, (7, 8)), Some(entries.clone()));

        // A crash mid-write leaves a prefix of the file: every whole line before the cut survives.
        let text: String = fs::read_to_string(&path).unwrap();
        let header: usize = text.find('\n').unwrap() + 1;
        let at: usize = header + cut.index(text.len() - header + 1);
        fs::write(&path, &text[..at]).unwrap();
        let kept: Vec<Vec<u16>> = read_jsonl(&path, (7, 8)).unwrap();
        let whole: usize = text[header..at].matches('\n').count();
        prop_assert!(kept.len() == whole || kept.len() == whole + 1); // + a last line cut before its "\n"
        prop_assert_eq!(&kept[..whole], &entries[..whole]);
    }
}
