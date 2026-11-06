//! Column stats and the one-plan computation behind them.

use polars::datatypes::DataType;
use polars::frame::{DataFrame, column::Column};
use polars::lazy::dsl::Expr;
use polars::lazy::frame::{IntoLazy, LazyFrame};
use polars::prelude::{DslPlan, Engine, col, len};
use serde::{Deserialize, Serialize};

use crate::metadata::error::MetadataError;
use crate::metadata::utils::unwrap_f64;

/// Stats of a numeric column, with nulls skipped. NaN when the column has no values.
/// `std` and `variance` use `ddof = 1`, like pandas; the quartiles interpolate linearly.
///
/// ```
/// use rusmlai::prelude::*;
/// let ds = read_csv("tests/fixtures/data/students.csv");
/// let cgpa = ds.try_numeric_stats("cgpa").unwrap();
/// assert_eq!((cgpa.min, cgpa.median, cgpa.max), (5.1, 7.4, 9.0));
/// assert!((cgpa.variance - cgpa.std * cgpa.std).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NumericStats {
    #[serde(deserialize_with = "nan_if_null")]
    /// Arithmetic mean.
    pub mean: f64,
    #[serde(deserialize_with = "nan_if_null")]
    /// The 50% quantile.
    pub median: f64,
    #[serde(deserialize_with = "nan_if_null")]
    /// Sample standard deviation.
    pub std: f64,
    #[serde(deserialize_with = "nan_if_null")]
    /// Sample variance.
    pub variance: f64,
    #[serde(deserialize_with = "nan_if_null")]
    /// Smallest value.
    pub min: f64,
    #[serde(deserialize_with = "nan_if_null")]
    /// Largest value.
    pub max: f64,
    #[serde(deserialize_with = "nan_if_null")]
    /// The 25% quantile.
    pub quantile_25: f64,
    #[serde(deserialize_with = "nan_if_null")]
    /// The 75% quantile.
    pub quantile_75: f64,
}

/// Stats of a text, boolean or other non-numeric column, with nulls skipped.
///
/// ```
/// use rusmlai::prelude::*;
/// let ds = read_csv("tests/fixtures/data/students.csv");
/// let passed = ds.try_categorical_stats("passed").unwrap();
/// assert_eq!(passed.categories, [("true".to_string(), 8), ("false".to_string(), 4)]);
/// assert_eq!(passed.mode, "true");
/// ```
///
/// Ties sort by name, so the mode is always the same:
///
/// ```
/// use rusmlai::prelude::*;
/// let batch = read_csv("tests/fixtures/data/students.csv").try_categorical_stats("batch").unwrap().clone();
/// assert_eq!(batch.unique_count, 3);
/// assert_eq!(batch.mode, "B13"); // B13, B14 and B15 have 4 rows each
/// assert!((batch.entropy - 3f64.ln()).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CategoricalStats {
    /// Distinct non-null values.
    pub unique_count: usize,
    /// `(value, count)`, most frequent first, ties by name.
    pub categories: Vec<(String, u64)>,
    /// The most frequent value; empty if the column has no values.
    pub mode: String,
    /// Shannon entropy of the counts, in nats.
    pub entropy: f64,
}

/// The stats of one column: numeric dtypes get [`NumericStats`], all others [`CategoricalStats`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColumnStats {
    /// An integer or float column.
    Numeric(NumericStats),
    /// Any other column.
    Categorical(CategoricalStats),
}

fn nan_if_null<'de, D: serde::Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
    Ok(Option::<f64>::deserialize(d)?.unwrap_or(f64::NAN))
}

impl ColumnStats {
    /// The stats of every column of `dataframe`, in column order, from one Polars plan.
    ///
    /// ```
    /// use polars::df;
    /// use rusmlai::metadata::dataset::stats::ColumnStats;
    /// let frame = df!("x" => [1.0, 2.0, 4.0], "s" => ["a", "b", "a"]).unwrap();
    /// let stats = ColumnStats::compute_stats(&frame).unwrap();
    /// assert!(matches!(&stats[0], ColumnStats::Numeric(n) if n.median == 2.0));
    /// assert!(matches!(&stats[1], ColumnStats::Categorical(c) if c.mode == "a"));
    /// ```
    pub fn compute_stats(dataframe: &DataFrame) -> Result<Vec<ColumnStats>, MetadataError> {
        let columns: &[Column] = dataframe.columns();
        let lazy: LazyFrame = dataframe.clone().lazy();
        let mut aggregates: Vec<Expr> = Vec::new();
        let mut plans: Vec<DslPlan> = Vec::new();
        for (i, column) in columns.iter().enumerate() {
            let name: &str = column.name().as_str();
            if column.dtype().is_numeric() {
                let x: Expr = col(name).cast(DataType::Float64);
                aggregates.extend([
                    x.clone().mean().alias(format!("{i}:mean")),
                    x.clone().std(1).alias(format!("{i}:std")),
                    x.clone().var(1).alias(format!("{i}:var")),
                    x.clone().min().alias(format!("{i}:min")),
                    x.max().alias(format!("{i}:max")),
                ]);
            } else {
                let counts: LazyFrame = lazy
                    .clone()
                    .select([col(name)])
                    .drop_nulls(None)
                    .group_by([col(name)])
                    .agg([len().alias("count")]);
                plans.push(counts.logical_plan);
            }
        }
        plans.push(lazy.clone().select(aggregates).logical_plan);

        let numeric: Vec<&Column> = columns.iter().filter(|c| c.dtype().is_numeric()).collect();
        let threads: usize = std::thread::available_parallelism().map_or(4, |n| n.get());
        let (mut outs, quartiles): (Vec<DataFrame>, Vec<[f64; 3]>) = std::thread::scope(|s| {
            let handles: Vec<_> = numeric
                .chunks(numeric.len().div_ceil(threads).max(1))
                .map(|chunk| {
                    s.spawn(move || {
                        chunk
                            .iter()
                            .map(|c| NumericStats::quartiles(c))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            let outs = LazyFrame::collect_all_with_engine(
                plans,
                Engine::Auto,
                lazy.get_current_optimizations(),
            );
            let quartiles: Result<Vec<[f64; 3]>, MetadataError> = handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect();
            Ok::<_, MetadataError>((outs?, quartiles?))
        })?;

        let aggregated: DataFrame = outs.pop().unwrap();
        let mut counts = outs.into_iter();
        let mut quartiles = quartiles.into_iter();
        columns
            .iter()
            .enumerate()
            .map(|(i, column)| {
                if column.dtype().is_numeric() {
                    let [median, q25, q75] = quartiles.next().unwrap();
                    Ok(ColumnStats::Numeric(NumericStats {
                        mean: unwrap_f64(&aggregated, &format!("{i}:mean"))?,
                        median,
                        std: unwrap_f64(&aggregated, &format!("{i}:std"))?,
                        variance: unwrap_f64(&aggregated, &format!("{i}:var"))?,
                        min: unwrap_f64(&aggregated, &format!("{i}:min"))?,
                        max: unwrap_f64(&aggregated, &format!("{i}:max"))?,
                        quantile_25: q25,
                        quantile_75: q75,
                    }))
                } else {
                    CategoricalStats::categorical(&counts.next().unwrap(), column.name().as_str())
                        .map(ColumnStats::Categorical)
                }
            })
            .collect()
    }
}

impl NumericStats {
    fn quartiles(column: &Column) -> Result<[f64; 3], MetadataError> {
        let column: Column = column.cast(&DataType::Float64)?;
        let ca = column.f64()?;
        let mut values: Vec<f64> = if ca.null_count() == 0 {
            ca.into_no_null_iter().collect()
        } else {
            ca.iter().flatten().collect()
        };
        Ok([
            Self::quantile(&mut values, 0.5),
            Self::quantile(&mut values, 0.25),
            Self::quantile(&mut values, 0.75),
        ])
    }

    fn quantile(values: &mut [f64], q: f64) -> f64 {
        if values.is_empty() {
            return f64::NAN;
        }
        let pos: f64 = q * (values.len() - 1) as f64;
        let lower: usize = pos as usize;
        let (_, low, above) = values.select_nth_unstable_by(lower, f64::total_cmp);
        let low: f64 = *low;
        let frac: f64 = pos - lower as f64;
        if frac == 0.0 {
            return low;
        }
        let high: f64 = above.iter().copied().min_by(f64::total_cmp).unwrap_or(low);
        low + (high - low) * frac
    }
}

impl CategoricalStats {
    /// Shannon entropy in nats of a list of counts. Zero counts are skipped.
    ///
    /// ```
    /// use rusmlai::metadata::dataset::stats::CategoricalStats;
    /// assert_eq!(CategoricalStats::entropy([5.0].into_iter()), 0.0);
    /// assert!((CategoricalStats::entropy([1.0, 1.0, 0.0].into_iter()) - 2f64.ln()).abs() < 1e-12);
    /// ```
    pub fn entropy(counts: impl Iterator<Item = f64> + Clone) -> f64 {
        let n: f64 = counts.clone().sum();
        counts
            .map(|count| count / n)
            .filter(|p| *p > 0.0)
            .fold(0.0, |h: f64, p: f64| h + p * (1.0 / p).ln())
    }

    fn categorical(counts: &DataFrame, name: &str) -> Result<CategoricalStats, MetadataError> {
        let labels: Column = counts.column(name)?.cast(&DataType::String)?;
        let n: Column = counts.column("count")?.cast(&DataType::UInt64)?;
        let mut categories: Vec<(String, u64)> = labels
            .str()?
            .iter()
            .zip(n.u64()?.iter())
            .filter_map(|(label, c)| Some((label?.to_string(), c?)))
            .collect();
        categories.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        Ok(CategoricalStats {
            unique_count: categories.len(),
            mode: categories
                .first()
                .map(|(s, _)| s.clone())
                .unwrap_or_default(),
            entropy: CategoricalStats::entropy(categories.iter().map(|&(_, count)| count as f64)),
            categories,
        })
    }
}

#[cfg(test)]
mod tests {
    use polars::df;
    use polars::prelude::{QuantileMethod, lit};

    use super::*;

    fn numeric(frame: &DataFrame) -> NumericStats {
        match ColumnStats::compute_stats(frame).unwrap().remove(0) {
            ColumnStats::Numeric(s) => s,
            other => panic!("expected numeric stats, got {other:?}"),
        }
    }

    fn categorical(frame: &DataFrame) -> CategoricalStats {
        match ColumnStats::compute_stats(frame).unwrap().remove(0) {
            ColumnStats::Categorical(s) => s,
            other => panic!("expected categorical stats, got {other:?}"),
        }
    }

    #[test]
    fn quantile_edge_cases() {
        let q = |v: &[f64], q: f64| NumericStats::quantile(&mut v.to_vec(), q);
        assert!(q(&[], 0.5).is_nan());
        assert_eq!(q(&[7.0], 0.25), 7.0);
        assert_eq!(q(&[1.0, 2.0], 0.5), 1.5);
        assert_eq!(q(&[3.0, 1.0, 2.0], 0.5), 2.0); // odd length
        assert_eq!(q(&[4.0, 1.0, 3.0, 2.0], 0.5), 2.5); // even length
        assert_eq!(q(&[5.0, 5.0, 5.0, 1.0], 0.5), 5.0); // duplicates
        assert_eq!(q(&[-3.0, -1.0, -2.0], 0.25), -2.5); // negatives
        assert_eq!(q(&[3.0, 1.0, 2.0], 0.0), 1.0);
        assert_eq!(q(&[3.0, 1.0, 2.0], 1.0), 3.0);
        assert!(q(&[1.0, f64::NAN], 1.0).is_nan()); // NaN sorts last, like Polars
    }

    #[test]
    fn quartiles_match_polars_linear() {
        let values: Vec<f64> = (0..101)
            .map(|i| ((i * 37) % 101) as f64 * 0.5 - 7.0)
            .collect();
        let frame: DataFrame = df!("x" => &values).unwrap();
        let polars: DataFrame = frame
            .clone()
            .lazy()
            .select([0.25, 0.5, 0.75].map(|q| {
                col("x")
                    .quantile(lit(q), QuantileMethod::Linear)
                    .alias(format!("{q}"))
            }))
            .collect()
            .unwrap();
        let ours: NumericStats = numeric(&frame);
        assert_eq!(ours.quantile_25, unwrap_f64(&polars, "0.25").unwrap());
        assert_eq!(ours.median, unwrap_f64(&polars, "0.5").unwrap());
        assert_eq!(ours.quantile_75, unwrap_f64(&polars, "0.75").unwrap());
    }

    #[test]
    fn numeric_stats_skip_nulls_and_use_ddof_1() {
        let frame: DataFrame =
            df!("x" => [Some(1.0), None, Some(2.0), Some(3.0), Some(4.0)]).unwrap();
        let s: NumericStats = numeric(&frame);
        assert_eq!((s.mean, s.median, s.min, s.max), (2.5, 2.5, 1.0, 4.0));
        assert!((s.variance - 5.0 / 3.0).abs() < 1e-12);
        assert!((s.std - (5.0f64 / 3.0).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn all_null_columns_give_nan_and_no_categories() {
        let s: NumericStats = numeric(&df!("x" => [None::<f64>, None]).unwrap());
        assert!(s.mean.is_nan() && s.median.is_nan() && s.min.is_nan());
        let c: CategoricalStats = categorical(&df!("s" => [None::<&str>, None]).unwrap());
        assert_eq!((c.unique_count, c.mode.as_str(), c.entropy), (0, "", 0.0));
    }

    #[test]
    fn categories_sort_by_count_then_name() {
        let frame: DataFrame = df!("s" => ["b", "c", "a", "c", "b", "d"]).unwrap();
        let c: CategoricalStats = categorical(&frame);
        let names: Vec<&str> = c.categories.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["b", "c", "a", "d"]);
        assert_eq!(c.mode, "b");
    }

    #[test]
    fn bool_columns_are_categorical() {
        let c: CategoricalStats = categorical(&df!("p" => [true, false, true]).unwrap());
        assert_eq!(
            c.categories,
            [("true".to_string(), 2), ("false".to_string(), 1)]
        );
    }

    #[test]
    fn entropy_is_in_nats() {
        assert_eq!(CategoricalStats::entropy([5.0].into_iter()), 0.0);
        assert!((CategoricalStats::entropy([1.0, 1.0].into_iter()) - 2f64.ln()).abs() < 1e-15);
        // scipy.stats.entropy([3, 1])
        assert!(
            (CategoricalStats::entropy([3.0, 1.0].into_iter()) - 0.562_335_144_618_808_3).abs()
                < 1e-15
        );
        assert_eq!(CategoricalStats::entropy([0.0, 4.0].into_iter()), 0.0); // 0 ln 0 = 0
    }

    #[test]
    fn zero_rows_and_zero_columns_do_not_fail() {
        let empty: DataFrame = df!("x" => Vec::<f64>::new(), "s" => Vec::<&str>::new()).unwrap();
        let stats: Vec<ColumnStats> = ColumnStats::compute_stats(&empty).unwrap();
        assert!(matches!(&stats[0], ColumnStats::Numeric(s) if s.mean.is_nan()));
        assert!(matches!(&stats[1], ColumnStats::Categorical(c) if c.unique_count == 0));
        assert!(
            ColumnStats::compute_stats(&DataFrame::empty())
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn text_only_frames_need_no_numeric_select() {
        let c: CategoricalStats = categorical(&df!("s" => ["x", "y", "x"]).unwrap());
        assert_eq!((c.mode.as_str(), c.unique_count), ("x", 2));
    }

    /// The textbook linear quantile on a sorted copy (numpy's default).
    fn reference(values: &[f64], q: f64) -> f64 {
        let mut sorted: Vec<f64> = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let pos: f64 = q * (sorted.len() - 1) as f64;
        let (lo, hi): (usize, usize) = (pos.floor() as usize, pos.ceil() as usize);
        sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
    }

    proptest::proptest! {
        #[test]
        fn quantile_matches_a_sort(
            values in proptest::collection::vec(-1e6f64..1e6, 1..200),
            dups in proptest::collection::vec(0usize..200, 0..50),
            q in 0.0f64..=1.0,
        ) {
            let mut values: Vec<f64> = values;
            let extra: Vec<f64> = dups.iter().map(|i| values[i % values.len()]).collect();
            values.extend(extra);
            let expected: f64 = reference(&values, q);
            let got: f64 = NumericStats::quantile(&mut values.clone(), q);
            proptest::prop_assert!((got - expected).abs() <= 1e-9 * expected.abs().max(1.0), "{got} vs {expected}");
        }
    }
}
