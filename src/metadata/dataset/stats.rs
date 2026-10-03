use polars::datatypes::DataType;
use polars::frame::{DataFrame, column::Column};
use polars::lazy::dsl::Expr;
use polars::lazy::frame::{IntoLazy, LazyFrame};
use polars::prelude::{DslPlan, Engine, col, len};
use serde::{Deserialize, Serialize};

use crate::metadata::error::MetadataError;
use crate::metadata::utils::unwrap_f64;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NumericStats {
    #[serde(deserialize_with = "nan_if_null")]
    pub mean: f64,
    #[serde(deserialize_with = "nan_if_null")]
    pub median: f64,
    #[serde(deserialize_with = "nan_if_null")]
    pub std: f64,
    #[serde(deserialize_with = "nan_if_null")]
    pub variance: f64,
    #[serde(deserialize_with = "nan_if_null")]
    pub min: f64,
    #[serde(deserialize_with = "nan_if_null")]
    pub max: f64,
    #[serde(deserialize_with = "nan_if_null")]
    pub quantile_25: f64,
    #[serde(deserialize_with = "nan_if_null")]
    pub quantile_75: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CategoricalStats {
    pub unique_count: usize,
    pub categories: Vec<(String, u64)>,
    pub mode: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColumnStats {
    Numeric(NumericStats),
    Categorical(CategoricalStats),
}

fn nan_if_null<'de, D: serde::Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
    Ok(Option::<f64>::deserialize(d)?.unwrap_or(f64::NAN))
}

impl ColumnStats {
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
            categories,
        })
    }
}
