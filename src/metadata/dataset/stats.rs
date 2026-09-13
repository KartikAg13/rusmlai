use core::f64;
use std::cmp::Reverse;

use polars::datatypes::DataType;
use polars::frame::{DataFrame, column::Column};
use polars::lazy::dsl::Expr;
use polars::lazy::frame::LazyFrame;
use polars::prelude::{QuantileMethod, col, len, lit};
use serde::{Deserialize, Serialize};

use crate::metadata::error::MetadataError;
use crate::metadata::utils::{column_to_lazy, unwrap_f64};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NumericStats {
    pub mean: f64,
    pub median: f64,
    pub std: f64,
    pub variance: f64,
    pub min: f64,
    pub max: f64,
    pub quantile_25: f64,
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

impl NumericStats {
    pub fn compute_stats(column: &Column) -> Result<Self, MetadataError> {
        let (name, lazy): (String, LazyFrame) = column_to_lazy(column)?;
        let x: Expr = col(&name).cast(DataType::Float64);
        let out: DataFrame = lazy
            .select([
                x.clone().mean().alias("mean"),
                x.clone().median().alias("median"),
                x.clone().std(1).alias("std"),
                x.clone().var(1).alias("variance"),
                x.clone().min().alias("min"),
                x.clone().max().alias("max"),
                x.clone()
                    .quantile(lit(0.25), QuantileMethod::Linear)
                    .alias("q25"),
                x.quantile(lit(0.75), QuantileMethod::Linear).alias("q75"),
            ])
            .collect()?;
        Ok(Self {
            mean: unwrap_f64(&out, "mean")?,
            median: unwrap_f64(&out, "median")?,
            std: unwrap_f64(&out, "std")?,
            variance: unwrap_f64(&out, "variance")?,
            min: unwrap_f64(&out, "min")?,
            max: unwrap_f64(&out, "max")?,
            quantile_25: unwrap_f64(&out, "q25")?,
            quantile_75: unwrap_f64(&out, "q75")?,
        })
    }
}

impl CategoricalStats {
    pub fn compute_stats(column: &Column) -> Result<Self, MetadataError> {
        let (name, lazy): (String, LazyFrame) = column_to_lazy(column)?;
        let counts: DataFrame = lazy
            .drop_nulls(None)
            .group_by_stable([col(&name)])
            .agg([len().alias("count")])
            .collect()?;

        let labels: Column = counts.column(&name)?.cast(&DataType::String)?;
        let n: Column = counts.column("count")?.cast(&DataType::UInt32)?;
        let mut categories: Vec<(String, u64)> = labels
            .str()?
            .iter()
            .zip(n.u32()?.iter())
            .filter_map(|(label, c)| Some((label?.to_string(), c? as u64)))
            .collect();
        categories.sort_unstable_by_key(|(_, c)| Reverse(*c));

        Ok(Self {
            unique_count: categories.len(),
            mode: categories
                .first()
                .map(|(s, _)| s.clone())
                .unwrap_or_default(),
            categories,
        })
    }
}
