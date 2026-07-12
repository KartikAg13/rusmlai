use polars::{
    datatypes::{StringType, UInt32Type},
    frame::DataFrame,
    lazy::{dsl::col, frame::IntoLazy},
    prelude::{ChunkedArray, SeriesMethods},
};
use serde::{Deserialize, Serialize};

use crate::data::error::DataError;

#[derive(Clone, Serialize, Deserialize)]
pub struct CategoricalStats {
    pub unique_count: usize,
    pub categories: Vec<(String, u64)>,
    pub mode: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct NumericStats {
    pub mean: f64,
    pub std: f64,
    pub variance: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
}

#[derive(Clone, Serialize, Deserialize)]
pub enum ColumnStats {
    Numeric(NumericStats),
    Categorical(CategoricalStats),
}

impl NumericStats {
    pub fn compute_stats(df: &DataFrame, column: &str) -> Result<Self, DataError> {
        if let Err(_) = df.column(column) {
            return Err(DataError::ColumnNotFound(column.to_string()));
        }

        let aggregate: DataFrame = df
            .clone()
            .lazy()
            .select([
                col(column).mean().alias("mean"),
                col(column).std(1).alias("std"),
                col(column).var(1).alias("variance"),
                col(column).min().alias("min"),
                col(column).max().alias("max"),
                col(column).median().alias("median"),
            ])
            .collect()?;

        Ok(NumericStats {
            mean: aggregate.column("mean")?.f64()?.get(0).unwrap_or(f64::NAN),
            std: aggregate.column("std")?.f64()?.get(0).unwrap_or(f64::NAN),
            variance: aggregate
                .column("variance")?
                .f64()?
                .get(0)
                .unwrap_or(f64::NAN),
            min: aggregate.column("min")?.f64()?.get(0).unwrap_or(f64::NAN),
            max: aggregate.column("max")?.f64()?.get(0).unwrap_or(f64::NAN),
            median: aggregate
                .column("median")?
                .f64()?
                .get(0)
                .unwrap_or(f64::NAN),
        })
    }
}

impl CategoricalStats {
    pub fn compute_stats(df: &DataFrame, column: &str) -> Result<Self, DataError> {
        let counts: DataFrame = df.column(column)?.as_materialized_series().value_counts(
            true,
            false,
            "count".into(),
            false,
        )?;

        let names: &ChunkedArray<StringType> = counts.column(column)?.str()?;
        let cnts: &ChunkedArray<UInt32Type> = counts.column("count")?.u32()?;

        let mut category_counts: Vec<(String, u64)> = names
            .iter()
            .zip(cnts.iter())
            .filter_map(|(n, c)| Some((n?.to_string(), c? as u64)))
            .collect();
        category_counts.sort_by(|a, b| b.1.cmp(&a.1));

        let mode: String = category_counts
            .first()
            .map(|(n, _)| n.clone())
            .unwrap_or_default();
        let unique_count: usize = category_counts.len();

        Ok(CategoricalStats {
            unique_count: unique_count,
            categories: category_counts,
            mode: mode,
        })
    }
}
