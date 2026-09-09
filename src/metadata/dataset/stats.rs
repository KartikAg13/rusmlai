use core::f64;

use polars::{
    chunked_array::ops::{ChunkAgg, ChunkQuantile, ChunkVar},
    datatypes::{Float64Type, StringType, UInt32Type},
    frame::{DataFrame, column::Column},
    prelude::{ChunkedArray, QuantileMethod, SeriesMethods},
    series::Series,
};
use serde::{Deserialize, Serialize};

use crate::dataset::error::DatasetError;

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    pub fn new() -> Self {
        Self {
            mean: 0.0,
            median: 0.0,
            std: 0.0,
            variance: 0.0,
            min: 0.0,
            max: 0.0,
            quantile_25: 0.0,
            quantile_75: 0.0,
        }
    }

    pub fn compute_stats(column: &Column) -> Result<Self, DatasetError> {
        let series: Series = column
            .as_materialized_series()
            .cast(&polars::datatypes::DataType::Float64)?;

        let values: &ChunkedArray<Float64Type> = series.f64()?;

        Ok(Self {
            mean: values.mean().unwrap_or(f64::NAN),
            median: values.median().unwrap_or(f64::NAN),
            std: values.std(1).unwrap_or(f64::NAN),
            variance: values.var(1).unwrap_or(f64::NAN),
            min: values.min().unwrap_or(f64::NAN),
            max: values.max().unwrap_or(f64::NAN),
            quantile_25: values
                .quantile(0.25, QuantileMethod::Linear)
                .unwrap()
                .unwrap_or(f64::NAN),
            quantile_75: values
                .quantile(0.75, QuantileMethod::Linear)
                .unwrap()
                .unwrap_or(f64::NAN),
        })
    }
}

impl CategoricalStats {
    pub fn new() -> Self {
        Self {
            unique_count: 0,
            categories: Vec::new(),
            mode: "".to_string(),
        }
    }

    pub fn compute_stats(column: &Column) -> Result<Self, DatasetError> {
        let counts: DataFrame =
            column
                .as_materialized_series()
                .value_counts(true, false, "count".into(), false)?;

        let names: &ChunkedArray<StringType> = counts.column(column.name())?.str()?;
        let cnts: &ChunkedArray<UInt32Type> = counts.column("count")?.u32()?;

        let mut category_counts: Vec<(String, u64)> = names
            .iter()
            .zip(cnts.iter())
            .filter_map(|(n, c)| Some((n?.to_string(), c? as u64)))
            .collect();

        category_counts.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        Ok(Self {
            unique_count: category_counts.len(),
            mode: category_counts
                .first()
                .map(|(name, _)| name.clone())
                .unwrap_or_default(),
            categories: category_counts,
        })
    }
}
