use serde::{Deserialize, Serialize};

use crate::plot::error::PlotError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Estimator {
    Mean,
    Median,
    Min,
    Max,
    Std,
    Count,
}

impl Estimator {
    pub fn parse(raw: &str) -> Result<Self, PlotError> {
        match raw {
            "mean" => Ok(Self::Mean),
            "median" => Ok(Self::Median),
            "min" => Ok(Self::Min),
            "max" => Ok(Self::Max),
            "std" => Ok(Self::Std),
            "count" => Ok(Self::Count),
            _ => Err(PlotError::UnknownEstimator(raw.to_string())),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlotSpec {
    Bar {
        x: String,
        y: Option<String>,
        hue: Option<String>,
        estimator: Estimator,
    },
    Heatmap {
        x: String,
        y: String,
        value: Option<String>,
        estimator: Estimator,
    },
    Hist {
        column: String,
        hue: Option<String>,
        bins: usize,
    },
    Corr,
}
