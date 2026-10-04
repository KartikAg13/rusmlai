use polars::lazy::dsl::{Expr, col};
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
    pub fn expr(self, column: &str) -> Expr {
        match self {
            Self::Mean => col(column).mean(),
            Self::Median => col(column).median(),
            Self::Min => col(column).min(),
            Self::Max => col(column).max(),
            Self::Std => col(column).std(1),
            Self::Count => col(column).count(),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::Median => "median",
            Self::Min => "min",
            Self::Max => "max",
            Self::Std => "std",
            Self::Count => "count",
        }
    }

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
pub struct BarSpec {
    pub x: String,
    pub y: Option<String>,
    pub hue: Option<String>,
    pub estimator: Estimator,
}

impl BarSpec {
    pub fn new(x: &str, y: &str, hue: &str, estimator: &str) -> Result<Self, PlotError> {
        if x.is_empty() {
            return Err(PlotError::EmptyX);
        }
        let estimator: Estimator = match (y.is_empty(), estimator) {
            (true, "" | "count") => Estimator::Count,
            (true, other) => return Err(PlotError::EstimatorNeedsY(other.to_string())),
            (false, "") => Estimator::Mean,
            (false, other) => Estimator::parse(other)?,
        };
        Ok(Self {
            x: x.to_string(),
            y: (!y.is_empty()).then(|| y.to_string()),
            hue: (!hue.is_empty()).then(|| hue.to_string()),
            estimator,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlotSpec {
    Bar(BarSpec),
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
