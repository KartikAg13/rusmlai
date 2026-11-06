use polars::lazy::dsl::{Expr, col};
use serde::{Deserialize, Serialize};

use crate::plot::{error::PlotError, optional};

/// How a bar or heatmap cell sums up its values: the `estimator` argument as a type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Estimator {
    Mean,
    Median,
    Min,
    Max,
    Std,
    Count,
    Sum,
}

impl Estimator {
    /// The Polars aggregation of `column`. Std is the sample std (ddof 1), like pandas.
    pub fn expr(self, column: &str) -> Expr {
        match self {
            Self::Mean => col(column).mean(),
            Self::Median => col(column).median(),
            Self::Min => col(column).min(),
            Self::Max => col(column).max(),
            Self::Std => col(column).std(1),
            Self::Count => col(column).count(),
            Self::Sum => col(column).sum(),
        }
    }

    /// The name users pass, for titles and hover text: "mean", "std", ...
    pub fn as_str(self) -> &'static str {
        ESTIMATORS
            .iter()
            .find(|(_, e)| *e == self)
            .map_or("", |(name, _)| name)
    }

    /// The estimator with this exact name, or `PlotError::UnknownEstimator`.
    pub fn parse(raw: &str) -> Result<Self, PlotError> {
        ESTIMATORS
            .iter()
            .find(|(name, _)| *name == raw)
            .map(|(_, e)| *e)
            .ok_or_else(|| PlotError::UnknownEstimator(raw.to_string()))
    }

    /// The estimator for `raw`, where `""` means the default: "count" without a value column,
    /// "mean" with one. Only "count" works without a value column.
    pub fn pick(has_value: bool, raw: &str) -> Result<Self, PlotError> {
        match (has_value, raw) {
            (false, "" | "count") => Ok(Self::Count),
            (false, other) => Err(PlotError::EstimatorNeedsY(other.to_string())),
            (true, "") => Ok(Self::Mean),
            (true, other) => Self::parse(other),
        }
    }
}

const ESTIMATORS: [(&str, Estimator); 7] = [
    ("mean", Estimator::Mean),
    ("median", Estimator::Median),
    ("min", Estimator::Min),
    ("max", Estimator::Max),
    ("std", Estimator::Std),
    ("count", Estimator::Count),
    ("sum", Estimator::Sum),
];

/// The cache key of a `bar` or `count` chart.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BarSpec {
    pub x: String,
    pub y: Option<String>,
    pub hue: Option<String>,
    pub estimator: Estimator,
}

impl BarSpec {
    /// The key for these arguments, with `""` as None and the estimator parsed.
    pub fn new(x: &str, y: &str, hue: &str, estimator: &str) -> Result<Self, PlotError> {
        if x.is_empty() {
            return Err(PlotError::EmptyX);
        }
        Ok(Self {
            x: x.to_string(),
            y: optional(y),
            // hue = x adds nothing: each bar already gets its own color, so don't dodge them.
            hue: optional(hue).filter(|hue| hue != x),
            estimator: Estimator::pick(!y.is_empty(), estimator)?,
        })
    }
}

/// The cache key of a chart: one line of `{stem}.plots.jsonl` per key.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlotSpec {
    Bar(BarSpec),
    Heatmap {
        x: String,
        y: String,
        value: Option<String>,
        estimator: Estimator,
    },
    /// Counts in fine bins (a multiple of `bins`, at least 1,000), so `.xlim` can regroup them.
    Histogram {
        column: String,
        hue: Option<String>,
        bins: usize,
    },
    Boxplot {
        x: String,
        y: String,
        hue: Option<String>,
    },
    /// The numeric columns go in the key, because `select` shares the plot cache.
    Corrplot {
        columns: Vec<String>,
    },
    /// Null counts per band of rows, for `missing_matrix`. The columns go in the key too.
    Missing {
        columns: Vec<String>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_estimator_name_parses_and_unknown_names_fail() {
        for (name, estimator) in ESTIMATORS {
            assert_eq!(Estimator::parse(name).unwrap(), estimator);
        }
        assert!(
            matches!(Estimator::parse("avg"), Err(PlotError::UnknownEstimator(n)) if n == "avg")
        );
        assert!(matches!(
            Estimator::parse("Mean"),
            Err(PlotError::UnknownEstimator(_))
        ));
    }

    #[test]
    fn pick_defaults_to_count_or_mean() {
        assert_eq!(Estimator::pick(false, "").unwrap(), Estimator::Count);
        assert_eq!(Estimator::pick(true, "").unwrap(), Estimator::Mean);
        assert!(matches!(
            Estimator::pick(false, "sum"),
            Err(PlotError::EstimatorNeedsY(_))
        ));
    }

    #[test]
    fn bar_spec_treats_empty_as_none_and_drops_hue_equal_to_x() {
        let spec: BarSpec = BarSpec::new("a", "", "a", "").unwrap();
        assert_eq!((spec.y, spec.hue), (None, None));
        assert!(matches!(
            BarSpec::new("", "", "", ""),
            Err(PlotError::EmptyX)
        ));
    }
}
