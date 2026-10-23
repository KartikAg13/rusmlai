use serde::{Deserialize, Deserializer, Serialize};

/// The cached numbers of a chart, not its SVG. A missing group is NaN, never 0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlotResult {
    Series {
        labels: Vec<String>,
        #[serde(deserialize_with = "nan_vec")]
        values: Vec<f64>,
    },
    Clustered {
        x_labels: Vec<String>,
        hue_labels: Vec<String>,
        #[serde(deserialize_with = "nan_grid")]
        grid: Vec<Vec<f64>>,
    },
    Matrix {
        x_labels: Vec<String>,
        y_labels: Vec<String>,
        #[serde(deserialize_with = "nan_grid")]
        grid: Vec<Vec<f64>>,
    },
    Hist {
        lo: f64,
        hi: f64,
        names: Vec<String>,
        counts: Vec<Vec<f64>>,
    },
    /// `boxes[x][hue]` = lower fence, q25, median, q75, upper fence (a missing group is all
    /// NaN); `outliers[x][hue]` = the values past the fences, at most 100.
    Boxes {
        x_labels: Vec<String>,
        hue_labels: Vec<String>,
        #[serde(deserialize_with = "nan_boxes")]
        boxes: Vec<Vec<[f64; 5]>>,
        outliers: Vec<Vec<Vec<f64>>>,
    },
}

impl PlotResult {
    /// True when there is nothing to draw: every value is NaN (or every count 0).
    pub(crate) fn is_empty(&self) -> bool {
        let finite = |v: &f64| v.is_finite();
        !match self {
            PlotResult::Series { values, .. } => values.iter().any(finite),
            PlotResult::Clustered { grid, .. } | PlotResult::Matrix { grid, .. } => {
                grid.iter().flatten().any(finite)
            }
            PlotResult::Hist { counts, .. } => counts.iter().flatten().any(|&n| n > 0.0),
            PlotResult::Boxes { boxes, .. } => {
                boxes.iter().flatten().any(|five| five[2].is_finite())
            }
        }
    }
}

fn nan_vec<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<f64>, D::Error> {
    let values: Vec<Option<f64>> = Vec::deserialize(d)?;
    Ok(values.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect())
}

fn nan_boxes<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<Vec<[f64; 5]>>, D::Error> {
    let rows: Vec<Vec<[Option<f64>; 5]>> = Vec::deserialize(d)?;
    Ok(rows
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|five| five.map(|v| v.unwrap_or(f64::NAN)))
                .collect()
        })
        .collect())
}

fn nan_grid<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<Vec<f64>>, D::Error> {
    let rows: Vec<Vec<Option<f64>>> = Vec::deserialize(d)?;
    Ok(rows
        .into_iter()
        .map(|row| row.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect())
        .collect())
}
