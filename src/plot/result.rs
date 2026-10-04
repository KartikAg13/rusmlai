use serde::{Deserialize, Deserializer, Serialize};

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
}

fn nan_vec<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<f64>, D::Error> {
    let values: Vec<Option<f64>> = Vec::deserialize(d)?;
    Ok(values.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect())
}

fn nan_grid<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<Vec<f64>>, D::Error> {
    let rows: Vec<Vec<Option<f64>>> = Vec::deserialize(d)?;
    Ok(rows
        .into_iter()
        .map(|row| row.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect())
        .collect())
}
