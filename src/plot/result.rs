use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlotResult {
    Series {
        labels: Vec<String>,
        values: Vec<f64>,
    },
    Clustered {
        x_labels: Vec<String>,
        hue_labels: Vec<String>,
        grid: Vec<Vec<f64>>,
    },
    Matrix {
        x_labels: Vec<String>,
        y_labels: Vec<String>,
        grid: Vec<Vec<f64>>,
    },
    Hist {
        lo: f64,
        hi: f64,
        names: Vec<String>,
        counts: Vec<Vec<f64>>,
    },
}
