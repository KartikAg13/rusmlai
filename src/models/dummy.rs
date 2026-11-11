//! [`DummyRegressor`], the baseline every model should beat (scikit-learn's `sklearn.dummy`).

use polars::prelude::{ChunkFull, Float64Chunked, IntoSeries, Series};

use crate::{
    dataset::Dataset,
    error::OrFatal,
    models::{ModelError, Regressor, numeric_target},
    preprocess::one_of,
};

/// Predicts one number for every row, learned from the target alone, like scikit-learn's
/// `DummyRegressor`.
///
/// `strategy` is `"mean"` (the default), `"median"` or `"constant"`, which predicts the value
/// set by [`DummyRegressor::constant`]. `fit` reads the target's cached stats, so it makes no
/// pass over the data. Its R² on new data is close to 0: a real model should score above it.
///
/// ```
/// use polars::df;
/// use rusmlai::prelude::*;
/// let ds = Dataset::new(df!("x" => [1, 2, 3, 4], "y" => [1.0, 2.0, 3.0, 10.0]).unwrap(), "");
/// let mut model = DummyRegressor::new();
/// model.fit(&ds, "y");
/// let predicted: Vec<f64> = model.predict(&ds).f64().unwrap().into_no_null_iter().collect();
/// assert_eq!(predicted, [4.0; 4]);
///
/// let mut median = DummyRegressor::new().strategy("median");
/// median.fit(&ds, "y");
/// assert_eq!(median.predict(&ds).f64().unwrap().get(0), Some(2.5));
/// ```
#[derive(Debug, Clone)]
pub struct DummyRegressor {
    strategy: String,
    constant: Option<f64>,
    /// The value predicted for every row, learned by `fit`.
    pub constant_: Option<f64>,
    target: String,
}

impl Default for DummyRegressor {
    fn default() -> Self {
        Self {
            strategy: "mean".into(),
            constant: None,
            constant_: None,
            target: String::new(),
        }
    }
}

impl DummyRegressor {
    /// A baseline with `strategy = "mean"`.
    pub fn new() -> Self {
        Self::default()
    }

    /// `"mean"`, `"median"` or `"constant"`.
    #[track_caller]
    pub fn strategy(mut self, strategy: &str) -> Self {
        self.strategy = one_of("strategy", strategy, &["mean", "median", "constant"]).or_fatal();
        self
    }

    /// The value `strategy("constant")` predicts.
    pub fn constant(mut self, value: f64) -> Self {
        self.constant = Some(value);
        self
    }
}

impl Regressor for DummyRegressor {
    fn try_fit(&mut self, dataset: &Dataset, target: &str) -> Result<(), ModelError> {
        let stats = numeric_target(dataset, target)?;
        let value: f64 = match self.strategy.as_str() {
            "mean" => stats.mean,
            "median" => stats.median,
            _ => self.constant.ok_or(ModelError::MissingSetting {
                setting: "constant",
                reason: "strategy(\"constant\") predicts it",
            })?,
        };
        self.constant_ = Some(value);
        self.target = target.into();
        Ok(())
    }

    fn try_predict(&self, dataset: &Dataset) -> Result<Series, ModelError> {
        let value: f64 = self.constant_.ok_or(ModelError::NotFitted {
            model: "DummyRegressor",
        })?;
        let target: &str = self.target.as_str();
        Ok(Float64Chunked::full(target.into(), value, dataset.height()).into_series())
    }
}

#[cfg(test)]
mod tests {
    use polars::df;

    use super::*;

    #[test]
    fn errors_name_the_problem() {
        let ds: Dataset = Dataset::new(
            df!("y" => [Some(1.0), None], "s" => ["a", "b"], "z" => [1.0, 2.0]).unwrap(),
            "",
        );
        let fit =
            |mut m: DummyRegressor, target: &str| m.try_fit(&ds, target).unwrap_err().to_string();
        assert!(fit(DummyRegressor::new(), "y").starts_with("target 'y' has 1 nulls"));
        assert!(fit(DummyRegressor::new(), "s").starts_with("column 's' is not numeric"));
        assert_eq!(
            fit(DummyRegressor::new().strategy("constant"), "z"),
            "constant must be set: strategy(\"constant\") predicts it"
        );
        assert!(matches!(
            DummyRegressor::new().try_predict(&ds),
            Err(ModelError::NotFitted { .. })
        ));
    }
}
