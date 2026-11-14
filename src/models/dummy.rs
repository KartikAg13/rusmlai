//! [`DummyRegressor`], the baseline every model should beat (scikit-learn's `sklearn.dummy`).

use polars::prelude::{
    ChunkFull, DataType, Float64Chunked, IntoLazy, IntoSeries, QuantileMethod, Series, lit,
};
use serde_json::{Value, json};

use crate::{
    dataset::Dataset,
    error::OrFatal,
    models::{Model, ModelError, Regressor, numeric_target},
    preprocess::one_of,
    utils::col,
};

/// Predicts one number for every row, learned from the target alone, like scikit-learn's
/// `DummyRegressor`.
///
/// `strategy` is `"mean"` (the default), `"median"`, `"quantile"`, which predicts the
/// [`DummyRegressor::quantile`] of the target, or `"constant"`, which predicts the value set by
/// [`DummyRegressor::constant`]. `fit` reads the target's cached stats (a quantile other than
/// the median makes one pass over the target). Its R² on new data is close to 0: a real model
/// should score above it.
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
///
/// let mut q = DummyRegressor::new().strategy("quantile").quantile(0.9);
/// q.fit(&ds, "y");
/// assert!((q.constant_.unwrap() - 7.9).abs() < 1e-12);
/// ```
///
/// Python:
///
/// ```python
/// import pandas as pd
/// from sklearn.dummy import DummyRegressor
/// ds = pd.DataFrame({"x": [1, 2, 3, 4], "y": [1.0, 2.0, 3.0, 10.0]})
/// model = DummyRegressor().fit(ds[["x"]], ds["y"])
/// assert model.predict(ds[["x"]]).tolist() == [4.0] * 4
///
/// median = DummyRegressor(strategy="median").fit(ds[["x"]], ds["y"])
/// assert median.predict(ds[["x"]])[0] == 2.5
///
/// q = DummyRegressor(strategy="quantile", quantile=0.9).fit(ds[["x"]], ds["y"])
/// assert abs(q.constant_[0][0] - 7.9) < 1e-12
/// ```
#[derive(Debug, Clone)]
pub struct DummyRegressor {
    strategy: String,
    constant: Option<f64>,
    quantile: Option<f64>,
    /// The value predicted for every row, learned by `fit`.
    pub constant_: Option<f64>,
    target: String,
    run: Option<String>,
    /// ½ · mean squared error on the training rows, for the journal.
    loss: Option<f64>,
}

impl Default for DummyRegressor {
    fn default() -> Self {
        Self {
            strategy: "mean".into(),
            constant: None,
            quantile: None,
            constant_: None,
            target: String::new(),
            run: None,
            loss: None,
        }
    }
}

impl DummyRegressor {
    /// A baseline with `strategy = "mean"`.
    pub fn new() -> Self {
        Self::default()
    }

    /// `"mean"`, `"median"`, `"quantile"` or `"constant"`.
    #[track_caller]
    pub fn strategy(mut self, strategy: &str) -> Self {
        let allowed: [&str; 4] = ["mean", "median", "quantile", "constant"];
        self.strategy = one_of("strategy", strategy, &allowed).or_fatal();
        self
    }

    /// The quantile that `strategy("quantile")` predicts, from 0 (the min) to 1 (the max).
    #[track_caller]
    pub fn quantile(mut self, quantile: f64) -> Self {
        if !(0.0..=1.0).contains(&quantile) {
            Err::<(), ModelError>(ModelError::InvalidSetting {
                setting: "quantile",
                value: quantile.to_string(),
                expected: "a number from 0 to 1",
            })
            .or_fatal();
        }
        self.quantile = Some(quantile);
        self
    }

    /// The value `strategy("constant")` predicts.
    pub fn constant(mut self, value: f64) -> Self {
        self.constant = Some(value);
        self
    }
}

impl Model for DummyRegressor {
    fn try_train(&mut self, dataset: &Dataset, target: &str) -> Result<(), ModelError> {
        let stats = numeric_target(dataset, target)?;
        let value: f64 = match self.strategy.as_str() {
            "mean" => stats.mean,
            "median" => stats.median,
            "quantile" => {
                let q: f64 = self.quantile.ok_or(ModelError::MissingSetting {
                    setting: "quantile",
                    reason: "strategy(\"quantile\") predicts it",
                })?;
                let x = col(target).cast(DataType::Float64);
                let frame = (dataset.dataframe.clone().lazy())
                    .select([x.quantile(lit(q), QuantileMethod::Linear)])
                    .collect()?;
                frame.columns()[0].f64()?.get(0).unwrap_or(f64::NAN)
            }
            _ => self.constant.ok_or(ModelError::MissingSetting {
                setting: "constant",
                reason: "strategy(\"constant\") predicts it",
            })?,
        };
        self.constant_ = Some(value);
        // mean((y - c)²) = population variance + (mean - c)², from the cached stats.
        let n: f64 = dataset.height() as f64;
        let variance: f64 = stats.variance * (n - 1.0) / n;
        self.loss = Some(0.5 * (variance + (stats.mean - value).powi(2)));
        self.target = target.into();
        Ok(())
    }

    fn params(&self) -> Value {
        json!({"strategy": self.strategy, "constant": self.constant, "quantile": self.quantile})
    }

    fn learned(&self) -> Value {
        json!({ "constant_": self.constant_ })
    }

    fn try_set_params(&mut self, params: Value) -> Result<(), ModelError> {
        #[derive(serde::Deserialize)]
        struct Params {
            strategy: String,
            constant: Option<f64>,
            quantile: Option<f64>,
        }
        let p: Params = serde_json::from_value(params)?;
        (self.strategy, self.constant, self.quantile) = (p.strategy, p.constant, p.quantile);
        Ok(())
    }

    fn try_load(&mut self, learned: Value, target: &str) -> Result<(), ModelError> {
        #[derive(serde::Deserialize)]
        struct Learned {
            constant_: Option<f64>,
        }
        self.constant_ = serde_json::from_value::<Learned>(learned)?.constant_;
        self.target = target.into();
        Ok(())
    }

    fn run(&self) -> Option<&str> {
        self.run.as_deref()
    }

    fn loss_curve(&self) -> Option<Vec<f64>> {
        self.loss.map(|l| vec![l])
    }

    fn set_run(&mut self, run: Option<String>) {
        self.run = run;
    }
}

impl Regressor for DummyRegressor {
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
            |mut m: DummyRegressor, target: &str| m.try_train(&ds, target).unwrap_err().to_string();
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
