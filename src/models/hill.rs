//! [`HillClimbing`]: a weighted blend of models' out-of-fold predictions, grown one model at a
//! time, as the Kaggle winners' notebooks build their last ensemble.

use polars::prelude::{DataType, NamedFrom, Series};
use polars_core::runtime::THREAD_POOL;
use rayon::prelude::*;

use crate::{
    dataset::Dataset,
    error::OrFatal,
    metrics::positive,
    models::{ModelError, check, design::column, hist::feature},
};

const SCORINGS: &str = "\"roc_auc\", \"neg_log_loss\", \"neg_root_mean_squared_error\", \
                        \"neg_mean_squared_error\", \"neg_mean_absolute_error\" or \"r2\"";

/// A blend of models' predictions, grown greedily. Each step tries every model at every
/// weight w of a grid (−0.5 to 0.5 by 0.01, the default), as `blend × (1 − w) + model × w`,
/// and keeps the best; it stops when no step adds `tol` to the score. A model can come back
/// in a later step. The blend starts from the best single model.
///
/// This is the hill climbing of Chris Deotte's notebooks (Kaggle S6E3, first place), which
/// most S6 winners use for their last blend: no Python library has it. `ranks(true)` blends
/// the ranks of each column, scaled to 0 to 1, instead of its values (his choice for
/// `roc_auc`, where only the order counts).
///
/// `fit` reads one column per model (out-of-fold predictions, from [`cross_val_predict`])
/// and the target; `predict` blends the same columns of other rows (the models' test
/// predictions) with the weights found.
///
/// [`cross_val_predict`]: crate::models::cross_val_predict
///
/// ```
/// use rusmlai::prelude::*;
/// let oof = Dataset::new(
///     df!(
///         "good" => [0.1, 0.2, 0.7, 0.4, 0.8, 0.9],
///         "other" => [0.3, 0.1, 0.0, 0.9, 0.2, 0.4],
///         "y" => [0, 0, 0, 1, 1, 1],
///     )
///     .unwrap(),
///     "",
/// );
/// let mut blend = HillClimbing::new().scoring("roc_auc");
/// blend.fit(&oof, "y");
/// // Each model alone ranks one positive below a negative; a quarter of `other` fixes it.
/// assert_eq!(blend.single_scores_[0], ("good".to_string(), 8.0 / 9.0));
/// assert_eq!(blend.score_, 1.0);
/// // The climb starts from `good`; the weights sum to 1.
/// assert_eq!(blend.history_[0].0, "good");
/// let total: f64 = blend.weights_.iter().map(|(_, w)| w).sum();
/// assert!((total - 1.0).abs() < 1e-12);
/// let blended = blend.predict(&oof);
/// assert_eq!(blended.len(), 6);
/// ```
///
/// Python (Deotte's loop, with numpy):
///
/// ```python
/// import numpy as np
/// from sklearn.metrics import roc_auc_score
/// x = np.array([[0.1, 0.3], [0.2, 0.1], [0.7, 0.0], [0.4, 0.9], [0.8, 0.2], [0.9, 0.4]])
/// y = np.array([0, 0, 0, 1, 1, 1])
/// ww = np.arange(-0.5, 0.5 + 1e-9, 0.01)
/// singles = [roc_auc_score(y, x[:, k]) for k in range(2)]
/// assert abs(singles[0] - 8 / 9) < 1e-12
/// blend, score = x[:, 0].copy(), singles[0]
/// while True:
///     cands = [(roc_auc_score(y, blend * (1 - w) + x[:, k] * w), k, w) for k in range(2) for w in ww]
///     best = max(cands, key=lambda c: c[0])
///     if best[0] - score < 1e-5:
///         break
///     score, blend = best[0], blend * (1 - best[2]) + x[:, best[1]] * best[2]
/// assert score == 1.0
/// ```
#[derive(Debug, Clone)]
pub struct HillClimbing {
    scoring: String,
    start: f64,
    stop: f64,
    step: f64,
    tol: f64,
    max_models: usize,
    ranks: bool,
    first: String,
    /// The model columns `fit` saw.
    pub feature_names_in_: Vec<String>,
    /// Each model's weight in the blend, largest first; they sum to 1. Only models the
    /// blend took are listed.
    pub weights_: Vec<(String, f64)>,
    /// The blend's score on the `fit` rows.
    pub score_: f64,
    /// Each model's own score, in column order.
    pub single_scores_: Vec<(String, f64)>,
    /// Each step: the model added, its step weight w and the blend's score after it. The
    /// first entry is the start (w = 1).
    pub history_: Vec<(String, f64, f64)>,
}

impl Default for HillClimbing {
    fn default() -> Self {
        Self {
            scoring: "neg_root_mean_squared_error".into(),
            start: -0.5,
            stop: 0.5,
            step: 0.01,
            tol: 1e-5,
            max_models: 1000,
            ranks: false,
            first: String::new(),
            feature_names_in_: Vec::new(),
            weights_: Vec::new(),
            score_: f64::NAN,
            single_scores_: Vec::new(),
            history_: Vec::new(),
        }
    }
}

/// What a candidate blend is scored against.
enum Truth {
    /// A 0/1 target, true for the positive class.
    Binary(Vec<bool>),
    Number(Vec<f64>),
}

/// The ranks of `x` (ties get their mean rank), scaled to 0 to 1: scipy's
/// `rankdata(x, "average")`, then (r − 1) / (n − 1).
fn rank01(x: &[f64]) -> Vec<f64> {
    let n: usize = x.len();
    if n == 1 {
        return vec![0.5];
    }
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.par_sort_unstable_by(|a, b| x[*a as usize].total_cmp(&x[*b as usize]));
    let mut out: Vec<f64> = vec![0.0; n];
    let mut i: usize = 0;
    while i < n {
        let mut j: usize = i;
        while j + 1 < n && x[order[j + 1] as usize] == x[order[i] as usize] {
            j += 1;
        }
        // Ranks i + 1 to j + 1, their mean, minus one.
        let rank: f64 = (i + j) as f64 / 2.0;
        order[i..=j].iter().for_each(|r| out[*r as usize] = rank / (n - 1) as f64);
        i = j + 1;
    }
    out
}

/// The area under the ROC curve of `score` against `y`: the share of (positive, negative)
/// pairs that the score orders right, ties counting half (Mann-Whitney), as scikit-learn's.
fn auc(y: &[bool], score: &[f64]) -> f64 {
    // Each score as a key that sorts as the number (IEEE bits, the negatives flipped), next to
    // its label: one contiguous sort.
    let key = |v: f64| -> u64 {
        let bits: u64 = (v + 0.0).to_bits(); // -0.0 sorts as 0.0
        if bits >> 63 == 1 { !bits } else { bits | 1 << 63 }
    };
    let mut pairs: Vec<(u64, bool)> = score.iter().zip(y).map(|(v, y)| (key(*v), *y)).collect();
    pairs.sort_unstable_by_key(|p| p.0);
    let (mut sum, mut positives): (f64, f64) = (0.0, 0.0);
    let mut i: usize = 0;
    while i < pairs.len() {
        let mut j: usize = i;
        while j + 1 < pairs.len() && pairs[j + 1].0 == pairs[i].0 {
            j += 1;
        }
        let rank: f64 = (i + j) as f64 / 2.0 + 1.0;
        let hits: usize = pairs[i..=j].iter().filter(|p| p.1).count();
        sum += rank * hits as f64;
        positives += hits as f64;
        i = j + 1;
    }
    let negatives: f64 = y.len() as f64 - positives;
    (sum - positives * (positives + 1.0) / 2.0) / (positives * negatives)
}

impl Truth {
    /// The score of predictions `p`: higher is better.
    fn score(&self, scoring: &str, p: &[f64]) -> f64 {
        let n: f64 = p.len() as f64;
        match (self, scoring) {
            (Truth::Binary(y), "roc_auc") => auc(y, p),
            (Truth::Binary(y), _) => {
                // neg_log_loss, the probabilities kept inside (ε, 1 − ε) like scikit-learn.
                let eps: f64 = f64::EPSILON;
                let sum: f64 = y.iter().zip(p).map(|(y, p)| {
                    let p: f64 = p.clamp(eps, 1.0 - eps);
                    if *y { p.ln() } else { (1.0 - p).ln() }
                }).sum();
                sum / n
            }
            (Truth::Number(y), _) => {
                let error = |f: fn(f64) -> f64| y.iter().zip(p).map(|(y, p)| f(y - p)).sum::<f64>() / n;
                match scoring {
                    "neg_mean_absolute_error" => -error(f64::abs),
                    "neg_mean_squared_error" => -error(|d| d * d),
                    "r2" => {
                        let mean: f64 = y.iter().sum::<f64>() / n;
                        let total: f64 = y.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
                        1.0 - error(|d| d * d) / total
                    }
                    _ => -error(|d| d * d).sqrt(),
                }
            }
        }
    }
}

impl HillClimbing {
    /// Scored by `"neg_root_mean_squared_error"`, weights −0.5 to 0.5 by 0.01, `tol` 1e-5.
    pub fn new() -> Self {
        Self::default()
    }

    /// What the blend maximizes, a scikit-learn scorer name: `"roc_auc"` or
    /// `"neg_log_loss"` (a target of two classes, the columns scores or probabilities of
    /// the second class), or `"neg_root_mean_squared_error"` (the default),
    /// `"neg_mean_squared_error"`, `"neg_mean_absolute_error"` or `"r2"` (a number target).
    #[track_caller]
    pub fn scoring(mut self, scoring: &str) -> Self {
        let ok: bool = matches!(
            scoring,
            "roc_auc" | "neg_log_loss" | "neg_root_mean_squared_error" | "neg_mean_squared_error" | "neg_mean_absolute_error" | "r2"
        );
        check(ok, "scoring", scoring, SCORINGS);
        self.scoring = scoring.into();
        self
    }

    /// The grid of step weights: `start` to `stop` by `step`. Default −0.5 to 0.5 by 0.01.
    /// A negative weight takes some of a model away.
    #[track_caller]
    pub fn weights(mut self, start: f64, stop: f64, step: f64) -> Self {
        let ok: bool = start.is_finite() && stop.is_finite() && start <= stop && step > 0.0 && (stop - start) / step < 1e6;
        check(ok, "weights", format!("({start}, {stop}, {step})"), "start ≤ stop and a step above 0");
        (self.start, self.stop, self.step) = (start, stop, step);
        self
    }

    /// The least gain of a step; below it the climb stops. Default 1e-5.
    #[track_caller]
    pub fn tol(mut self, tol: f64) -> Self {
        check(tol.is_finite() && tol >= 0.0, "tol", tol, "a finite number of 0 or more");
        self.tol = tol;
        self
    }

    /// The most distinct models in the blend. Default 1000.
    #[track_caller]
    pub fn max_models(mut self, max_models: usize) -> Self {
        check(max_models >= 1, "max_models", max_models, "1 or more");
        self.max_models = max_models;
        self
    }

    /// `true` blends each column's ranks, scaled to 0 to 1, instead of its values. Default
    /// `false`.
    pub fn ranks(mut self, ranks: bool) -> Self {
        self.ranks = ranks;
        self
    }

    /// The model to start from; `""` (the default) is the best single model.
    pub fn first(mut self, first: &str) -> Self {
        self.first = first.into();
        self
    }

    /// The model columns of `dataset` (every column but `target`), as numbers, ranked when
    /// `ranks` is set.
    fn columns(&self, dataset: &Dataset, names: &[String]) -> Result<Vec<Vec<f64>>, ModelError> {
        THREAD_POOL.install(|| {
            names.par_iter().map(|name| {
                let values: Vec<f64> = column(&feature(dataset, name)?).to_vec();
                if values.iter().any(|v| !v.is_finite()) {
                    return Err(ModelError::NotFinite { column: name.clone() });
                }
                Ok(if self.ranks { rank01(&values) } else { values })
            }).collect()
        })
    }

    /// Climbs on the model columns of `dataset` against `target`.
    #[track_caller]
    pub fn fit(&mut self, dataset: &Dataset, target: &str) -> &mut Self {
        self.try_fit(dataset, target).or_fatal();
        self
    }

    /// [`HillClimbing::fit`], or why it could not.
    pub fn try_fit(&mut self, dataset: &Dataset, target: &str) -> Result<&mut Self, ModelError> {
        let names: Vec<String> = dataset.columns().into_iter().filter(|c| c != target).collect();
        if names.is_empty() {
            return Err(ModelError::NoFeatures { target: target.into() });
        }
        let y: Series = dataset.column(target)?.as_materialized_series().clone();
        let truth: Truth = match self.scoring.as_str() {
            "roc_auc" | "neg_log_loss" => Truth::Binary(positive(&y, "")?),
            _ => Truth::Number(column(&y.cast(&DataType::Float64)?.rechunk()).to_vec()),
        };
        let x: Vec<Vec<f64>> = self.columns(dataset, &names)?;
        let scoring: &str = &self.scoring;
        let singles: Vec<f64> = THREAD_POOL.install(|| x.par_iter().map(|c| truth.score(scoring, c)).collect());
        let start: usize = match self.first.as_str() {
            "" => (0..names.len()).fold(0, |best, k| if singles[k] > singles[best] { k } else { best }),
            first => names.iter().position(|n| n == first).ok_or_else(|| ModelError::InvalidSetting {
                setting: "first",
                value: first.into(),
                expected: "one of the model columns, or \"\" for the best single model",
            })?,
        };
        let grid: Vec<f64> = (0..).map(|j| self.start + j as f64 * self.step).take_while(|w| *w <= self.stop + 1e-9).collect();
        let mut blend: Vec<f64> = x[start].clone();
        let mut score: f64 = singles[start];
        let mut steps: Vec<(usize, f64)> = Vec::new();
        self.history_ = vec![(names[start].clone(), 1.0, score)];
        let mut used: Vec<bool> = vec![false; names.len()];
        used[start] = true;
        while used.iter().filter(|u| **u).count() < self.max_models {
            // Every (model, weight), scored in parallel; the first best wins.
            let candidates: Vec<(usize, f64)> = (0..names.len()).flat_map(|k| grid.iter().map(move |w| (k, *w))).collect();
            let scores: Vec<f64> = THREAD_POOL.install(|| {
                candidates.par_iter().map(|(k, w)| {
                    let mixed: Vec<f64> = blend.iter().zip(&x[*k]).map(|(b, v)| b * (1.0 - w) + v * w).collect();
                    truth.score(scoring, &mixed)
                }).collect()
            });
            let best: usize = (0..scores.len()).fold(0, |best, i| if scores[i] > scores[best] { i } else { best });
            if scores[best] - score < self.tol {
                break;
            }
            let (k, w): (usize, f64) = candidates[best];
            blend.iter_mut().zip(&x[k]).for_each(|(b, v)| *b = *b * (1.0 - w) + v * w);
            score = scores[best];
            steps.push((k, w));
            used[k] = true;
            self.history_.push((names[k].clone(), w, score));
        }
        // Step weights to each model's share: every step scales what came before by 1 − w.
        let mut shares: Vec<f64> = vec![0.0; names.len()];
        let mut chain: Vec<(usize, f64)> = vec![(start, 1.0)];
        for (k, w) in &steps {
            chain.iter_mut().for_each(|(_, share)| *share *= 1.0 - w);
            chain.push((*k, *w));
        }
        chain.iter().for_each(|(k, share)| shares[*k] += share);
        let mut weights: Vec<(String, f64)> = (0..names.len()).filter(|k| used[*k]).map(|k| (names[k].clone(), shares[k])).collect();
        weights.sort_by(|a, b| b.1.total_cmp(&a.1));
        self.weights_ = weights;
        self.score_ = score;
        self.single_scores_ = names.iter().cloned().zip(singles).collect();
        self.feature_names_in_ = names;
        Ok(self)
    }

    /// The blend of the model columns of `dataset` (the names `fit` saw), with the weights
    /// found.
    #[track_caller]
    pub fn predict(&self, dataset: &Dataset) -> Series {
        self.try_predict(dataset).or_fatal()
    }

    /// [`HillClimbing::predict`], or why it could not.
    pub fn try_predict(&self, dataset: &Dataset) -> Result<Series, ModelError> {
        if self.weights_.is_empty() {
            return Err(ModelError::NotFitted { model: "HillClimbing" });
        }
        let names: Vec<String> = self.weights_.iter().map(|(n, _)| n.clone()).collect();
        let x: Vec<Vec<f64>> = self.columns(dataset, &names)?;
        let mut out: Vec<f64> = vec![0.0; dataset.height()];
        for (values, (_, weight)) in x.iter().zip(&self.weights_) {
            out.iter_mut().zip(values).for_each(|(o, v)| *o += weight * v);
        }
        Ok(Series::new("blend".into(), out))
    }
}

#[cfg(test)]
mod tests {
    use polars::prelude::{Column, DataFrame};

    use super::*;

    /// 300 rows and four "models" of a 0/1 target: the formulas of
    /// `analysis/2026-11-20/hill/parity.py`.
    pub(crate) fn models(n: usize) -> Dataset {
        let i = |r: usize| r as f64;
        let y: Vec<i64> = (0..n).map(|r| i64::from((i(r) * 0.37).sin() + (i(r) * 0.11).cos() > 0.3)).collect();
        let truth: Vec<f64> = y.iter().map(|v| *v as f64).collect();
        let m = |a: f64, b: f64| -> Vec<f64> {
            (0..n).map(|r| (truth[r] * a + ((i(r) * b).sin() * 0.5 + 0.5) * (1.0 - a)).clamp(0.0, 1.0)).collect()
        };
        let frame: DataFrame = DataFrame::new(n, vec![
            Column::new("m1".into(), m(0.3, 1.7)),
            Column::new("m2".into(), m(0.25, 2.3)),
            Column::new("m3".into(), m(0.2, 0.9)),
            Column::new("m4".into(), m(0.15, 3.1)),
            Column::new("y".into(), y),
        ]).unwrap();
        Dataset::new(frame, "")
    }

    #[test]
    fn rank01_is_scipys_average_rank_scaled() {
        assert_eq!(rank01(&[3.0, 1.0, 3.0, 2.0, 5.0]), [0.625, 0.0, 0.625, 0.25, 1.0]);
    }

    /// Against Deotte's loop in numpy, from `analysis/2026-11-20/hill/parity.py`.
    #[test]
    fn the_blend_matches_the_notebooks_loop() {
        let ds: Dataset = models(300);
        let mut auc = HillClimbing::new().scoring("roc_auc");
        auc.fit(&ds, "y");
        let mut ranked = HillClimbing::new().scoring("roc_auc").ranks(true);
        ranked.fit(&ds, "y");
        let mut rmse = HillClimbing::new();
        rmse.fit(&ds, "y");
        let path = |b: &HillClimbing| -> Vec<(String, i64)> {
            b.history_.iter().map(|(m, w, _)| (m.clone(), (w * 100.0).round() as i64)).collect()
        };
        let want = |steps: &[(&str, i64)]| -> Vec<(String, i64)> { steps.iter().map(|(m, w)| (m.to_string(), *w)).collect() };
        let cases = [
            (&auc, 0.9014728288471304, want(&[("m1", 100), ("m2", 42), ("m3", 26), ("m4", 9), ("m3", -4), ("m1", 6), ("m4", 2), ("m3", -1), ("m2", -1)])),
            (&ranked, 0.891361558705388, want(&[("m1", 100), ("m2", 39), ("m3", 28), ("m4", 17), ("m3", -3), ("m4", -6), ("m3", -2), ("m1", 3)])),
            (&rmse, -0.39506343255007487, want(&[("m1", 100), ("m2", 38), ("m3", 17), ("m4", 4), ("m2", -3)])),
        ];
        for (blend, score, steps) in cases {
            assert!((blend.score_ - score).abs() < 1e-12, "{} against {score}", blend.score_);
            assert_eq!(path(blend), steps);
        }
        let total: f64 = rmse.weights_.iter().map(|w| w.1).sum();
        assert!((total - 1.0).abs() < 1e-12);
        // The blend of the fit rows scores what the climb said.
        let blended: Vec<f64> = column(&rmse.predict(&ds)).to_vec();
        let y: Vec<f64> = ds.to_list("y");
        let score: f64 = Truth::Number(y).score("neg_root_mean_squared_error", &blended);
        assert!((score - rmse.score_).abs() < 1e-12);
    }

}
