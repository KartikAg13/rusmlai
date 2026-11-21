//! [`OlsSummary`]: statsmodels' OLS results table for a fitted [`LinearRegression`].

use std::fmt;

use faer::{Mat, Side};

use crate::{
    dataset::Dataset,
    display::TableView,
    error::OrFatal,
    models::{LinearRegression, ModelError, Regressor, design::DesignMatrix},
};

/// ln Γ(x) for x > 0 (Lanczos, g = 7, 9 terms): 15 digits.
fn ln_gamma(x: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        let pi: f64 = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x: f64 = x - 1.0;
    let t: f64 = x + 7.5;
    let sum: f64 = C[1..].iter().enumerate().fold(C[0], |s, (i, c)| s + c / (x + i as f64 + 1.0));
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

/// The regularized incomplete beta function I_x(a, b), by its continued fraction (Lentz).
fn beta_inc(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let front: f64 = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    // The fraction converges fast for x below (a + 1) / (a + b + 2); else use the symmetry.
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - beta_inc(b, a, 1.0 - x);
    }
    let tiny: f64 = 1e-300;
    let (mut c, mut d): (f64, f64) = (1.0, 1.0 - (a + b) * x / (a + 1.0));
    d = 1.0 / if d.abs() < tiny { tiny } else { d };
    let mut f: f64 = d;
    for m in 1..1000 {
        let m: f64 = m as f64;
        for numerator in [
            m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m)),
            -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0)),
        ] {
            d = 1.0 + numerator * d;
            d = 1.0 / if d.abs() < tiny { tiny } else { d };
            c = 1.0 + numerator / c;
            c = if c.abs() < tiny { tiny } else { c };
            f *= c * d;
        }
        if (c * d - 1.0).abs() < 1e-16 {
            break;
        }
    }
    front * f / a
}

/// P(|T| > |t|) for Student's t with `df` degrees of freedom.
fn t_two_sided(t: f64, df: f64) -> f64 {
    beta_inc(df / 2.0, 0.5, df / (df + t * t))
}

/// The t with P(T ≤ t) = q, for q above one half: bisection on the distribution.
fn t_quantile(q: f64, df: f64) -> f64 {
    let (mut lo, mut hi): (f64, f64) = (0.0, 1e4);
    for _ in 0..200 {
        let mid: f64 = (lo + hi) / 2.0;
        if 1.0 - t_two_sided(mid, df) / 2.0 < q { lo = mid } else { hi = mid }
    }
    (lo + hi) / 2.0
}

/// P(F > f) for the F distribution with `d1` and `d2` degrees of freedom.
fn f_upper(f: f64, d1: f64, d2: f64) -> f64 {
    beta_inc(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * f))
}

/// scipy's `skewtest` and `kurtosistest` statistics, and the omnibus K² = Zs² + Zk² of
/// `normaltest`, which statsmodels reports, with its p-value (χ² with 2 degrees of freedom).
fn omnibus(skew: f64, kurtosis: f64, n: f64) -> (f64, f64) {
    let y: f64 = skew * ((n + 1.0) * (n + 3.0) / (6.0 * (n - 2.0))).sqrt();
    let beta2: f64 = 3.0 * (n * n + 27.0 * n - 70.0) * (n + 1.0) * (n + 3.0) / ((n - 2.0) * (n + 5.0) * (n + 7.0) * (n + 9.0));
    let w2: f64 = -1.0 + (2.0 * (beta2 - 1.0)).sqrt();
    let delta: f64 = 1.0 / (0.5 * w2.ln()).sqrt();
    let alpha: f64 = (2.0 / (w2 - 1.0)).sqrt();
    let y: f64 = if y == 0.0 { 1.0 } else { y };
    let zs: f64 = delta * (y / alpha + ((y / alpha).powi(2) + 1.0).sqrt()).ln();
    let e: f64 = 3.0 * (n - 1.0) / (n + 1.0);
    let var: f64 = 24.0 * n * (n - 2.0) * (n - 3.0) / ((n + 1.0).powi(2) * (n + 3.0) * (n + 5.0));
    let x: f64 = (kurtosis - e) / var.sqrt();
    let sqrt_beta1: f64 = 6.0 * (n * n - 5.0 * n + 2.0) / ((n + 7.0) * (n + 9.0)) * (6.0 * (n + 3.0) * (n + 5.0) / (n * (n - 2.0) * (n - 3.0))).sqrt();
    let a: f64 = 6.0 + 8.0 / sqrt_beta1 * (2.0 / sqrt_beta1 + (1.0 + 4.0 / sqrt_beta1.powi(2)).sqrt());
    let term1: f64 = 1.0 - 2.0 / (9.0 * a);
    let denom: f64 = 1.0 + x * (2.0 / (a - 4.0)).sqrt();
    let term2: f64 = denom.signum() * ((1.0 - 2.0 / a) / denom.abs()).cbrt();
    let zk: f64 = (term1 - term2) / (2.0 / (9.0 * a)).sqrt();
    let k2: f64 = zs * zs + zk * zk;
    (k2, (-k2 / 2.0).exp())
}

/// statsmodels' `OLS(...).fit().summary()`: the fit's numbers, the coefficients' standard
/// errors, t values, p values and 95% intervals, and the residuals' tests. Every field has
/// the name of the statsmodels result attribute it matches.
#[derive(Debug, Clone)]
pub struct OlsSummary {
    pub dep_variable: String,
    /// The intercept first (`"const"`, when the model fits one), then the features.
    pub names: Vec<String>,
    pub params: Vec<f64>,
    pub bse: Vec<f64>,
    pub tvalues: Vec<f64>,
    pub pvalues: Vec<f64>,
    /// The 95% interval of each coefficient.
    pub conf_int: Vec<(f64, f64)>,
    pub nobs: usize,
    pub df_model: f64,
    pub df_resid: f64,
    pub rsquared: f64,
    pub rsquared_adj: f64,
    pub fvalue: f64,
    pub f_pvalue: f64,
    pub llf: f64,
    pub aic: f64,
    pub bic: f64,
    pub omnibus: f64,
    pub omnibus_pvalue: f64,
    pub durbin_watson: f64,
    pub jarque_bera: f64,
    pub jarque_bera_pvalue: f64,
    pub skew: f64,
    pub kurtosis: f64,
    pub condition_number: f64,
}

/// A number as statsmodels' summary prints it: 3 decimals, or 3 significant digits in
/// scientific form when it is large or small.
fn short(v: f64) -> String {
    if v != 0.0 && (v.abs() >= 1e5 || v.abs() < 1e-3) { format!("{v:.3e}") } else { format!("{v:.4}") }
}

impl OlsSummary {
    /// The three tables of statsmodels' summary: the fit, the coefficients, the residuals.
    pub fn tables(&self) -> [TableView; 3] {
        let s = |v: &str| v.to_string();
        let fit: Vec<Vec<String>> = vec![
            vec![s("Dep. Variable:"), self.dep_variable.clone(), s("R-squared:"), short(self.rsquared)],
            vec![s("Model:"), s("OLS"), s("Adj. R-squared:"), short(self.rsquared_adj)],
            vec![s("Method:"), s("Least Squares"), s("F-statistic:"), short(self.fvalue)],
            vec![s("No. Observations:"), self.nobs.to_string(), s("Prob (F-statistic):"), short(self.f_pvalue)],
            vec![s("Df Residuals:"), self.df_resid.to_string(), s("Log-Likelihood:"), short(self.llf)],
            vec![s("Df Model:"), self.df_model.to_string(), s("AIC:"), short(self.aic)],
            vec![s("Covariance Type:"), s("nonrobust"), s("BIC:"), short(self.bic)],
        ];
        let headers: Vec<String> = ["", "coef", "std err", "t", "P>|t|", "[0.025", "0.975]"].map(s).to_vec();
        let coefficients: Vec<Vec<String>> = (0..self.names.len())
            .map(|i| {
                vec![
                    self.names[i].clone(),
                    short(self.params[i]),
                    short(self.bse[i]),
                    format!("{:.3}", self.tvalues[i]),
                    format!("{:.3}", self.pvalues[i]),
                    short(self.conf_int[i].0),
                    short(self.conf_int[i].1),
                ]
            })
            .collect();
        let residuals: Vec<Vec<String>> = vec![
            vec![s("Omnibus:"), short(self.omnibus), s("Durbin-Watson:"), short(self.durbin_watson)],
            vec![s("Prob(Omnibus):"), short(self.omnibus_pvalue), s("Jarque-Bera (JB):"), short(self.jarque_bera)],
            vec![s("Skew:"), short(self.skew), s("Prob(JB):"), short(self.jarque_bera_pvalue)],
            vec![s("Kurtosis:"), short(self.kurtosis), s("Cond. No."), short(self.condition_number)],
        ];
        [
            TableView::new("OLS Regression Results", &["", "", "", ""].map(s), &fit),
            TableView::new("", &headers, &coefficients),
            TableView::new("", &["", "", "", ""].map(s), &residuals),
        ]
    }

    /// Prints the three tables: HTML in a notebook, text elsewhere.
    pub fn show(&self) {
        self.tables().iter().for_each(TableView::show);
    }
}

impl fmt::Display for OlsSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for table in self.tables() {
            writeln!(f, "{table}")?;
        }
        Ok(())
    }
}

impl LinearRegression {
    /// statsmodels' OLS summary of this fitted model on the rows it was fitted on: R² and
    /// adjusted R², the F test, the log-likelihood, AIC and BIC; each coefficient's standard
    /// error, t, p value and 95% interval; and the residuals' omnibus, Durbin-Watson and
    /// Jarque-Bera tests, skew, kurtosis and condition number. The covariance is the
    /// pseudo-inverse of XᵀX, as statsmodels' default. Prints the tables and returns the
    /// numbers. (statsmodels: `sm.OLS(y, sm.add_constant(X)).fit().summary()`)
    ///
    /// ```
    /// use rusmlai::prelude::*;
    /// let ds = Dataset::new(
    ///     df!(
    ///         "hours" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    ///         "sleep" => [8.0, 7.5, 7.0, 7.5, 6.0, 6.5, 6.0, 5.0],
    ///         "score" => [52.0, 55.0, 61.0, 63.0, 70.0, 71.0, 79.0, 84.0],
    ///     )
    ///     .unwrap(),
    ///     "",
    /// );
    /// let mut model = LinearRegression::new();
    /// model.fit(&ds, "score");
    /// let summary = model.summary(&ds, "score");
    /// assert_eq!(summary.names, ["const", "hours", "sleep"]);
    /// assert!(summary.rsquared > 0.98);
    /// assert!(summary.pvalues[1] < 0.01); // hours matter
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// import statsmodels.api as sm
    /// ds = pd.DataFrame({
    ///     "hours": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    ///     "sleep": [8.0, 7.5, 7.0, 7.5, 6.0, 6.5, 6.0, 5.0],
    ///     "score": [52.0, 55.0, 61.0, 63.0, 70.0, 71.0, 79.0, 84.0],
    /// })
    /// result = sm.OLS(ds["score"], sm.add_constant(ds[["hours", "sleep"]])).fit()
    /// print(result.summary())
    /// assert result.params.index.tolist() == ["const", "hours", "sleep"]
    /// assert result.rsquared > 0.98
    /// assert result.pvalues["hours"] < 0.01
    /// ```
    #[track_caller]
    pub fn summary(&self, dataset: &Dataset, target: &str) -> OlsSummary {
        let summary: OlsSummary = self.try_summary(dataset, target).or_fatal();
        summary.show();
        summary
    }

    /// [`LinearRegression::summary`] without printing, or why it could not: not fitted, a
    /// feature missing, or as many coefficients as rows.
    pub fn try_summary(&self, dataset: &Dataset, target: &str) -> Result<OlsSummary, ModelError> {
        if self.feature_names_in_.is_empty() {
            return Err(ModelError::NotFitted { model: "LinearRegression" });
        }
        let mut columns: Vec<&str> = self.feature_names_in_.iter().map(String::as_str).collect();
        columns.push(target);
        let data: Dataset = dataset.try_select(&columns)?;
        let matrix: DesignMatrix = DesignMatrix::try_new(&data, target)?;
        let constant: bool = self.fits_intercept();
        let (n, p): (usize, usize) = (matrix.rows(), matrix.names.len());
        let (x_means, y_mean): (Vec<f64>, f64) = matrix.means();
        let centers: Vec<f64> = if constant { x_means.clone() } else { vec![0.0; p] };
        let (g, _) = matrix.gram(&centers, 0.0);
        let k: usize = p + usize::from(constant);
        let (df_model, df_resid): (f64, f64) = (p as f64, n as f64 - k as f64);
        if df_resid <= 0.0 {
            return Err(ModelError::InvalidSetting {
                setting: "summary",
                value: format!("{n} rows for {k} coefficients"),
                expected: "more rows than coefficients",
            });
        }
        let predicted: Vec<f64> = crate::models::design::column(&self.try_predict(&data)?).to_vec();
        let y: &[f64] = matrix.y();
        let residuals: Vec<f64> = y.iter().zip(&predicted).map(|(y, p)| y - p).collect();
        let rss: f64 = residuals.iter().map(|e| e * e).sum();
        let tss: f64 = match constant {
            true => y.iter().map(|v| (v - y_mean).powi(2)).sum(),
            false => y.iter().map(|v| v * v).sum(),
        };
        let sigma2: f64 = rss / df_resid;
        // (XᵀX)⁺ of the centered features, from its eigendecomposition (statsmodels' pinv).
        let eigen = g.self_adjoint_eigen(Side::Lower).map_err(|_| ModelError::Solve)?;
        let (u, s) = (eigen.U(), eigen.S().column_vector());
        let largest: f64 = (0..p).map(|i| s[i].abs()).fold(0.0, f64::max);
        let cut: f64 = largest * f64::EPSILON * (n.max(p) as f64);
        let mut inverse: Mat<f64> = Mat::zeros(p, p);
        for m in (0..p).filter(|m| s[*m] > cut) {
            for i in 0..p {
                for j in 0..p {
                    inverse[(i, j)] += u[(i, m)] * u[(j, m)] / s[m];
                }
            }
        }
        let mut names: Vec<String> = Vec::new();
        let (mut params, mut variances): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
        if constant {
            // Var(b) = σ² (1/n + x̄ᵀ (XcᵀXc)⁺ x̄).
            let quad: f64 = (0..p).map(|i| (0..p).map(|j| x_means[i] * inverse[(i, j)] * x_means[j]).sum::<f64>()).sum();
            names.push("const".into());
            params.push(self.intercept_);
            variances.push(sigma2 * (1.0 / n as f64 + quad));
        }
        for (i, name) in matrix.names.iter().enumerate() {
            names.push(name.clone());
            params.push(self.coef_[i]);
            variances.push(sigma2 * inverse[(i, i)]);
        }
        let bse: Vec<f64> = variances.iter().map(|v| v.sqrt()).collect();
        let tvalues: Vec<f64> = params.iter().zip(&bse).map(|(b, s)| b / s).collect();
        let pvalues: Vec<f64> = tvalues.iter().map(|t| t_two_sided(*t, df_resid)).collect();
        let q: f64 = t_quantile(0.975, df_resid);
        let conf_int: Vec<(f64, f64)> = params.iter().zip(&bse).map(|(b, s)| (b - q * s, b + q * s)).collect();
        let nobs: f64 = n as f64;
        let llf: f64 = -nobs / 2.0 * ((2.0 * std::f64::consts::PI).ln() + (rss / nobs).ln() + 1.0);
        let rsquared: f64 = 1.0 - rss / tss;
        let fvalue: f64 = ((tss - rss) / df_model) / sigma2;
        // The residuals' moments (biased, as scipy's skew and kurtosis).
        let mean: f64 = residuals.iter().sum::<f64>() / nobs;
        let moment = |r: i32| residuals.iter().map(|e| (e - mean).powi(r)).sum::<f64>() / nobs;
        let (m2, m3, m4): (f64, f64, f64) = (moment(2), moment(3), moment(4));
        let (skew, kurtosis): (f64, f64) = (m3 / m2.powf(1.5), m4 / (m2 * m2));
        let (omnibus, omnibus_pvalue): (f64, f64) = omnibus(skew, kurtosis, nobs);
        let jarque_bera: f64 = nobs / 6.0 * (skew * skew + (kurtosis - 3.0).powi(2) / 4.0);
        let durbin_watson: f64 = residuals.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum::<f64>() / rss;
        // np.linalg.cond of the design with its column of ones: √(λmax / λmin) of its XᵀX.
        let size: usize = k;
        let mut full: Mat<f64> = Mat::zeros(size, size);
        for i in 0..p {
            for j in 0..p {
                full[(i, j)] = g[(i, j)] + if constant { nobs * x_means[i] * x_means[j] } else { 0.0 };
            }
            if constant {
                full[(i, p)] = nobs * x_means[i];
                full[(p, i)] = nobs * x_means[i];
            }
        }
        if constant {
            full[(p, p)] = nobs;
        }
        let values = full.self_adjoint_eigen(Side::Lower).map_err(|_| ModelError::Solve)?;
        let lambdas = values.S().column_vector();
        let condition_number: f64 = (lambdas[size - 1] / lambdas[0]).sqrt();
        Ok(OlsSummary {
            dep_variable: target.into(),
            names,
            params,
            bse,
            tvalues,
            pvalues,
            conf_int,
            nobs: n,
            df_model,
            df_resid,
            rsquared,
            rsquared_adj: 1.0 - (1.0 - rsquared) * (nobs - f64::from(u8::from(constant))) / df_resid,
            fvalue,
            f_pvalue: f_upper(fvalue, df_model, df_resid),
            llf,
            aic: -2.0 * llf + 2.0 * k as f64,
            bic: -2.0 * llf + nobs.ln() * k as f64,
            omnibus,
            omnibus_pvalue,
            durbin_watson,
            jarque_bera,
            jarque_bera_pvalue: (-jarque_bera / 2.0).exp(),
            skew,
            kurtosis,
            condition_number,
        })
    }
}

#[cfg(test)]
mod tests {
    use polars::df;

    use super::*;
    use crate::models::Model;

    /// Against `analysis/2026-11-20/summary/parity.py` (statsmodels' formulas on numpy and
    /// scipy).
    #[test]
    fn summary_matches_statsmodels() {
        let i: Vec<f64> = (0..50).map(f64::from).collect();
        let a: Vec<f64> = i.iter().map(|i| (i * 0.7).sin() * 3.0).collect();
        let b: Vec<f64> = i.iter().map(|i| (i * 13.0) % 11.0).collect();
        let c: Vec<f64> = i.iter().map(|i| (i * 0.3).cos() + i / 10.0).collect();
        let y: Vec<f64> = (0..50).map(|r| 2.0 + 1.5 * a[r] - 0.4 * b[r] + 0.8 * c[r] + (i[r] * 1.9).sin() * 0.7).collect();
        let ds = Dataset::new(df!("a" => a, "b" => b, "c" => c, "y" => y).unwrap(), "");
        let near = |got: f64, want: f64, what: &str| {
            assert!((got - want).abs() <= 1e-7 * want.abs().max(1e-12), "{what}: {got} against {want}");
        };
        type Want<'a> = (&'a [f64], &'a [f64], &'a [f64], &'a [f64], [f64; 12], [f64; 3]);
        let cases: [(bool, Want); 2] = [
            (true, (
                &[2.0488456980560965, 1.5013143375993858, -0.4067536948347207, 0.7925389529485666],
                &[0.17061551993052215, 0.03459487529219656, 0.022797601002299543, 0.04432838521451741],
                &[8.846798854751981e-16, 5.65524031978703e-39, 2.6614242489458737e-22, 2.4488993328314874e-22],
                &[1.7054144688805983, 1.4316784653785568, -0.4526428855581706, 0.7033105414430594],
                [0.9817741323183691, 0.9805854887739149, 825.9617754195499, 5.398852414906505e-40, -35.56755928818696, 79.13511857637393, 86.78321059808651, 37.8001435962874, 6.1916031210942556e-09, 2.5974756362448956, 4.56518790637211, 0.10201923008870394],
                [0.04973962420252257, 1.5230456763341056, 15.331226848078284],
            )),
            (false, (
                &[1.5083272518983948, -0.2349011719146869, 1.1178624526315315],
                &[0.0695844726807208, 0.035700126568885014, 0.0705840212162099],
                &[4.106576508971031e-26, 3.550729272261246e-08, 1.8012737128414426e-20],
                &[1.3683413490800667, -0.3067205628785933, 0.9758657175289207],
                [0.9455058453094712, 0.9420274950100758, 271.82594158897757, 1.0899025120974923e-29, -71.05413409984413, 148.10826819968827, 153.8443372159727, 0.21481756555419596, 0.8981584462335073, 1.073291489665715, 0.41550744102302795, 0.8124070912791422],
                [0.023479088423421404, 2.555884585588077, 3.296621546589294],
            )),
        ];
        for (constant, (params, bse, pvalues, lo, fit, moments)) in cases {
            let mut model = LinearRegression::new().fit_intercept(constant);
            model.try_train(&ds, "y").unwrap();
            let s: OlsSummary = model.try_summary(&ds, "y").unwrap();
            for k in 0..params.len() {
                near(s.params[k], params[k], "params");
                near(s.bse[k], bse[k], "bse");
                near(s.pvalues[k], pvalues[k], "pvalues");
                near(s.conf_int[k].0, lo[k], "conf_int");
            }
            let got: [f64; 12] = [s.rsquared, s.rsquared_adj, s.fvalue, s.f_pvalue, s.llf, s.aic, s.bic, s.omnibus, s.omnibus_pvalue, s.durbin_watson, s.jarque_bera, s.jarque_bera_pvalue];
            let labels = ["rsquared", "rsquared_adj", "fvalue", "f_pvalue", "llf", "aic", "bic", "omnibus", "omnibus_pvalue", "durbin_watson", "jarque_bera", "jarque_bera_pvalue"];
            for ((g, w), what) in got.iter().zip(fit).zip(labels) {
                near(*g, w, what);
            }
            near(s.skew, moments[0], "skew");
            near(s.kurtosis, moments[1], "kurtosis");
            near(s.condition_number, moments[2], "condition_number");
        }
    }

    #[test]
    fn the_distributions_match_scipy() {
        // scipy.stats: t.sf(2.0, 10) * 2, t.ppf(0.975, 46), f.sf(3.0, 2, 20).
        assert!((t_two_sided(2.0, 10.0) - 0.07338803477074037).abs() < 1e-14);
        assert!((t_quantile(0.975, 46.0) - 2.012895598919429).abs() < 1e-10);
        assert!((f_upper(3.0, 2.0, 20.0) - 0.07253815028640571).abs() < 1e-14);
    }

    #[test]
    fn as_many_coefficients_as_rows_is_refused() {
        let ds = Dataset::new(df!("x" => [1.0, 2.0], "y" => [1.0, 3.0]).unwrap(), "");
        let mut model = LinearRegression::new();
        model.try_train(&ds, "y").unwrap();
        assert!(model.try_summary(&ds, "y").unwrap_err().to_string().contains("more rows than coefficients"));
        assert!(LinearRegression::new().try_summary(&ds, "y").is_err());
    }
}
