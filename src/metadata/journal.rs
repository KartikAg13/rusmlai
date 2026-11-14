//! [`Dataset::journal`]: one HTML page with every model fitted on a CSV, read from its
//! [`ModelCache`]. It works offline: the charts are rusmlai's own SVG, whose points show their
//! values on hover, and the table sorts with a few lines of inline JavaScript.

use std::{
    fs,
    path::{Path, PathBuf},
};

use serde_json::Value;

use crate::{
    dataset::Dataset,
    display::{Figure, escape, in_notebook, open_in_viewer},
    error::OrFatal,
    metadata::{
        constants::JOURNAL,
        models::{Entry, FitRecord, ModelCache, ScoreRecord},
        utils::{cache_path, print_saved},
    },
    models::ModelError,
    plot::{
        chart::{Body, Chart, Texts},
        result::PlotResult,
        xy::line_figure,
    },
};

/// One fit and its latest scores.
struct Run<'a> {
    /// `r01`, `r02`, ... in the order the fits ran.
    label: String,
    fit: &'a FitRecord,
    train: Option<&'a ScoreRecord>,
    test: Option<&'a ScoreRecord>,
}

/// The runs in the file, oldest first. A score is "train" when it scored the data the model
/// trained on, "test" otherwise; the latest of each counts.
fn runs(entries: &[Entry]) -> Vec<Run<'_>> {
    let fits = entries.iter().filter_map(|e| match e {
        Entry::Fit(fit) => Some(fit),
        Entry::Score(_) => None,
    });
    fits.enumerate()
        .map(|(i, fit)| {
            let latest = |train: bool| {
                entries.iter().rev().find_map(|e| match e {
                    Entry::Score(s) if s.id == fit.id && (s.data == fit.data) == train => Some(s),
                    _ => None,
                })
            };
            Run {
                label: format!("r{:02}", i + 1),
                fit,
                train: latest(true),
                test: latest(false),
            }
        })
        .collect()
}

/// `YYYY-MM-DD HH:MM` (UTC) of seconds since 1970, by Howard Hinnant's `civil_from_days`.
fn utc(seconds: u64) -> String {
    let days: i64 = (seconds / 86_400) as i64 + 719_468;
    let era: i64 = days.div_euclid(146_097);
    let doe: i64 = days - era * 146_097;
    let yoe: i64 = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy: i64 = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp: i64 = (5 * doy + 2) / 153;
    let day: i64 = doy - (153 * mp + 2) / 5 + 1;
    let month: i64 = if mp < 10 { mp + 3 } else { mp - 9 };
    let year: i64 = yoe + era * 400 + i64::from(month <= 2);
    let minutes: u64 = seconds % 86_400 / 60;
    format!(
        "{year:04}-{month:02}-{day:02} {:02}:{:02}",
        minutes / 60,
        minutes % 60
    )
}

/// A setting as table text.
fn text(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// The left third of a card: the training and validation loss per round. A run with no curve
/// of two or more rounds (an exact solve, a baseline) shows its train and test r2 instead.
/// The charts are built from the numbers as they are: no `Dataset` and no Polars query, which
/// cost 0.5 ms per card for a few rows.
fn chart(run: &Run) -> String {
    let lines: Vec<(&str, Vec<(f64, f64)>)> = [
        ("train", &run.fit.loss_curve),
        ("validation", &run.fit.validation_loss_curve),
    ]
    .into_iter()
    .map(|(name, curve)| {
        let line: Vec<(f64, f64)> = (curve.iter().flatten().enumerate())
            .filter(|(_, loss)| loss.is_finite())
            .map(|(i, loss)| ((i + 1) as f64, *loss))
            .collect();
        (name, line)
    })
    .filter(|(_, line)| !line.is_empty())
    .collect();
    let figure: Option<Figure> = if lines.iter().any(|(_, line)| line.len() >= 2) {
        let top: f64 = (lines.iter().flat_map(|(_, line)| line))
            .fold(f64::MIN_POSITIVE, |top, point| top.max(point.1));
        let two: bool = lines.iter().any(|(name, _)| *name == "validation");
        let (names, lines): (Vec<String>, Vec<Vec<(f64, f64)>>) =
            lines.into_iter().map(|(n, l)| (n.to_string(), l)).unzip();
        let (hue, names) = if two {
            (Some("curve".to_string()), names)
        } else {
            (None, Vec::new())
        };
        line_figure("round", "loss", hue, names, &lines, None)
            .ok()
            // From 0, so the drop reads at its true size: 41.2 to 40.1 is 3%, not a cliff.
            .map(|fig| fig.title("Loss per round").ylim(0.0, top))
    } else {
        let scores: Vec<(&str, f64)> = [("train", run.train), ("test", run.test)]
            .into_iter()
            .filter_map(|(split, s)| s.map(|s| (split, main(s).1)))
            .filter(|score| score.1.is_finite())
            .collect();
        let metric: &str = run.test.or(run.train).map_or("score", |s| main(s).0);
        let low: f64 = scores.iter().map(|s| s.1).fold(0.0, f64::min);
        let texts: Texts = Texts {
            x: "split".into(),
            y: metric.into(),
            value: format!("mean {metric}"),
            ..Texts::default()
        };
        let body: Body = Body::Bar(PlotResult::Series {
            labels: scores.iter().map(|s| s.0.to_string()).collect(),
            values: scores.iter().map(|s| s.1).collect(),
        });
        let chart: Chart = Chart {
            name: "bar",
            texts,
            body,
        };
        // r2 and accuracy are at most 1: a fixed top keeps a baseline's -0.000002 a sliver,
        // not a full bar.
        (!scores.is_empty()).then(|| {
            Figure::new(chart, None)
                .sort("name")
                .title(&format!("{metric} (no loss curve)"))
                .ylim(low, 1.0)
        })
    };
    match figure {
        Some(fig) => fig.subtitle("").size(480.0, 320.0).svg(),
        None => "<p class=\"empty\">Nothing to draw yet: no loss curve and no score.</p>".into(),
    }
}

/// `name: value` rows.
fn pairs(rows: &[(String, String)]) -> String {
    rows.iter()
        .map(|(k, v)| format!("<dt>{}</dt><dd>{}</dd>", escape(k), escape(v)))
        .collect()
}

/// A score's main metric, the one `score` returns: the first one recorded (`r2` for a
/// regressor, `accuracy` for a classifier).
fn main(score: &ScoreRecord) -> (&str, f64) {
    score
        .metrics
        .first()
        .map_or(("score", f64::NAN), |(k, v)| (k.as_str(), *v))
}

/// The journal's name of a metric.
fn label(metric: &str) -> &str {
    match metric {
        "mse" => "MSE",
        "rmse" => "RMSE",
        "mae" => "MAE",
        "log_loss" => "log loss",
        other => other,
    }
}

/// One cell of the train / test table: rows, the loss (½ · MSE) or a recorded metric.
fn stat(score: Option<&ScoreRecord>, row: &str) -> String {
    let Some(s) = score else {
        return "—".into();
    };
    let value: Option<f64> = match row {
        "rows" => return s.rows.to_string(),
        "loss" => s.metrics.get("mse").map(|mse| mse / 2.0),
        metric => s.metrics.get(metric).copied(),
    };
    value.map_or("—".into(), |v| format!("{v:.6}"))
}

/// The train and test columns: rows, loss and every metric the scores recorded.
fn stats(run: &Run) -> String {
    let scores: Vec<&ScoreRecord> = [run.train, run.test].into_iter().flatten().collect();
    let mut rows: Vec<(&str, &str)> = vec![("rows", "rows")];
    if scores.iter().any(|s| s.metrics.contains_key("mse")) {
        rows.push(("loss (½ MSE)", "loss"));
    }
    for metric in scores.iter().flat_map(|s| s.metrics.keys()) {
        if !rows.iter().any(|(_, row)| row == metric) {
            rows.push((label(metric), metric));
        }
    }
    let body: String = rows
        .iter()
        .map(|(name, row)| {
            format!(
                "<tr><th>{name}</th><td>{}</td><td>{}</td></tr>",
                stat(run.train, row),
                stat(run.test, row)
            )
        })
        .collect();
    let hint: &str = if run.test.is_none() {
        "<p class=\"hint\">No test score yet: call <code>model.score(&amp;test, target)</code>.</p>"
    } else {
        ""
    };
    format!(
        "<table class=\"stats\"><thead><tr><th></th><th>Train</th><th>Test</th></tr></thead>\
         <tbody>{body}</tbody></table>{hint}"
    )
}

/// One run: the chart on the left third, the numbers on the right two thirds.
fn card(run: &Run) -> String {
    let fit: &FitRecord = run.fit;
    let total: String = fit
        .loss_curve
        .as_ref()
        .and_then(|c| c.last())
        .map_or("—".into(), |l| format!("{l:.6}"));
    let settings: Vec<(String, String)> = fit
        .params
        .as_object()
        .map(|p| p.iter().map(|(k, v)| (k.clone(), text(v))).collect())
        .unwrap_or_default();
    let metrics: String = stats(run);
    format!(
        "<section class=\"run\" id=\"{label}\"><div class=\"chart\">{chart}</div><div class=\"info\">\
         <header><h2><span class=\"label\">{label}</span> {model}</h2>\
         <p class=\"meta\">{time} UTC · target <b>{target}</b> · {rows} rows · {features} features</p></header>\
         <div class=\"total\"><span>Total loss</span><strong>{total}</strong>\
         <em>the training objective: the loss (½ · MSE, or log loss) plus the penalty</em></div>\
         <div class=\"cols\"><div><h3>Hyperparameters</h3><dl>{settings}</dl></div>\
         <div><h3>Train and test</h3>{metrics}\
         <p class=\"fit\">fit time {seconds:.3} s</p></div></div>\
         <p class=\"load\">Back to this model: <code>model.load_run(&amp;ds, \"{label}\")</code></p>\
         </div></section>",
        label = run.label,
        chart = chart(run),
        model = escape(&fit.model),
        time = utc(fit.time),
        target = escape(&fit.target),
        rows = fit.rows,
        features = fit.features.len(),
        settings = pairs(&settings),
        seconds = fit.fit_seconds,
    )
}

const STYLE: &str = "
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { background: #15171C; color: #F2F3F5; font: 14px/1.5 system-ui, sans-serif; margin: 0; padding: 24px; }
h1 { font-size: 22px; margin: 0 0 4px; }
.lede { color: #8B8F9A; margin: 0 0 24px; }
.run { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 2fr); gap: 24px; padding: 20px;
  margin-bottom: 20px; background: #1B1E24; border: 1px solid #2A2D35; border-radius: 8px; }
.chart svg { width: 100%; height: auto; display: block; border-radius: 6px; }
.empty { color: #8B8F9A; }
.info header h2 { font-size: 18px; margin: 0; }
.label { color: #199E70; font-family: ui-monospace, monospace; margin-right: 6px; }
.meta { color: #8B8F9A; margin: 2px 0 16px; }
.total { display: flex; align-items: baseline; gap: 12px; margin-bottom: 16px; }
.total span { color: #8B8F9A; text-transform: uppercase; font-size: 12px; letter-spacing: 0.06em; }
.total strong { font-size: 28px; font-variant-numeric: tabular-nums; }
.total em { color: #8B8F9A; font-style: normal; font-size: 12px; }
.cols { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 24px; }
h3 { font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: #8B8F9A; margin: 0 0 6px; }
dl { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 4px 16px; margin: 0; }
dt { color: #8B8F9A; }
dd { margin: 0; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.load { margin: 16px 0 0; color: #8B8F9A; }
.stats { border-collapse: collapse; font-variant-numeric: tabular-nums; width: 100%; }
.stats th, .stats td { text-align: left; padding: 3px 12px 3px 0; font-weight: normal; }
.stats thead th { color: #8B8F9A; font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; }
.stats tbody th { color: #8B8F9A; }
.fit, .hint { color: #8B8F9A; margin: 8px 0 0; }
code { font-family: ui-monospace, monospace; background: #22262D; padding: 2px 6px; border-radius: 4px; color: #F2F3F5; }
@media (max-width: 860px) { .run, .cols { grid-template-columns: minmax(0, 1fr); } body { padding: 16px; } }
";

/// The page for the runs of `csv`: one card per run, newest first.
fn page(csv: &Path, runs: &[Run]) -> String {
    let name: String = csv
        .file_name()
        .map_or(String::new(), |n| n.to_string_lossy().into_owned());
    // The best test score among the runs scored with the newest test score's metric.
    let tested: Vec<(&Run, (&str, f64))> = runs
        .iter()
        .filter_map(|r| r.test.map(|s| (r, main(s))))
        .collect();
    let metric: &str = tested.last().map_or("", |t| t.1.0);
    let best: String = tested
        .iter()
        .filter(|t| t.1.0 == metric)
        .max_by(|a, b| a.1.1.total_cmp(&b.1.1))
        .map_or(String::new(), |(r, (name, v))| {
            format!(
                " Best test {name}: <a href=\"#{0}\">{0}</a> ({v:.6}).",
                r.label
            )
        });
    let cards: String = runs.iter().rev().map(card).collect();
    format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\
         <title>Journal: {name}</title><style>{STYLE}</style></head><body>\
         <h1>Training journal: {name}</h1>\
         <p class=\"lede\">{} models trained on this dataset, newest first.{best} Hover a chart point for its values.</p>\
         {cards}</body></html>",
        runs.len(),
        name = escape(&name),
    )
}

impl Dataset {
    /// Opens the training journal of this dataset's CSV, `{csv dir}/metadata/{stem}.journal.html`:
    /// in a notebook it shows inline, in a program it opens in the browser. Every `fit` that
    /// trains and every `score` already rewrites the page, so this only opens it (after one more
    /// write, for a journal made by an older version).
    ///
    /// The page lists every model fitted on the CSV or on a split or transform of it, read from
    /// the model cache: settings, rows, fit time, the latest train and test stats (a score of
    /// the rows the model trained on is "train", any other is "test"), loss curves when the model
    /// trains in rounds, and each numeric setting against the score. It needs no internet.
    ///
    /// ```no_run
    /// use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv").select(&["age", "cgpa"]);
    /// let ds = SimpleImputer::new().fit_transform(&ds);
    /// let (train, test) = train_test_split(&ds, 0.25, 42, "");
    /// for alpha in [0.0, 0.1, 1.0] {
    ///     let mut model = LinearRegression::new().alpha(alpha);
    ///     model.fit(&train, "cgpa");
    ///     model.score(&test, "cgpa");
    /// }
    /// ds.journal();
    /// ```
    ///
    /// Python has no journal of its own; MLflow's tracking UI is the closest:
    ///
    /// ```python
    /// # mlflow.sklearn.autolog(); model.fit(X, y); then run `mlflow ui` and open the browser.
    /// ```
    #[track_caller]
    pub fn journal(&self) {
        let path: PathBuf = self.try_journal().or_fatal();
        if in_notebook() {
            let html: String = fs::read_to_string(&path).or_fatal();
            println!("EVCXR_BEGIN_CONTENT text/html\n{html}\nEVCXR_END_CONTENT");
        } else {
            open_in_viewer(&path);
        }
    }

    /// Writes the journal page and returns its path, without opening it. `fit` and `score` do
    /// this already; call it to rebuild the page by hand.
    ///
    /// # Errors
    ///
    /// - [`ModelError::NoCache`] when the dataset was made in memory, so it has no CSV.
    /// - [`ModelError::Io`] when the page can't be written.
    ///
    /// ```
    /// use polars::df;
    /// use rusmlai::{models::ModelError, prelude::*};
    /// let ds = Dataset::new(df!("x" => [1.0]).unwrap(), "");
    /// assert!(matches!(ds.try_journal(), Err(ModelError::NoCache)));
    /// ```
    pub fn try_journal(&self) -> Result<PathBuf, ModelError> {
        write(&ModelCache::of(self).ok_or(ModelError::NoCache)?)
    }
}

/// Writes the journal page of `cache` next to its CSV and returns its path. The model cache
/// calls it after every line it appends, so the page is always up to date.
// ponytail: rewrites the whole page each time, so a write costs O(runs); keep each run's card
// if a search with hundreds of fits shows the cost.
pub(crate) fn write(cache: &ModelCache) -> Result<PathBuf, ModelError> {
    let entries: Vec<Entry> = cache.entries();
    let path: PathBuf = cache_path(&cache.csv, JOURNAL);
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    fs::write(&path, page(&cache.csv, &runs(&entries)))?;
    print_saved(&path);
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dataset::read_csv,
        models::{
            Classifier, DummyRegressor, LinearRegression, LogisticRegression, Model, Regressor,
        },
    };

    #[test]
    fn utc_dates_match_known_days() {
        assert_eq!(utc(0), "1970-01-01 00:00");
        assert_eq!(utc(951_782_400), "2000-02-29 00:00");
        assert_eq!(utc(1_794_571_230), "2026-11-13 12:00"); // 30 s past
    }

    #[test]
    fn the_page_lists_every_fit_with_its_scores_and_settings() {
        let dir: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/tmp/journal");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let csv: PathBuf = dir.join("runs.csv");
        let rows: String = (0..20)
            .map(|i| {
                let y: f64 = 2.0 * f64::from(i) + f64::from(i % 3);
                format!(
                    "{i},{y},{}\n",
                    if (y + f64::from(i % 4)) > 20.0 {
                        "big"
                    } else {
                        "small"
                    }
                )
            })
            .collect();
        fs::write(&csv, format!("x,y,size\n{rows}")).unwrap();
        let ds: Dataset = read_csv(&csv);
        let (all_train, all_test) = crate::dataset::split::train_test_split(&ds, 0.25, 42, "");
        let (train, test) = (all_train.drop(&["size"]), all_test.drop(&["size"]));
        for alpha in [0.0, 0.5] {
            let mut model: LinearRegression = LinearRegression::new().alpha(alpha).l1_ratio(1.0);
            model.fit(&train, "y");
            model.score(&train, "y");
            model.score(&test, "y");
        }
        let mut dummy: DummyRegressor = DummyRegressor::new();
        dummy.fit(&train, "y");
        let mut classifier: LogisticRegression = LogisticRegression::new();
        classifier.fit(&all_train.drop(&["y"]), "size");
        classifier.score(&all_test.drop(&["y"]), "size");

        // Every fit and score rewrote the page already.
        let page: PathBuf = cache_path(&csv, JOURNAL);
        let html: String = fs::read_to_string(&page).unwrap();
        assert!(
            html.contains("<th>Test</th>"),
            "the last score is on the page"
        );
        assert_eq!(ds.try_journal().unwrap(), page);
        assert!(
            html.find("id=\"r04\"") < html.find("id=\"r01\""),
            "newest first"
        );
        assert!(html.contains("LogisticRegression") && html.contains("<th>log loss</th>"));
        assert!(
            html.contains("Best test accuracy"),
            "the newest test metric picks the best"
        );
        assert!(html.contains("DummyRegressor") && html.contains("<dt>alpha</dt>"));
        assert!(html.contains("<dt>strategy</dt><dd>mean</dd>"));
        assert!(
            html.contains("Loss per round"),
            "coordinate descent records its loss"
        );
        assert!(html.contains("model.load_run(&amp;ds, \"r02\")"));
        assert!(html.contains("<th>Test</th>") && html.contains("<th>RMSE</th>"));
        assert!(
            html.contains("class=\"pt\""),
            "line points are hover targets only"
        );
        assert!(
            !html.contains("src=\"http") && !html.contains("<link"),
            "the page works offline"
        );
    }
}
