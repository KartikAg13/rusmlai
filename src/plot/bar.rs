//! `bar`, `count`, `describe_bar` and `pie`: compute the numbers, then render them.

use polars::prelude::{DataFrame, DataType, Expr, col, len};

use crate::{
    dataset::Dataset,
    display::{Figure, escape},
    error::OrFatal,
    metadata::dataset::stats::{CategoricalStats, NumericStats},
    plot::{
        MAX_GROUPS, VALUE,
        chart::{Body, Texts},
        error::PlotError,
        f64s, labels,
        result::PlotResult,
        rows, slots,
        spec::{BarSpec, Estimator, PlotSpec},
        style::{Legend, Style},
        svg::{
            self, HEIGHT, Layout, MIN_CLUSTER_BAR, TOP, Tick, WIDTH, XScale, YScale,
            bottom_legend_h, finish, hlines, layout, legend_at, ordered, percent_factor, pick,
            range, tick_mode, unit, value_label,
        },
        unique,
    },
};

const PIE_TOP: usize = 8; // slices before "Other"

impl Dataset {
    /// One bar per `x` category: the mean of `y`, or another `estimator`, split by `hue`.
    /// (seaborn: `barplot`)
    ///
    /// `estimator = ""` means "mean". The others are median, sum, min, max, std and count.
    /// `hue = ""` draws one bar per category, and `y = ""` counts rows like [`Dataset::count`].
    /// Categories come in the metadata order (most rows first). A number column as `x` sorts
    /// ascending. The numbers are cached next to the CSV, so the next call reads them back.
    ///
    /// Options: `.title`, `.subtitle`, `.xlabel`, `.ylabel`, `.theme`, `.palette`, `.colors`,
    /// `.color`, `.color_map`, `.size`, `.order`, `.hue_order`, `.top`, `.values`, `.ylim`,
    /// `.sort`, `.horizontal`, `.hline` and `.legend`.
    ///
    /// # Panics
    ///
    /// If a column doesn't exist, `y` isn't numeric, the estimator is unknown, or no row has
    /// values. [`Dataset::try_bar`] returns these as errors.
    ///
    /// # Examples
    ///
    /// The mean CGPA of each batch:
    /// ```
    /// use rusmlai::prelude::*;
    ///
    /// let ds = read_csv("tests/data/students.csv");
    /// ds.bar("batch", "cgpa", "", ""); // saves metadata/plots/Bar batch vs cgpa.png
    /// ```
    ///
    /// The median, split by `passed`, with most options:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/data/students.csv");
    /// let svg = ds
    ///     .bar("batch", "cgpa", "passed", "median")
    ///     .title("Median CGPA by batch")
    ///     .subtitle("12 students")
    ///     .xlabel("batch")
    ///     .ylabel("CGPA")
    ///     .theme("light")
    ///     .palette("colorblind")
    ///     .color_map(&[("false", "tomato")])
    ///     .size(800.0, 500.0)
    ///     .order(&["B15", "B14", "B13"])
    ///     .hue_order(&["true", "false"])
    ///     .values(true)
    ///     .ylim(0.0, 10.0)
    ///     .hline(6.0, "pass mark")
    ///     .legend("bottom")
    ///     .svg(); // the SVG text, and no auto-save
    /// assert!(svg.contains("Median CGPA by batch"));
    /// assert!(svg.contains("#FF6347")); // tomato
    /// ```
    ///
    /// Edge case: std needs 2 values in a group, and every name belongs to one student:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/data/students.csv");
    /// ds.bar("name", "cgpa", "", "std"); // panics: no name group has 2 or more values of cgpa
    /// ```
    #[track_caller]
    pub fn bar(&self, x: &str, y: &str, hue: &str, estimator: &str) -> Figure {
        self.try_bar(x, y, hue, estimator).or_fatal()
    }

    /// [`Dataset::bar`] that returns an error instead of panicking.
    ///
    /// # Errors
    ///
    /// - `PlotError::Dataset` when a column doesn't exist or `y` isn't numeric.
    /// - `PlotError::UnknownEstimator` for an estimator name it doesn't know.
    /// - `PlotError::EmptyX` when `x` is `""`.
    /// - `PlotError::EmptyResult` when no row has values, and `PlotError::Undefined` when std
    ///   finds no group with 2 values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # use rusmlai::plot::error::PlotError;
    /// let ds = read_csv("tests/data/students.csv");
    /// let svg = ds.try_bar("batch", "cgpa", "", "sum")?.svg();
    /// assert!(svg.contains("sum cgpa"));
    /// # Ok::<(), PlotError>(())
    /// ```
    ///
    /// An estimator it doesn't know:
    /// ```
    /// # use rusmlai::prelude::*;
    /// use rusmlai::plot::error::PlotError;
    ///
    /// let ds = read_csv("tests/data/students.csv");
    /// let result = ds.try_bar("batch", "cgpa", "", "mode");
    /// assert!(matches!(result, Err(PlotError::UnknownEstimator(name)) if name == "mode"));
    /// ```
    pub fn try_bar(
        &self,
        x: &str,
        y: &str,
        hue: &str,
        estimator: &str,
    ) -> Result<Figure, PlotError> {
        let spec: BarSpec = BarSpec::new(x, y, hue, estimator)?;
        let nulls: usize = self.try_column_info(&spec.x)?.null_count;
        if let Some(y) = &spec.y {
            self.try_numeric_stats(y)?;
        }
        if let Some(hue) = &spec.hue {
            self.try_column_info(hue)?;
        }
        let x_stats: Option<&CategoricalStats> = self.try_categorical_stats(&spec.x).ok();

        let result: PlotResult = match (x_stats, &spec.y, &spec.hue) {
            // A plain count of a text column is already in the metadata: no group-by, no cache line.
            (Some(stats), None, None) => top_counts(stats),
            _ => self
                .metadata
                .plot(PlotSpec::Bar(spec.clone()), || compute(self, &spec))?,
        };
        if result.is_empty() && spec.estimator == Estimator::Std {
            return Err(PlotError::Undefined(format!(
                "no {} group has 2 or more values of {}, and std needs 2",
                spec.x,
                spec.y.as_deref().unwrap_or_default()
            )));
        }
        if result.is_empty() {
            let columns: Vec<&str> = [
                Some(spec.x.as_str()),
                spec.y.as_deref(),
                spec.hue.as_deref(),
            ]
            .into_iter()
            .flatten()
            .collect();
            return Err(PlotError::EmptyResult(columns.join(", ")));
        }
        let name: &'static str = if spec.y.is_none() { "count" } else { "bar" };
        Ok(self.figure(name, bar_texts(&spec, x_stats, nulls), Body::Bar(result)))
    }

    /// The number of rows in each `x` category, split by `hue`. (seaborn: `countplot`)
    ///
    /// A text column needs no pass over the data: its counts are in the cached stats. A number
    /// column (codes like 0 / 1, ages) gets one group-by and sorts ascending. Nulls are left out.
    /// `hue = x` draws plain bars, the same as `hue = ""`.
    ///
    /// Options: the ones of [`Dataset::bar`], plus `.percent` and `.stacked`.
    ///
    /// # Panics
    ///
    /// If a column doesn't exist. `count` has no `try_` twin: for a `Result`, call
    /// [`try_bar(x, "", hue, "")`](Dataset::try_bar).
    ///
    /// # Examples
    ///
    /// Students per batch:
    /// ```
    /// use rusmlai::prelude::*;
    ///
    /// let ds = read_csv("tests/data/students.csv");
    /// ds.count("batch", ""); // saves metadata/plots/Count batch vs count.png
    /// ```
    ///
    /// Pass and fail stacked in each batch, as percentages, with most options:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/data/students.csv");
    /// let svg = ds
    ///     .count("batch", "passed")
    ///     .stacked()
    ///     .percent()
    ///     .title("Who passed")
    ///     .colors(&["#566E3D", "tab:orange"])
    ///     .hue_order(&["true", "false"])
    ///     .sort("name")
    ///     .horizontal()
    ///     .values(true)
    ///     .hline(25.0, "a quarter")
    ///     .legend("bottom")
    ///     .svg(); // the SVG text, and no auto-save
    /// assert!(svg.contains("Who passed"));
    /// assert!(svg.contains("#566E3D") && svg.contains("#FF7F0E")); // tab:orange
    /// ```
    ///
    /// Edge case: a number column with a null. The null row is left out, and the ages sort
    /// ascending:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/data/students.csv");
    /// let svg = ds.count("age", "").svg();
    /// assert!(svg.contains("age = 19\ncount = 3")); // the hover text of the first bar
    /// ```
    #[track_caller]
    pub fn count(&self, x: &str, hue: &str) -> Figure {
        self.try_bar(x, "", hue, "").or_fatal()
    }

    /// Statistics of the numeric columns as bars, read from the cached stats with no pass over
    /// the data. (pandas: `df.describe()`)
    ///
    /// With a `column` it draws seven bars: min, q25, median, mean, q75, max and std. An
    /// `estimator` (mean, median, min, max, std, count, sum) draws only that one. With
    /// `column = ""` it draws one bar per numeric column, for `estimator` ("mean" when `""`).
    /// Text columns are skipped, and count is the number of values that aren't null.
    ///
    /// Options: the ones of [`Dataset::bar`].
    ///
    /// # Panics
    ///
    /// If the column doesn't exist or isn't numeric, or the estimator is unknown.
    /// [`Dataset::try_describe_bar`] returns these as errors.
    ///
    /// # Examples
    ///
    /// The seven statistics of CGPA:
    /// ```
    /// use rusmlai::prelude::*;
    ///
    /// let ds = read_csv("tests/data/students.csv");
    /// ds.describe_bar("cgpa", ""); // saves metadata/plots/Describe statistic vs cgpa.png
    /// ```
    ///
    /// The median of every numeric column, with most options:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/data/students.csv");
    /// let svg = ds
    ///     .describe_bar("", "median")
    ///     .title("Medians")
    ///     .subtitle("id, age and cgpa")
    ///     .xlabel("column")
    ///     .ylabel("median")
    ///     .theme("light")
    ///     .color("teal")
    ///     .size(700.0, 420.0)
    ///     .sort("value")
    ///     .top(2)
    ///     .values(false)
    ///     .hline(10.0, "")
    ///     .svg(); // the SVG text, and no auto-save
    /// assert!(svg.contains("Medians"));
    /// assert!(svg.contains("#008080")); // teal
    /// ```
    ///
    /// Edge case: with `column = ""`, text columns are skipped and count leaves out nulls:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/data/students.csv");
    /// let svg = ds.describe_bar("", "count").svg();
    /// assert!(svg.contains("column = age\ncount = 11")); // one age is null
    /// assert!(!svg.contains("column = batch"));
    /// ```
    #[track_caller]
    pub fn describe_bar(&self, column: &str, estimator: &str) -> Figure {
        self.try_describe_bar(column, estimator).or_fatal()
    }

    /// [`Dataset::describe_bar`] that returns an error instead of panicking.
    ///
    /// # Errors
    ///
    /// - `PlotError::Dataset` when the column doesn't exist or isn't numeric.
    /// - `PlotError::UnknownEstimator` for an estimator name it doesn't know.
    /// - `PlotError::EmptyResult` when the dataset has no numeric column.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # use rusmlai::plot::error::PlotError;
    /// let ds = read_csv("tests/data/students.csv");
    /// let svg = ds.try_describe_bar("cgpa", "max")?.svg();
    /// assert!(svg.contains("statistic = max\ncgpa = 9"));
    /// # Ok::<(), PlotError>(())
    /// ```
    ///
    /// A text column has no statistics to draw:
    /// ```
    /// # use rusmlai::prelude::*;
    /// use rusmlai::{dataset::error::DatasetError, plot::error::PlotError};
    ///
    /// let ds = read_csv("tests/data/students.csv");
    /// let result = ds.try_describe_bar("batch", "");
    /// assert!(matches!(
    ///     result,
    ///     Err(PlotError::Dataset(DatasetError::ColumnNotNumeric { .. }))
    /// ));
    /// ```
    pub fn try_describe_bar(&self, column: &str, estimator: &str) -> Result<Figure, PlotError> {
        let (texts, labels, values): (Texts, Vec<String>, Vec<f64>) = if column.is_empty() {
            let est: Estimator = Estimator::pick(true, estimator)?;
            let mut labels: Vec<String> = Vec::new();
            let mut values: Vec<f64> = Vec::new();
            for name in self.columns() {
                let Ok(stats) = self.try_numeric_stats(&name) else {
                    continue; // text column
                };
                values.push(stat(self, &name, stats, est)?);
                labels.push(name);
            }
            let texts: Texts = Texts {
                title: format!("Describe: column vs {}", est.as_str()),
                subtitle: format!("n={}", self.metadata.height),
                x: "column".to_string(),
                y: est.as_str().to_string(),
                value: est.as_str().to_string(),
                ..Texts::default()
            };
            (texts, labels, values)
        } else {
            let s: &NumericStats = self.try_numeric_stats(column)?;
            let (labels, values): (Vec<String>, Vec<f64>) = match estimator {
                "" => (
                    ["Min", "Q25", "Median", "Mean", "Q75", "Max", "Std"]
                        .map(String::from)
                        .to_vec(),
                    vec![
                        s.min,
                        s.quantile_25,
                        s.median,
                        s.mean,
                        s.quantile_75,
                        s.max,
                        s.std,
                    ],
                ),
                name => {
                    let est: Estimator = Estimator::parse(name)?;
                    (
                        vec![est.as_str().to_string()],
                        vec![stat(self, column, s, est)?],
                    )
                }
            };
            let nulls: usize = self.try_column_info(column)?.null_count;
            let texts: Texts = Texts {
                title: format!("Describe: statistic vs {column}"),
                subtitle: format!("n={},  nulls={nulls}", self.metadata.height - nulls),
                x: "statistic".to_string(),
                y: column.to_string(),
                value: column.to_string(),
                ..Texts::default()
            };
            (texts, labels, values)
        };
        let result: PlotResult = PlotResult::Series { labels, values };
        if result.is_empty() {
            return Err(PlotError::EmptyResult(if column.is_empty() {
                "the numeric columns".to_string()
            } else {
                column.to_string()
            }));
        }
        Ok(self.figure("describe_bar", texts, Body::Bar(result)))
    }

    /// The share of nulls in each column, in percent, from the cached null counts with no pass
    /// over the data. (pandas: `df.isnull().mean() * 100`, missingno: `bar`)
    ///
    /// The bars lie sideways, so long column names read down the left.
    ///
    /// Options: the ones of [`Dataset::bar`].
    ///
    /// # Examples
    ///
    /// Three of the six columns have one null each:
    /// ```
    /// use rusmlai::prelude::*;
    ///
    /// let ds = read_csv("tests/data/students.csv");
    /// ds.missing(); // saves metadata/plots/Missing values.png
    /// ```
    ///
    /// With most options:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/data/students.csv");
    /// let svg = ds
    ///     .missing()
    ///     .title("Nulls per column")
    ///     .subtitle("students.csv")
    ///     .xlabel("column")
    ///     .ylabel("null %")
    ///     .theme("light")
    ///     .color("tomato")
    ///     .size(700.0, 400.0)
    ///     .sort("value")
    ///     .top(3)
    ///     .values(true)
    ///     .ylim(0.0, 20.0)
    ///     .hline(5.0, "5%")
    ///     .svg(); // the SVG text, and no auto-save
    /// assert!(svg.contains("Nulls per column"));
    /// assert!(svg.contains("#FF6347")); // tomato
    /// ```
    ///
    /// Edge case: no nulls at all still draws a chart, with zero-length bars:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/data/students.csv");
    /// let svg = ds.select(&["id", "batch"]).missing().svg();
    /// assert!(svg.contains("no missing values"));
    /// ```
    #[track_caller]
    pub fn missing(&self) -> Figure {
        self.try_missing().or_fatal()
    }

    /// [`Dataset::missing`] that returns an error instead of panicking.
    ///
    /// # Errors
    ///
    /// `PlotError::Dataset` if the metadata lacks a column. Datasets from `read_csv` and
    /// `select` always have every column, so in practice this returns `Ok`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # use rusmlai::plot::error::PlotError;
    /// let ds = read_csv("tests/data/students.csv");
    /// let svg = ds.try_missing()?.svg();
    /// assert!(svg.contains("3 of 6 columns have nulls"));
    /// # Ok::<(), PlotError>(())
    /// ```
    pub fn try_missing(&self) -> Result<Figure, PlotError> {
        let rows: usize = self.metadata.height;
        let (labels, values): (Vec<String>, Vec<f64>) = self
            .columns()
            .into_iter()
            .map(|c| {
                let nulls: usize = self.try_column_info(&c)?.null_count;
                Ok((c, nulls as f64 / rows.max(1) as f64 * 100.0))
            })
            .collect::<Result<Vec<(String, f64)>, PlotError>>()?
            .into_iter()
            .unzip();
        let with_nulls: usize = values.iter().filter(|v| **v > 0.0).count();
        let texts: Texts = Texts {
            title: "Missing values".to_string(),
            subtitle: match with_nulls {
                0 => format!("no missing values,  n={rows}"),
                k => format!("{k} of {} columns have nulls,  n={rows}", labels.len()),
            },
            x: "column".to_string(),
            y: "missing (%)".to_string(),
            value: "missing %".to_string(),
            ..Texts::default()
        };
        // Column names read best down the left, like `df.isnull().mean().plot.barh()`.
        let mut figure: Figure = self.figure(
            "missing",
            texts,
            Body::Bar(PlotResult::Series { labels, values }),
        );
        figure.style.horizontal = true;
        Ok(figure)
    }

    /// The 8 biggest categories of `x` as slices, and one "Other" slice for the rest.
    /// (matplotlib: `plt.pie`)
    ///
    /// A text column uses the cached counts. A number column (codes like 0 / 1) is counted like
    /// [`Dataset::count`]. Nulls are left out. The legend shows each slice's count and share.
    ///
    /// Options: `.title`, `.subtitle`, `.theme`, `.size`, `.palette`, `.colors`, `.color_map`,
    /// `.order`, `.top`, `.values` and `.sort`. `.top(n)` sets the number of slices before
    /// "Other", and `.values(false)` hides the counts in the legend.
    ///
    /// # Panics
    ///
    /// If the column doesn't exist or has no values. [`Dataset::try_pie`] returns these as
    /// errors.
    ///
    /// # Examples
    ///
    /// The three batches, a third each:
    /// ```
    /// use rusmlai::prelude::*;
    ///
    /// let ds = read_csv("tests/data/students.csv");
    /// ds.pie("batch"); // saves metadata/plots/Distribution batch.png
    /// ```
    ///
    /// With most options. B14 is not in `.order`, so it goes into "Other":
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/data/students.csv");
    /// let svg = ds
    ///     .pie("batch")
    ///     .title("Batches")
    ///     .subtitle("12 students")
    ///     .theme("light")
    ///     .size(700.0, 450.0)
    ///     .palette("pastel")
    ///     .color_map(&[("B13", "gold")])
    ///     .order(&["B13", "B15"])
    ///     .values(false)
    ///     .svg(); // the SVG text, and no auto-save
    /// assert!(svg.contains("Batches") && svg.contains("Other"));
    /// assert!(svg.contains("#FFD700")); // gold
    /// ```
    ///
    /// Edge case: 11 names (and one null). The 8 most common names get a slice, and the other
    /// 3 share "Other":
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/data/students.csv");
    /// let svg = ds.pie("name").svg();
    /// assert!(svg.contains("Other  3  (27%)"));
    /// ```
    #[track_caller]
    pub fn pie(&self, x: &str) -> Figure {
        self.try_pie(x).or_fatal()
    }

    /// [`Dataset::pie`] that returns an error instead of panicking.
    ///
    /// # Errors
    ///
    /// - `PlotError::Dataset` when the column doesn't exist.
    /// - `PlotError::EmptyResult` when the column has no values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # use rusmlai::plot::error::PlotError;
    /// let ds = read_csv("tests/data/students.csv");
    /// let svg = ds.try_pie("passed")?.svg();
    /// assert!(svg.contains("true  8  (67%)"));
    /// # Ok::<(), PlotError>(())
    /// ```
    ///
    /// A column that doesn't exist:
    /// ```
    /// # use rusmlai::prelude::*;
    /// use rusmlai::{dataset::error::DatasetError, plot::error::PlotError};
    ///
    /// let ds = read_csv("tests/data/students.csv");
    /// let result = ds.try_pie("grade");
    /// assert!(matches!(
    ///     result,
    ///     Err(PlotError::Dataset(DatasetError::ColumnNotFound { .. }))
    /// ));
    /// ```
    pub fn try_pie(&self, x: &str) -> Result<Figure, PlotError> {
        self.try_column_info(x)?;
        let result: PlotResult = match self.try_categorical_stats(x) {
            Ok(stats) => top_counts(stats),
            Err(_) => {
                let spec: BarSpec = BarSpec::new(x, "", "", "")?;
                self.metadata
                    .plot(PlotSpec::Bar(spec.clone()), || compute(self, &spec))?
            }
        };
        if result.is_empty() {
            return Err(PlotError::EmptyResult(x.to_string()));
        }
        let texts: Texts = Texts {
            title: format!("Distribution: {x}"),
            x: x.to_string(),
            value: "count".to_string(),
            ..Texts::default()
        };
        Ok(self.figure("pie", texts, Body::Pie(result)))
    }
}

/// The old texts: `Bar: gender vs exam_score` / `Count: course vs count`, and for a text `x`
/// the subtitle `unique=7,  mode='b.tech' (14%),  entropy=1.94,  n=630000,  nulls=0,  hue=gender`.
fn bar_texts(spec: &BarSpec, x_stats: Option<&CategoricalStats>, nulls: usize) -> Texts {
    let (title, y, value): (String, String, String) = match &spec.y {
        Some(y) => (
            format!("Bar: {} vs {y}", spec.x),
            y.clone(),
            format!("{} {y}", spec.estimator.as_str()),
        ),
        None => (
            format!("Count: {} vs count", spec.x),
            "count".to_string(),
            "count".to_string(),
        ),
    };
    let hue: String = spec
        .hue
        .as_ref()
        .map(|h| format!("hue={h}"))
        .unwrap_or_default();
    let subtitle: String = match x_stats {
        None => hue,
        Some(stats) => {
            let n: u64 = stats.categories.iter().map(|(_, count)| count).sum();
            let mode_count: u64 = stats.categories.first().map_or(0, |&(_, count)| count);
            let facts: String = format!(
                "unique={},  mode='{}' ({:.0}%),  entropy={:.2},  n={n},  nulls={nulls}",
                stats.unique_count,
                svg::shorten(&stats.mode, 160.0),
                mode_count as f64 / n.max(1) as f64 * 100.0,
                entropy(&stats.categories),
            );
            if hue.is_empty() {
                facts
            } else {
                format!("{facts},  {hue}")
            }
        }
    };
    Texts {
        title,
        subtitle,
        x: spec.x.clone(),
        y,
        value,
        hue: spec.hue.clone().unwrap_or_default(),
    }
}

/// Shannon entropy of the category counts in nats, like `scipy.stats.entropy` of the value
/// counts: 0 for one category, ln(unique) when every category is equally common.
fn entropy(categories: &[(String, u64)]) -> f64 {
    let n: f64 = categories.iter().map(|&(_, count)| count as f64).sum();
    categories
        .iter()
        .map(|&(_, count)| count as f64 / n)
        .filter(|p| *p > 0.0)
        .fold(0.0, |h, p| h + p * (1.0 / p).ln()) // p ln(1/p) is +0 at p = 1, so never "-0.00"
}

fn stat(
    dataset: &Dataset,
    column: &str,
    s: &NumericStats,
    est: Estimator,
) -> Result<f64, PlotError> {
    let n: f64 = (dataset.metadata.height - dataset.try_column_info(column)?.null_count) as f64;
    Ok(match est {
        Estimator::Mean => s.mean,
        Estimator::Median => s.median,
        Estimator::Min => s.min,
        Estimator::Max => s.max,
        Estimator::Std => s.std,
        Estimator::Count => n,
        Estimator::Sum => s.mean * n,
    })
}

/// The cached category counts of a text column, at most `MAX_GROUPS`.
fn top_counts(stats: &CategoricalStats) -> PlotResult {
    let top: &[(String, u64)] = &stats.categories[..stats.categories.len().min(MAX_GROUPS)];
    PlotResult::Series {
        labels: top.iter().map(|(name, _)| name.clone()).collect(),
        values: top.iter().map(|&(_, n)| n as f64).collect(),
    }
}

/// One lazy group-by on x (and hue). A group with no rows stays NaN, never 0.
fn compute(dataset: &Dataset, spec: &BarSpec) -> Result<PlotResult, PlotError> {
    let keys: Vec<&str> = unique(
        [Some(spec.x.as_str()), spec.hue.as_deref()]
            .into_iter()
            .flatten(),
    );
    let mut columns: Vec<&str> = keys.clone();
    columns.extend(spec.y.as_deref());
    let value: Expr = match &spec.y {
        Some(y) => spec.estimator.expr(y),
        None => len(),
    };
    let grouped: DataFrame = rows(dataset, &columns)
        .group_by(keys.iter().map(|k| col(*k)).collect::<Vec<Expr>>())
        .agg([value.cast(DataType::Float64).alias(VALUE)])
        .collect()?;

    let x_labels: Vec<String> = labels(dataset, &grouped, &spec.x)?;
    let hue_labels: Vec<String> = match &spec.hue {
        Some(hue) => labels(dataset, &grouped, hue)?,
        None => Vec::new(),
    };
    let mut grid: Vec<Vec<f64>> = vec![vec![f64::NAN; hue_labels.len().max(1)]; x_labels.len()];
    let cells = slots(&grouped, Some(&spec.x), &x_labels)?
        .into_iter()
        .zip(slots(&grouped, spec.hue.as_deref(), &hue_labels)?);
    for ((i, j), value) in cells.zip(f64s(&grouped, VALUE)?) {
        if let (Some(i), Some(j)) = (i, j) {
            grid[i][j] = value;
        }
    }
    Ok(match spec.hue {
        Some(_) => PlotResult::Clustered {
            x_labels,
            hue_labels,
            grid,
        },
        None => PlotResult::Series {
            labels: x_labels,
            values: grid.into_iter().map(|row| row[0]).collect(),
        },
    })
}

/// The old `render_simple` and `render_clustered` in one: a single series is one group.
/// `per_category`: a single series gets one palette color per bar (not `describe_bar`).
/// `.horizontal()` lays the same bars down; `.stacked()` piles hue groups.
pub(crate) fn render(
    texts: &Texts,
    result: &PlotResult,
    per_category: bool,
    style: &Style,
) -> String {
    let theme = style.theme();
    let top: usize = style.top.unwrap_or(TOP);

    let (all, factor, x_labels, hue_labels, grid): (
        usize,
        f64,
        Vec<&String>,
        Vec<&String>,
        Vec<Vec<f64>>,
    ) = match result {
        PlotResult::Series { labels, values } => {
            let rows: Vec<usize> = pick(labels, style, values, top);
            (
                labels.len(),
                percent_factor(style, values.iter().copied()),
                rows.iter().map(|&i| &labels[i]).collect(),
                Vec::new(),
                rows.iter().map(|&i| vec![values[i]]).collect(),
            )
        }
        PlotResult::Clustered {
            x_labels,
            hue_labels,
            grid,
        } => {
            let totals: Vec<f64> = grid
                .iter()
                .map(|row| row.iter().filter(|v| v.is_finite()).sum())
                .collect();
            let rows: Vec<usize> = pick(x_labels, style, &totals, top);
            let cols: Vec<usize> = ordered(hue_labels, style.hue_order.as_deref());
            (
                x_labels.len(),
                percent_factor(style, grid.iter().flatten().copied()),
                rows.iter().map(|&i| &x_labels[i]).collect(),
                cols.iter().map(|&j| &hue_labels[j]).collect(),
                rows.iter()
                    .map(|&i| cols.iter().map(|&j| grid[i][j]).collect())
                    .collect(),
            )
        }
        _ => unreachable!("bar charts hold Series or Clustered results"),
    };
    let grid: Vec<Vec<f64>> = grid
        .into_iter()
        .map(|row| row.into_iter().map(|v| v * factor).collect())
        .collect();

    let title: String = texts.title(style);
    let base_subtitle: &str = texts.subtitle(style);
    // The cache keeps MAX_GROUPS categories, so a capped list has more than it holds.
    let of: String = if all >= MAX_GROUPS {
        format!("{all}+")
    } else {
        all.to_string()
    };
    let subtitle: String = match (x_labels.len() < all, base_subtitle.is_empty()) {
        (true, true) => format!("showing {} of {of}", x_labels.len()),
        (true, false) => format!("{base_subtitle},  showing {} of {of}", x_labels.len()),
        (false, _) => base_subtitle.to_string(),
    };

    let clustered: bool = !hue_labels.is_empty();
    let stacked: bool = clustered && style.stacked;
    let what: &str = if style.percent {
        "percent"
    } else {
        &texts.value
    };
    let fill_of = |i: usize, j: usize| -> String {
        if clustered {
            style.group_color(j, hue_labels[j])
        } else if per_category {
            style.category_color(i, x_labels[i])
        } else {
            style.single_color(x_labels[i])
        }
    };
    let hover = |i: usize, j: usize, v: f64| -> String {
        let mut text: String = format!("{} = {}", texts.x, x_labels[i]);
        if clustered {
            text += &format!("\n{} = {}", texts.hue, hue_labels[j]);
        }
        text + &format!("\n{what} = {}", unit(style, v))
    };
    let items: Vec<(String, String)> = if clustered {
        hue_labels
            .iter()
            .enumerate()
            .map(|(j, hue)| (hue.to_string(), style.group_color(j, hue)))
            .collect()
    } else {
        Vec::new()
    };
    let sums: Vec<f64> = grid
        .iter()
        .map(|row| row.iter().filter(|v| v.is_finite() && **v > 0.0).sum())
        .collect();
    let (lo, hi): (f64, f64) = if stacked {
        range(style, sums.iter().copied())
    } else {
        range(style, grid.iter().flatten().copied())
    };
    let show_values: bool = style.values.unwrap_or(!clustered);
    let (x_name, y_name): (&str, &str) = texts.names(style);
    let y_name: &str = if style.percent && style.ylabel.is_none() {
        "percent"
    } else {
        y_name
    };
    let n: usize = x_labels.len().max(1);
    let k: usize = if clustered && !stacked {
        hue_labels.len()
    } else {
        1
    };

    if !style.horizontal {
        let min_slot: f64 = if k > 1 {
            k as f64 * MIN_CLUSTER_BAR / 0.62
        } else {
            48.0
        }
        .max(svg::label_width(&x_labels));
        let mut l: Layout = layout(style, &subtitle, n, min_slot, &items);
        let scale: YScale = YScale::new(lo, hi, l.top, l.plot_h);
        let slot: f64 = l.plot_w / n as f64;
        let tick: Tick = tick_mode(&mut l, &x_labels, slot);
        let bar_w: f64 = if k > 1 {
            slot * 0.70 / k as f64
        } else {
            slot * 0.58
        };
        let mut out: String = svg::card(&theme, &title, &subtitle, l.width, l.height);
        scale.draw(&mut out, &theme, l.left, l.width - l.right);
        for (i, row) in grid.iter().enumerate() {
            let slot_x: f64 = l.left + i as f64 * slot;
            let start: f64 = slot_x + (slot - bar_w * k as f64) / 2.0;
            if stacked {
                let mut sum: f64 = 0.0;
                for (j, &v) in row.iter().enumerate() {
                    if v.is_finite() && v > 0.0 {
                        let (y0, y1): (f64, f64) = (scale.y(sum + v), scale.y(sum));
                        out.push_str(&svg::rect(
                            start,
                            y0,
                            bar_w,
                            y1 - y0,
                            &fill_of(i, j),
                            &hover(i, j, v),
                        ));
                        sum += v;
                    }
                }
                if show_values && sum > 0.0 {
                    let text: String = unit(style, sum);
                    value_label(
                        &mut out,
                        &theme,
                        start + bar_w / 2.0,
                        scale.y(sum) - 8.0,
                        &text,
                    );
                }
            } else {
                for (j, &v) in row.iter().enumerate() {
                    let x: f64 = start + j as f64 * bar_w;
                    let w: f64 = if clustered {
                        (bar_w - 1.5).max(1.0)
                    } else {
                        bar_w
                    };
                    out.push_str(&svg::bar(&scale, x, w, v, &fill_of(i, j), &hover(i, j, v)));
                    if show_values && v.is_finite() {
                        let y: f64 = if v >= 0.0 {
                            scale.y(v) - 8.0
                        } else {
                            scale.y(v) + 16.0
                        };
                        value_label(&mut out, &theme, x + w / 2.0, y, &unit(style, v));
                    }
                }
            }
            let cx: f64 = slot_x + slot / 2.0;
            svg::x_tick(
                &mut out,
                &theme,
                x_labels[i],
                cx,
                scale.bottom() + 22.0,
                slot,
                tick,
            );
        }
        hlines(&mut out, &theme, style, &scale, &l);
        finish(&mut out, &theme, style, (x_name, y_name), &items, &l);
        return out;
    }

    // Horizontal: categories down the left, values along the bottom.
    let (card_w, card_h): (f64, f64) = style.size.unwrap_or((WIDTH, HEIGHT));
    let at: Legend = legend_at(style, &items);
    let longest: f64 = x_labels
        .iter()
        .map(|label| svg::text_px(label))
        .fold(0.0, f64::max);
    let left: f64 = (longest.min(svg::MAX_LABEL) + 40.0).max(72.0);
    let right: f64 = if at == Legend::Right {
        svg::legend_width(&items)
    } else {
        36.0
    };
    let top_pad: f64 = if subtitle.is_empty() { 92.0 } else { 110.0 };
    let width: f64 = card_w
        .max(left + 200.0 + right)
        .max(svg::subtitle_width(&subtitle));
    let legend_h: f64 = bottom_legend_h(at, &items, width);
    let min_slot: f64 = if k > 1 { k as f64 * 12.0 / 0.7 } else { 28.0 };
    let plot_h: f64 = (card_h - top_pad - 70.0 - legend_h).max(n as f64 * min_slot);
    let l: Layout = Layout {
        left,
        right,
        top: top_pad,
        plot_w: width - left - right,
        plot_h,
        width,
        height: top_pad + plot_h + 70.0 + legend_h,
        legend_h,
    };
    let room: f64 = if show_values { 52.0 } else { 0.0 }; // value labels past the longest bar
    let scale: XScale = XScale::new(lo, hi, l.left, l.plot_w - room);
    let slot: f64 = l.plot_h / n as f64;
    let bar_h: f64 = if k > 1 {
        slot * 0.70 / k as f64
    } else {
        slot * 0.58
    };
    let bottom: f64 = l.top + l.plot_h;
    let mut out: String = svg::card(&theme, &title, &subtitle, l.width, l.height);
    scale.grid(&mut out, &theme, l.top, bottom);
    scale.draw(&mut out, &theme, bottom + 22.0);
    let end_label = |out: &mut String, x: f64, y: f64, anchor: &str, v: f64| {
        out.push_str(&format!(
            r##"<text x="{x:.2}" y="{:.2}" font-family="sans-serif" font-size="12" fill="{}" text-anchor="{anchor}">{}</text>"##,
            y + 4.0,
            theme.value,
            unit(style, v)
        ));
    };
    for (i, row) in grid.iter().enumerate() {
        let slot_y: f64 = l.top + i as f64 * slot;
        let start: f64 = slot_y + (slot - bar_h * k as f64) / 2.0;
        if stacked {
            let mut sum: f64 = 0.0;
            for (j, &v) in row.iter().enumerate() {
                if v.is_finite() && v > 0.0 {
                    let (x0, x1): (f64, f64) = (scale.x(sum), scale.x(sum + v));
                    out.push_str(&svg::rect(
                        x0,
                        start,
                        x1 - x0,
                        bar_h,
                        &fill_of(i, j),
                        &hover(i, j, v),
                    ));
                    sum += v;
                }
            }
            if show_values && sum > 0.0 {
                end_label(
                    &mut out,
                    scale.x(sum) + 6.0,
                    start + bar_h / 2.0,
                    "start",
                    sum,
                );
            }
        } else {
            for (j, &v) in row.iter().enumerate() {
                let y: f64 = start + j as f64 * bar_h;
                let h: f64 = if clustered {
                    (bar_h - 1.5).max(1.0)
                } else {
                    bar_h
                };
                out.push_str(&svg::hbar(&scale, y, h, v, &fill_of(i, j), &hover(i, j, v)));
                if show_values && v.is_finite() {
                    let (x, anchor): (f64, &str) = if v >= 0.0 {
                        (scale.x(v) + 6.0, "start")
                    } else {
                        (scale.x(v) - 6.0, "end")
                    };
                    end_label(&mut out, x, y + h / 2.0, anchor, v);
                }
            }
        }
        let cy: f64 = slot_y + slot / 2.0 + 4.0;
        svg::y_tick(
            &mut out,
            &theme,
            x_labels[i],
            l.left - 10.0,
            cy,
            l.left - 24.0,
        );
    }
    for (v, label) in &style.hlines {
        svg::ref_line(
            &mut out,
            &theme,
            false,
            scale.x(*v),
            l.top,
            bottom,
            label,
            *v,
        );
    }
    finish(&mut out, &theme, style, (y_name, x_name), &items, &l);
    out
}

/// The old pie: slices on the left, a legend with counts and shares on the right.
pub(crate) fn render_pie(texts: &Texts, result: &PlotResult, style: &Style) -> String {
    let PlotResult::Series { labels, values } = result else {
        unreachable!("pie charts hold Series results");
    };
    let theme = style.theme();
    let rows: Vec<usize> = pick(labels, style, values, style.top.unwrap_or(PIE_TOP));
    let mut slices: Vec<(&str, f64)> = rows
        .iter()
        .map(|&i| (labels[i].as_str(), values[i]))
        .collect();
    let other: f64 = (0..labels.len())
        .filter(|i| !rows.contains(i))
        .map(|i| values[i])
        .sum();
    if other > 0.0 {
        slices.push(("Other", other));
    }
    let total: f64 = slices.iter().map(|&(_, v)| v).sum::<f64>().max(1.0);

    let (width, height): (f64, f64) = style.size.unwrap_or((WIDTH, HEIGHT));
    let (sx, sy): (f64, f64) = (width / WIDTH, height / HEIGHT);
    let (cx, cy, r): (f64, f64, f64) = (250.0 * sx, 270.0 * sy, 130.0 * sx.min(sy));
    let title: String = texts.title(style);
    let subtitle: &str = texts.subtitle(style);
    let show_values: bool = style.values.unwrap_or(true);

    let mut out: String = svg::card(&theme, &title, subtitle, width, height);
    let mut angle: f64 = -std::f64::consts::FRAC_PI_2;
    for (i, &(name, value)) in slices.iter().enumerate() {
        let share: f64 = value / total;
        let next: f64 = angle + share * std::f64::consts::TAU;
        let color: String = style.group_color(i, name);
        let hover: String = escape(&format!(
            "{} = {name}\n{} = {} ({:.0}%)",
            texts.x,
            texts.value,
            svg::format_value(value),
            share * 100.0
        ));
        let bg: &str = theme.background;
        if share >= 0.9999 {
            // One slice: an arc from a point back to itself draws nothing, so draw the circle.
            out.push_str(&format!(
                r##"<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}"><title>{hover}</title></circle>"##
            ));
        } else {
            let large: u8 = u8::from(share > 0.5);
            let (x1, y1): (f64, f64) = (cx + r * angle.cos(), cy + r * angle.sin());
            let (x2, y2): (f64, f64) = (cx + r * next.cos(), cy + r * next.sin());
            out.push_str(&format!(
                r##"<path d="M {cx} {cy} L {x1:.2} {y1:.2} A {r} {r} 0 {large} 1 {x2:.2} {y2:.2} Z" fill="{color}" stroke="{bg}" stroke-width="1.5"><title>{hover}</title></path>"##
            ));
        }
        let row: f64 = 140.0 * sy + i as f64 * 28.0;
        let text: String = if show_values {
            format!(
                "{name}  {}  ({:.0}%)",
                svg::format_value(value),
                share * 100.0
            )
        } else {
            name.to_string()
        };
        out.push_str(&format!(
            r##"<rect x="{}" y="{}" width="12" height="12" rx="2" fill="{color}"/><text x="{}" y="{row}" font-family="sans-serif" font-size="13" fill="{}">{}</text>"##,
            430.0 * sx,
            row - 10.0,
            452.0 * sx,
            theme.tick,
            escape(&text)
        ));
        angle = next;
    }
    out.push_str("</svg>");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_subtitle_reads_the_cached_counts() {
        let stats: CategoricalStats = CategoricalStats {
            unique_count: 2,
            categories: vec![("a".to_string(), 3), ("b".to_string(), 1)],
            mode: "a".to_string(),
        };
        let spec: BarSpec = BarSpec::new("c", "", "", "").unwrap();
        // scipy.stats.entropy([3, 1]) = 0.5623
        assert_eq!(
            bar_texts(&spec, Some(&stats), 2).subtitle,
            "unique=2,  mode='a' (75%),  entropy=0.56,  n=4,  nulls=2"
        );
        assert_eq!(format!("{:.2}", entropy(&[("a".to_string(), 5)])), "0.00");
    }
}
