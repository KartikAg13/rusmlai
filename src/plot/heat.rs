//! `heatmap` and `corrplot`: a grid of colored cells.

use polars::prelude::{DataFrame, DataType, Expr, IntoLazy, SortMultipleOptions, len, lit};

use crate::utils::col;
use crate::{
    dataset::Dataset,
    display::{Figure, escape},
    error::OrFatal,
    metadata::dataset::stats::NumericStats,
    plot::{
        VALUE,
        chart::{Body, Texts},
        error::PlotError,
        f64s, labels, optional,
        result::PlotResult,
        rows, slots,
        spec::{Estimator, PlotSpec},
        style::Style,
        svg::{self, HEIGHT, Tick, WIDTH, extent},
        unique,
    },
};

impl Dataset {
    /// A grid of colored cells: the mean of `value`, or another `estimator`, for each pair of
    /// `x` and `y` categories. (seaborn: `heatmap` of a pivot table)
    ///
    /// `value = ""` counts rows, and then `estimator` must be `""` or "count". With a `value`,
    /// `estimator = ""` means "mean"; the others are median, sum, min, max, std and count. A pair
    /// with no rows stays empty. A color bar shows the scale. The cells are cached next to the
    /// CSV.
    ///
    /// Options: `.title`, `.subtitle`, `.xlabel`, `.ylabel`, `.theme`, `.size`, `.cmap` and
    /// `.values`.
    ///
    /// # Panics
    ///
    /// If a column doesn't exist, `value` isn't numeric, the estimator is unknown or needs a
    /// `value`, or no row has values. [`Dataset::try_heatmap`] returns these as errors.
    ///
    /// # Examples
    ///
    /// Students in each batch who passed and failed:
    /// ```
    /// use rusmlai::prelude::*;
    ///
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.heatmap("batch", "passed", "", ""); // saves metadata/plots/Heatmap batch vs passed.png
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import matplotlib.pyplot as plt
    /// import pandas as pd
    /// import seaborn as sns
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// sns.heatmap(pd.crosstab(ds["passed"], ds["batch"]), annot=True)  # rows per cell
    /// plt.show()
    /// ```
    ///
    /// The mean CGPA of each cell, with every option:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds
    ///     .heatmap("batch", "passed", "cgpa", "mean")
    ///     .title("Mean CGPA")
    ///     .subtitle("by batch and result")
    ///     .xlabel("batch")
    ///     .ylabel("passed")
    ///     .theme("light")
    ///     .size(700.0, 400.0)
    ///     .cmap("viridis")
    ///     .values(true)
    ///     .svg(); // the SVG text, and no auto-save
    /// assert!(svg.contains("Mean CGPA"));
    /// assert!(svg.contains("batch = B13\npassed = true\nmean cgpa = 8.33"));
    /// ```
    ///
    /// Edge case: a mean needs a column of values:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.heatmap("batch", "passed", "", "mean"); // panics: estimator needs a column of values
    /// ```
    #[track_caller]
    pub fn heatmap(&self, x: &str, y: &str, value: &str, estimator: &str) -> Figure {
        self.try_heatmap(x, y, value, estimator).or_fatal()
    }

    /// [`Dataset::heatmap`] that returns an error instead of panicking.
    ///
    /// # Errors
    ///
    /// - `PlotError::Dataset` when a column doesn't exist or `value` isn't numeric.
    /// - `PlotError::UnknownEstimator` for an estimator name it doesn't know.
    /// - `PlotError::EstimatorNeedsY` for an estimator other than count without a `value`.
    /// - `PlotError::EmptyResult` when no row has values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # use rusmlai::plot::error::PlotError;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.try_heatmap("batch", "passed", "", "")?.svg();
    /// assert!(svg.contains("batch = B13\npassed = true\ncount = 3"));
    /// # Ok::<(), PlotError>(())
    /// ```
    ///
    /// An estimator it doesn't know:
    /// ```
    /// # use rusmlai::prelude::*;
    /// use rusmlai::plot::error::PlotError;
    ///
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let result = ds.try_heatmap("batch", "passed", "cgpa", "mode");
    /// assert!(matches!(result, Err(PlotError::UnknownEstimator(_))));
    /// ```
    pub fn try_heatmap(
        &self,
        x: &str,
        y: &str,
        value: &str,
        estimator: &str,
    ) -> Result<Figure, PlotError> {
        self.try_column_info(x)?;
        self.try_column_info(y)?;
        let value: Option<String> = optional(value);
        let estimator: Estimator = Estimator::pick(value.is_some(), estimator)?;
        if let Some(value) = &value {
            self.try_numeric_stats(value)?;
        }
        let spec: PlotSpec = PlotSpec::Heatmap {
            x: x.to_string(),
            y: y.to_string(),
            value: value.clone(),
            estimator,
        };
        let result: PlotResult = self.metadata.plot(spec, || {
            compute_heatmap(self, x, y, value.as_deref(), estimator)
        })?;
        if result.is_empty() {
            return Err(PlotError::EmptyResult(format!("{x}, {y}")));
        }
        let (title, what): (String, String) = match &value {
            Some(v) => (
                format!("Heatmap: {x} × {y} ({} {v})", estimator.as_str()),
                format!("{} {v}", estimator.as_str()),
            ),
            None => (format!("Heatmap: {x} vs {y}"), "count".to_string()),
        };
        let texts: Texts = Texts {
            title,
            x: x.to_string(),
            y: y.to_string(),
            value: what,
            ..Texts::default()
        };
        Ok(self.figure(
            "heatmap",
            texts,
            Body::Matrix {
                result,
                fixed: None,
            },
        ))
    }

    /// Where the nulls are: one band per column, rows top to bottom in file order, with gaps
    /// where values are missing. The rows are grouped into 250 bands. (missingno: `matrix`)
    ///
    /// One group-by counts the nulls of every column in every band, and the result is cached
    /// next to the CSV. Hover a column for its null count.
    ///
    /// Options: `.title`, `.subtitle`, `.theme`, `.size`, `.palette`, `.colors` and `.color`.
    /// The color is the color of the values.
    ///
    /// # Examples
    ///
    /// Where the 3 nulls are:
    /// ```
    /// use rusmlai::prelude::*;
    ///
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.missing_matrix(); // saves metadata/plots/Missing values by row.png
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import matplotlib.pyplot as plt
    /// import pandas as pd
    /// import seaborn as sns
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// sns.heatmap(ds.isna(), cbar=False)  # or missingno.matrix(ds)
    /// plt.show()
    /// ```
    ///
    /// With every option:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds
    ///     .missing_matrix()
    ///     .title("Nulls by row")
    ///     .subtitle("students.csv")
    ///     .theme("light")
    ///     .size(700.0, 500.0)
    ///     .color("steelblue")
    ///     .svg(); // the SVG text, and no auto-save
    /// assert!(svg.contains("Nulls by row"));
    /// assert!(svg.contains("#4682B4")); // steelblue
    /// ```
    ///
    /// Edge case: columns with no nulls draw full bands:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.select(&["id", "batch"]).missing_matrix().svg();
    /// assert!(svg.contains("no missing values"));
    /// ```
    #[track_caller]
    pub fn missing_matrix(&self) -> Figure {
        self.try_missing_matrix().or_fatal()
    }

    /// [`Dataset::missing_matrix`] that returns an error instead of panicking.
    ///
    /// # Errors
    ///
    /// `PlotError::EmptyResult` for a dataset with no columns. `read_csv` and `select` never
    /// make one, so in practice this returns `Ok`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # use rusmlai::plot::error::PlotError;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.try_missing_matrix()?.svg();
    /// assert!(svg.contains("3 nulls in 3 of 6 columns"));
    /// # Ok::<(), PlotError>(())
    /// ```
    pub fn try_missing_matrix(&self) -> Result<Figure, PlotError> {
        let columns: Vec<String> = self.columns();
        if columns.is_empty() {
            return Err(PlotError::EmptyResult("the dataset".to_string()));
        }
        let spec: PlotSpec = PlotSpec::Missing {
            columns: columns.clone(),
        };
        let result: PlotResult = self
            .metadata
            .plot(spec, || compute_missing(self, &columns))?;
        let nulls: Vec<usize> = columns
            .iter()
            .map(|c| Ok(self.try_column_info(c)?.null_count))
            .collect::<Result<_, PlotError>>()?;
        let with_nulls: usize = nulls.iter().filter(|n| **n > 0).count();
        let texts: Texts = Texts {
            title: "Missing values by row".to_string(),
            subtitle: match with_nulls {
                0 => "no missing values".to_string(),
                k => format!(
                    "{} nulls in {k} of {} columns",
                    nulls.iter().sum::<usize>(),
                    columns.len()
                ),
            },
            y: "row".to_string(),
            ..Texts::default()
        };
        let body: Body = Body::Missing {
            result,
            nulls,
            rows: self.metadata.height,
        };
        Ok(self.figure("missing_matrix", texts, body))
    }

    /// The Pearson correlation of every pair of numeric columns. (seaborn: `heatmap(df.corr())`)
    ///
    /// Like pandas, each pair uses the rows where both columns have values. The color range is
    /// fixed at -1 to 1. Text columns are skipped. To pick columns, call it on a
    /// [`Dataset::select`]. The matrix is cached next to the CSV.
    ///
    /// Options: `.title`, `.subtitle`, `.xlabel`, `.ylabel`, `.theme`, `.size`, `.cmap` and
    /// `.values`.
    ///
    /// # Panics
    ///
    /// If the dataset has no numeric column, or no pair has a correlation.
    /// [`Dataset::try_corrplot`] returns these as errors.
    ///
    /// # Examples
    ///
    /// Every numeric column against every other:
    /// ```
    /// use rusmlai::prelude::*;
    ///
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.corrplot(); // saves metadata/plots/Corrplot Correlation Matrix.png
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import matplotlib.pyplot as plt
    /// import pandas as pd
    /// import seaborn as sns
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")
    /// sns.heatmap(ds.corr(numeric_only=True), annot=True, vmin=-1, vmax=1, cmap="RdBu_r")
    /// plt.show()
    /// ```
    ///
    /// Age and CGPA only, with every option:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds
    ///     .select(&["age", "cgpa"])
    ///     .corrplot()
    ///     .title("Age vs CGPA")
    ///     .subtitle("Pearson")
    ///     .xlabel("column")
    ///     .ylabel("column")
    ///     .theme("light")
    ///     .size(500.0, 420.0)
    ///     .cmap("coolwarm")
    ///     .values(true)
    ///     .svg(); // the SVG text, and no auto-save
    /// assert!(svg.contains("Age vs CGPA"));
    /// assert!(svg.contains("age × age\ncorrelation = 1"));
    /// ```
    ///
    /// Edge case: text columns have no correlation:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.select(&["name", "batch"]).corrplot(); // panics: nothing to plot
    /// ```
    #[track_caller]
    pub fn corrplot(&self) -> Figure {
        self.try_corrplot().or_fatal()
    }

    /// [`Dataset::corrplot`] that returns an error instead of panicking.
    ///
    /// # Errors
    ///
    /// - `PlotError::EmptyResult` when the dataset has no numeric column.
    /// - `PlotError::Undefined` when no pair has 2 or more rows with different values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # use rusmlai::plot::error::PlotError;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.try_corrplot()?.svg();
    /// assert!(svg.contains("cgpa × cgpa\ncorrelation = 1"));
    /// # Ok::<(), PlotError>(())
    /// ```
    ///
    /// No numeric column:
    /// ```
    /// # use rusmlai::prelude::*;
    /// use rusmlai::plot::error::PlotError;
    ///
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let result = ds.select(&["name", "batch"]).try_corrplot();
    /// assert!(matches!(result, Err(PlotError::EmptyResult(_))));
    /// ```
    pub fn try_corrplot(&self) -> Result<Figure, PlotError> {
        let columns: Vec<String> = self
            .columns()
            .into_iter()
            .filter(|name| self.try_numeric_stats(name).is_ok())
            .collect();
        if columns.is_empty() {
            return Err(PlotError::EmptyResult("the numeric columns".to_string()));
        }
        let spec: PlotSpec = PlotSpec::Corrplot {
            columns: columns.clone(),
        };
        let result: PlotResult = self.metadata.plot(spec, || compute_corr(self, &columns))?;
        if result.is_empty() {
            return Err(PlotError::Undefined(
                "no two numeric columns have a correlation: each needs 2 or more rows with different values"
                    .to_string(),
            ));
        }
        let texts: Texts = Texts {
            title: "Corrplot: Correlation Matrix".to_string(),
            value: "correlation".to_string(),
            ..Texts::default()
        };
        let fixed: Option<(f64, f64)> = Some((-1.0, 1.0));
        Ok(self.figure("corrplot", texts, Body::Matrix { result, fixed }))
    }
}

fn compute_heatmap(
    dataset: &Dataset,
    x: &str,
    y: &str,
    value: Option<&str>,
    estimator: Estimator,
) -> Result<PlotResult, PlotError> {
    let columns: Vec<&str> = [Some(x), Some(y), value].into_iter().flatten().collect();
    let keys: Vec<Expr> = unique([x, y]).into_iter().map(col).collect();
    let agg: Expr = match value {
        Some(value) => estimator.expr(value),
        None => len(),
    };
    let grouped: DataFrame = rows(dataset, &columns)
        .group_by(keys)
        .agg([agg.cast(DataType::Float64).alias(VALUE)])
        .collect()?;
    let (x_labels, y_labels): (Vec<String>, Vec<String>) =
        (labels(dataset, &grouped, x)?, labels(dataset, &grouped, y)?);
    let mut grid: Vec<Vec<f64>> = vec![vec![f64::NAN; x_labels.len()]; y_labels.len()];
    let cells =
        slots(&grouped, Some(x), &x_labels)?
            .into_iter()
            .zip(slots(&grouped, Some(y), &y_labels)?);
    for ((i, j), value) in cells.zip(f64s(&grouped, VALUE)?) {
        if let (Some(i), Some(j)) = (i, j) {
            grid[j][i] = value;
        }
    }
    Ok(PlotResult::Matrix {
        x_labels,
        y_labels,
        grid,
    })
}

const BANDS: usize = 250; // row bands of missing_matrix, like missingno's usual sample
const ROW: &str = "__rusmlai_row";
const BAND: &str = "__rusmlai_band";

/// The share of nulls in each band of rows, per column: `grid[band][column]`.
fn compute_missing(dataset: &Dataset, columns: &[String]) -> Result<PlotResult, PlotError> {
    let height: usize = dataset.metadata.height;
    let size: usize = height.div_ceil(BANDS).max(1);
    let band: Expr = (col(ROW).cast(DataType::Float64) / lit(size as f64)).cast(DataType::Int64);
    let grouped: DataFrame = dataset
        .dataframe
        .clone()
        .lazy()
        .with_row_index(ROW, None)
        .group_by([band.alias(BAND)])
        .agg(
            columns
                .iter()
                .map(|c| col(c.as_str()).null_count().cast(DataType::Float64))
                .collect::<Vec<Expr>>(),
        )
        .sort([BAND], SortMultipleOptions::default())
        .collect()?;
    let bands: Vec<usize> = f64s(&grouped, BAND)?
        .into_iter()
        .map(|b| b as usize)
        .collect();
    let rows_in = |b: usize| (height - b * size).min(size) as f64;
    let mut grid: Vec<Vec<f64>> = vec![Vec::with_capacity(columns.len()); bands.len()];
    for c in columns {
        for (k, nulls) in f64s(&grouped, c)?.into_iter().enumerate() {
            grid[k].push(nulls / rows_in(bands[k]));
        }
    }
    let y_labels: Vec<String> = bands
        .iter()
        .map(|&b| format!("rows {} to {}", b * size + 1, (b * size + size).min(height)))
        .collect();
    Ok(PlotResult::Matrix {
        x_labels: columns.to_vec(),
        y_labels,
        grid,
    })
}

/// Pearson correlation of each pair on the rows where both values are present, like pandas.
/// One Polars select for all pairs.
fn compute_corr(dataset: &Dataset, columns: &[String]) -> Result<PlotResult, PlotError> {
    let n: usize = columns.len();
    // The cached (mean, variance) of a column with a finite value in every row. A pair of two
    // such columns uses every row, so it needs one pass and no filter: Σ(da·db) / ((n - 1)·sa·sb).
    let whole: Vec<Option<(f64, f64)>> = columns
        .iter()
        .map(|c| {
            let stats: &NumericStats = dataset.try_numeric_stats(c).ok()?;
            let finite: bool = [stats.mean, stats.variance, stats.min, stats.max]
                .iter()
                .all(|v| v.is_finite());
            (finite && dataset.try_column_info(c).ok()?.null_count == 0)
                .then_some((stats.mean, stats.variance))
        })
        .collect();
    let freedom: f64 = dataset.metadata.height as f64 - 1.0;
    let pairs: Vec<(usize, usize)> = (0..n).flat_map(|i| (i..n).map(move |j| (i, j))).collect();
    let rs: DataFrame = dataset
        .dataframe
        .clone()
        .lazy()
        .select(
            pairs
                .iter()
                .map(|&(i, j)| {
                    let (a, b): (&str, &str) = (&columns[i], &columns[j]);
                    let r: Expr = match (whole[i], whole[j]) {
                        (Some((mean_a, var_a)), Some((mean_b, var_b))) => {
                            let d = |c: &str, mean: f64| col(c).cast(DataType::Float64) - lit(mean);
                            (d(a, mean_a) * d(b, mean_b)).sum()
                                / lit(freedom * var_a.sqrt() * var_b.sqrt())
                        }
                        _ => pearson(a, b),
                    };
                    r.alias(format!("{i},{j}"))
                })
                .collect::<Vec<Expr>>(),
        )
        .collect()?;
    let mut grid: Vec<Vec<f64>> = vec![vec![f64::NAN; n]; n];
    for (i, j) in pairs {
        // Rounding can leave 1.0000000000000002, which is past the color scale.
        let r: f64 = f64s(&rs, &format!("{i},{j}"))?[0].clamp(-1.0, 1.0);
        grid[i][j] = r;
        grid[j][i] = r;
    }
    Ok(PlotResult::Matrix {
        x_labels: columns.to_vec(),
        y_labels: columns.to_vec(),
        grid,
    })
}

/// Σ(da·db) / √(Σda² · Σdb²) over the rows where `a` and `b` are both present, two passes. A
/// constant column gives 0 / 0 = NaN (no correlation), not 0.
fn pearson(a: &str, b: &str) -> Expr {
    let finite = |c: &str| col(c).cast(DataType::Float64).is_finite(); // null and NaN are false
    let both: Expr = finite(a).and(finite(b));
    let centered = |c: &str| {
        let v: Expr = col(c).cast(DataType::Float64).filter(both.clone());
        v.clone() - v.mean()
    };
    let (da, db): (Expr, Expr) = (centered(a), centered(b));
    (da.clone() * db.clone()).sum() / ((da.clone() * da).sum() * (db.clone() * db).sum()).sqrt()
}

/// The old matrix: y labels on the left, x labels on top, rounded cells with numbers.
/// `fixed` pins the color range (corrplot uses -1 to 1).
pub(crate) fn render_matrix(
    texts: &Texts,
    result: &PlotResult,
    fixed: Option<(f64, f64)>,
    style: &Style,
) -> String {
    let PlotResult::Matrix {
        x_labels,
        y_labels,
        grid,
    } = result
    else {
        unreachable!("heatmaps hold Matrix results");
    };
    let theme = style.theme();
    let (card_w, card_h): (f64, f64) = style.size.unwrap_or((WIDTH, HEIGHT));
    // Row names fit the left margin; column names tilt up when wider than a cell, and the top
    // and right margins grow for them.
    let longest: f64 = y_labels
        .iter()
        .map(|label| svg::text_px(label))
        .fold(0.0, f64::max);
    let (left, bottom): (f64, f64) = ((longest.min(svg::MAX_LABEL) + 40.0).max(110.0), 70.0);
    let (nx, ny): (usize, usize) = (x_labels.len().max(1), y_labels.len().max(1));
    let cell = |right: f64| ((card_w - left - right) / nx as f64).max(28.0);
    let tick: Tick = if svg::tilts(x_labels, cell(100.0)) {
        Tick::Up
    } else {
        Tick::Flat
    };
    let room: f64 = if tick == Tick::Up {
        svg::tilt_room(x_labels)
    } else {
        0.0
    };
    let (top, right): (f64, f64) = (100.0f64.max(92.0 + room), 100.0f64.max(room + 8.0));
    let cell_w: f64 = cell(right);
    let cell_h: f64 = ((card_h - top - bottom) / ny as f64).max(28.0);
    let width: f64 = (left + cell_w * nx as f64 + right).max(card_w);
    let height: f64 = (top + cell_h * ny as f64 + bottom).max(card_h);

    let (vmin, vmax): (f64, f64) = fixed.unwrap_or_else(|| extent(grid.iter().flatten().copied()));
    let diverging: bool = style.cmap.map_or(vmin < 0.0, |c| c.diverging);
    let stops: Vec<&str> = match style.cmap {
        Some(cmap) => cmap.stops.to_vec(),
        None if diverging => vec![theme.negative, theme.cell_mid, theme.bar],
        None => vec![theme.cell_low, theme.bar],
    };
    let reach: f64 = vmin.abs().max(vmax.abs()).max(1e-12);
    let span: f64 = (vmax - vmin).max(1e-12);
    let color = |v: f64| {
        let t: f64 = if diverging {
            0.5 + v / (2.0 * reach) // 0 sits in the middle stop
        } else {
            (v - vmin) / span
        };
        svg::mix(&stops, t)
    };
    let show_values: bool = style.values.unwrap_or(true);
    let subtitle: &str = texts.subtitle(style);

    let mut out: String = svg::card(&theme, &texts.title(style), subtitle, width, height);
    for (c, x_label) in x_labels.iter().enumerate() {
        svg::x_tick(
            &mut out,
            &theme,
            x_label,
            left + (c as f64 + 0.5) * cell_w,
            top - 12.0,
            cell_w,
            tick,
        );
    }
    for (r, y_label) in y_labels.iter().enumerate() {
        let y: f64 = top + (r as f64 + 0.55) * cell_h;
        svg::y_tick(&mut out, &theme, y_label, left - 10.0, y, left - 16.0);
        for (c, x_label) in x_labels.iter().enumerate() {
            let v: f64 = grid[r][c];
            if !v.is_finite() {
                continue; // a missing pair stays empty
            }
            let fill: String = color(v);
            let (x, y): (f64, f64) = (
                left + c as f64 * cell_w + 2.0,
                top + r as f64 * cell_h + 2.0,
            );
            let (w, h): (f64, f64) = ((cell_w - 4.0).max(4.0), (cell_h - 4.0).max(4.0));
            let hover: String = if texts.x.is_empty() {
                format!(
                    "{x_label} × {y_label}\n{} = {}",
                    texts.value,
                    svg::format_value(v)
                )
            } else {
                format!(
                    "{} = {x_label}\n{} = {y_label}\n{} = {}",
                    texts.x,
                    texts.y,
                    texts.value,
                    svg::format_value(v)
                )
            };
            out.push_str(&format!(
                r##"<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}"><title>{}</title></rect>"##,
                escape(&hover)
            ));
            if show_values {
                out.push_str(&format!(
                    r##"<text x="{}" y="{}" font-family="sans-serif" font-size="11" fill="{}" text-anchor="middle">{}</text>"##,
                    x + w / 2.0,
                    y + h / 2.0 + 4.0,
                    svg::text_on(&fill),
                    svg::format_value(v)
                ));
            }
        }
    }
    let (lo, hi): (f64, f64) = if diverging {
        (-reach, reach)
    } else {
        (vmin, vmax)
    };
    svg::colorbar(
        &mut out,
        &theme,
        &stops,
        lo,
        hi,
        left + cell_w * nx as f64 + 24.0,
        top,
        (cell_h * ny as f64).max(120.0),
    );
    let (x_name, y_name): (&str, &str) = texts.names(style);
    if !x_name.is_empty() || !y_name.is_empty() {
        svg::axis_names(
            &mut out,
            &theme,
            x_name,
            y_name,
            left + cell_w * nx as f64 / 2.0,
            top + cell_h * ny as f64 / 2.0,
            height,
        );
    }
    out.push_str("</svg>");
    out
}

/// One band per column in the palette's first color, with a gap in the card color where a band
/// of rows has nulls (darker for more nulls). Column names on top, tilted when they don't fit.
pub(crate) fn render_missing(
    texts: &Texts,
    result: &PlotResult,
    nulls: &[usize],
    rows: usize,
    style: &Style,
) -> String {
    let PlotResult::Matrix {
        x_labels,
        y_labels,
        grid,
    } = result
    else {
        unreachable!("missing_matrix holds a Matrix result");
    };
    let theme = style.theme();
    let (card_w, card_h): (f64, f64) = style.size.unwrap_or((WIDTH, HEIGHT));
    let (left, right): (f64, f64) = (72.0, 36.0);
    let n: usize = x_labels.len().max(1);
    let cell_w: f64 = ((card_w - left - right) / n as f64).max(14.0);
    let tick: Tick = if svg::tilts(x_labels, cell_w) {
        Tick::Up
    } else {
        Tick::Flat
    };
    let room: f64 = if tick == Tick::Up {
        svg::tilt_room(x_labels)
    } else {
        0.0
    };
    let top: f64 = 100f64.max(92.0 + room);
    let plot_h: f64 = (card_h - top - 40.0).max(200.0);
    let width: f64 = (left + cell_w * n as f64 + right).max(card_w);
    let height: f64 = top + plot_h + 40.0;
    let band_h: f64 = plot_h / grid.len().max(1) as f64;
    let color: String = style.single_color("");
    let subtitle: &str = texts.subtitle(style);

    let mut out: String = svg::card(&theme, &texts.title(style), subtitle, width, height);
    for (c, name) in x_labels.iter().enumerate() {
        let x: f64 = left + c as f64 * cell_w + 1.0;
        let w: f64 = (cell_w - 2.0).max(1.0);
        let share: f64 = nulls[c] as f64 / rows.max(1) as f64 * 100.0;
        let hover: String = format!(
            "{name}\nnulls = {} ({}%)",
            nulls[c],
            svg::format_value(share)
        );
        out.push_str(&svg::rect(x, top, w, plot_h, &color, &hover));
        for (b, row) in grid.iter().enumerate().filter(|(_, row)| row[c] > 0.0) {
            out.push_str(&format!(
                r##"<rect x="{x:.2}" y="{:.2}" width="{w:.2}" height="{:.2}" fill="{}" fill-opacity="{:.2}"><title>{}</title></rect>"##,
                top + b as f64 * band_h,
                band_h.max(1.0),
                theme.background,
                0.35 + 0.65 * row[c],
                escape(&format!("{name}, {}: {}% missing", y_labels[b], svg::format_value(row[c] * 100.0)))
            ));
        }
        svg::x_tick(
            &mut out,
            &theme,
            name,
            x + w / 2.0,
            top - 12.0,
            cell_w,
            tick,
        );
    }
    for (y, label) in [
        (top + 12.0, "1".to_string()),
        (top + plot_h, rows.to_string()),
    ] {
        out.push_str(&format!(
            r##"<text x="{}" y="{y:.2}" font-family="sans-serif" font-size="12" fill="{}" text-anchor="end">{label}</text>"##,
            left - 8.0,
            theme.tick
        ));
    }
    let (x_name, y_name): (&str, &str) = texts.names(style);
    svg::axis_names(
        &mut out,
        &theme,
        x_name,
        y_name,
        left + cell_w * n as f64 / 2.0,
        top + plot_h / 2.0,
        height,
    );
    out.push_str("</svg>");
    out
}

#[cfg(test)]
mod tests {
    use polars::df;

    use super::*;

    #[test]
    fn corr_drops_nulls_per_pair() {
        let frame: DataFrame = df!(
            "a" => [1.0, 2.0, 3.0, 4.0],
            "b" => [1.0, 3.0, 2.0, 4.0],
            "c" => [None, Some(1.0), Some(2.0), Some(3.0)]
        )
        .unwrap();
        let ds: Dataset = Dataset::new(frame, "");
        let PlotResult::Matrix { grid, .. } =
            compute_corr(&ds, &["a", "b", "c"].map(String::from)).unwrap()
        else {
            unreachable!()
        };
        assert!((grid[0][1] - 0.8).abs() < 1e-12); // all 4 rows; dropping row 1 would give 0.5
        assert!((grid[0][2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn corr_from_the_cached_stats_agrees_with_the_filtered_pass() {
        let frame: DataFrame = df!(
            "a" => [1.0, 2.0, 4.0, 8.0],
            "b" => [3.0, 1.0, 2.0, 9.0],
            "same" => [5.0, 5.0, 5.0, 5.0],
            "nan" => [1.0, f64::NAN, 2.0, 3.0]
        )
        .unwrap();
        let ds: Dataset = Dataset::new(frame.clone(), "");
        let PlotResult::Matrix { grid, .. } =
            compute_corr(&ds, &["a", "b", "same", "nan"].map(String::from)).unwrap()
        else {
            unreachable!()
        };
        // a and b have a value in every row: one pass with the cached mean and variance.
        let filtered: DataFrame = frame
            .lazy()
            .select([pearson("a", "b").alias("r")])
            .collect()
            .unwrap();
        assert!((grid[0][1] - f64s(&filtered, "r").unwrap()[0]).abs() < 1e-12);
        assert!((grid[0][0] - 1.0).abs() < 1e-12);
        assert!(grid[0][2].is_nan() && grid[2][2].is_nan()); // a constant column: no correlation
        // The NaN row is left out: a = 1, 4, 8 against 1, 2, 3.
        assert!(
            (grid[0][3] - 0.996_615_895_540_123_9).abs() < 1e-12,
            "{}",
            grid[0][3]
        );
    }

    #[test]
    fn missing_bands_hold_null_shares() {
        let frame: DataFrame = df!(
            "a" => [Some(1.0), None, Some(3.0)],
            "b" => [None, None, Some("x")]
        )
        .unwrap();
        let ds: Dataset = Dataset::new(frame, "");
        let PlotResult::Matrix { y_labels, grid, .. } =
            compute_missing(&ds, &["a", "b"].map(String::from)).unwrap()
        else {
            unreachable!()
        };
        assert_eq!(grid, vec![vec![0.0, 1.0], vec![1.0, 1.0], vec![0.0, 0.0]]);
        assert_eq!(y_labels[1], "rows 2 to 2");
    }
}
