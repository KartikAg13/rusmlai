//! `hist`, `boxplot` and `stripplot`: how a numeric column is spread out.

use polars::prelude::{DataFrame, DataType, Expr, LazyFrame, QuantileMethod, len, lit, when};

use crate::utils::col;
use crate::{
    dataset::Dataset,
    display::{Figure, escape},
    error::OrFatal,
    metadata::dataset::stats::NumericStats,
    plot::{
        BOUND, SAMPLE, VALUE,
        chart::{Body, Texts},
        check_columns, density,
        error::PlotError,
        f64s, labels, optional,
        result::PlotResult,
        rows, sample, sample_subtitle, slots,
        spec::PlotSpec,
        style::{Style, Theme},
        svg::{
            self, Layout, MIN_CLUSTER_BAR, TOP, Tick, YScale, finish, hlines, layout, legend_items,
            ordered, percent_factor, pick, range, tick_mode, unit, value_label,
        },
        thin, unique,
    },
};

pub(crate) const MAX_BINS: usize = 1000;
// At most 100 outlier rings per box, evenly spread by rank with both ends kept: past that they
// pile up on one axis line anyway, and the SVG and the cache line stay small (like `SAMPLE`).
const MAX_OUTLIERS: usize = 100;
const BIN: &str = "__rusmlai_bin";
const OUTLIERS: &str = "__rusmlai_outliers";
const ALL: &str = "__rusmlai_all"; // the one x group of a box plot without x
const FIVE: [&str; 5] = [
    "__rusmlai_min",
    "__rusmlai_q25",
    "__rusmlai_median",
    "__rusmlai_q75",
    "__rusmlai_max",
];

impl Dataset {
    /// How many values of `column` fall in each of `bins` equal bins, split by `hue`.
    /// (seaborn: `histplot`)
    ///
    /// `bins` goes from 1 to 1,000. The counts are cached in at least 1,000 fine bins, so
    /// `.xlim` can re-bin inside a new range without a pass over the data. Hue groups share the
    /// same edges and overlap; `.stacked()` piles them. Nulls, NaN and infinities are left out.
    ///
    /// Options: `.title`, `.subtitle`, `.xlabel`, `.ylabel`, `.theme`, `.palette`, `.colors`,
    /// `.color`, `.color_map`, `.size`, `.hue_order`, `.values`, `.ylim`, `.xlim`, `.percent`,
    /// `.stacked`, `.kde`, `.hline`, `.vline` and `.legend`.
    ///
    /// # Panics
    ///
    /// If a column doesn't exist, `column` isn't numeric, or `bins` is 0 or above 1,000.
    /// [`Dataset::try_hist`] returns these as errors.
    ///
    /// # Examples
    ///
    /// CGPA in 10 bins:
    /// ```
    /// use rusmlai::prelude::*;
    ///
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.hist("cgpa", "", 10); // saves metadata/plots/Histogram cgpa.png
    /// ```
    ///
    /// Split by `passed`, stacked, in percent, with a density line and most options:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds
    ///     .hist("cgpa", "passed", 8)
    ///     .stacked()
    ///     .percent()
    ///     .kde()
    ///     .title("CGPA of passed and failed")
    ///     .subtitle("8 bins")
    ///     .xlabel("CGPA")
    ///     .ylabel("share of students")
    ///     .theme("light")
    ///     .colors(&["seagreen", "tomato"])
    ///     .hue_order(&["true", "false"])
    ///     .size(800.0, 450.0)
    ///     .xlim(5.0, 9.5)
    ///     .vline(6.0, "pass mark")
    ///     .hline(10.0, "")
    ///     .values(false)
    ///     .legend("bottom")
    ///     .svg(); // the SVG text, and no auto-save
    /// assert!(svg.contains("CGPA of passed and failed"));
    /// assert!(svg.contains("#2E8B57") && svg.contains("#FF6347")); // seagreen, tomato
    /// ```
    ///
    /// Edge case: a text column has no bins:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.hist("batch", "", 10); // panics: column 'batch' is not numeric (found str)
    /// ```
    #[track_caller]
    pub fn hist(&self, column: &str, hue: &str, bins: usize) -> Figure {
        self.try_hist(column, hue, bins).or_fatal()
    }

    /// [`Dataset::hist`] that returns an error instead of panicking.
    ///
    /// # Errors
    ///
    /// - `PlotError::InvalidBins` when `bins` is 0 or above 1,000.
    /// - `PlotError::Dataset` when a column doesn't exist or `column` isn't numeric.
    /// - `PlotError::EmptyResult` when the column has no finite values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # use rusmlai::plot::error::PlotError;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.try_hist("age", "", 6)?.svg();
    /// assert!(svg.contains("Histogram: age"));
    /// # Ok::<(), PlotError>(())
    /// ```
    ///
    /// Zero bins:
    /// ```
    /// # use rusmlai::prelude::*;
    /// use rusmlai::plot::error::PlotError;
    ///
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// assert!(matches!(ds.try_hist("cgpa", "", 0), Err(PlotError::InvalidBins(0))));
    /// ```
    pub fn try_hist(&self, column: &str, hue: &str, bins: usize) -> Result<Figure, PlotError> {
        if !(1..=MAX_BINS).contains(&bins) {
            return Err(PlotError::InvalidBins(bins));
        }
        let hue: Option<String> = optional(hue);
        let result: PlotResult = fine_bins(self, column, hue.as_deref(), bins)?;
        let texts: Texts = Texts {
            title: format!("Histogram: {column}"),
            x: column.to_string(),
            y: "count".to_string(),
            value: "count".to_string(),
            hue: hue.unwrap_or_default(),
            ..Texts::default()
        };
        Ok(self.figure("hist", texts, Body::Hist { result, bins }))
    }

    /// A box of `y` for each `x` category, split by `hue`: the quartiles, the median, and
    /// whiskers. (seaborn: `boxplot`)
    ///
    /// Like seaborn, the whiskers stop at the last value within 1.5 × IQR of the box, and the
    /// values past them are drawn as points (at most 100 per box). `x = ""` draws one box for
    /// the whole column. The boxes are cached next to the CSV.
    ///
    /// Options: `.title`, `.subtitle`, `.xlabel`, `.ylabel`, `.theme`, `.palette`, `.colors`,
    /// `.color`, `.color_map`, `.size`, `.order`, `.hue_order`, `.top`, `.ylim`, `.sort`,
    /// `.hline` and `.legend`.
    ///
    /// # Panics
    ///
    /// If a column doesn't exist, `y` isn't numeric, `hue` is set without `x`, or no row has
    /// values. [`Dataset::try_boxplot`] returns these as errors.
    ///
    /// # Examples
    ///
    /// CGPA in each batch:
    /// ```
    /// use rusmlai::prelude::*;
    ///
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.boxplot("batch", "cgpa", ""); // saves metadata/plots/Box batch vs cgpa.png
    /// ```
    ///
    /// Split by `passed`, with most options:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds
    ///     .boxplot("batch", "cgpa", "passed")
    ///     .title("CGPA by batch")
    ///     .subtitle("whiskers at 1.5 × IQR")
    ///     .xlabel("batch")
    ///     .ylabel("CGPA")
    ///     .theme("light")
    ///     .palette("deep")
    ///     .color_map(&[("false", "tomato")])
    ///     .size(800.0, 480.0)
    ///     .order(&["B13", "B14", "B15"])
    ///     .hue_order(&["true", "false"])
    ///     .ylim(4.0, 10.0)
    ///     .hline(6.0, "pass mark")
    ///     .legend("bottom")
    ///     .svg(); // the SVG text, and no auto-save
    /// assert!(svg.contains("CGPA by batch"));
    /// assert!(svg.contains("#FF6347")); // tomato
    /// ```
    ///
    /// Edge case: `x = ""` draws one box of all 11 CGPA values (one is null):
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.boxplot("", "cgpa", "").svg();
    /// assert!(svg.contains("Box: cgpa") && svg.contains("median = 7.40"));
    /// ```
    #[track_caller]
    pub fn boxplot(&self, x: &str, y: &str, hue: &str) -> Figure {
        self.try_boxplot(x, y, hue).or_fatal()
    }

    /// [`Dataset::boxplot`] that returns an error instead of panicking.
    ///
    /// # Errors
    ///
    /// - `PlotError::Dataset` when a column doesn't exist or `y` isn't numeric.
    /// - `PlotError::HueNeedsX` when `hue` is set and `x` is `""`.
    /// - `PlotError::EmptyResult` when no row has values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # use rusmlai::plot::error::PlotError;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.try_boxplot("passed", "age", "")?.svg();
    /// assert!(svg.contains("Box: passed vs age"));
    /// # Ok::<(), PlotError>(())
    /// ```
    ///
    /// A hue without an x column:
    /// ```
    /// # use rusmlai::prelude::*;
    /// use rusmlai::plot::error::PlotError;
    ///
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let result = ds.try_boxplot("", "cgpa", "passed");
    /// assert!(matches!(result, Err(PlotError::HueNeedsX)));
    /// ```
    pub fn try_boxplot(&self, x: &str, y: &str, hue: &str) -> Result<Figure, PlotError> {
        let (result, x, hue): (PlotResult, Option<String>, Option<String>) =
            boxes(self, x, y, hue)?;
        Ok(self.figure("boxplot", spread_texts("Box", x, y, hue), Body::Box(result)))
    }

    /// One dot per row: the values of `y` in each `x` category, spread sideways and colored by
    /// `hue`. (seaborn: `stripplot`)
    ///
    /// It draws 2,500 sampled rows by default, from up to 50,000 kept. The seed is fixed, so the
    /// same data draws the same picture. `x = ""` draws one strip. Raw rows aren't cached.
    ///
    /// Options: `.title`, `.subtitle`, `.xlabel`, `.ylabel`, `.theme`, `.palette`, `.colors`,
    /// `.color`, `.color_map`, `.size`, `.order`, `.hue_order`, `.top`, `.ylim`, `.sort`,
    /// `.sample`, `.hline` and `.legend`.
    ///
    /// # Panics
    ///
    /// If a column doesn't exist, `y` isn't numeric, `hue` is set without `x`, or no row has
    /// values. [`Dataset::try_stripplot`] returns these as errors.
    ///
    /// # Examples
    ///
    /// Each student's CGPA, by batch:
    /// ```
    /// use rusmlai::prelude::*;
    ///
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.stripplot("batch", "cgpa", ""); // saves metadata/plots/Strip batch vs cgpa.png
    /// ```
    ///
    /// Colored by `passed`, with 6 sampled dots and most options:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds
    ///     .stripplot("batch", "cgpa", "passed")
    ///     .sample(6)
    ///     .title("CGPA of each student")
    ///     .subtitle("6 of 11 rows")
    ///     .xlabel("batch")
    ///     .ylabel("CGPA")
    ///     .theme("light")
    ///     .colors(&["#2E8B57", "tomato"])
    ///     .size(700.0, 450.0)
    ///     .order(&["B13", "B15"])
    ///     .hue_order(&["true", "false"])
    ///     .ylim(4.0, 10.0)
    ///     .hline(6.0, "pass mark")
    ///     .legend("bottom")
    ///     .svg(); // the SVG text, and no auto-save
    /// assert!(svg.contains("CGPA of each student"));
    /// assert!(svg.contains("#2E8B57"));
    /// ```
    ///
    /// Edge case: a hue needs an x column to split:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.stripplot("", "cgpa", "passed"); // panics: hue needs an x column
    /// ```
    #[track_caller]
    pub fn stripplot(&self, x: &str, y: &str, hue: &str) -> Figure {
        self.try_stripplot(x, y, hue).or_fatal()
    }

    /// [`Dataset::stripplot`] that returns an error instead of panicking.
    ///
    /// # Errors
    ///
    /// - `PlotError::Dataset` when a column doesn't exist or `y` isn't numeric.
    /// - `PlotError::HueNeedsX` when `hue` is set and `x` is `""`.
    /// - `PlotError::EmptyResult` when no row has values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # use rusmlai::plot::error::PlotError;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.try_stripplot("", "age", "")?.svg();
    /// assert!(svg.contains("Strip: age"));
    /// # Ok::<(), PlotError>(())
    /// ```
    ///
    /// A text column as `y`:
    /// ```
    /// # use rusmlai::prelude::*;
    /// use rusmlai::{dataset::error::DatasetError, plot::error::PlotError};
    ///
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let result = ds.try_stripplot("batch", "name", "");
    /// assert!(matches!(
    ///     result,
    ///     Err(PlotError::Dataset(DatasetError::ColumnNotNumeric { .. }))
    /// ));
    /// ```
    pub fn try_stripplot(&self, x: &str, y: &str, hue: &str) -> Result<Figure, PlotError> {
        self.try_numeric_stats(y)?;
        let (x, hue): (Option<String>, Option<String>) = (optional(x), optional(hue));
        check_columns(self, x.as_deref(), hue.as_deref())?;
        let columns: Vec<&str> = [Some(y), x.as_deref(), hue.as_deref()]
            .into_iter()
            .flatten()
            .collect();
        let (sample, rows): (DataFrame, usize) = sample(self, &columns)?;

        let x_labels: Vec<String> = match &x {
            Some(x) => labels(self, &sample, x)?,
            None => vec![y.to_string()],
        };
        let hue_labels: Vec<String> = match &hue {
            Some(hue) => labels(self, &sample, hue)?,
            None => Vec::new(),
        };
        let cells = slots(&sample, x.as_deref(), &x_labels)?
            .into_iter()
            .zip(slots(&sample, hue.as_deref(), &hue_labels)?);
        let points: Vec<(usize, usize, f64)> = cells
            .zip(f64s(&sample, y)?)
            .filter_map(|((i, j), y)| Some((i?, j?, y)))
            .collect();
        if points.is_empty() {
            return Err(PlotError::EmptyResult(y.to_string()));
        }
        let body: Body = Body::Strip {
            x_labels,
            hue_labels,
            points,
            rows,
        };
        Ok(self.figure("stripplot", spread_texts("Strip", x, y, hue), body))
    }
}

/// The cached counts of `column` in fine bins (a multiple of `bins`, at least 1,000), split by
/// `hue`. The edges come from the cached min and max, so every hue group shares them.
pub(crate) fn fine_bins(
    dataset: &Dataset,
    column: &str,
    hue: Option<&str>,
    bins: usize,
) -> Result<PlotResult, PlotError> {
    let stats: &NumericStats = dataset.try_numeric_stats(column)?;
    if let Some(hue) = hue {
        dataset.try_column_info(hue)?;
    }
    // The cached min and max keep ±inf (and values past BOUND); the bins need the drawn range.
    let (lo, hi): (f64, f64) = if stats.min.abs() <= BOUND && stats.max.abs() <= BOUND {
        (stats.min, stats.max)
    } else {
        let v = || col(column).cast(DataType::Float64);
        let edges: DataFrame = rows(dataset, &[column])
            .select([v().min().alias("lo"), v().max().alias("hi")])
            .collect()?;
        (f64s(&edges, "lo")?[0], f64s(&edges, "hi")?[0])
    };
    if !lo.is_finite() {
        return Err(PlotError::EmptyResult(column.to_string()));
    }
    let spec: PlotSpec = PlotSpec::Histogram {
        column: column.to_string(),
        hue: hue.map(str::to_string),
        bins,
    };
    let fine: usize = bins * MAX_BINS.div_ceil(bins);
    dataset
        .metadata
        .plot(spec, || compute_hist(dataset, column, hue, fine, lo, hi))
}

/// The cached boxes of `y` per `x` category and `hue` group (also a violin's inner box), and
/// the two columns as options.
pub(crate) fn boxes(
    dataset: &Dataset,
    x: &str,
    y: &str,
    hue: &str,
) -> Result<(PlotResult, Option<String>, Option<String>), PlotError> {
    dataset.try_numeric_stats(y)?;
    let (x, hue): (Option<String>, Option<String>) = (optional(x), optional(hue));
    check_columns(dataset, x.as_deref(), hue.as_deref())?;
    let spec: PlotSpec = PlotSpec::Boxplot {
        x: x.clone().unwrap_or_default(),
        y: y.to_string(),
        hue: hue.clone(),
    };
    let result: PlotResult = dataset.metadata.plot(spec, || {
        compute_boxes(dataset, x.as_deref(), y, hue.as_deref())
    })?;
    if result.is_empty() {
        return Err(PlotError::EmptyResult(y.to_string()));
    }
    Ok((result, x, hue))
}

/// The hover text of one box: its category, hue group, fences and quartiles.
pub(crate) fn box_hover(texts: &Texts, x: &str, hue: Option<&str>, five: [f64; 5]) -> String {
    let mut hover: String = format!("{} = {x}", texts.x);
    if let Some(hue) = hue {
        hover += &format!("\n{} = {hue}", texts.hue);
    }
    for (name, v) in ["upper fence", "q75", "median", "q25", "lower fence"]
        .iter()
        .zip([five[4], five[3], five[2], five[1], five[0]])
    {
        hover += &format!("\n{name} = {}", svg::format_value(v));
    }
    hover
}

/// `Box: day vs tip` / `Box: tip` (no x).
pub(crate) fn spread_texts(kind: &str, x: Option<String>, y: &str, hue: Option<String>) -> Texts {
    Texts {
        title: match &x {
            Some(x) => format!("{kind}: {x} vs {y}"),
            None => format!("{kind}: {y}"),
        },
        x: x.unwrap_or_else(|| y.to_string()),
        y: y.to_string(),
        value: y.to_string(),
        hue: hue.unwrap_or_default(),
        ..Texts::default()
    }
}

/// `fine` bin counts over [`lo`, `hi`] regrouped into `bins` equal bins over `range`, by each fine
/// bin's center. Returns the counts and how many rows fell outside `range`. Without `.xlim`
/// the fine bins split evenly, so the counts are exact; with it an edge can be off by one fine
/// bin (1/1,000 of the data range).
fn regroup(fine: &[f64], (lo, hi): (f64, f64), range: (f64, f64), bins: usize) -> (Vec<f64>, f64) {
    let width: f64 = (hi - lo).max(1e-9) / fine.len().max(1) as f64;
    let span: f64 = (range.1 - range.0).max(1e-9);
    let mut counts: Vec<f64> = vec![0.0; bins];
    let mut outside: f64 = 0.0;
    for (f, &n) in fine.iter().enumerate() {
        let center: f64 = lo + (f as f64 + 0.5) * width;
        if center < range.0 || center > range.1 {
            outside += n;
        } else {
            let b: usize = ((center - range.0) / span * bins as f64) as usize;
            counts[b.min(bins - 1)] += n;
        }
    }
    (counts, outside)
}

fn compute_hist(
    dataset: &Dataset,
    column: &str,
    hue: Option<&str>,
    bins: usize,
    lo: f64,
    hi: f64,
) -> Result<PlotResult, PlotError> {
    let span: f64 = (hi - lo).max(1e-9);
    let last: i64 = bins as i64 - 1;
    let raw: Expr = ((col(column).cast(DataType::Float64) - lit(lo)) / lit(span)
        * lit(bins as f64))
    .cast(DataType::Int64);
    let bin: Expr = when(raw.clone().gt(lit(last)))
        .then(lit(last))
        .otherwise(raw)
        .alias(BIN);
    let mut keys: Vec<Expr> = vec![col(BIN)];
    keys.extend(hue.map(col));
    let columns: Vec<&str> = [Some(column), hue].into_iter().flatten().collect();
    let grouped: DataFrame = rows(dataset, &columns)
        .with_column(bin)
        .group_by(keys)
        .agg([len().cast(DataType::Float64).alias(VALUE)])
        .collect()?;

    let names: Vec<String> = match hue {
        Some(hue) => labels(dataset, &grouped, hue)?,
        None => vec![column.to_string()],
    };
    let mut counts: Vec<Vec<f64>> = vec![vec![0.0; bins]; names.len()];
    let bins_column = grouped.column(BIN)?.i64()?.iter();
    for ((bin, s), n) in bins_column
        .zip(slots(&grouped, hue, &names)?)
        .zip(f64s(&grouped, VALUE)?)
    {
        // NaN values have no bin.
        if let (Some(bin), Some(s)) = (bin, s) {
            counts[s][bin as usize] = n;
        }
    }
    Ok(PlotResult::Hist {
        lo,
        hi,
        names,
        counts,
    })
}

fn compute_boxes(
    dataset: &Dataset,
    x: Option<&str>,
    y: &str,
    hue: Option<&str>,
) -> Result<PlotResult, PlotError> {
    let keys: Vec<Expr> = unique([Some(x.unwrap_or(ALL)), hue].into_iter().flatten())
        .into_iter()
        .map(col)
        .collect();
    let columns: Vec<&str> = [Some(y), x, hue].into_iter().flatten().collect();
    let mut frame: LazyFrame = rows(dataset, &columns);
    if x.is_none() {
        frame = frame.with_column(lit(y).alias(ALL));
    }
    let v = || col(y).cast(DataType::Float64);
    let q = |p: f64| v().quantile(lit(p), QuantileMethod::Linear);
    let reach: Expr = (q(0.75) - q(0.25)) * lit(1.5);
    let inside: Expr = v()
        .gt_eq(q(0.25) - reach.clone())
        .and(v().lt_eq(q(0.75) + reach));
    let grouped: DataFrame = frame
        .group_by(keys)
        .agg([
            v().filter(inside.clone()).min().alias(FIVE[0]),
            q(0.25).alias(FIVE[1]),
            v().median().alias(FIVE[2]),
            q(0.75).alias(FIVE[3]),
            v().filter(inside.clone()).max().alias(FIVE[4]),
            v().filter(inside.not()).alias(OUTLIERS),
        ])
        .collect()?;

    let x_labels: Vec<String> = match x {
        Some(x) => labels(dataset, &grouped, x)?,
        None => vec![y.to_string()],
    };
    let hue_labels: Vec<String> = match hue {
        Some(hue) => labels(dataset, &grouped, hue)?,
        None => Vec::new(),
    };
    let five: Vec<Vec<f64>> = FIVE
        .iter()
        .map(|name| f64s(&grouped, name))
        .collect::<Result<_, _>>()?;
    let mut past: Vec<Vec<f64>> = Vec::with_capacity(grouped.height());
    for group in grouped.column(OUTLIERS)?.list()?.series_iter() {
        let mut values: Vec<f64> = match group {
            Some(values) => values
                .f64()?
                .iter()
                .flatten()
                .filter(|v| v.is_finite())
                .collect(),
            None => Vec::new(),
        };
        values.sort_by(f64::total_cmp);
        if values.len() > MAX_OUTLIERS {
            let last: usize = values.len() - 1;
            values = (0..MAX_OUTLIERS)
                .map(|i| values[i * last / (MAX_OUTLIERS - 1)])
                .collect();
        }
        past.push(values);
    }
    let k: usize = hue_labels.len().max(1);
    let mut boxes: Vec<Vec<[f64; 5]>> = vec![vec![[f64::NAN; 5]; k]; x_labels.len()];
    let mut outliers: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); k]; x_labels.len()];
    let cells = slots(&grouped, x, &x_labels)?
        .into_iter()
        .zip(slots(&grouped, hue, &hue_labels)?);
    for (row, cell) in cells.enumerate() {
        if let (Some(i), Some(j)) = cell {
            boxes[i][j] = [0, 1, 2, 3, 4].map(|k| five[k][row]);
            outliers[i][j] = std::mem::take(&mut past[row]);
        }
    }
    Ok(PlotResult::Boxes {
        x_labels,
        hue_labels,
        boxes,
        outliers,
    })
}

/// The old histogram: bars per bin (grouped or stacked by hue), bin ticks, five edge labels.
pub(crate) fn render_hist(
    texts: &Texts,
    result: &PlotResult,
    bins: usize,
    style: &Style,
) -> String {
    let PlotResult::Hist {
        lo: data_lo,
        hi: data_hi,
        names: groups,
        counts: fine,
    } = result
    else {
        unreachable!("histograms hold Hist results");
    };
    let theme: Theme = style.theme();
    let grouped: bool = !texts.hue.is_empty();
    let stacked: bool = grouped && style.stacked;
    let cols: Vec<usize> = if grouped {
        ordered(groups, style.hue_order.as_deref())
    } else {
        vec![0]
    };
    let (lo, hi): (f64, f64) = style.xlim.unwrap_or((*data_lo, *data_hi));
    let bins: usize = bins.max(1);
    let mut outside: f64 = 0.0;
    let counts: Vec<Vec<f64>> = fine
        .iter()
        .enumerate()
        .map(|(j, series)| {
            let (counts, out): (Vec<f64>, f64) =
                regroup(series, (*data_lo, *data_hi), (lo, hi), bins);
            if cols.contains(&j) {
                outside += out;
            }
            counts
        })
        .collect();
    let factor: f64 = percent_factor(style, counts.iter().flatten().copied());
    let counts: Vec<Vec<f64>> = counts
        .iter()
        .map(|c| c.iter().map(|v| v * factor).collect())
        .collect();
    let what: &str = if style.percent { "percent" } else { "count" };
    let k: usize = if stacked { 1 } else { cols.len().max(1) };
    let fill: f64 = if k == 1 { 0.82 } else { 0.70 };
    let min_slot: f64 = if k == 1 {
        16.0
    } else {
        k as f64 * MIN_CLUSTER_BAR / fill
    };
    let items: Vec<(String, String)> = if grouped {
        legend_items(style, groups, &cols)
    } else {
        Vec::new()
    };
    let left_out: String = if outside > 0.0 {
        format!("{} rows outside the range", svg::format_value(outside))
    } else {
        texts.subtitle.clone()
    };
    let subtitle: &str = style.subtitle.as_deref().unwrap_or(&left_out);
    let l: Layout = layout(style, subtitle, bins, min_slot, &items);
    let heights: Vec<f64> = if stacked {
        (0..bins)
            .map(|b| cols.iter().map(|&j| counts[j][b]).sum())
            .collect()
    } else {
        cols.iter()
            .flat_map(|&j| counts[j].iter().copied())
            .collect()
    };
    // `.kde()`: each series' density at 200 points of the drawn range, in rows per drawn bin.
    let curves: Vec<density::Curve> = if style.kde {
        cols.iter()
            .enumerate()
            .filter_map(|(s, &j)| {
                let h: f64 = density::scott(&fine[j], (*data_lo, *data_hi))?;
                let per_bin: f64 = fine[j].iter().sum::<f64>() * (hi - lo) / bins as f64 * factor;
                let xs: Vec<f64> = density::linspace(lo, hi);
                let ys: Vec<f64> = density::kde(&fine[j], (*data_lo, *data_hi), h, &xs);
                Some((
                    s,
                    j,
                    xs.into_iter()
                        .zip(ys.into_iter().map(|d| d * per_bin))
                        .collect(),
                ))
            })
            .collect()
    } else {
        Vec::new()
    };
    let curve_tops = curves.iter().flat_map(|c| c.2.iter().map(|p| p.1));
    let (y_lo, y_hi): (f64, f64) = range(style, heights.into_iter().chain(curve_tops));
    let scale: YScale = YScale::new(y_lo, y_hi, l.top, l.plot_h);
    let (slot, span): (f64, f64) = (l.plot_w / bins as f64, (hi - lo).max(1e-9));
    let (inner, base): (f64, f64) = (slot * fill, scale.bottom());
    let bar_w: f64 = inner / k as f64;
    let show_values: bool = style.values.unwrap_or(false);
    let edge = |b: usize| lo + span * b as f64 / bins as f64;
    let hover = |b: usize, j: usize, v: f64| {
        let mut text: String = format!(
            "{} = {} to {}",
            texts.x,
            svg::format_value(edge(b)),
            svg::format_value(edge(b + 1))
        );
        if grouped {
            text += &format!("\n{} = {}", texts.hue, groups[j]);
        }
        text + &format!("\n{what} = {}", unit(style, v))
    };
    let color = |s: usize, j: usize| {
        if grouped {
            style.group_color(s, &groups[j])
        } else {
            style.single_color(&groups[j])
        }
    };

    let mut out: String = svg::card(&theme, &texts.title(style), subtitle, l.width, l.height);
    scale.draw(&mut out, &theme, l.left, l.width - l.right);
    for b in 0..=bins {
        let x: f64 = l.left + b as f64 * slot;
        out.push_str(&format!(
            r##"<line x1="{x}" y1="{}" x2="{x}" y2="{base}" stroke="{}" stroke-width="1"/>"##,
            base + 6.0,
            theme.grid
        ));
    }
    if stacked {
        #[allow(clippy::needless_range_loop)] // b is the bin: it indexes every group's counts
        for b in 0..bins {
            let x: f64 = l.left + b as f64 * slot + (slot - inner) / 2.0;
            let w: f64 = (inner - 1.2).max(1.0);
            let mut sum: f64 = 0.0;
            for (s, &j) in cols.iter().enumerate() {
                let v: f64 = counts[j][b];
                if v > 0.0 {
                    let (y0, y1): (f64, f64) = (scale.y(sum + v), scale.y(sum));
                    out.push_str(&svg::rect(x, y0, w, y1 - y0, &color(s, j), &hover(b, j, v)));
                    sum += v;
                }
            }
            if show_values && sum > 0.0 {
                value_label(
                    &mut out,
                    &theme,
                    x + w / 2.0,
                    scale.y(sum) - 8.0,
                    &unit(style, sum),
                );
            }
        }
    } else {
        for (s, &j) in cols.iter().enumerate() {
            for (b, &value) in counts[j].iter().enumerate() {
                let x: f64 = l.left + b as f64 * slot + (slot - inner) / 2.0 + s as f64 * bar_w;
                let w: f64 = (bar_w - 1.2).max(1.0);
                out.push_str(&svg::bar(
                    &scale,
                    x,
                    w,
                    value,
                    &color(s, j),
                    &hover(b, j, value),
                ));
                if show_values && value > 0.0 {
                    let text: String = unit(style, value);
                    value_label(&mut out, &theme, x + w / 2.0, scale.y(value) - 8.0, &text);
                }
            }
        }
    }
    for (s, j, points) in &curves {
        let d: String = svg::polyline(
            points
                .iter()
                .map(|&(x, y)| (l.left + (x - lo) / span * l.plot_w, scale.y(y))),
        );
        let mut hover: String = format!("density of {}", texts.x);
        if grouped {
            hover += &format!("\n{} = {}", texts.hue, groups[*j]);
        }
        density::halo_path(&mut out, &theme, &d, &color(*s, *j), &hover);
    }
    let mut ticks: Vec<usize> = (0..=4).map(|t| t * bins / 4).collect();
    ticks.dedup();
    for b in ticks {
        out.push_str(&format!(
            r##"<text x="{}" y="{}" font-family="sans-serif" font-size="11" fill="{}" text-anchor="middle">{}</text>"##,
            l.left + b as f64 * slot,
            base + 22.0,
            theme.tick,
            svg::format_value(edge(b))
        ));
    }
    hlines(&mut out, &theme, style, &scale, &l);
    // A .vline outside the drawn range is skipped; `.xlim` can widen the range to include it.
    for (v, label) in style.vlines.iter().filter(|(v, _)| (lo..=hi).contains(v)) {
        let x: f64 = l.left + (v - lo) / span * l.plot_w;
        svg::ref_line(&mut out, &theme, false, x, l.top, base, label, *v);
    }
    let (x_name, y_name): (&str, &str) = texts.names(style);
    let y_name: &str = if style.percent && style.ylabel.is_none() {
        "percent"
    } else {
        y_name
    };
    finish(&mut out, &theme, style, (x_name, y_name), &items, &l);
    out
}

/// The categories, hue groups and legend of a box plot or a violin plot.
pub(crate) struct BoxGroups<'a> {
    pub grouped: bool,
    /// The shown categories, by index into the x labels.
    pub rows: Vec<usize>,
    /// The shown hue groups, by index into the hue labels (`[0]` without hue).
    pub cols: Vec<usize>,
    /// The number of boxes per category.
    pub k: usize,
    pub shown: Vec<&'a String>,
    pub min_slot: f64,
    pub items: Vec<(String, String)>,
}

/// Picks and orders the categories by median and the hue groups by `hue_order`.
pub(crate) fn box_groups<'a>(
    x_labels: &'a [String],
    hue_labels: &[String],
    boxes: &[Vec<[f64; 5]>],
    style: &Style,
) -> BoxGroups<'a> {
    let grouped: bool = !hue_labels.is_empty();
    let medians: Vec<f64> = boxes
        .iter()
        .map(|row| mean(row.iter().map(|five| five[2])))
        .collect();
    let rows: Vec<usize> = pick(x_labels, style, &medians, style.top.unwrap_or(TOP));
    let cols: Vec<usize> = if grouped {
        ordered(hue_labels, style.hue_order.as_deref())
    } else {
        vec![0]
    };
    let k: usize = cols.len().max(1);
    let shown: Vec<&String> = rows.iter().map(|&i| &x_labels[i]).collect();
    let min_slot: f64 = if grouped {
        k as f64 * MIN_CLUSTER_BAR / 0.62
    } else {
        56.0
    }
    .max(svg::label_width(&shown));
    let items: Vec<(String, String)> = if grouped {
        legend_items(style, hue_labels, &cols)
    } else {
        Vec::new()
    };
    BoxGroups {
        grouped,
        rows,
        cols,
        k,
        shown,
        min_slot,
        items,
    }
}

/// The box from q25 to q75 with a median line, whiskers to the fences, outliers as rings.
pub(crate) fn render_box(texts: &Texts, result: &PlotResult, style: &Style) -> String {
    let PlotResult::Boxes {
        x_labels,
        hue_labels,
        boxes,
        outliers,
    } = result
    else {
        unreachable!("box plots hold Boxes results");
    };
    let theme: Theme = style.theme();
    let BoxGroups {
        grouped,
        rows,
        cols,
        k,
        shown,
        min_slot,
        items,
    } = box_groups(x_labels, hue_labels, boxes, style);
    let subtitle: &str = texts.subtitle(style);
    let mut l: Layout = layout(style, subtitle, rows.len(), min_slot, &items);
    let (lo, hi): (f64, f64) = range(
        style,
        rows.iter().flat_map(|&i| {
            cols.iter().flat_map(move |&j| {
                [boxes[i][j][0], boxes[i][j][4]]
                    .into_iter()
                    .chain(outliers[i][j].iter().copied())
            })
        }),
    );
    let scale: YScale = YScale::new(lo, hi, l.top, l.plot_h);
    let slot: f64 = l.plot_w / rows.len().max(1) as f64;
    let tick: Tick = tick_mode(&mut l, &shown, slot);
    let box_w: f64 = if grouped {
        slot * 0.70 / k as f64
    } else {
        slot * 0.42
    };

    let mut out: String = svg::card(&theme, &texts.title(style), subtitle, l.width, l.height);
    scale.draw(&mut out, &theme, l.left, l.width - l.right);
    for (n, &i) in rows.iter().enumerate() {
        let slot_x: f64 = l.left + n as f64 * slot;
        let start: f64 = slot_x + (slot - box_w * k as f64) / 2.0;
        for (s, &j) in cols.iter().enumerate() {
            let five: [f64; 5] = boxes[i][j];
            if !five[2].is_finite() {
                continue; // a missing group stays empty
            }
            let color: String = if grouped {
                style.group_color(s, &hue_labels[j])
            } else {
                style.category_color(n, &x_labels[i])
            };
            let hover: String = box_hover(
                texts,
                &x_labels[i],
                grouped.then(|| hue_labels[j].as_str()),
                five,
            );
            let w: f64 = if grouped { box_w * 0.8 } else { box_w };
            let cx: f64 = start + (s as f64 + 0.5) * box_w;
            out.push_str(&box_svg(&scale, cx, w, five, &color, theme.value, &hover));
            for &v in &outliers[i][j] {
                out.push_str(&format!(
                    r##"<circle cx="{cx}" cy="{:.2}" r="2.5" fill="none" stroke="{color}" stroke-width="1.2"><title>{} = {}</title></circle>"##,
                    scale.y(v),
                    escape(&texts.y),
                    svg::format_value(v)
                ));
            }
        }
        svg::x_tick(
            &mut out,
            &theme,
            &x_labels[i],
            slot_x + slot / 2.0,
            scale.bottom() + 22.0,
            slot,
            tick,
        );
    }
    hlines(&mut out, &theme, style, &scale, &l);
    finish(&mut out, &theme, style, texts.names(style), &items, &l);
    out
}

fn box_svg(
    scale: &YScale,
    cx: f64,
    w: f64,
    five: [f64; 5],
    color: &str,
    median_color: &str,
    hover: &str,
) -> String {
    let [min, q25, median, q75, max] = five.map(|v| scale.y(v));
    let (x0, cap): (f64, f64) = (cx - w / 2.0, w * 0.25);
    let (left, right): (f64, f64) = (cx - cap, cx + cap);
    format!(
        r##"<line x1="{cx}" y1="{min}" x2="{cx}" y2="{max}" stroke="{color}" stroke-width="1.5"/><line x1="{left}" y1="{min}" x2="{right}" y2="{min}" stroke="{color}" stroke-width="1.5"/><line x1="{left}" y1="{max}" x2="{right}" y2="{max}" stroke="{color}" stroke-width="1.5"/><rect x="{x0}" y="{q75}" width="{w}" height="{}" fill="{color}" fill-opacity="0.35" stroke="{color}" stroke-width="1.5" rx="2"><title>{}</title></rect><line x1="{x0}" y1="{median}" x2="{}" y2="{median}" stroke="{median_color}" stroke-width="2"/>"##,
        (q25 - q75).abs().max(1.0),
        escape(hover),
        x0 + w
    )
}

/// The old strip plot: sampled points, spread sideways inside each category.
pub(crate) fn render_strip(
    texts: &Texts,
    x_labels: &[String],
    hue_labels: &[String],
    points: &[(usize, usize, f64)],
    rows_with_values: usize,
    style: &Style,
) -> String {
    let theme: Theme = style.theme();
    let grouped: bool = !hue_labels.is_empty();
    let points: Vec<(usize, usize, f64)> = thin(points, style.sample.unwrap_or(SAMPLE));
    let means: Vec<f64> = (0..x_labels.len())
        .map(|i| mean(points.iter().filter(|p| p.0 == i).map(|p| p.2)))
        .collect();
    let rows: Vec<usize> = pick(x_labels, style, &means, style.top.unwrap_or(TOP));
    let mut slot_of: Vec<Option<usize>> = vec![None; x_labels.len()];
    for (n, &i) in rows.iter().enumerate() {
        slot_of[i] = Some(n);
    }
    let cols: Vec<usize> = if grouped {
        ordered(hue_labels, style.hue_order.as_deref())
    } else {
        Vec::new()
    };
    let mut hue_color: Vec<Option<String>> = vec![None; hue_labels.len()];
    for (s, &j) in cols.iter().enumerate() {
        hue_color[j] = Some(style.group_color(s, &hue_labels[j]));
    }
    let x_color: Vec<String> = x_labels
        .iter()
        .enumerate()
        .map(|(i, x)| style.category_color(slot_of[i].unwrap_or(0), x))
        .collect();
    let shown: Vec<&String> = rows.iter().map(|&i| &x_labels[i]).collect();
    let subtitle: String = sample_subtitle(style, &texts.subtitle, points.len(), rows_with_values);
    let subtitle: &str = &subtitle;
    let items: Vec<(String, String)> = cols
        .iter()
        .filter_map(|&j| Some((hue_labels[j].clone(), hue_color[j].clone()?)))
        .collect();
    let mut l: Layout = layout(
        style,
        subtitle,
        rows.len(),
        svg::label_width(&shown).max(48.0),
        &items,
    );
    let (lo, hi): (f64, f64) = range(
        style,
        points
            .iter()
            .filter(|p| slot_of[p.0].is_some())
            .map(|p| p.2),
    );
    let scale: YScale = YScale::new(lo, hi, l.top, l.plot_h);
    let slot: f64 = l.plot_w / rows.len().max(1) as f64;
    let tick: Tick = tick_mode(&mut l, &shown, slot);

    let mut out: String = svg::card(&theme, &texts.title(style), subtitle, l.width, l.height);
    scale.draw(&mut out, &theme, l.left, l.width - l.right);
    for (k, &(i, j, y)) in points.iter().enumerate() {
        let Some(n) = slot_of[i] else {
            continue;
        };
        let color: &str = if grouped {
            match &hue_color[j] {
                Some(color) => color,
                None => continue, // left out by .hue_order
            }
        } else {
            &x_color[i]
        };
        // Fixed jitter from the point number, so the same data draws the same picture.
        let jitter: f64 = ((k as f64 * 12.9898).sin() * 43758.5453).fract().abs() - 0.5;
        let px: f64 = l.left + (n as f64 + 0.5) * slot + jitter * slot * 0.35;
        let mut hover: String = format!(
            "{} = {}\n{} = {}",
            texts.x,
            x_labels[i],
            texts.y,
            svg::format_value(y)
        );
        if grouped {
            hover += &format!("\n{} = {}", texts.hue, hue_labels[j]);
        }
        out.push_str(&format!(
            r##"<circle cx="{px:.2}" cy="{:.2}" r="3" fill="{color}" fill-opacity="0.6"><title>{}</title></circle>"##,
            scale.y(y),
            escape(&hover)
        ));
    }
    for (n, &i) in rows.iter().enumerate() {
        svg::x_tick(
            &mut out,
            &theme,
            &x_labels[i],
            l.left + (n as f64 + 0.5) * slot,
            scale.bottom() + 22.0,
            slot,
            tick,
        );
    }
    hlines(&mut out, &theme, style, &scale, &l);
    finish(&mut out, &theme, style, texts.names(style), &items, &l);
    out
}

/// Mean of the finite values; NaN when there are none.
pub(crate) fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let (sum, n): (f64, usize) = values
        .filter(|v| v.is_finite())
        .fold((0.0, 0), |(sum, n), v| (sum + v, n + 1));
    if n == 0 { f64::NAN } else { sum / n as f64 }
}

#[cfg(test)]
mod tests {
    use polars::df;

    use super::*;

    #[test]
    fn whiskers_stop_at_the_fences() {
        let ds: Dataset = Dataset::new(df!("y" => [1.0, 2.0, 3.0, 4.0, 100.0]).unwrap(), "");
        let PlotResult::Boxes {
            boxes, outliers, ..
        } = compute_boxes(&ds, None, "y", None).unwrap()
        else {
            unreachable!()
        };
        assert_eq!(boxes, vec![vec![[1.0, 2.0, 3.0, 4.0, 4.0]]]);
        assert_eq!(outliers, vec![vec![vec![100.0]]]);
    }

    #[test]
    fn regroup_is_exact_without_xlim_and_counts_rows_outside_it() {
        let fine: Vec<f64> = vec![1.0; 1000]; // 1,000 fine bins over 0..100, one row each
        let (counts, outside): (Vec<f64>, f64) = regroup(&fine, (0.0, 100.0), (0.0, 100.0), 4);
        assert_eq!((counts, outside), (vec![250.0; 4], 0.0));
        let (counts, outside): (Vec<f64>, f64) = regroup(&fine, (0.0, 100.0), (25.0, 75.0), 2);
        assert_eq!((counts, outside), (vec![250.0; 2], 500.0));
    }
}
