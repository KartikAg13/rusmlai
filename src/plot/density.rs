//! `kdeplot`, `violinplot` and `qqplot`: the smooth shape of a numeric column, and the density
//! math that `hist(..).kde()` and `pairplot` share.

use std::f64::consts::TAU;

use polars::prelude::{DataFrame, DataType, SortMultipleOptions, col};

use crate::{
    dataset::Dataset,
    display::{Figure, escape},
    error::OrFatal,
    plot::{
        MAX_SAMPLE,
        chart::{Body, Texts},
        dist::{box_hover, boxes, fine_bins, mean, spread_texts},
        error::PlotError,
        f64s,
        result::PlotResult,
        rows, sample, sample_subtitle, slots,
        style::{Style, Theme},
        svg::{
            self, Layout, MIN_CLUSTER_BAR, TOP, Tick, XScale, YScale, extent, finish, hlines,
            layout, legend_items, ordered, pick, range, tick_mode,
        },
    },
};

/// One curve: its palette slot, its group, and its (x, y) points.
pub(crate) type Curve = (usize, usize, Vec<(f64, f64)>);

const GRID: usize = 200; // points on a curve (seaborn's gridsize)
const KDE_BINS: usize = 256; // bins that sampled rows are counted in before smoothing
const MAX_BINS: usize = 1000; // the `hist` cache key that `kdeplot` reads (1,000 fine bins)

/// Scott's rule, as in scipy's `gaussian_kde` (seaborn's default): the rows' std times
/// n^(-1/5), both read from the bins. None for fewer than 2 rows or no spread; seaborn skips
/// those too.
pub(crate) fn scott(counts: &[f64], (lo, hi): (f64, f64)) -> Option<f64> {
    let w: f64 = (hi - lo) / counts.len() as f64;
    let center = |b: usize| lo + (b as f64 + 0.5) * w;
    let n: f64 = counts.iter().sum();
    if n < 2.0 {
        return None;
    }
    let mean: f64 = counts
        .iter()
        .enumerate()
        .map(|(b, c)| c * center(b))
        .sum::<f64>()
        / n;
    let var: f64 = counts
        .iter()
        .enumerate()
        .map(|(b, c)| c * (center(b) - mean).powi(2))
        .sum::<f64>()
        / (n - 1.0);
    let h: f64 = var.sqrt() * n.powf(-0.2);
    (h > 0.0 && h.is_finite()).then_some(h)
}

/// The density at each of `xs`: every bin's rows spread with a Gaussian of width `h` (a binned
/// KDE), so the curve's area is 1.
pub(crate) fn kde(counts: &[f64], (lo, hi): (f64, f64), h: f64, xs: &[f64]) -> Vec<f64> {
    let w: f64 = (hi - lo) / counts.len() as f64;
    let n: f64 = counts.iter().sum();
    let norm: f64 = 1.0 / (n * h * TAU.sqrt());
    let bins: Vec<(f64, f64)> = counts
        .iter()
        .enumerate()
        .filter(|(_, c)| **c > 0.0)
        .map(|(b, &c)| (lo + (b as f64 + 0.5) * w, c))
        .collect();
    xs.iter()
        .map(|&x| {
            let spread = |&(m, c): &(f64, f64)| c * (-0.5 * ((x - m) / h).powi(2)).exp();
            bins.iter().map(spread).sum::<f64>() * norm
        })
        .collect()
}

/// A curve's x values: from `cut` bandwidths below the lowest row to `cut` above the highest
/// (seaborn: `cut=3` for kdeplot, `cut=2` for violins).
pub(crate) fn grid(counts: &[f64], (lo, hi): (f64, f64), h: f64, cut: f64) -> Vec<f64> {
    let w: f64 = (hi - lo) / counts.len() as f64;
    let first: usize = counts.iter().position(|c| *c > 0.0).unwrap_or(0);
    let last: usize = counts.iter().rposition(|c| *c > 0.0).unwrap_or(0);
    linspace(
        lo + first as f64 * w - cut * h,
        lo + (last + 1) as f64 * w + cut * h,
    )
}

/// `GRID` evenly spaced values from `a` to `b`.
pub(crate) fn linspace(a: f64, b: f64) -> Vec<f64> {
    (0..GRID)
        .map(|i| a + (b - a) * i as f64 / (GRID - 1) as f64)
        .collect()
}

/// Counts of `values` in `bins` equal bins over [lo, hi].
pub(crate) fn bin(
    values: impl Iterator<Item = f64>,
    (lo, hi): (f64, f64),
    bins: usize,
) -> Vec<f64> {
    let span: f64 = (hi - lo).max(1e-12);
    let mut counts: Vec<f64> = vec![0.0; bins];
    for v in values.filter(|v| v.is_finite()) {
        let b: usize = ((v - lo) / span * bins as f64) as usize;
        counts[b.min(bins - 1)] += 1.0;
    }
    counts
}

/// A curve with a halo in the card color, so it shows over bars of its own color.
pub(crate) fn halo_path(out: &mut String, theme: &Theme, d: &str, color: &str, hover: &str) {
    out.push_str(&format!(
        r##"<g><title>{}</title><path d="{d}" fill="none" stroke="{}" stroke-width="4.5" stroke-opacity="0.8"/><path d="{d}" fill="none" stroke="{color}" stroke-width="2"/></g>"##,
        escape(hover),
        theme.background
    ));
}

/// The standard normal quantile (Acklam's approximation, relative error below 1.2e-9).
pub(crate) fn normal_quantile(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.38357751867269e2,
        -3.066479806614716e1,
        2.506628277459239,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838,
        -2.549732539343734,
        4.374664141464968,
        2.938163982698783,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996,
        3.754408661907416,
    ];
    let tail = |q: f64| {
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    if p < 0.02425 {
        tail((-2.0 * p.ln()).sqrt())
    } else if p > 1.0 - 0.02425 {
        -tail((-2.0 * (1.0 - p).ln()).sqrt())
    } else {
        let (q, r): (f64, f64) = (p - 0.5, (p - 0.5) * (p - 0.5));
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    }
}

/// Where the i-th of n sorted values sits on a normal Q-Q plot (scipy `probplot`'s Filliben
/// positions).
fn filliben(i: usize, n: usize) -> f64 {
    let last: f64 = 0.5f64.powf(1.0 / n as f64);
    match i {
        0 => 1.0 - last,
        _ if i == n - 1 => last,
        _ => (i as f64 + 1.0 - 0.3175) / (n as f64 + 0.365),
    }
}

impl Dataset {
    /// A smooth density curve of `column`, one per `hue` group. (seaborn: `kdeplot`)
    ///
    /// It smooths the cached 1,000 fine bins of [`Dataset::hist`] with Scott's bandwidth, like
    /// scipy. Like seaborn, each group's curve is scaled by its share of the rows, so the areas
    /// add up to 1. A group with fewer than 2 distinct values has no curve.
    ///
    /// Options: `.title`, `.subtitle`, `.xlabel`, `.ylabel`, `.theme`, `.size`, `.palette`,
    /// `.colors`, `.color`, `.color_map`, `.hue_order`, `.xlim`, `.ylim`, `.hline`, `.vline`
    /// and `.legend`.
    ///
    /// # Panics
    ///
    /// If a column doesn't exist, `column` isn't numeric, or no group has a shape.
    /// [`Dataset::try_kdeplot`] returns these as errors.
    ///
    /// # Examples
    ///
    /// The shape of CGPA:
    /// ```
    /// use rusmlai::prelude::*;
    ///
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.kdeplot("cgpa", ""); // saves metadata/plots/Density cgpa.png
    /// ```
    ///
    /// One curve for passed and one for failed, with most options:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds
    ///     .kdeplot("cgpa", "passed")
    ///     .title("CGPA density")
    ///     .subtitle("the areas add up to 1")
    ///     .xlabel("CGPA")
    ///     .ylabel("density")
    ///     .theme("light")
    ///     .size(800.0, 450.0)
    ///     .palette("colorblind")
    ///     .color_map(&[("false", "tomato")])
    ///     .hue_order(&["true", "false"])
    ///     .xlim(4.0, 10.0)
    ///     .ylim(0.0, 0.6)
    ///     .vline(6.0, "pass mark")
    ///     .hline(0.1, "")
    ///     .legend("bottom")
    ///     .svg(); // the SVG text, and no auto-save
    /// assert!(svg.contains("CGPA density"));
    /// assert!(svg.contains("#FF6347")); // tomato
    /// ```
    ///
    /// Edge case: every name belongs to one student, so no group has a shape:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.kdeplot("cgpa", "name"); // panics: cgpa has a single value, so it has no shape to draw
    /// ```
    #[track_caller]
    pub fn kdeplot(&self, column: &str, hue: &str) -> Figure {
        self.try_kdeplot(column, hue).or_fatal()
    }

    /// [`Dataset::kdeplot`] that returns an error instead of panicking.
    ///
    /// # Errors
    ///
    /// - `PlotError::Dataset` when a column doesn't exist or `column` isn't numeric.
    /// - `PlotError::NoSpread` when no group has 2 or more distinct values.
    /// - `PlotError::EmptyResult` when the column has no finite values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # use rusmlai::plot::error::PlotError;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.try_kdeplot("age", "batch")?.svg();
    /// assert!(svg.contains("Density: age"));
    /// # Ok::<(), PlotError>(())
    /// ```
    ///
    /// A column that doesn't exist:
    /// ```
    /// # use rusmlai::prelude::*;
    /// use rusmlai::{dataset::error::DatasetError, plot::error::PlotError};
    ///
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let result = ds.try_kdeplot("gpa", "");
    /// assert!(matches!(
    ///     result,
    ///     Err(PlotError::Dataset(DatasetError::ColumnNotFound { .. }))
    /// ));
    /// ```
    pub fn try_kdeplot(&self, column: &str, hue: &str) -> Result<Figure, PlotError> {
        let hue_column: Option<&str> = (!hue.is_empty()).then_some(hue);
        let result: PlotResult = fine_bins(self, column, hue_column, MAX_BINS)?;
        let PlotResult::Hist { lo, hi, counts, .. } = &result else {
            unreachable!("histograms hold Hist results");
        };
        if !counts.iter().any(|c| scott(c, (*lo, *hi)).is_some()) {
            return Err(PlotError::NoSpread(column.to_string()));
        }
        let texts: Texts = Texts {
            title: format!("Density: {column}"),
            x: column.to_string(),
            y: "density".to_string(),
            value: "density".to_string(),
            hue: hue.to_string(),
            ..Texts::default()
        };
        Ok(self.figure("kdeplot", texts, Body::Kde(result)))
    }

    /// The shape of `y` in each `x` category, as a mirrored density with a small box inside,
    /// split by `hue`. (seaborn: `violinplot`)
    ///
    /// The inner box is the cached [`Dataset::boxplot`] of all rows. The shapes come from up to
    /// 50,000 sampled rows. A group with fewer than 2 distinct values draws only its box.
    /// `x = ""` draws one violin.
    ///
    /// Options: the ones of [`Dataset::boxplot`].
    ///
    /// # Panics
    ///
    /// If a column doesn't exist, `y` isn't numeric, `hue` is set without `x`, or no row has
    /// values. [`Dataset::try_violinplot`] returns these as errors.
    ///
    /// # Examples
    ///
    /// CGPA in each batch:
    /// ```
    /// use rusmlai::prelude::*;
    ///
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.violinplot("batch", "cgpa", ""); // saves metadata/plots/Violin batch vs cgpa.png
    /// ```
    ///
    /// Split by `passed`, with most options:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds
    ///     .violinplot("batch", "cgpa", "passed")
    ///     .title("CGPA by batch")
    ///     .subtitle("passed and failed")
    ///     .xlabel("batch")
    ///     .ylabel("CGPA")
    ///     .theme("light")
    ///     .colors(&["seagreen", "tomato"])
    ///     .size(800.0, 480.0)
    ///     .sort("name")
    ///     .top(2)
    ///     .hue_order(&["true", "false"])
    ///     .ylim(4.0, 10.0)
    ///     .hline(6.0, "pass mark")
    ///     .legend("bottom")
    ///     .svg(); // the SVG text, and no auto-save
    /// assert!(svg.contains("CGPA by batch"));
    /// assert!(svg.contains("#2E8B57")); // seagreen
    /// ```
    ///
    /// Edge case: one row per name, so each violin is only its box, a flat line at that
    /// student's CGPA:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.violinplot("name", "cgpa", "").svg();
    /// assert!(svg.contains("name = Aarav Shah\nupper fence = 8.10"));
    /// ```
    #[track_caller]
    pub fn violinplot(&self, x: &str, y: &str, hue: &str) -> Figure {
        self.try_violinplot(x, y, hue).or_fatal()
    }

    /// [`Dataset::violinplot`] that returns an error instead of panicking.
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
    /// let svg = ds.try_violinplot("", "age", "")?.svg();
    /// assert!(svg.contains("Violin: age"));
    /// # Ok::<(), PlotError>(())
    /// ```
    ///
    /// A hue without an x column:
    /// ```
    /// # use rusmlai::prelude::*;
    /// use rusmlai::plot::error::PlotError;
    ///
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let result = ds.try_violinplot("", "cgpa", "batch");
    /// assert!(matches!(result, Err(PlotError::HueNeedsX)));
    /// ```
    pub fn try_violinplot(&self, x: &str, y: &str, hue: &str) -> Result<Figure, PlotError> {
        let (result, x, hue): (PlotResult, Option<String>, Option<String>) =
            boxes(self, x, y, hue)?;
        let PlotResult::Boxes {
            x_labels,
            hue_labels,
            ..
        } = &result
        else {
            unreachable!("box plots hold Boxes results");
        };
        let columns: Vec<&str> = [Some(y), x.as_deref(), hue.as_deref()]
            .into_iter()
            .flatten()
            .collect();
        let (sample, rows): (DataFrame, usize) = sample(self, &columns)?;
        let ys: Vec<f64> = f64s(&sample, y)?;
        let (lo, hi): (f64, f64) = extent(ys.iter().copied());
        let mut groups: Vec<Vec<Vec<f64>>> =
            vec![vec![Vec::new(); hue_labels.len().max(1)]; x_labels.len()];
        let cells = slots(&sample, x.as_deref(), x_labels)?
            .into_iter()
            .zip(slots(&sample, hue.as_deref(), hue_labels)?);
        for ((i, j), v) in cells.zip(ys) {
            if let (Some(i), Some(j)) = (i, j) {
                groups[i][j].push(v);
            }
        }
        let shapes: Vec<Vec<Vec<(f64, f64)>>> = groups
            .iter()
            .map(|row| {
                row.iter()
                    .map(|values| {
                        let counts: Vec<f64> = bin(values.iter().copied(), (lo, hi), KDE_BINS);
                        let Some(h) = scott(&counts, (lo, hi)) else {
                            return Vec::new(); // fewer than 2 rows or one value: the box only
                        };
                        let ys: Vec<f64> = grid(&counts, (lo, hi), h, 2.0);
                        let ds: Vec<f64> = kde(&counts, (lo, hi), h, &ys);
                        ys.into_iter().zip(ds).collect()
                    })
                    .collect()
            })
            .collect();
        let body: Body = Body::Violin {
            result,
            shapes,
            shown: sample.height(),
            rows,
        };
        Ok(self.figure("violinplot", spread_texts("Violin", x, y, hue), body))
    }

    /// The sorted values of `column` against normal quantiles, with the line of a normal
    /// distribution with the column's mean and std: points on the line mean "normal".
    /// (statsmodels: `qqplot(line="s")`, scipy: `probplot`)
    ///
    /// It sorts the whole column once and keeps up to 50,000 evenly spaced values, the lowest
    /// and highest included. `.sample(n)` draws n of them (1,000 by default).
    ///
    /// Options: `.title`, `.subtitle`, `.xlabel`, `.ylabel`, `.theme`, `.size`, `.palette`,
    /// `.colors`, `.color`, `.xlim`, `.ylim`, `.hline`, `.vline` and `.sample`.
    ///
    /// # Panics
    ///
    /// If the column doesn't exist, isn't numeric, or has fewer than 2 values.
    /// [`Dataset::try_qqplot`] returns these as errors.
    ///
    /// # Examples
    ///
    /// Is CGPA normal?
    /// ```
    /// use rusmlai::prelude::*;
    ///
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.qqplot("cgpa"); // saves metadata/plots/Q-Q plot cgpa.png
    /// ```
    ///
    /// With 5 points and most options:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds
    ///     .qqplot("cgpa")
    ///     .sample(5)
    ///     .title("CGPA against a normal")
    ///     .subtitle("5 of 11 values")
    ///     .xlabel("normal quantile")
    ///     .ylabel("CGPA")
    ///     .theme("light")
    ///     .size(600.0, 450.0)
    ///     .color("tab:purple")
    ///     .xlim(-2.5, 2.5)
    ///     .ylim(4.0, 10.0)
    ///     .hline(6.0, "pass mark")
    ///     .vline(0.0, "")
    ///     .svg(); // the SVG text, and no auto-save
    /// assert!(svg.contains("CGPA against a normal"));
    /// assert!(svg.contains("#9467BD")); // tab:purple
    /// ```
    ///
    /// Edge case: a text column has no quantiles:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.qqplot("batch"); // panics: column 'batch' is not numeric (found str)
    /// ```
    #[track_caller]
    pub fn qqplot(&self, column: &str) -> Figure {
        self.try_qqplot(column).or_fatal()
    }

    /// [`Dataset::qqplot`] that returns an error instead of panicking.
    ///
    /// # Errors
    ///
    /// - `PlotError::Dataset` when the column doesn't exist or isn't numeric.
    /// - `PlotError::EmptyResult` when it has fewer than 2 finite values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # use rusmlai::plot::error::PlotError;
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.try_qqplot("age")?.svg();
    /// assert!(svg.contains("Q-Q plot: age"));
    /// # Ok::<(), PlotError>(())
    /// ```
    ///
    /// A text column:
    /// ```
    /// # use rusmlai::prelude::*;
    /// use rusmlai::{dataset::error::DatasetError, plot::error::PlotError};
    ///
    /// let ds = read_csv("tests/fixtures/data/students.csv");
    /// let result = ds.try_qqplot("name");
    /// assert!(matches!(
    ///     result,
    ///     Err(PlotError::Dataset(DatasetError::ColumnNotNumeric { .. }))
    /// ));
    /// ```
    pub fn try_qqplot(&self, column: &str) -> Result<Figure, PlotError> {
        self.try_numeric_stats(column)?;
        let sorted: DataFrame = rows(self, &[column])
            .select([col(column).cast(DataType::Float64)])
            .sort([column], SortMultipleOptions::default())
            .collect()?;
        let values: Vec<f64> = f64s(&sorted, column)?;
        let n: usize = values.len();
        if n < 2 {
            return Err(PlotError::EmptyResult(column.to_string()));
        }
        // The line through the finite values: the cached mean is NaN when the column holds NaN.
        let mean: f64 = values.iter().sum::<f64>() / n as f64;
        let std: f64 =
            (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();
        let points: Vec<(f64, f64)> = spread(n, MAX_SAMPLE)
            .map(|i| (normal_quantile(filliben(i, n)), values[i]))
            .collect();
        let texts: Texts = Texts {
            title: format!("Q-Q plot: {column}"),
            x: "normal quantile".to_string(),
            y: column.to_string(),
            value: column.to_string(),
            ..Texts::default()
        };
        let body: Body = Body::Qq {
            points,
            line: (mean, std),
            rows: n,
        };
        Ok(self.figure("qqplot", texts, body))
    }
}

/// `k` evenly spaced indices of `0..n`, both ends included (all of them when n <= k).
fn spread(n: usize, k: usize) -> impl Iterator<Item = usize> {
    let k: usize = k.min(n);
    (0..k).map(move |i| if k < 2 { 0 } else { i * (n - 1) / (k - 1) })
}

/// Density curves (lines), one per hue group, over the data range or `.xlim`.
pub(crate) fn render_kde(texts: &Texts, result: &PlotResult, style: &Style) -> String {
    let PlotResult::Hist {
        lo,
        hi,
        names,
        counts,
    } = result
    else {
        unreachable!("density charts hold Hist results");
    };
    let theme: Theme = style.theme();
    let grouped: bool = !texts.hue.is_empty();
    let cols: Vec<usize> = if grouped {
        ordered(names, style.hue_order.as_deref())
    } else {
        vec![0]
    };
    let total: f64 = cols.iter().map(|&j| counts[j].iter().sum::<f64>()).sum();
    let color = |s: usize, j: usize| {
        if grouped {
            style.group_color(s, &names[j])
        } else {
            style.single_color(&names[j])
        }
    };
    // (palette slot, group, points): seaborn's common_norm scales each curve by its share.
    let curves: Vec<Curve> = cols
        .iter()
        .enumerate()
        .filter_map(|(s, &j)| {
            let h: f64 = scott(&counts[j], (*lo, *hi))?;
            let xs: Vec<f64> = match style.xlim {
                Some((a, b)) => linspace(a, b),
                None => grid(&counts[j], (*lo, *hi), h, 3.0),
            };
            let share: f64 = counts[j].iter().sum::<f64>() / total;
            let ds: Vec<f64> = kde(&counts[j], (*lo, *hi), h, &xs);
            Some((
                s,
                j,
                xs.into_iter()
                    .zip(ds.into_iter().map(|d| d * share))
                    .collect(),
            ))
        })
        .collect();
    let items: Vec<(String, String)> = if grouped {
        curves
            .iter()
            .map(|(s, j, _)| (names[*j].clone(), color(*s, *j)))
            .collect()
    } else {
        Vec::new()
    };
    let subtitle: &str = texts.subtitle(style);
    let l: Layout = layout(style, subtitle, 1, 0.0, &items);
    let (x_lo, x_hi): (f64, f64) = style.xlim.unwrap_or_else(|| {
        let xs = curves.iter().flat_map(|c| c.2.iter().map(|p| p.0));
        extent(xs.chain(style.vlines.iter().map(|line| line.0)))
    });
    let (y_lo, y_hi): (f64, f64) =
        range(style, curves.iter().flat_map(|c| c.2.iter().map(|p| p.1)));
    let scale: YScale = YScale::new(y_lo, y_hi, l.top, l.plot_h);
    let xscale: XScale = XScale::new(x_lo, x_hi, l.left, l.plot_w);

    let mut out: String = svg::card(&theme, &texts.title(style), subtitle, l.width, l.height);
    scale.draw(&mut out, &theme, l.left, l.width - l.right);
    xscale.draw(&mut out, &theme, scale.bottom() + 22.0);
    for (s, j, points) in &curves {
        let d: String = svg::polyline(points.iter().map(|&(x, y)| (xscale.x(x), scale.y(y))));
        let mut hover: String = format!("density of {}", texts.x);
        if grouped {
            hover += &format!("\n{} = {}", texts.hue, names[*j]);
        }
        halo_path(&mut out, &theme, &d, &color(*s, *j), &hover);
    }
    hlines(&mut out, &theme, style, &scale, &l);
    for (v, label) in &style.vlines {
        svg::ref_line(
            &mut out,
            &theme,
            false,
            xscale.x(*v),
            l.top,
            scale.bottom(),
            label,
            *v,
        );
    }
    finish(&mut out, &theme, style, texts.names(style), &items, &l);
    out
}

/// Mirrored density shapes per category (and hue group), each with a small box plot inside.
/// Every violin has the same area (seaborn's `density_norm="area"`).
pub(crate) fn render_violin(
    texts: &Texts,
    result: &PlotResult,
    shapes: &[Vec<Vec<(f64, f64)>>],
    shown: usize,
    rows_with_values: usize,
    style: &Style,
) -> String {
    let PlotResult::Boxes {
        x_labels,
        hue_labels,
        boxes,
        ..
    } = result
    else {
        unreachable!("violins hold Boxes results");
    };
    let theme: Theme = style.theme();
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
    let shown_labels: Vec<&String> = rows.iter().map(|&i| &x_labels[i]).collect();
    let min_slot: f64 = if grouped {
        k as f64 * MIN_CLUSTER_BAR / 0.62
    } else {
        56.0
    }
    .max(svg::label_width(&shown_labels));
    let items: Vec<(String, String)> = if grouped {
        legend_items(style, hue_labels, &cols)
    } else {
        Vec::new()
    };
    let subtitle: String = sample_subtitle(style, &texts.subtitle, shown, rows_with_values);
    let mut l: Layout = layout(style, &subtitle, rows.len(), min_slot, &items);
    let (lo, hi): (f64, f64) = range(
        style,
        rows.iter().flat_map(|&i| {
            cols.iter().flat_map(move |&j| {
                let fences = [boxes[i][j][0], boxes[i][j][4]];
                fences.into_iter().chain(shapes[i][j].iter().map(|p| p.0))
            })
        }),
    );
    let scale: YScale = YScale::new(lo, hi, l.top, l.plot_h);
    let slot: f64 = l.plot_w / rows.len().max(1) as f64;
    let tick: Tick = tick_mode(&mut l, &shown_labels, slot);
    let width: f64 = if grouped {
        slot * 0.8 / k as f64
    } else {
        slot * 0.7
    };
    let peak: f64 = shapes
        .iter()
        .flatten()
        .flatten()
        .map(|p| p.1)
        .fold(0.0, f64::max)
        .max(1e-300);

    let mut out: String = svg::card(&theme, &texts.title(style), &subtitle, l.width, l.height);
    scale.draw(&mut out, &theme, l.left, l.width - l.right);
    for (n, &i) in rows.iter().enumerate() {
        let slot_x: f64 = l.left + n as f64 * slot;
        let start: f64 = slot_x + (slot - width * k as f64) / 2.0;
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
            let cx: f64 = start + (s as f64 + 0.5) * width;
            let half = |d: f64| d / peak * width * 0.48;
            let shape: &[(f64, f64)] = &shapes[i][j];
            if !shape.is_empty() {
                let left = shape.iter().map(|&(y, d)| (cx - half(d), scale.y(y)));
                let right = shape.iter().rev().map(|&(y, d)| (cx + half(d), scale.y(y)));
                out.push_str(&format!(
                    r##"<path d="{} Z" fill="{color}" fill-opacity="0.7" stroke="{color}" stroke-width="1.2"><title>{}</title></path>"##,
                    svg::polyline(left.chain(right)),
                    escape(&hover)
                ));
            }
            // The inner box: whiskers to the fences, a thick bar from q25 to q75, the median dot.
            let [low, q25, median, q75, high] = five.map(|v| scale.y(v));
            let ink: &str = theme.value;
            out.push_str(&format!(
                r##"<g><title>{}</title><line x1="{cx:.2}" y1="{low:.2}" x2="{cx:.2}" y2="{high:.2}" stroke="{ink}" stroke-width="1.2"/><line x1="{cx:.2}" y1="{q25:.2}" x2="{cx:.2}" y2="{q75:.2}" stroke="{ink}" stroke-width="5"/><circle cx="{cx:.2}" cy="{median:.2}" r="2.6" fill="{}"/></g>"##,
                escape(&hover),
                theme.background
            ));
        }
        let cx: f64 = slot_x + slot / 2.0;
        svg::x_tick(
            &mut out,
            &theme,
            &x_labels[i],
            cx,
            scale.bottom() + 22.0,
            slot,
            tick,
        );
    }
    hlines(&mut out, &theme, style, &scale, &l);
    finish(&mut out, &theme, style, texts.names(style), &items, &l);
    out
}

/// Sorted values against normal quantiles, and the line of a normal with the column's mean and
/// std. `.sample(n)` draws n evenly spaced points, the lowest and highest included.
pub(crate) fn render_qq(
    texts: &Texts,
    points: &[(f64, f64)],
    (mean, std): (f64, f64),
    rows: usize,
    style: &Style,
) -> String {
    let theme: Theme = style.theme();
    let drawn: Vec<(f64, f64)> = spread(points.len(), style.sample.unwrap_or(QQ_POINTS))
        .map(|i| points[i])
        .collect();
    let subtitle: String = sample_subtitle(style, &texts.subtitle, drawn.len(), rows);
    let l: Layout = layout(style, &subtitle, 1, 0.0, &[]);
    let (x_lo, x_hi): (f64, f64) = style.xlim.unwrap_or_else(|| {
        extent(
            drawn
                .iter()
                .map(|p| p.0)
                .chain(style.vlines.iter().map(|line| line.0)),
        )
    });
    let (y_lo, y_hi): (f64, f64) = style.ylim.unwrap_or_else(|| {
        let values = drawn.iter().map(|p| p.1);
        extent(values.chain(style.hlines.iter().map(|line| line.0)))
    });
    let scale: YScale = YScale::new(y_lo, y_hi, l.top, l.plot_h);
    let xscale: XScale = XScale::new(x_lo, x_hi, l.left, l.plot_w);
    let color: String = style.single_color(&texts.y);

    let mut out: String = svg::card(&theme, &texts.title(style), &subtitle, l.width, l.height);
    scale.draw(&mut out, &theme, l.left, l.width - l.right);
    xscale.draw(&mut out, &theme, scale.bottom() + 22.0);
    for &(z, v) in &drawn {
        out.push_str(&format!(
            r##"<circle cx="{:.2}" cy="{:.2}" r="2.6" fill="{color}" fill-opacity="0.75"><title>{}</title></circle>"##,
            xscale.x(z),
            scale.y(v),
            escape(&format!(
                "{} = {}\n{} = {}",
                texts.x,
                svg::format_value(z),
                texts.y,
                svg::format_value(v)
            ))
        ));
    }
    // The part of the line inside the plot. Pinning its ends to the edges would bend its slope.
    // No line when the spread overflows (values near ±1e300): its ends would be NaN.
    let (mut a, mut b): (f64, f64) = (x_lo, x_hi);
    if std > 0.0 {
        a = a.max((y_lo - mean) / std);
        b = b.min((y_hi - mean) / std);
    }
    if mean.is_finite() && std.is_finite() {
        out.push_str(&format!(
        r##"<line x1="{:.2}" y1="{:.2}" x2="{:.2}" y2="{:.2}" stroke="{}" stroke-width="2"><title>{}</title></line>"##,
        xscale.x(a),
        scale.y(mean + std * a),
        xscale.x(b),
        scale.y(mean + std * b),
        theme.fit,
        escape(&format!(
            "normal: {} = {} + {} × {}",
            texts.y,
            svg::format_value(mean),
            svg::format_value(std),
            texts.x
        ))
    ));
    }
    hlines(&mut out, &theme, style, &scale, &l);
    for (v, label) in &style.vlines {
        svg::ref_line(
            &mut out,
            &theme,
            false,
            xscale.x(*v),
            l.top,
            scale.bottom(),
            label,
            *v,
        );
    }
    finish(&mut out, &theme, style, texts.names(style), &[], &l);
    out
}

const QQ_POINTS: usize = 1000; // points a Q-Q plot draws by default

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kde_matches_scipy_and_has_area_one() {
        // Normal quantiles from scipy.stats.norm.ppf.
        assert!((normal_quantile(0.975) - 1.959_963_985).abs() < 1e-8);
        assert!((normal_quantile(0.001) + 3.090_232_306).abs() < 1e-8);
        assert!(normal_quantile(0.5).abs() < 1e-12);
        // Rows 0..=9 once each: scipy's gaussian_kde uses std 3.0277 · 10^(-1/5) = 1.9104.
        let counts: Vec<f64> = bin((0..10).map(f64::from), (0.0, 9.0), 1000);
        let h: f64 = scott(&counts, (0.0, 9.0)).unwrap();
        assert!((h - 1.9104).abs() < 0.01, "{h}");
        let xs: Vec<f64> = grid(&counts, (0.0, 9.0), h, 6.0);
        let step: f64 = xs[1] - xs[0];
        let area: f64 = kde(&counts, (0.0, 9.0), h, &xs).iter().sum::<f64>() * step;
        assert!((area - 1.0).abs() < 0.01, "{area}");
    }
}
