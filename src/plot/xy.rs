//! `scatter`, `line` and `regplot`: one numeric column against another.

use polars::prelude::{DataFrame, DataType, Expr, SortMultipleOptions, col};

use crate::{
    dataset::Dataset,
    display::{Figure, escape},
    error::OrFatal,
    plot::{
        SAMPLE, VALUE,
        chart::{Body, Texts},
        check_columns,
        error::PlotError,
        f64s, labels, optional, rows, sample, sample_subtitle, slots,
        style::{Style, Theme},
        svg::{self, Layout, XScale, YScale, extent, finish, hlines, layout, ordered, range},
        thin, unique,
    },
};

const MAX_LINE_POINTS: usize = 2000;

impl Dataset {
    /// One dot per row: `y` against `x`, colored by `hue`. (seaborn: `scatterplot`)
    ///
    /// It draws 2,500 sampled rows by default, from up to 50,000 kept (one random row from each
    /// run of rows). The seed is fixed, so the same data draws the same picture. Rows with a
    /// null in any of the columns are left out. Raw rows aren't cached.
    ///
    /// Options: `.title`, `.subtitle`, `.xlabel`, `.ylabel`, `.theme`, `.size`, `.palette`,
    /// `.colors`, `.color`, `.color_map`, `.hue_order`, `.xlim`, `.ylim`, `.sample`, `.hline`,
    /// `.vline` and `.legend`.
    ///
    /// # Panics
    ///
    /// If a column doesn't exist, `x` or `y` isn't numeric, or no row has values.
    /// [`Dataset::try_scatter`] returns these as errors.
    ///
    /// # Examples
    ///
    /// CGPA against age:
    /// ```
    /// use rusmlai::prelude::*;
    ///
    /// let ds = read_csv("tests/data/students.csv");
    /// ds.scatter("age", "cgpa", ""); // saves metadata/plots/Scatter age vs cgpa.png
    /// ```
    ///
    /// Colored by `passed`, with 8 sampled dots and most options:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/data/students.csv");
    /// let svg = ds
    ///     .scatter("age", "cgpa", "passed")
    ///     .sample(8)
    ///     .title("Age and CGPA")
    ///     .subtitle("8 of 10 students")
    ///     .xlabel("age (years)")
    ///     .ylabel("CGPA")
    ///     .theme("light")
    ///     .size(700.0, 450.0)
    ///     .palette("bright")
    ///     .color_map(&[("false", "tomato")])
    ///     .hue_order(&["true", "false"])
    ///     .xlim(18.0, 25.0)
    ///     .ylim(4.0, 10.0)
    ///     .hline(6.0, "pass mark")
    ///     .vline(21.0, "")
    ///     .legend("bottom")
    ///     .svg(); // the SVG text, and no auto-save
    /// assert!(svg.contains("Age and CGPA"));
    /// assert!(svg.contains("#FF6347")); // tomato
    /// ```
    ///
    /// Edge case: one row has no age and another no CGPA, so 10 of the 12 rows get a dot:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/data/students.csv");
    /// let svg = ds.scatter("age", "cgpa", "").svg();
    /// assert_eq!(svg.matches("<circle").count(), 10);
    /// ```
    #[track_caller]
    pub fn scatter(&self, x: &str, y: &str, hue: &str) -> Figure {
        self.try_scatter(x, y, hue).or_fatal()
    }

    /// [`Dataset::scatter`] that returns an error instead of panicking.
    ///
    /// # Errors
    ///
    /// - `PlotError::Dataset` when a column doesn't exist, or `x` or `y` isn't numeric.
    /// - `PlotError::EmptyResult` when no row has values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # use rusmlai::plot::error::PlotError;
    /// let ds = read_csv("tests/data/students.csv");
    /// let svg = ds.try_scatter("id", "cgpa", "batch")?.svg();
    /// assert!(svg.contains("Scatter: id vs cgpa"));
    /// # Ok::<(), PlotError>(())
    /// ```
    ///
    /// A column that doesn't exist:
    /// ```
    /// # use rusmlai::prelude::*;
    /// use rusmlai::{dataset::error::DatasetError, plot::error::PlotError};
    ///
    /// let ds = read_csv("tests/data/students.csv");
    /// let result = ds.try_scatter("age", "gpa", "");
    /// assert!(matches!(
    ///     result,
    ///     Err(PlotError::Dataset(DatasetError::ColumnNotFound { .. }))
    /// ));
    /// ```
    pub fn try_scatter(&self, x: &str, y: &str, hue: &str) -> Result<Figure, PlotError> {
        try_sampled(self, "scatter", x, y, hue)
    }

    /// A scatter plus a least-squares line. (seaborn: `regplot`)
    ///
    /// The line is fitted on all rows with both values, not only the sampled ones. With a `hue`,
    /// the dots get the group colors and the one line still fits every row. The dots are
    /// sampled like [`Dataset::scatter`].
    ///
    /// Options: the ones of [`Dataset::scatter`].
    ///
    /// # Panics
    ///
    /// If a column doesn't exist, `x` or `y` isn't numeric, or no row has values.
    /// [`Dataset::try_regplot`] returns these as errors.
    ///
    /// # Examples
    ///
    /// CGPA against age, with the fitted line:
    /// ```
    /// use rusmlai::prelude::*;
    ///
    /// let ds = read_csv("tests/data/students.csv");
    /// ds.regplot("age", "cgpa", ""); // saves metadata/plots/Regplot age vs cgpa.png
    /// ```
    ///
    /// With most options:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/data/students.csv");
    /// let svg = ds
    ///     .regplot("age", "cgpa", "batch")
    ///     .sample(10)
    ///     .title("Does age predict CGPA?")
    ///     .subtitle("least squares on all rows")
    ///     .xlabel("age")
    ///     .ylabel("CGPA")
    ///     .theme("light")
    ///     .size(700.0, 450.0)
    ///     .colors(&["#1F77B4", "tab:orange", "C2"])
    ///     .hue_order(&["B13", "B14", "B15"])
    ///     .xlim(18.0, 25.0)
    ///     .ylim(4.0, 10.0)
    ///     .hline(6.0, "pass mark")
    ///     .vline(21.0, "")
    ///     .legend("right")
    ///     .svg(); // the SVG text, and no auto-save
    /// assert!(svg.contains("Does age predict CGPA?"));
    /// assert!(svg.contains("#FF7F0E")); // tab:orange
    /// ```
    ///
    /// Edge case: a column against itself fits the line y = x:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/data/students.csv");
    /// let svg = ds.regplot("age", "age", "").svg();
    /// assert!(svg.contains("Regplot: age vs age"));
    /// ```
    #[track_caller]
    pub fn regplot(&self, x: &str, y: &str, hue: &str) -> Figure {
        self.try_regplot(x, y, hue).or_fatal()
    }

    /// [`Dataset::regplot`] that returns an error instead of panicking.
    ///
    /// # Errors
    ///
    /// - `PlotError::Dataset` when a column doesn't exist, or `x` or `y` isn't numeric.
    /// - `PlotError::EmptyResult` when no row has values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # use rusmlai::plot::error::PlotError;
    /// let ds = read_csv("tests/data/students.csv");
    /// let svg = ds.try_regplot("id", "age", "")?.svg();
    /// assert!(svg.contains("Regplot: id vs age"));
    /// # Ok::<(), PlotError>(())
    /// ```
    ///
    /// A text column:
    /// ```
    /// # use rusmlai::prelude::*;
    /// use rusmlai::{dataset::error::DatasetError, plot::error::PlotError};
    ///
    /// let ds = read_csv("tests/data/students.csv");
    /// let result = ds.try_regplot("batch", "cgpa", "");
    /// assert!(matches!(
    ///     result,
    ///     Err(PlotError::Dataset(DatasetError::ColumnNotNumeric { .. }))
    /// ));
    /// ```
    pub fn try_regplot(&self, x: &str, y: &str, hue: &str) -> Result<Figure, PlotError> {
        try_sampled(self, "regplot", x, y, hue)
    }

    /// The mean of `y` at each `x`, as a line sorted by `x`, one line per `hue` group.
    /// (seaborn: `lineplot`)
    ///
    /// Past 2,000 points a line keeps the lowest and highest `y` of each run of points, so
    /// spikes stay visible. Nothing is sampled, so `.sample` doesn't apply.
    ///
    /// Options: `.title`, `.subtitle`, `.xlabel`, `.ylabel`, `.theme`, `.size`, `.palette`,
    /// `.colors`, `.color`, `.color_map`, `.hue_order`, `.xlim`, `.ylim`, `.hline`, `.vline`
    /// and `.legend`.
    ///
    /// # Panics
    ///
    /// If a column doesn't exist, `x` or `y` isn't numeric, or no row has values.
    /// [`Dataset::try_line`] returns these as errors.
    ///
    /// # Examples
    ///
    /// The mean CGPA at each age:
    /// ```
    /// use rusmlai::prelude::*;
    ///
    /// let ds = read_csv("tests/data/students.csv");
    /// ds.line("age", "cgpa", ""); // saves metadata/plots/Line mean cgpa vs age.png
    /// ```
    ///
    /// One line for passed and one for failed, with most options:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/data/students.csv");
    /// let svg = ds
    ///     .line("age", "cgpa", "passed")
    ///     .title("CGPA by age")
    ///     .subtitle("mean at each age")
    ///     .xlabel("age")
    ///     .ylabel("mean CGPA")
    ///     .theme("light")
    ///     .size(700.0, 450.0)
    ///     .colors(&["seagreen", "tomato"])
    ///     .hue_order(&["true", "false"])
    ///     .xlim(18.0, 25.0)
    ///     .ylim(4.0, 10.0)
    ///     .hline(6.0, "pass mark")
    ///     .vline(21.0, "")
    ///     .legend("bottom")
    ///     .svg(); // the SVG text, and no auto-save
    /// assert!(svg.contains("CGPA by age"));
    /// assert!(svg.contains("#FF6347")); // tomato
    /// ```
    ///
    /// Edge case: three students are 19, so the line passes through their mean, 8.1:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/data/students.csv");
    /// let svg = ds.line("age", "cgpa", "").svg();
    /// assert!(svg.contains("age = 19\ncgpa = 8.10"));
    /// ```
    #[track_caller]
    pub fn line(&self, x: &str, y: &str, hue: &str) -> Figure {
        self.try_line(x, y, hue).or_fatal()
    }

    /// [`Dataset::line`] that returns an error instead of panicking.
    ///
    /// # Errors
    ///
    /// - `PlotError::Dataset` when a column doesn't exist, or `x` or `y` isn't numeric.
    /// - `PlotError::EmptyResult` when no row has values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # use rusmlai::plot::error::PlotError;
    /// let ds = read_csv("tests/data/students.csv");
    /// let svg = ds.try_line("id", "cgpa", "")?.svg();
    /// assert!(svg.contains("Line: mean cgpa vs id"));
    /// # Ok::<(), PlotError>(())
    /// ```
    ///
    /// A text column:
    /// ```
    /// # use rusmlai::prelude::*;
    /// use rusmlai::{dataset::error::DatasetError, plot::error::PlotError};
    ///
    /// let ds = read_csv("tests/data/students.csv");
    /// let result = ds.try_line("age", "name", "");
    /// assert!(matches!(
    ///     result,
    ///     Err(PlotError::Dataset(DatasetError::ColumnNotNumeric { .. }))
    /// ));
    /// ```
    pub fn try_line(&self, x: &str, y: &str, hue: &str) -> Result<Figure, PlotError> {
        self.try_numeric_stats(x)?;
        self.try_numeric_stats(y)?;
        let hue: Option<String> = optional(hue);
        check_columns(self, Some(x), hue.as_deref())?;
        let keys: Vec<Expr> = unique([Some(x), hue.as_deref()].into_iter().flatten())
            .into_iter()
            .map(col)
            .collect();
        let columns: Vec<&str> = [Some(x), Some(y), hue.as_deref()]
            .into_iter()
            .flatten()
            .collect();
        let grouped: DataFrame = rows(self, &columns)
            .group_by(keys)
            .agg([col(y).cast(DataType::Float64).mean().alias(VALUE)])
            .sort([x], SortMultipleOptions::default())
            .collect()?;
        let hue_labels: Vec<String> = match &hue {
            Some(hue) => labels(self, &grouped, hue)?,
            None => Vec::new(),
        };
        let mut lines: Vec<Vec<(f64, f64)>> = vec![Vec::new(); hue_labels.len().max(1)];
        let rows = slots(&grouped, hue.as_deref(), &hue_labels)?
            .into_iter()
            .zip(f64s(&grouped, x)?.into_iter().zip(f64s(&grouped, VALUE)?));
        for (j, point) in rows {
            if let Some(j) = j {
                lines[j].push(point);
            }
        }
        let mut points: Vec<(f64, f64, usize)> = Vec::new();
        for (j, line) in lines.iter().enumerate() {
            points.extend(thin_line(line).into_iter().map(|(px, py)| (px, py, j)));
        }
        if points.is_empty() {
            return Err(PlotError::EmptyResult(format!("{x}, {y}")));
        }
        let texts: Texts = Texts {
            title: format!("Line: mean {y} vs {x}"),
            subtitle: "each point is the mean of y at that x".to_string(),
            ..xy_texts(x, y, hue)
        };
        let body: Body = Body::Xy {
            hue_labels,
            points,
            rows: 0,
            fit: None,
        };
        Ok(self.figure("line", texts, body))
    }
}

/// At most `MAX_LINE_POINTS`: past that, the lowest and highest y of each run of k points, in x
/// order, so spikes survive. At the default size the 1,000 runs outnumber the plot's pixel
/// columns, so this draws the same picture as every point (the M4 idea); LTTB keeps one point
/// per run and can drop a peak.
fn thin_line(line: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let k: usize = line.len().div_ceil(MAX_LINE_POINTS / 2).max(1);
    let mut kept: Vec<(f64, f64)> = Vec::new();
    for run in line.chunks(k) {
        let by_y = |a: &&(f64, f64), b: &&(f64, f64)| a.1.total_cmp(&b.1);
        let mut ends: [&(f64, f64); 2] = [
            run.iter().min_by(by_y).unwrap(), // chunks are never empty
            run.iter().max_by(by_y).unwrap(),
        ];
        ends.sort_by(|a, b| a.0.total_cmp(&b.0));
        kept.push(*ends[0]);
        if !std::ptr::eq(ends[0], ends[1]) {
            kept.push(*ends[1]);
        }
    }
    kept
}

/// `scatter` or `regplot`: sampled rows, plus the fit line for `regplot`.
fn try_sampled(
    dataset: &Dataset,
    name: &'static str,
    x: &str,
    y: &str,
    hue: &str,
) -> Result<Figure, PlotError> {
    dataset.try_numeric_stats(x)?;
    dataset.try_numeric_stats(y)?;
    let hue: Option<String> = optional(hue);
    check_columns(dataset, Some(x), hue.as_deref())?;
    let columns: Vec<&str> = [Some(x), Some(y), hue.as_deref()]
        .into_iter()
        .flatten()
        .collect();
    let (picked, rows): (DataFrame, usize) = sample(dataset, &columns)?;
    let hue_labels: Vec<String> = match &hue {
        Some(hue) => labels(dataset, &picked, hue)?,
        None => Vec::new(),
    };
    let xys = f64s(&picked, x)?.into_iter().zip(f64s(&picked, y)?);
    let points: Vec<(f64, f64, usize)> = slots(&picked, hue.as_deref(), &hue_labels)?
        .into_iter()
        .zip(xys)
        .filter_map(|(j, (px, py))| Some((px, py, j?)))
        .collect();
    if points.is_empty() {
        return Err(PlotError::EmptyResult(format!("{x}, {y}")));
    }
    let (fit, kind): (Option<(f64, f64)>, &str) = if name == "regplot" {
        (least_squares(dataset, x, y)?, "Regplot")
    } else {
        (None, "Scatter")
    };
    let texts: Texts = Texts {
        title: format!("{kind}: {x} vs {y}"),
        ..xy_texts(x, y, hue)
    };
    let body: Body = Body::Xy {
        hue_labels,
        points,
        rows,
        fit,
    };
    Ok(dataset.figure(name, texts, body))
}

fn xy_texts(x: &str, y: &str, hue: Option<String>) -> Texts {
    Texts {
        x: x.to_string(),
        y: y.to_string(),
        value: y.to_string(),
        hue: hue.unwrap_or_default(),
        ..Texts::default()
    }
}

/// Intercept and slope of y = a + b·x over all rows with both values, in one select.
fn least_squares(dataset: &Dataset, x: &str, y: &str) -> Result<Option<(f64, f64)>, PlotError> {
    // Cast inside each expression: a select of col(x) and col(y) fails with x = y (same name).
    let (fx, fy): (Expr, Expr) = (
        col(x).cast(DataType::Float64),
        col(y).cast(DataType::Float64),
    );
    let (dx, dy): (Expr, Expr) = (
        fx.clone() - fx.clone().mean(),
        fy.clone() - fy.clone().mean(),
    );
    let sums: DataFrame = rows(dataset, &[x, y])
        .select([
            fx.mean().alias("mx"),
            fy.mean().alias("my"),
            (dx.clone() * dy).sum().alias("sxy"),
            (dx.clone() * dx).sum().alias("sxx"),
        ])
        .collect()?;
    let [mx, my, sxy, sxx] = ["mx", "my", "sxy", "sxx"].map(|name| f64s(&sums, name).map(|v| v[0]));
    let (mx, my, sxy, sxx): (f64, f64, f64, f64) = (mx?, my?, sxy?, sxx?);
    Ok((sxx > 0.0 && sxy.is_finite()).then(|| (my - sxy / sxx * mx, sxy / sxx)))
}

/// Points (scatter, regplot) or lines with markers (line), plus the fit line for regplot.
/// `rows > 0` marks sampled points: `.sample(n)` draws n of them.
pub(crate) fn render_xy(
    name: &str,
    texts: &Texts,
    hue_labels: &[String],
    points: &[(f64, f64, usize)],
    rows: usize,
    fit: Option<(f64, f64)>,
    style: &Style,
) -> String {
    let theme: Theme = style.theme();
    let grouped: bool = !hue_labels.is_empty();
    let points: Vec<(f64, f64, usize)> = if rows > 0 {
        thin(points, style.sample.unwrap_or(SAMPLE))
    } else {
        points.to_vec()
    };
    let cols: Vec<usize> = if grouped {
        ordered(hue_labels, style.hue_order.as_deref())
    } else {
        vec![0]
    };
    let mut color_of: Vec<Option<String>> = vec![None; hue_labels.len().max(1)];
    for (s, &j) in cols.iter().enumerate() {
        color_of[j] = Some(if grouped {
            style.group_color(s, &hue_labels[j])
        } else {
            style.single_color(&texts.y)
        });
    }
    let shown = || points.iter().filter(|p| color_of[p.2].is_some());
    let (x_lo, x_hi): (f64, f64) = style.xlim.unwrap_or_else(|| {
        extent(
            shown()
                .map(|p| p.0)
                .chain(style.vlines.iter().map(|line| line.0)),
        )
    });
    let subtitle: String = sample_subtitle(style, &texts.subtitle, points.len(), rows);
    let subtitle: &str = &subtitle;
    let mut items: Vec<(String, String)> = if grouped {
        cols.iter()
            .filter_map(|&j| Some((hue_labels[j].clone(), color_of[j].clone()?)))
            .collect()
    } else {
        Vec::new()
    };
    if grouped && fit.is_some() {
        items.push(("fit".to_string(), theme.fit.to_string()));
    }
    let l: Layout = layout(style, subtitle, 1, 0.0, &items);
    // Lines use the data range like seaborn; points keep the old axis from 0.
    let (y_lo, y_hi): (f64, f64) = match (name, style.ylim) {
        ("line", None) => extent(
            shown()
                .map(|p| p.1)
                .chain(style.hlines.iter().map(|line| line.0)),
        ),
        _ => range(style, shown().map(|p| p.1)),
    };
    let scale: YScale = YScale::new(y_lo, y_hi, l.top, l.plot_h);
    let xscale: XScale = XScale::new(x_lo, x_hi, l.left, l.plot_w);

    let mut out: String = svg::card(&theme, &texts.title(style), subtitle, l.width, l.height);
    scale.draw(&mut out, &theme, l.left, l.width - l.right);
    xscale.draw(&mut out, &theme, scale.bottom() + 22.0);
    if name == "line" {
        for (j, color) in color_of.iter().enumerate() {
            let Some(color) = color else { continue };
            let d: String = points
                .iter()
                .filter(|p| p.2 == j)
                .enumerate()
                .map(|(k, p)| {
                    let cmd: &str = if k == 0 { "M" } else { "L" };
                    format!("{cmd} {:.2} {:.2} ", xscale.x(p.0), scale.y(p.1))
                })
                .collect();
            out.push_str(&format!(
                r##"<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>"##
            ));
        }
    }
    let (radius, opacity): (&str, &str) = if name == "line" {
        ("3", "1")
    } else {
        ("3.5", "0.6")
    };
    for &(px, py, j) in &points {
        let Some(color) = &color_of[j] else { continue };
        let mut hover: String = format!(
            "{} = {}\n{} = {}",
            texts.x,
            svg::format_value(px),
            texts.y,
            svg::format_value(py)
        );
        if grouped {
            hover += &format!("\n{} = {}", texts.hue, hue_labels[j]);
        }
        out.push_str(&format!(
            r##"<circle cx="{:.2}" cy="{:.2}" r="{radius}" fill="{color}" fill-opacity="{opacity}"><title>{}</title></circle>"##,
            xscale.x(px),
            scale.y(py),
            escape(&hover)
        ));
    }
    if let Some((a, b)) = fit {
        out.push_str(&format!(
            r##"<line x1="{:.2}" y1="{:.2}" x2="{:.2}" y2="{:.2}" stroke="{}" stroke-width="2.5"><title>{} = {} + {} × {}</title></line>"##,
            xscale.x(x_lo),
            scale.y(a + b * x_lo),
            xscale.x(x_hi),
            scale.y(a + b * x_hi),
            theme.fit,
            escape(&texts.y),
            svg::format_value(a),
            svg::format_value(b),
            escape(&texts.x)
        ));
    }
    hlines(&mut out, &theme, style, &scale, &l);
    for (v, label) in &style.vlines {
        let x: f64 = xscale.x(*v);
        svg::ref_line(&mut out, &theme, false, x, l.top, scale.bottom(), label, *v);
    }
    finish(&mut out, &theme, style, texts.names(style), &items, &l);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thin_keeps_spikes() {
        let mut line: Vec<(f64, f64)> = (0..10_000).map(|i| (f64::from(i), 0.0)).collect();
        line[5_001].1 = 100.0;
        let kept: Vec<(f64, f64)> = thin_line(&line);
        assert!(kept.len() <= MAX_LINE_POINTS);
        assert!(kept.contains(&(5_001.0, 100.0)));
        assert!(kept.windows(2).all(|w| w[0].0 < w[1].0));
    }
}
