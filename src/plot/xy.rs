//! `scatter`, `line` and `regplot`: one numeric column against another.

use polars::prelude::{DataFrame, DataType, Expr, IntoLazy, SortMultipleOptions, col, lit};

use crate::{
    dataset::Dataset,
    display::{Figure, escape},
    plot::{
        bar::{index, labels, strings},
        chart::{Chart, Texts},
        dist::{Layout, check_columns, f64s, finish, layout, optional, range, sample, title},
        error::PlotError,
        style::{Style, Theme},
        svg::{self, XScale, YScale},
    },
};

const VALUE: &str = "__rusmlai_value";
const MAX_LINE_POINTS: usize = 2000;

/// About 2,000 sampled rows of `y` against `x`, colored by `hue`. (seaborn: `scatterplot`)
pub(crate) fn try_scatter(
    dataset: &Dataset,
    x: &str,
    y: &str,
    hue: &str,
) -> Result<Figure, PlotError> {
    let (points, hue_labels, subtitle) = sampled_points(dataset, x, y, hue)?;
    Ok(xy_figure(
        dataset,
        "scatter",
        format!("Scatter: {x} vs {y}"),
        (x, y, hue),
        hue_labels,
        points,
        None,
        subtitle,
    ))
}

/// The scatter plus a least-squares line fitted on all rows (not only the sampled ones).
pub(crate) fn try_regplot(
    dataset: &Dataset,
    x: &str,
    y: &str,
    hue: &str,
) -> Result<Figure, PlotError> {
    let (points, hue_labels, subtitle) = sampled_points(dataset, x, y, hue)?;
    let fit: Option<(f64, f64)> = least_squares(dataset, x, y)?;
    Ok(xy_figure(
        dataset,
        "regplot",
        format!("Regplot: {x} vs {y}"),
        (x, y, hue),
        hue_labels,
        points,
        fit,
        subtitle,
    ))
}

/// The mean of `y` at each `x`, sorted by `x`, one line per `hue` group. (seaborn: `lineplot`)
pub(crate) fn try_line(
    dataset: &Dataset,
    x: &str,
    y: &str,
    hue: &str,
) -> Result<Figure, PlotError> {
    dataset.try_numeric_stats(x)?;
    dataset.try_numeric_stats(y)?;
    let hue: Option<String> = optional(hue);
    check_columns(dataset, Some(x), hue.as_deref())?;
    let mut keys: Vec<Expr> = vec![col(x)];
    let mut columns: Vec<Expr> = vec![col(x), col(y)];
    if let Some(hue) = &hue {
        keys.push(col(hue.as_str()));
        columns.push(col(hue.as_str()));
    }
    let grouped: DataFrame = dataset
        .dataframe
        .clone()
        .lazy()
        .select(columns)
        .drop_nulls(None)
        .group_by(keys)
        .agg([col(y).cast(DataType::Float64).mean().alias(VALUE)])
        .sort([x], SortMultipleOptions::default())
        .collect()?;
    let (xs, ys): (Vec<f64>, Vec<f64>) = (f64s(&grouped, x)?, f64s(&grouped, VALUE)?);
    let hue_labels: Vec<String> = match &hue {
        Some(hue) => labels(dataset, &grouped, hue)?,
        None => Vec::new(),
    };
    let hues: Option<Vec<String>> = hue.as_deref().map(|h| strings(&grouped, h)).transpose()?;
    let hue_index = index(&hue_labels);
    let mut lines: Vec<Vec<(f64, f64)>> = vec![Vec::new(); hue_labels.len().max(1)];
    for row in 0..grouped.height() {
        let j: usize = match &hues {
            Some(hues) => match hue_index.get(hues[row].as_str()) {
                Some(&j) => j,
                None => continue,
            },
            None => 0,
        };
        lines[j].push((xs[row], ys[row]));
    }
    let mut points: Vec<(f64, f64, usize)> = Vec::new();
    for (j, line) in lines.iter().enumerate() {
        points.extend(thin(line).into_iter().map(|(px, py)| (px, py, j)));
    }
    if points.is_empty() {
        return Err(PlotError::EmptyResult(format!("{x}, {y}")));
    }
    Ok(xy_figure(
        dataset,
        "line",
        format!("Line: mean {y} vs {x}"),
        (x, y, hue.as_deref().unwrap_or("")),
        hue_labels,
        points,
        None,
        "each point is the mean of y at that x".to_string(),
    ))
}

/// At most `MAX_LINE_POINTS`: past that, the lowest and highest y of each run of k points, in x
/// order, so spikes survive.
// ponytail: keeps the peaks of a run but not its shape; LTTB if long lines look off.
fn thin(line: &[(f64, f64)]) -> Vec<(f64, f64)> {
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

type Sampled = (Vec<(f64, f64, usize)>, Vec<String>, String);

fn sampled_points(dataset: &Dataset, x: &str, y: &str, hue: &str) -> Result<Sampled, PlotError> {
    dataset.try_numeric_stats(x)?;
    dataset.try_numeric_stats(y)?;
    let hue: Option<String> = optional(hue);
    check_columns(dataset, Some(x), hue.as_deref())?;
    let columns: Vec<&str> = [Some(x), Some(y), hue.as_deref()]
        .into_iter()
        .flatten()
        .collect();
    let (rows, step): (DataFrame, usize) = sample(dataset, &columns)?;
    let (xs, ys): (Vec<f64>, Vec<f64>) = (f64s(&rows, x)?, f64s(&rows, y)?);
    let hue_labels: Vec<String> = match &hue {
        Some(hue) => labels(dataset, &rows, hue)?,
        None => Vec::new(),
    };
    let hues: Option<Vec<String>> = hue.as_deref().map(|h| strings(&rows, h)).transpose()?;
    let hue_index = index(&hue_labels);
    let points: Vec<(f64, f64, usize)> = (0..rows.height())
        .filter_map(|row| {
            let j: usize = match &hues {
                Some(hues) => *hue_index.get(hues[row].as_str())?,
                None => 0,
            };
            Some((xs[row], ys[row], j))
        })
        .collect();
    if points.is_empty() {
        return Err(PlotError::EmptyResult(format!("{x}, {y}")));
    }
    let subtitle: String = if step > 1 {
        format!("{} of {} rows", points.len(), dataset.metadata.height)
    } else {
        String::new()
    };
    Ok((points, hue_labels, subtitle))
}

/// Intercept and slope of y = a + b·x over all rows with both values, in two passes.
fn least_squares(dataset: &Dataset, x: &str, y: &str) -> Result<Option<(f64, f64)>, PlotError> {
    let pairs = || {
        dataset
            .dataframe
            .clone()
            .lazy()
            .select([
                col(x).cast(DataType::Float64),
                col(y).cast(DataType::Float64),
            ])
            .drop_nulls(None)
    };
    let means: DataFrame = pairs()
        .select([col(x).mean().alias("mx"), col(y).mean().alias("my")])
        .collect()?;
    let (mx, my): (f64, f64) = (f64s(&means, "mx")?[0], f64s(&means, "my")?[0]);
    let sums: DataFrame = pairs()
        .select([
            ((col(x) - lit(mx)) * (col(y) - lit(my))).sum().alias("sxy"),
            ((col(x) - lit(mx)) * (col(x) - lit(mx))).sum().alias("sxx"),
        ])
        .collect()?;
    let (sxy, sxx): (f64, f64) = (f64s(&sums, "sxy")?[0], f64s(&sums, "sxx")?[0]);
    Ok((sxx > 0.0 && sxy.is_finite()).then(|| (my - sxy / sxx * mx, sxy / sxx)))
}

#[allow(clippy::too_many_arguments)]
fn xy_figure(
    dataset: &Dataset,
    name: &'static str,
    title: String,
    (x, y, hue): (&str, &str, &str),
    hue_labels: Vec<String>,
    points: Vec<(f64, f64, usize)>,
    fit: Option<(f64, f64)>,
    subtitle: String,
) -> Figure {
    Figure::new(
        Chart::Xy {
            name,
            texts: Texts {
                title,
                x: x.to_string(),
                y: y.to_string(),
                value: y.to_string(),
                hue: hue.to_string(),
            },
            hue_labels,
            points,
            fit,
            subtitle,
        },
        dataset.metadata.filepath.as_deref(),
    )
}

/// Points (scatter, regplot) or lines with markers (line), plus the fit line for regplot.
pub(crate) fn render_xy(
    name: &str,
    texts: &Texts,
    hue_labels: &[String],
    points: &[(f64, f64, usize)],
    fit: Option<(f64, f64)>,
    subtitle: &str,
    style: &Style,
) -> String {
    let theme: Theme = style.theme();
    let grouped: bool = !hue_labels.is_empty();
    let cols: Vec<usize> = if grouped {
        crate::plot::bar::pick(hue_labels, style.hue_order.as_deref(), usize::MAX)
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
        shown()
            .map(|p| p.0)
            .filter(|v| v.is_finite())
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
                (lo.min(v), hi.max(v))
            })
    });
    let subtitle: &str = style.subtitle.as_deref().unwrap_or(subtitle);
    let l: Layout = layout(style, subtitle, 1, 0.0, if grouped { 150.0 } else { 36.0 });
    // Lines use the data range like seaborn; points keep the old axis from 0.
    let (y_lo, y_hi): (f64, f64) = match (name, style.ylim) {
        ("line", None) => shown()
            .map(|p| p.1)
            .filter(|v| v.is_finite())
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
                (lo.min(v), hi.max(v))
            }),
        _ => range(style, shown().map(|p| p.1)),
    };
    let scale: YScale = YScale::new(y_lo, y_hi, l.top, l.plot_h);
    let xscale: XScale = XScale::new(x_lo, x_hi, l.left, l.plot_w);

    let mut out: String = svg::card(&theme, &title(texts, style), subtitle, l.width, l.height);
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
        ("3.5", "0.85")
    };
    for &(px, py, j) in points {
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
    finish(&mut out, &theme, style, texts, &items, &l);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thin_keeps_spikes() {
        let mut line: Vec<(f64, f64)> = (0..10_000).map(|i| (f64::from(i), 0.0)).collect();
        line[5_001].1 = 100.0;
        let kept: Vec<(f64, f64)> = thin(&line);
        assert!(kept.len() <= MAX_LINE_POINTS);
        assert!(kept.contains(&(5_001.0, 100.0)));
        assert!(kept.windows(2).all(|w| w[0].0 < w[1].0));
    }
}
