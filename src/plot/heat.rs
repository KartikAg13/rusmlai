//! `heatmap` and `corrplot`: a grid of colored cells.

use polars::prelude::{DataFrame, DataType, Expr, IntoLazy, col, len};

use crate::{
    dataset::Dataset,
    display::{Figure, escape},
    plot::{
        bar::{index, labels, strings},
        chart::{Chart, Texts},
        dist::{f64s, title},
        error::PlotError,
        result::PlotResult,
        spec::{Estimator, PlotSpec},
        style::Style,
        svg::{self, HEIGHT, Tick, WIDTH},
    },
};

const VALUE: &str = "__rusmlai_value";

/// `estimator(value)` for each pair of `x` and `y` categories. `value = ""` counts rows.
pub(crate) fn try_heatmap(
    dataset: &Dataset,
    x: &str,
    y: &str,
    value: &str,
    estimator: &str,
) -> Result<Figure, PlotError> {
    dataset.try_column_info(x)?;
    dataset.try_column_info(y)?;
    let value: Option<String> = (!value.is_empty()).then(|| value.to_string());
    let estimator: Estimator = match (&value, estimator) {
        (None, "" | "count") => Estimator::Count,
        (None, other) => return Err(PlotError::EstimatorNeedsY(other.to_string())),
        (Some(_), "") => Estimator::Mean,
        (Some(_), other) => Estimator::parse(other)?,
    };
    if let Some(value) = &value {
        dataset.try_numeric_stats(value)?;
    }
    let spec: PlotSpec = PlotSpec::Heatmap {
        x: x.to_string(),
        y: y.to_string(),
        value: value.clone(),
        estimator,
    };
    let result: PlotResult = dataset.metadata.plot(spec, || {
        compute_heatmap(dataset, x, y, value.as_deref(), estimator)
    })?;
    check_values(&result, &format!("{x}, {y}"))?;
    let (title, what): (String, String) = match &value {
        Some(v) => (
            format!("Heatmap: {x} × {y} ({} {v})", estimator.as_str()),
            format!("{} {v}", estimator.as_str()),
        ),
        None => (format!("Heatmap: {x} vs {y}"), "count".to_string()),
    };
    Ok(Figure::new(
        Chart::Matrix {
            name: "heatmap",
            texts: Texts {
                title,
                x: x.to_string(),
                y: y.to_string(),
                value: what,
                hue: String::new(),
            },
            result,
            fixed: None,
            subtitle: String::new(),
        },
        dataset.metadata.filepath.as_deref(),
    ))
}

fn compute_heatmap(
    dataset: &Dataset,
    x: &str,
    y: &str,
    value: Option<&str>,
    estimator: Estimator,
) -> Result<PlotResult, PlotError> {
    let mut columns: Vec<Expr> = vec![col(x), col(y)];
    columns.extend(value.map(col));
    let agg: Expr = match value {
        Some(value) => estimator.expr(value),
        None => len(),
    };
    let grouped: DataFrame = dataset
        .dataframe
        .clone()
        .lazy()
        .select(columns)
        .drop_nulls(None)
        .group_by([col(x), col(y)])
        .agg([agg.cast(DataType::Float64).alias(VALUE)])
        .collect()?;
    let (x_labels, y_labels): (Vec<String>, Vec<String>) =
        (labels(dataset, &grouped, x)?, labels(dataset, &grouped, y)?);
    let (xs, ys, values): (Vec<String>, Vec<String>, Vec<f64>) = (
        strings(&grouped, x)?,
        strings(&grouped, y)?,
        f64s(&grouped, VALUE)?,
    );
    let (x_index, y_index) = (index(&x_labels), index(&y_labels));
    let mut grid: Vec<Vec<f64>> = vec![vec![f64::NAN; x_labels.len()]; y_labels.len()];
    for row in 0..grouped.height() {
        if let (Some(&i), Some(&j)) = (x_index.get(xs[row].as_str()), y_index.get(ys[row].as_str()))
        {
            grid[j][i] = values[row];
        }
    }
    Ok(PlotResult::Matrix {
        x_labels,
        y_labels,
        grid,
    })
}

/// Pearson correlation of every pair of numeric columns, on the rows where none is null.
pub(crate) fn try_corrplot(dataset: &Dataset) -> Result<Figure, PlotError> {
    let columns: Vec<String> = dataset
        .columns()
        .into_iter()
        .filter(|name| dataset.try_numeric_stats(name).is_ok())
        .collect();
    if columns.is_empty() {
        return Err(PlotError::EmptyResult("the numeric columns".to_string()));
    }
    let spec: PlotSpec = PlotSpec::Corrplot {
        columns: columns.clone(),
    };
    let result: PlotResult = dataset
        .metadata
        .plot(spec, || compute_corr(dataset, &columns))?;
    check_values(&result, "the numeric columns")?;
    Ok(Figure::new(
        Chart::Matrix {
            name: "corrplot",
            texts: Texts {
                title: "Corrplot: Correlation Matrix".to_string(),
                x: String::new(),
                y: String::new(),
                value: "correlation".to_string(),
                hue: String::new(),
            },
            result,
            fixed: Some((-1.0, 1.0)),
            subtitle: String::new(),
        },
        dataset.metadata.filepath.as_deref(),
    ))
}

/// Pearson correlation of each pair on the rows where both values are present, like pandas.
/// One Polars select for all pairs.
fn compute_corr(dataset: &Dataset, columns: &[String]) -> Result<PlotResult, PlotError> {
    let n: usize = columns.len();
    let pairs: Vec<(usize, usize)> = (0..n).flat_map(|i| (i..n).map(move |j| (i, j))).collect();
    let rs: DataFrame = dataset
        .dataframe
        .clone()
        .lazy()
        .select(
            pairs
                .iter()
                .map(|&(i, j)| pearson(&columns[i], &columns[j]).alias(format!("{i},{j}")))
                .collect::<Vec<Expr>>(),
        )
        .collect()?;
    let mut grid: Vec<Vec<f64>> = vec![vec![f64::NAN; n]; n];
    for (i, j) in pairs {
        let r: f64 = f64s(&rs, &format!("{i},{j}"))?[0];
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
    let both: Expr = col(a).is_not_null().and(col(b).is_not_null());
    let centered = |c: &str| {
        let v: Expr = col(c).cast(DataType::Float64).filter(both.clone());
        v.clone() - v.mean()
    };
    let (da, db): (Expr, Expr) = (centered(a), centered(b));
    (da.clone() * db.clone()).sum() / ((da.clone() * da).sum() * (db.clone() * db).sum()).sqrt()
}

fn check_values(result: &PlotResult, columns: &str) -> Result<(), PlotError> {
    let PlotResult::Matrix { grid, .. } = result else {
        unreachable!("heatmaps hold Matrix results");
    };
    if grid.iter().flatten().any(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(PlotError::EmptyResult(columns.to_string()))
    }
}

/// The old matrix: y labels on the left, x labels on top, rounded cells with numbers.
/// `fixed` pins the color range (corrplot uses -1 to 1).
pub(crate) fn render_matrix(
    texts: &Texts,
    result: &PlotResult,
    fixed: Option<(f64, f64)>,
    subtitle: &str,
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
    let longest: f64 = y_labels.iter().map(|label| svg::text_px(label)).fold(0.0, f64::max);
    let (left, bottom): (f64, f64) = ((longest.min(svg::MAX_LABEL) + 40.0).max(110.0), 70.0);
    let (nx, ny): (usize, usize) = (x_labels.len().max(1), y_labels.len().max(1));
    let cell = |right: f64| ((card_w - left - right) / nx as f64).max(28.0);
    let tick: Tick = if svg::tilts(x_labels, cell(100.0)) { Tick::Up } else { Tick::Flat };
    let room: f64 = if tick == Tick::Up { svg::tilt_room(x_labels) } else { 0.0 };
    let (top, right): (f64, f64) = (100.0f64.max(92.0 + room), 100.0f64.max(room + 8.0));
    let cell_w: f64 = cell(right);
    let cell_h: f64 = ((card_h - top - bottom) / ny as f64).max(28.0);
    let width: f64 = (left + cell_w * nx as f64 + right).max(card_w);
    let height: f64 = (top + cell_h * ny as f64 + bottom).max(card_h);

    let (vmin, vmax): (f64, f64) = fixed.unwrap_or_else(|| {
        grid.iter()
            .flatten()
            .filter(|v| v.is_finite())
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            })
    });
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
    let subtitle: &str = style.subtitle.as_deref().unwrap_or(subtitle);

    let mut out: String = svg::card(&theme, &title(texts, style), subtitle, width, height);
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
    let (lo, hi): (f64, f64) = if diverging { (-reach, reach) } else { (vmin, vmax) };
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
    let x_name: &str = style.xlabel.as_deref().unwrap_or(&texts.x);
    let y_name: &str = style.ylabel.as_deref().unwrap_or(&texts.y);
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
}
