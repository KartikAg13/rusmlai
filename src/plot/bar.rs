//! `bar`, `count`, `describe_bar` and `pie`: compute the numbers, then render them.

use std::collections::HashMap;

use polars::prelude::{DataFrame, DataType, Expr, IntoLazy, SortOptions, col, len};

use crate::{
    dataset::Dataset,
    display::{Figure, escape},
    metadata::dataset::stats::{CategoricalStats, NumericStats},
    plot::{
        chart::{Chart, Texts},
        error::PlotError,
        result::PlotResult,
        spec::{BarSpec, Estimator, PlotSpec},
        style::Style,
        svg::{self, HEIGHT, MIN_CLUSTER_BAR, WIDTH, YScale},
    },
};

pub(crate) const TOP: usize = 30;
const PIE_TOP: usize = 8; // slices before "Other"
pub(crate) const MAX_GROUPS: usize = 1000; // categories kept in the cache, so a huge column can't fill the file
const VALUE: &str = "__rusmlai_value";

pub(crate) fn try_bar(
    dataset: &Dataset,
    x: &str,
    y: &str,
    hue: &str,
    estimator: &str,
) -> Result<Figure, PlotError> {
    let spec: BarSpec = BarSpec::new(x, y, hue, estimator)?;
    dataset.try_column_info(&spec.x)?;
    if let Some(y) = &spec.y {
        dataset.try_numeric_stats(y)?;
    }
    if let Some(hue) = &spec.hue {
        dataset.try_column_info(hue)?;
    }
    let x_stats: Option<CategoricalStats> = dataset.try_categorical_stats(&spec.x).ok();

    let result: PlotResult = match (&x_stats, &spec.y, &spec.hue) {
        // A plain count of a text column is already in the metadata: no group-by, no cache line.
        (Some(stats), None, None) => {
            let top: &[(String, u64)] = &stats.categories[..stats.categories.len().min(MAX_GROUPS)];
            PlotResult::Series {
                labels: top.iter().map(|(name, _)| name.clone()).collect(),
                values: top.iter().map(|&(_, n)| n as f64).collect(),
            }
        }
        _ => dataset
            .metadata
            .plot(PlotSpec::Bar(spec.clone()), || compute(dataset, &spec))?,
    };

    let has_values: bool = match &result {
        PlotResult::Series { values, .. } => values.iter().any(|v| v.is_finite()),
        PlotResult::Clustered { grid, .. } => grid.iter().flatten().any(|v| v.is_finite()),
        _ => unreachable!("bar charts hold Series or Clustered results"),
    };
    if !has_values {
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

    Ok(Figure::new(
        Chart::Bar {
            name: if spec.y.is_none() { "count" } else { "bar" },
            texts: bar_texts(&spec),
            result,
            subtitle: subtitle(x_stats.as_ref(), spec.hue.as_deref()),
        },
        dataset.metadata.filepath.as_deref(),
    ))
}

/// The old texts: `Bar: gender vs exam_score` / `Count: course vs count`.
fn bar_texts(spec: &BarSpec) -> Texts {
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
    Texts {
        title,
        x: spec.x.clone(),
        y,
        value,
        hue: spec.hue.clone().unwrap_or_default(),
    }
}

/// One bar per numeric column (`column = ""`, estimator default "mean"), or the summary
/// stats of one column. Read from the cached stats: no pass over the data.
pub(crate) fn try_describe_bar(
    dataset: &Dataset,
    column: &str,
    estimator: &str,
) -> Result<Figure, PlotError> {
    let (texts, labels, values): (Texts, Vec<String>, Vec<f64>) = if column.is_empty() {
        let est: Estimator = match estimator {
            "" => Estimator::Mean,
            name => Estimator::parse(name)?,
        };
        let mut labels: Vec<String> = Vec::new();
        let mut values: Vec<f64> = Vec::new();
        for name in dataset.columns() {
            let Ok(stats) = dataset.try_numeric_stats(&name) else {
                continue; // text column
            };
            values.push(stat(dataset, &name, &stats, est)?);
            labels.push(name);
        }
        let texts: Texts = Texts {
            title: format!("Describe: column vs {}", est.as_str()),
            x: "column".to_string(),
            y: est.as_str().to_string(),
            value: est.as_str().to_string(),
            hue: String::new(),
        };
        (texts, labels, values)
    } else {
        let s: NumericStats = dataset.try_numeric_stats(column)?;
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
                    vec![stat(dataset, column, &s, est)?],
                )
            }
        };
        let texts: Texts = Texts {
            title: format!("Describe: statistic vs {column}"),
            x: "statistic".to_string(),
            y: column.to_string(),
            value: column.to_string(),
            hue: String::new(),
        };
        (texts, labels, values)
    };
    if !values.iter().any(|v| v.is_finite()) {
        return Err(PlotError::EmptyResult(if column.is_empty() {
            "the numeric columns".to_string()
        } else {
            column.to_string()
        }));
    }
    Ok(Figure::new(
        Chart::Bar {
            name: "describe_bar",
            texts,
            result: PlotResult::Series { labels, values },
            subtitle: "from cached column stats".to_string(),
        },
        dataset.metadata.filepath.as_deref(),
    ))
}

fn stat(
    dataset: &Dataset,
    column: &str,
    s: &NumericStats,
    est: Estimator,
) -> Result<f64, PlotError> {
    Ok(match est {
        Estimator::Mean => s.mean,
        Estimator::Median => s.median,
        Estimator::Min => s.min,
        Estimator::Max => s.max,
        Estimator::Std => s.std,
        Estimator::Count => {
            (dataset.metadata.height - dataset.try_column_info(column)?.null_count) as f64
        }
    })
}

/// Slices from the cached category counts: the top 8 plus "Other" by default.
pub(crate) fn try_pie(dataset: &Dataset, x: &str) -> Result<Figure, PlotError> {
    let stats: CategoricalStats = dataset.try_categorical_stats(x)?;
    let top: &[(String, u64)] = &stats.categories[..stats.categories.len().min(MAX_GROUPS)];
    if top.iter().all(|&(_, n)| n == 0) {
        return Err(PlotError::EmptyResult(x.to_string()));
    }
    Ok(Figure::new(
        Chart::Pie {
            texts: Texts {
                title: format!("Distribution: {x}"),
                x: x.to_string(),
                y: String::new(),
                value: "count".to_string(),
                hue: String::new(),
            },
            result: PlotResult::Series {
                labels: top.iter().map(|(name, _)| name.clone()).collect(),
                values: top.iter().map(|&(_, n)| n as f64).collect(),
            },
            subtitle: String::new(),
        },
        dataset.metadata.filepath.as_deref(),
    ))
}

/// One lazy group-by on x (and hue). A group with no rows stays NaN, never 0.
fn compute(dataset: &Dataset, spec: &BarSpec) -> Result<PlotResult, PlotError> {
    let keys: Vec<&str> = [Some(spec.x.as_str()), spec.hue.as_deref()]
        .into_iter()
        .flatten()
        .collect();
    let mut columns: Vec<Expr> = keys.iter().map(|key| col(*key)).collect();
    columns.extend(spec.y.as_deref().map(col));
    let value: Expr = match &spec.y {
        Some(y) => spec.estimator.expr(y),
        None => len(),
    };
    let grouped: DataFrame = dataset
        .dataframe
        .clone()
        .lazy()
        .select(columns)
        .drop_nulls(None)
        .group_by(keys.iter().map(|key| col(*key)).collect::<Vec<Expr>>())
        .agg([value.cast(DataType::Float64).alias(VALUE)])
        .collect()?;

    let xs: Vec<String> = strings(&grouped, &spec.x)?;
    let values: Vec<f64> = grouped
        .column(VALUE)?
        .f64()?
        .iter()
        .map(|v| v.unwrap_or(f64::NAN))
        .collect();
    let x_labels: Vec<String> = labels(dataset, &grouped, &spec.x)?;
    let Some(hue) = &spec.hue else {
        let by_x: HashMap<&str, f64> = xs.iter().map(String::as_str).zip(values).collect();
        let values: Vec<f64> = x_labels
            .iter()
            .map(|label| by_x.get(label.as_str()).copied().unwrap_or(f64::NAN))
            .collect();
        return Ok(PlotResult::Series {
            labels: x_labels,
            values,
        });
    };
    let hs: Vec<String> = strings(&grouped, hue)?;
    let hue_labels: Vec<String> = labels(dataset, &grouped, hue)?;
    let x_index: HashMap<&str, usize> = index(&x_labels);
    let hue_index: HashMap<&str, usize> = index(&hue_labels);
    let mut grid: Vec<Vec<f64>> = vec![vec![f64::NAN; hue_labels.len()]; x_labels.len()];
    for ((x, h), value) in xs.iter().zip(&hs).zip(values) {
        if let (Some(&i), Some(&j)) = (x_index.get(x.as_str()), hue_index.get(h.as_str())) {
            grid[i][j] = value;
        }
    }
    Ok(PlotResult::Clustered {
        x_labels,
        hue_labels,
        grid,
    })
}

/// Category order: the metadata order (most frequent first, ties alphabetical) for text
/// columns, ascending for numbers. At most `MAX_GROUPS`.
pub(crate) fn labels(
    dataset: &Dataset,
    grouped: &DataFrame,
    column: &str,
) -> Result<Vec<String>, PlotError> {
    let mut labels: Vec<String> = match dataset.try_categorical_stats(column) {
        Ok(stats) => stats.categories.into_iter().map(|(name, _)| name).collect(),
        Err(_) => {
            let sorted: DataFrame = grouped
                .clone()
                .lazy()
                .select([col(column).unique().sort(SortOptions::default())])
                .collect()?;
            strings(&sorted, column)?
        }
    };
    labels.truncate(MAX_GROUPS);
    Ok(labels)
}

/// A column as text, the same way the metadata names categories.
pub(crate) fn strings(frame: &DataFrame, column: &str) -> Result<Vec<String>, PlotError> {
    Ok(frame
        .column(column)?
        .cast(&DataType::String)?
        .str()?
        .iter()
        .map(|v| v.unwrap_or_default().to_string())
        .collect())
}

pub(crate) fn index(labels: &[String]) -> HashMap<&str, usize> {
    labels
        .iter()
        .enumerate()
        .map(|(i, label)| (label.as_str(), i))
        .collect()
}

/// The old subtitle: `unique=7,  mode='b.tech',  n=630000,  hue=gender`.
fn subtitle(x: Option<&CategoricalStats>, hue: Option<&str>) -> String {
    let hue: String = hue.map(|h| format!("hue={h}")).unwrap_or_default();
    let Some(stats) = x else {
        return hue;
    };
    let n: u64 = stats.categories.iter().map(|(_, count)| count).sum();
    let facts: String = format!(
        "unique={},  mode='{}',  n={n}",
        stats.unique_count, stats.mode
    );
    if hue.is_empty() {
        facts
    } else {
        format!("{facts},  {hue}")
    }
}

/// The old `render_simple` and `render_clustered` in one: a single series is one group.
pub(crate) fn render(texts: &Texts, result: &PlotResult, subtitle: &str, style: &Style) -> String {
    let theme = style.theme();
    let top: usize = style.top.unwrap_or(TOP);

    let (all, x_labels, hue_labels, grid): (usize, Vec<&String>, Vec<&String>, Vec<Vec<f64>>) =
        match result {
            PlotResult::Series { labels, values } => {
                let rows: Vec<usize> = pick(labels, style.order.as_deref(), top);
                (
                    labels.len(),
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
                let rows: Vec<usize> = pick(x_labels, style.order.as_deref(), top);
                let cols: Vec<usize> = pick(hue_labels, style.hue_order.as_deref(), usize::MAX);
                (
                    x_labels.len(),
                    rows.iter().map(|&i| &x_labels[i]).collect(),
                    cols.iter().map(|&j| &hue_labels[j]).collect(),
                    rows.iter()
                        .map(|&i| cols.iter().map(|&j| grid[i][j]).collect())
                        .collect(),
                )
            }
            _ => unreachable!("bar charts hold Series or Clustered results"),
        };

    let title: String = style.title.clone().unwrap_or_else(|| texts.title.clone());
    let base_subtitle: &str = style.subtitle.as_deref().unwrap_or(subtitle);
    let subtitle: String = match (x_labels.len() < all, base_subtitle.is_empty()) {
        (true, true) => format!("showing {} of {all}", x_labels.len()),
        (true, false) => format!("{base_subtitle},  showing {} of {all}", x_labels.len()),
        (false, _) => base_subtitle.to_string(),
    };

    let clustered: bool = !hue_labels.is_empty();
    let k: usize = hue_labels.len().max(1);
    let n: usize = x_labels.len().max(1);
    let (card_w, height): (f64, f64) = style.size.unwrap_or((WIDTH, HEIGHT));
    let left: f64 = 72.0;
    let right: f64 = if clustered { 150.0 } else { 28.0 };
    let pad_top: f64 = if subtitle.is_empty() { 92.0 } else { 110.0 };
    let plot_h: f64 = height - pad_top - 70.0;
    let min_slot: f64 = if clustered {
        k as f64 * MIN_CLUSTER_BAR / 0.62
    } else {
        48.0
    }
    .max(svg::label_width(&x_labels));
    let plot_w: f64 = (card_w - left - right).max(n as f64 * min_slot);
    let width: f64 = left + plot_w + right;

    let (lo, hi): (f64, f64) = style.ylim.unwrap_or_else(|| {
        grid.iter()
            .flatten()
            .filter(|v| v.is_finite())
            .fold((0.0, 0.0), |(lo, hi): (f64, f64), &v| {
                (lo.min(v), hi.max(v))
            })
    });
    let scale: YScale = YScale::new(lo, hi, pad_top, plot_h);
    let slot: f64 = plot_w / n as f64;
    let bar_w: f64 = if clustered {
        slot * 0.70 / k as f64
    } else {
        slot * 0.58
    };
    let show_values: bool = style.values.unwrap_or(!clustered);

    let mut out: String = svg::card(&theme, &title, &subtitle, width, height);
    scale.draw(&mut out, &theme, left, width - right);
    for (i, (label, row)) in x_labels.iter().zip(&grid).enumerate() {
        let slot_x: f64 = left + i as f64 * slot;
        let start: f64 = slot_x + (slot - bar_w * k as f64) / 2.0;
        for (j, &value) in row.iter().enumerate() {
            let x: f64 = start + j as f64 * bar_w;
            let (w, fill, hover): (f64, String, String) = if clustered {
                let hue: &str = hue_labels[j];
                (
                    (bar_w - 1.5).max(1.0),
                    style.group_color(j, hue),
                    format!(
                        "{} = {label}\n{} = {hue}\n{} = {}",
                        texts.x,
                        texts.hue,
                        texts.value,
                        svg::format_value(value)
                    ),
                )
            } else {
                (
                    bar_w,
                    style.single_color(label),
                    format!(
                        "{} = {label}\n{} = {}",
                        texts.x,
                        texts.value,
                        svg::format_value(value)
                    ),
                )
            };
            out.push_str(&svg::bar(&scale, x, w, value, &fill, &hover));
            if show_values && value.is_finite() {
                let y: f64 = if value >= 0.0 {
                    scale.y(value) - 8.0
                } else {
                    scale.y(value) + 16.0
                };
                out.push_str(&format!(
                    r##"<text x="{}" y="{y}" font-family="sans-serif" font-size="12" fill="{}" text-anchor="middle">{}</text>"##,
                    x + w / 2.0,
                    theme.value,
                    svg::format_value(value)
                ));
            }
        }
        svg::x_tick(
            &mut out,
            &theme,
            label,
            slot_x + slot / 2.0,
            scale.bottom() + 22.0,
            slot,
        );
    }
    if clustered {
        let items: Vec<(String, String)> = hue_labels
            .iter()
            .enumerate()
            .map(|(j, hue)| (hue.to_string(), style.group_color(j, hue)))
            .collect();
        svg::legend(
            &mut out,
            &theme,
            &items,
            width - right + 16.0,
            pad_top + 8.0,
        );
    }
    let x_name: &str = style.xlabel.as_deref().unwrap_or(&texts.x);
    let y_axis: &str = style.ylabel.as_deref().unwrap_or(&texts.y);
    svg::axis_names(
        &mut out,
        &theme,
        x_name,
        y_axis,
        left + plot_w / 2.0,
        pad_top + plot_h / 2.0,
        height,
    );
    out.push_str("</svg>");
    out
}

/// Indices of the categories to draw: `.order` first (else the stored order), then at most `top`.
pub(crate) fn pick(labels: &[String], order: Option<&[String]>, top: usize) -> Vec<usize> {
    match order {
        Some(order) => order
            .iter()
            .filter_map(|name| labels.iter().position(|label| label == name))
            .take(top)
            .collect(),
        None => (0..labels.len().min(top)).collect(),
    }
}

/// The old pie: slices on the left, a legend with counts and shares on the right.
pub(crate) fn render_pie(
    texts: &Texts,
    result: &PlotResult,
    subtitle: &str,
    style: &Style,
) -> String {
    let PlotResult::Series { labels, values } = result else {
        unreachable!("pie charts hold Series results");
    };
    let theme = style.theme();
    let rows: Vec<usize> = pick(labels, style.order.as_deref(), style.top.unwrap_or(PIE_TOP));
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
    let title: String = style.title.clone().unwrap_or_else(|| texts.title.clone());
    let subtitle: &str = style.subtitle.as_deref().unwrap_or(subtitle);
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
