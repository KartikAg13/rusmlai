//! `hist`, `boxplot` and `stripplot`: how a numeric column is spread out.

use std::{
    collections::hash_map::DefaultHasher,
    hash::{BuildHasher, BuildHasherDefault},
};

use polars::prelude::{
    DataFrame, DataType, Expr, IdxCa, IdxSize, IntoLazy, PlSmallStr, QuantileMethod, col, len,
    lit, when,
};

use crate::{
    dataset::Dataset,
    display::{Figure, escape},
    metadata::dataset::stats::NumericStats,
    plot::{
        bar::{TOP, index, labels, ordered, pick, strings},
        chart::{Chart, Texts},
        error::PlotError,
        result::PlotResult,
        spec::PlotSpec,
        style::{Legend, Style, Theme},
        svg::{self, HEIGHT, MIN_CLUSTER_BAR, Tick, WIDTH, YScale},
    },
};

const MAX_BINS: usize = 1000;
pub(crate) const SAMPLE: usize = 2500; // points that strip, scatter and regplot draw by default
pub(crate) const MAX_SAMPLE: usize = 50_000; // points kept from the data, the most .sample(n) draws
// ponytail: at most 100 outliers per box, evenly spread by rank with both ends kept; raise it if
// the dots look sparse.
const MAX_OUTLIERS: usize = 100;
const BIN: &str = "__rusmlai_bin";
const COUNT: &str = "__rusmlai_count";
const OUTLIERS: &str = "__rusmlai_outliers";
const ALL: &str = "__rusmlai_all"; // the one x group of a box plot without x
const FIVE: [&str; 5] = [
    "__rusmlai_min",
    "__rusmlai_q25",
    "__rusmlai_median",
    "__rusmlai_q75",
    "__rusmlai_max",
];

/// Counts in `bins` equal bins. The edges come from the cached min and max, so every hue
/// group shares them.
pub(crate) fn try_hist(
    dataset: &Dataset,
    column: &str,
    hue: &str,
    bins: usize,
) -> Result<Figure, PlotError> {
    if !(1..=MAX_BINS).contains(&bins) {
        return Err(PlotError::InvalidBins(bins));
    }
    let stats: &NumericStats = dataset.try_numeric_stats(column)?;
    let hue: Option<String> = optional(hue);
    if let Some(hue) = &hue {
        dataset.try_column_info(hue)?;
    }
    if !stats.min.is_finite() {
        return Err(PlotError::EmptyResult(column.to_string()));
    }
    let spec: PlotSpec = PlotSpec::Histogram {
        column: column.to_string(),
        hue: hue.clone(),
        bins,
    };
    let fine: usize = bins * MAX_BINS.div_ceil(bins);
    let result: PlotResult = dataset.metadata.plot(spec, || {
        compute_hist(dataset, column, hue.as_deref(), fine, stats.min, stats.max)
    })?;
    Ok(Figure::new(
        Chart::Hist {
            texts: Texts {
                title: format!("Histogram: {column}"),
                x: column.to_string(),
                y: "count".to_string(),
                value: "count".to_string(),
                hue: hue.unwrap_or_default(),
            },
            result,
            bins,
            subtitle: String::new(),
        },
        dataset.metadata.filepath.as_deref(),
    ))
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
    let mut columns: Vec<Expr> = vec![col(column)];
    let mut keys: Vec<Expr> = vec![col(BIN)];
    if let Some(hue) = hue {
        columns.push(col(hue));
        keys.push(col(hue));
    }
    let grouped: DataFrame = dataset
        .dataframe
        .clone()
        .lazy()
        .select(columns)
        .drop_nulls(None)
        .with_column(bin)
        .group_by(keys)
        .agg([len().cast(DataType::Float64).alias(COUNT)])
        .collect()?;

    let names: Vec<String> = match hue {
        Some(hue) => labels(dataset, &grouped, hue)?,
        None => vec![column.to_string()],
    };
    let hues: Option<Vec<String>> = hue.map(|hue| strings(&grouped, hue)).transpose()?;
    let name_index = index(&names);
    let counts_column: Vec<f64> = f64s(&grouped, COUNT)?;
    let mut counts: Vec<Vec<f64>> = vec![vec![0.0; bins]; names.len()];
    for (row, bin) in grouped.column(BIN)?.i64()?.iter().enumerate() {
        let Some(bin) = bin else {
            continue; // NaN values have no bin
        };
        let series: Option<usize> = match &hues {
            Some(hues) => name_index.get(hues[row].as_str()).copied(),
            None => Some(0),
        };
        if let Some(series) = series {
            counts[series][bin as usize] = counts_column[row];
        }
    }
    Ok(PlotResult::Hist {
        lo,
        hi,
        names,
        counts,
    })
}

/// One box per `x` category (split by `hue`), or one box with no `x`. Like seaborn, the
/// whiskers stop at the last value within 1.5 × IQR of the box; values past them are points.
pub(crate) fn try_boxplot(
    dataset: &Dataset,
    x: &str,
    y: &str,
    hue: &str,
) -> Result<Figure, PlotError> {
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
    let PlotResult::Boxes { boxes, .. } = &result else {
        unreachable!("box plots hold Boxes results");
    };
    if !boxes.iter().flatten().any(|five| five[2].is_finite()) {
        return Err(PlotError::EmptyResult(y.to_string()));
    }
    Ok(Figure::new(
        Chart::Box {
            texts: Texts {
                title: match &x {
                    Some(x) => format!("Box: {x} vs {y}"),
                    None => format!("Box: {y}"),
                },
                x: x.unwrap_or_else(|| y.to_string()),
                y: y.to_string(),
                value: y.to_string(),
                hue: hue.unwrap_or_default(),
            },
            result,
            subtitle: String::new(),
        },
        dataset.metadata.filepath.as_deref(),
    ))
}

fn compute_boxes(
    dataset: &Dataset,
    x: Option<&str>,
    y: &str,
    hue: Option<&str>,
) -> Result<PlotResult, PlotError> {
    let key: &str = x.unwrap_or(ALL);
    let mut columns: Vec<Expr> = vec![col(y)];
    let mut keys: Vec<Expr> = vec![col(key)];
    columns.extend(x.map(col));
    if let Some(hue) = hue {
        columns.push(col(hue));
        keys.push(col(hue));
    }
    let mut rows = dataset.dataframe.clone().lazy().select(columns).drop_nulls(None);
    if x.is_none() {
        rows = rows.with_column(lit(y).alias(ALL));
    }
    let v = || col(y).cast(DataType::Float64);
    let q = |p: f64| v().quantile(lit(p), QuantileMethod::Linear);
    let reach: Expr = (q(0.75) - q(0.25)) * lit(1.5);
    let inside: Expr = v()
        .gt_eq(q(0.25) - reach.clone())
        .and(v().lt_eq(q(0.75) + reach));
    let grouped: DataFrame = rows
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
    let xs: Vec<String> = strings(&grouped, key)?;
    let hues: Option<Vec<String>> = hue.map(|hue| strings(&grouped, hue)).transpose()?;
    let five: Vec<Vec<f64>> = FIVE
        .iter()
        .map(|name| f64s(&grouped, name))
        .collect::<Result<_, _>>()?;
    let mut past: Vec<Vec<f64>> = Vec::with_capacity(grouped.height());
    for group in grouped.column(OUTLIERS)?.list()?.series_iter() {
        let mut values: Vec<f64> = match group {
            Some(values) => values.f64()?.iter().flatten().filter(|v| v.is_finite()).collect(),
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
    let (x_index, hue_index) = (index(&x_labels), index(&hue_labels));
    let k: usize = hue_labels.len().max(1);
    let mut boxes: Vec<Vec<[f64; 5]>> = vec![vec![[f64::NAN; 5]; k]; x_labels.len()];
    let mut outliers: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); k]; x_labels.len()];
    for row in 0..grouped.height() {
        let Some(&i) = x_index.get(xs[row].as_str()) else {
            continue;
        };
        let j: usize = match &hues {
            Some(hues) => match hue_index.get(hues[row].as_str()) {
                Some(&j) => j,
                None => continue,
            },
            None => 0,
        };
        boxes[i][j] = [0, 1, 2, 3, 4].map(|k| five[k][row]);
        outliers[i][j] = std::mem::take(&mut past[row]);
    }
    Ok(PlotResult::Boxes {
        x_labels,
        hue_labels,
        boxes,
        outliers,
    })
}

/// Sampled rows, spread sideways inside each `x` category. Raw rows, so nothing is cached.
pub(crate) fn try_stripplot(
    dataset: &Dataset,
    x: &str,
    y: &str,
    hue: &str,
) -> Result<Figure, PlotError> {
    dataset.try_numeric_stats(y)?;
    let (x, hue): (Option<String>, Option<String>) = (optional(x), optional(hue));
    check_columns(dataset, x.as_deref(), hue.as_deref())?;
    let columns: Vec<&str> = [Some(y), x.as_deref(), hue.as_deref()]
        .into_iter()
        .flatten()
        .collect();
    let (sample, rows): (DataFrame, usize) = sample(dataset, &columns)?;

    let ys: Vec<f64> = f64s(&sample, y)?;
    let x_labels: Vec<String> = match &x {
        Some(x) => labels(dataset, &sample, x)?,
        None => vec![y.to_string()],
    };
    let hue_labels: Vec<String> = match &hue {
        Some(hue) => labels(dataset, &sample, hue)?,
        None => Vec::new(),
    };
    let xs: Option<Vec<String>> = x.as_deref().map(|x| strings(&sample, x)).transpose()?;
    let hues: Option<Vec<String>> = hue.as_deref().map(|h| strings(&sample, h)).transpose()?;
    let (x_index, hue_index) = (index(&x_labels), index(&hue_labels));
    let points: Vec<(usize, usize, f64)> = (0..sample.height())
        .filter_map(|row| {
            let i: usize = match &xs {
                Some(xs) => *x_index.get(xs[row].as_str())?,
                None => 0,
            };
            let j: usize = match &hues {
                Some(hues) => *hue_index.get(hues[row].as_str())?,
                None => 0,
            };
            Some((i, j, ys[row]))
        })
        .collect();
    if points.is_empty() {
        return Err(PlotError::EmptyResult(y.to_string()));
    }
    Ok(Figure::new(
        Chart::Strip {
            texts: Texts {
                title: match &x {
                    Some(x) => format!("Strip: {x} vs {y}"),
                    None => format!("Strip: {y}"),
                },
                x: x.unwrap_or_else(|| y.to_string()),
                y: y.to_string(),
                value: y.to_string(),
                hue: hue.unwrap_or_default(),
            },
            x_labels,
            hue_labels,
            points,
            rows,
            subtitle: String::new(),
        },
        dataset.metadata.filepath.as_deref(),
    ))
}

/// Up to `MAX_SAMPLE` rows with no nulls in `columns`: one random row from each run of k rows.
/// The seed is fixed, so the same data draws the same picture. Returns them and how many rows
/// had values.
pub(crate) fn sample(dataset: &Dataset, columns: &[&str]) -> Result<(DataFrame, usize), PlotError> {
    let rows: DataFrame = dataset
        .dataframe
        .clone()
        .lazy()
        .select(columns.iter().map(|c| col(*c)).collect::<Vec<Expr>>())
        .drop_nulls(None)
        .collect()?;
    let step: usize = rows.height().div_ceil(MAX_SAMPLE).max(1);
    if step == 1 {
        let n: usize = rows.height();
        return Ok((rows, n));
    }
    let seed: BuildHasherDefault<DefaultHasher> = BuildHasherDefault::default();
    let picks: Vec<IdxSize> = (0..rows.height() / step)
        .map(|run| (run * step + (seed.hash_one(run) % step as u64) as usize) as IdxSize)
        .collect();
    Ok((rows.take(&IdxCa::from_vec(PlSmallStr::EMPTY, picks))?, rows.height()))
}

/// `n` evenly spaced items, or all of them when there are fewer (`.sample(n)` on the stored
/// sample points).
pub(crate) fn thin<T: Copy>(items: &[T], n: usize) -> Vec<T> {
    if items.len() <= n {
        return items.to_vec();
    }
    (0..n).map(|i| items[i * items.len() / n]).collect()
}

/// "2500 of 630000 rows" when a sample is drawn, unless a subtitle is set.
pub(crate) fn sample_subtitle(style: &Style, subtitle: &str, shown: usize, rows: usize) -> String {
    match &style.subtitle {
        Some(text) => text.clone(),
        None if subtitle.is_empty() && shown < rows => format!("{shown} of {rows} rows"),
        None => subtitle.to_string(),
    }
}

pub(crate) fn optional(column: &str) -> Option<String> {
    (!column.is_empty()).then(|| column.to_string())
}

pub(crate) fn check_columns(
    dataset: &Dataset,
    x: Option<&str>,
    hue: Option<&str>,
) -> Result<(), PlotError> {
    if hue.is_some() && x.is_none() {
        return Err(PlotError::HueNeedsX);
    }
    for column in x.into_iter().chain(hue) {
        dataset.try_column_info(column)?;
    }
    Ok(())
}

pub(crate) fn f64s(frame: &DataFrame, column: &str) -> Result<Vec<f64>, PlotError> {
    Ok(frame
        .column(column)?
        .cast(&DataType::Float64)?
        .f64()?
        .iter()
        .map(|v| v.unwrap_or(f64::NAN))
        .collect())
}

/// Padding, card size and plot area shared by the charts. `legend_h` is the room a bottom
/// legend takes under the axis name.
pub(crate) struct Layout {
    pub left: f64,
    pub right: f64,
    pub top: f64,
    pub plot_w: f64,
    pub plot_h: f64,
    pub width: f64,
    pub height: f64,
    pub legend_h: f64,
}

/// `items` is the legend: right of the plot, under it, or nowhere (`.legend`).
pub(crate) fn layout(
    style: &Style,
    subtitle: &str,
    slots: usize,
    min_slot: f64,
    items: &[(String, String)],
) -> Layout {
    let (card_w, height): (f64, f64) = style.size.unwrap_or((WIDTH, HEIGHT));
    let at: Legend = legend_at(style, items);
    let (left, top): (f64, f64) = (72.0, if subtitle.is_empty() { 92.0 } else { 110.0 });
    let right: f64 = if at == Legend::Right { svg::legend_width(items) } else { 36.0 };
    let plot_w: f64 = (card_w - left - right).max(slots.max(1) as f64 * min_slot);
    let width: f64 = left + plot_w + right;
    let legend_h: f64 = bottom_legend_h(at, items, width);
    Layout {
        left,
        right,
        top,
        plot_w,
        plot_h: height - top - 70.0 - legend_h,
        width,
        height,
        legend_h,
    }
}

/// Tilts the category names when one is wider than its slot. The card grows taller for them
/// (the flat row already has about 18 px), and the plot moves right when a name near the left
/// would hang past the card's edge.
pub(crate) fn tick_mode(l: &mut Layout, labels: &[impl AsRef<str>], slot: f64) -> Tick {
    if !svg::tilts(labels, slot) {
        return Tick::Flat;
    }
    l.height += (svg::tilt_room(labels) - 18.0).max(0.0);
    let shift: f64 = labels
        .iter()
        .enumerate()
        .map(|(i, label)| svg::tilt_reach(label.as_ref()) - (i as f64 + 0.5) * slot + 8.0 - l.left)
        .fold(0.0, f64::max);
    l.left += shift;
    l.width += shift;
    Tick::Down
}

/// Where the legend goes: nowhere without items, else `.legend` (right by default).
pub(crate) fn legend_at(style: &Style, items: &[(String, String)]) -> Legend {
    if items.is_empty() {
        Legend::None
    } else {
        style.legend.unwrap_or(Legend::Right)
    }
}

pub(crate) fn bottom_legend_h(at: Legend, items: &[(String, String)], width: f64) -> f64 {
    if at == Legend::Bottom {
        svg::legend_rows(items, width - 72.0) as f64 * 20.0 + 8.0
    } else {
        0.0
    }
}

/// The value axis: `.ylim`, else the data range with 0 and every `.hline` included.
pub(crate) fn range(style: &Style, values: impl Iterator<Item = f64>) -> (f64, f64) {
    style.ylim.unwrap_or_else(|| {
        values
            .chain(style.hlines.iter().map(|line| line.0))
            .filter(|v| v.is_finite())
            .fold((0.0, 0.0), |(lo, hi): (f64, f64), v| (lo.min(v), hi.max(v)))
    })
}

pub(crate) fn title(texts: &Texts, style: &Style) -> String {
    style.title.clone().unwrap_or_else(|| texts.title.clone())
}

/// The axis names: `.xlabel` / `.ylabel`, else the column names.
pub(crate) fn names<'a>(style: &'a Style, texts: &'a Texts) -> (&'a str, &'a str) {
    (
        style.xlabel.as_deref().unwrap_or(&texts.x),
        style.ylabel.as_deref().unwrap_or(&texts.y),
    )
}

/// `.hline` lines across the plot.
pub(crate) fn hlines(out: &mut String, theme: &Theme, style: &Style, scale: &YScale, l: &Layout) {
    for (v, label) in &style.hlines {
        svg::ref_line(out, theme, true, scale.y(*v), l.left, l.width - l.right, label, *v);
    }
}

/// 100 / the sum of the finite values with `.percent`, else 1.
pub(crate) fn percent_factor(style: &Style, values: impl Iterator<Item = f64>) -> f64 {
    if !style.percent {
        return 1.0;
    }
    let total: f64 = values.filter(|v| v.is_finite()).sum();
    if total > 0.0 { 100.0 / total } else { 0.0 }
}

/// A value as a label: "12.5%" with `.percent`.
pub(crate) fn unit(style: &Style, v: f64) -> String {
    if style.percent {
        format!("{}%", svg::format_value(v))
    } else {
        svg::format_value(v)
    }
}

/// Legend, axis names, and the closing tag.
pub(crate) fn finish(
    out: &mut String,
    theme: &Theme,
    style: &Style,
    (x_name, y_name): (&str, &str),
    items: &[(String, String)],
    l: &Layout,
) {
    match legend_at(style, items) {
        Legend::Right => svg::legend(out, theme, items, l.width - l.right + 16.0, l.top + 8.0),
        Legend::Bottom => svg::legend_flow(
            out,
            theme,
            items,
            36.0,
            l.height - l.legend_h + 12.0,
            l.width - 72.0,
        ),
        Legend::None => {}
    }
    svg::axis_names(
        out,
        theme,
        x_name,
        y_name,
        l.left + l.plot_w / 2.0,
        l.top + l.plot_h / 2.0,
        l.height - l.legend_h,
    );
    out.push_str("</svg>");
}

/// The old histogram: bars per bin (grouped or stacked by hue), bin ticks, five edge labels.
pub(crate) fn render_hist(
    texts: &Texts,
    result: &PlotResult,
    bins: usize,
    subtitle: &str,
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
            let (counts, out): (Vec<f64>, f64) = regroup(series, (*data_lo, *data_hi), (lo, hi), bins);
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
        cols.iter()
            .enumerate()
            .map(|(s, &j)| (groups[j].clone(), style.group_color(s, &groups[j])))
            .collect()
    } else {
        Vec::new()
    };
    let left_out: String = if outside > 0.0 {
        format!("{} rows outside the range", svg::format_value(outside))
    } else {
        subtitle.to_string()
    };
    let subtitle: &str = style.subtitle.as_deref().unwrap_or(&left_out);
    let l: Layout = layout(style, subtitle, bins, min_slot, &items);
    let heights: Vec<f64> = if stacked {
        (0..bins)
            .map(|b| cols.iter().map(|&j| counts[j][b]).sum())
            .collect()
    } else {
        cols.iter().flat_map(|&j| counts[j].iter().copied()).collect()
    };
    let (y_lo, y_hi): (f64, f64) = range(style, heights.into_iter());
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

    let mut out: String = svg::card(&theme, &title(texts, style), subtitle, l.width, l.height);
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
                value_label(&mut out, &theme, x + w / 2.0, scale.y(sum) - 8.0, &unit(style, sum));
            }
        }
    } else {
        for (s, &j) in cols.iter().enumerate() {
            for (b, &value) in counts[j].iter().enumerate() {
                let x: f64 = l.left + b as f64 * slot + (slot - inner) / 2.0 + s as f64 * bar_w;
                let w: f64 = (bar_w - 1.2).max(1.0);
                out.push_str(&svg::bar(&scale, x, w, value, &color(s, j), &hover(b, j, value)));
                if show_values && value > 0.0 {
                    let text: String = unit(style, value);
                    value_label(&mut out, &theme, x + w / 2.0, scale.y(value) - 8.0, &text);
                }
            }
        }
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
    let (x_name, y_name): (&str, &str) = names(style, texts);
    let y_name: &str = if style.percent && style.ylabel.is_none() { "percent" } else { y_name };
    finish(&mut out, &theme, style, (x_name, y_name), &items, &l);
    out
}

/// The box from q25 to q75 with a median line, whiskers to the fences, outliers as rings.
pub(crate) fn render_box(
    texts: &Texts,
    result: &PlotResult,
    subtitle: &str,
    style: &Style,
) -> String {
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
    let grouped: bool = !hue_labels.is_empty();
    let medians: Vec<f64> = boxes.iter().map(|row| mean(row.iter().map(|five| five[2]))).collect();
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
        cols.iter()
            .enumerate()
            .map(|(s, &j)| (hue_labels[j].clone(), style.group_color(s, &hue_labels[j])))
            .collect()
    } else {
        Vec::new()
    };
    let subtitle: &str = style.subtitle.as_deref().unwrap_or(subtitle);
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

    let mut out: String = svg::card(&theme, &title(texts, style), subtitle, l.width, l.height);
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
            let mut hover: String = format!("{} = {}", texts.x, x_labels[i]);
            if grouped {
                hover += &format!("\n{} = {}", texts.hue, hue_labels[j]);
            }
            for (name, v) in ["upper fence", "q75", "median", "q25", "lower fence"]
                .iter()
                .zip([five[4], five[3], five[2], five[1], five[0]])
            {
                hover += &format!("\n{name} = {}", svg::format_value(v));
            }
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
    finish(&mut out, &theme, style, names(style, texts), &items, &l);
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
    subtitle: &str,
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
    let subtitle: String = sample_subtitle(style, subtitle, points.len(), rows_with_values);
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

    let mut out: String = svg::card(&theme, &title(texts, style), subtitle, l.width, l.height);
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
    finish(&mut out, &theme, style, names(style, texts), &items, &l);
    out
}

/// Mean of the finite values; NaN when there are none.
fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let (sum, n): (f64, usize) = values
        .filter(|v| v.is_finite())
        .fold((0.0, 0), |(sum, n), v| (sum + v, n + 1));
    if n == 0 { f64::NAN } else { sum / n as f64 }
}

pub(crate) fn value_label(out: &mut String, theme: &Theme, x: f64, y: f64, text: &str) {
    out.push_str(&format!(
        r##"<text x="{x}" y="{y}" font-family="sans-serif" font-size="12" fill="{}" text-anchor="middle">{text}</text>"##,
        theme.value
    ));
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

    #[test]
    fn sample_takes_one_row_from_each_run() {
        let ys: Vec<f64> = (0..100_000).map(f64::from).collect();
        let ds: Dataset = Dataset::new(df!("y" => ys).unwrap(), "");
        let (sample, rows): (DataFrame, usize) = sample(&ds, &["y"]).unwrap();
        assert_eq!((sample.height(), rows), (MAX_SAMPLE, 100_000));
        let picked: Vec<f64> = f64s(&sample, "y").unwrap();
        assert!(picked.iter().enumerate().all(|(run, &y)| y as usize / 2 == run));
        assert_eq!(thin(&picked, 8000).len(), 8000);
    }
}
