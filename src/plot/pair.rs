//! `pairplot`: a scatter for every pair of numeric columns, each column's shape on the diagonal.

use polars::prelude::DataFrame;

use crate::{
    dataset::Dataset,
    display::{Figure, escape},
    error::OrFatal,
    plot::{
        chart::{Body, Texts},
        density::{self, halo_path},
        error::PlotError,
        f64s, labels, optional, sample, slots,
        style::{Legend, Style, Theme},
        svg::{self, HEIGHT, Tick, WIDTH, extent, legend_at, ordered},
        thin,
    },
};

const PAIR_COLUMNS: usize = 8; // numeric columns drawn when none are named
const PAIR_SAMPLE: usize = 1000; // dots per panel by default
const BINS: usize = 20; // bars of a diagonal histogram
const MIN_PANEL: f64 = 110.0;

impl Dataset {
    /// A scatter for every pair of `columns` and a histogram of each one on the diagonal (a
    /// density per group with a `hue`). `columns = &[]` takes the first 8 numeric columns.
    /// (seaborn: `pairplot`)
    ///
    /// One sample of up to 50,000 rows feeds every panel. Each scatter draws 1,000 of them
    /// (`.sample(n)` changes it), and the diagonals use them all. A column named twice is drawn
    /// once.
    ///
    /// Options: `.title`, `.subtitle`, `.theme`, `.size`, `.palette`, `.colors`, `.color`,
    /// `.color_map`, `.hue_order`, `.sample` and `.legend`.
    ///
    /// # Panics
    ///
    /// If a column doesn't exist, a named column isn't numeric, or there is no numeric column.
    /// [`Dataset::try_pairplot`] returns these as errors.
    ///
    /// # Examples
    ///
    /// Every numeric column (id, age and cgpa):
    /// ```
    /// use rusmlai::prelude::*;
    ///
    /// let ds = read_csv("tests/data/students.csv");
    /// ds.pairplot(&[], ""); // saves metadata/plots/Pairs 3 numeric columns.png
    /// ```
    ///
    /// Age and CGPA colored by `passed`, with every option:
    /// ```
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/data/students.csv");
    /// let svg = ds
    ///     .pairplot(&["age", "cgpa"], "passed")
    ///     .sample(8)
    ///     .title("Age and CGPA")
    ///     .subtitle("8 dots per panel")
    ///     .theme("light")
    ///     .size(600.0, 600.0)
    ///     .palette("muted")
    ///     .color_map(&[("false", "tomato")])
    ///     .hue_order(&["true", "false"])
    ///     .legend("bottom")
    ///     .svg(); // the SVG text, and no auto-save
    /// assert!(svg.contains("Age and CGPA"));
    /// assert!(svg.contains("#FF6347")); // tomato
    /// ```
    ///
    /// Edge case: a text column in the list:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// let ds = read_csv("tests/data/students.csv");
    /// ds.pairplot(&["age", "batch"], ""); // panics: column 'batch' is not numeric (found str)
    /// ```
    #[track_caller]
    pub fn pairplot(&self, columns: &[&str], hue: &str) -> Figure {
        self.try_pairplot(columns, hue).or_fatal()
    }

    /// [`Dataset::pairplot`] that returns an error instead of panicking.
    ///
    /// # Errors
    ///
    /// - `PlotError::Dataset` when a column doesn't exist or a named column isn't numeric.
    /// - `PlotError::EmptyResult` when there is no numeric column.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # use rusmlai::plot::error::PlotError;
    /// let ds = read_csv("tests/data/students.csv");
    /// let svg = ds.try_pairplot(&["age", "cgpa"], "batch")?.svg();
    /// assert!(svg.contains("Pairs: 2 numeric columns"));
    /// # Ok::<(), PlotError>(())
    /// ```
    ///
    /// No numeric column:
    /// ```
    /// # use rusmlai::prelude::*;
    /// use rusmlai::plot::error::PlotError;
    ///
    /// let ds = read_csv("tests/data/students.csv");
    /// let result = ds.select(&["name", "batch"]).try_pairplot(&[], "");
    /// assert!(matches!(result, Err(PlotError::EmptyResult(_))));
    /// ```
    pub fn try_pairplot(&self, columns: &[&str], hue: &str) -> Result<Figure, PlotError> {
        let hue: Option<String> = optional(hue);
        if let Some(hue) = &hue {
            self.try_column_info(hue)?;
        }
        let numeric: Vec<String> = if columns.is_empty() {
            let is_numeric = |c: &String| self.try_numeric_stats(c).is_ok();
            self.columns()
                .into_iter()
                .filter(|c| is_numeric(c) && Some(c) != hue.as_ref())
                .collect()
        } else {
            for column in columns {
                self.try_numeric_stats(column)?;
            }
            columns.iter().map(|c| c.to_string()).collect()
        };
        if numeric.is_empty() {
            return Err(PlotError::EmptyResult("the numeric columns".to_string()));
        }
        let all: usize = numeric.len();
        let picked: Vec<String> = numeric
            .into_iter()
            .take(if columns.is_empty() {
                PAIR_COLUMNS
            } else {
                usize::MAX
            })
            .collect();
        let mut needed: Vec<&str> = picked.iter().map(String::as_str).collect();
        if let Some(hue) = hue.as_deref().filter(|h| !needed.contains(h)) {
            needed.push(hue);
        }
        let (sample, rows): (DataFrame, usize) = sample(self, &needed)?;
        let values: Vec<Vec<f64>> = picked
            .iter()
            .map(|c| f64s(&sample, c))
            .collect::<Result<_, _>>()?;
        let hue_labels: Vec<String> = match &hue {
            Some(hue) => labels(self, &sample, hue)?,
            None => Vec::new(),
        };
        let hues: Vec<Option<usize>> = slots(&sample, hue.as_deref(), &hue_labels)?;
        let texts: Texts = Texts {
            title: format!("Pairs: {} numeric columns", picked.len()),
            subtitle: if picked.len() < all {
                format!("the first {} of {all} numeric columns", picked.len())
            } else {
                String::new()
            },
            hue: hue.unwrap_or_default(),
            ..Texts::default()
        };
        let body: Body = Body::Pair {
            columns: picked,
            values,
            hue_labels,
            hues,
            rows,
        };
        Ok(self.figure("pairplot", texts, body))
    }
}

/// A k × k grid of panels: scatters off the diagonal, histograms (or densities per hue group)
/// on it. Each column's scale is shared down its column and across its row.
pub(crate) fn render_pair(
    texts: &Texts,
    columns: &[String],
    values: &[Vec<f64>],
    hue_labels: &[String],
    hues: &[Option<usize>],
    rows: usize,
    style: &Style,
) -> String {
    let theme: Theme = style.theme();
    let k: usize = columns.len();
    let grouped: bool = !hue_labels.is_empty();
    let cols: Vec<usize> = if grouped {
        ordered(hue_labels, style.hue_order.as_deref())
    } else {
        vec![0]
    };
    // A group left out by .hue_order has no color, and its rows are not drawn.
    let mut color_of: Vec<Option<String>> = vec![None; hue_labels.len().max(1)];
    for (s, &j) in cols.iter().enumerate() {
        color_of[j] = Some(if grouped {
            style.group_color(s, &hue_labels[j])
        } else {
            style.single_color("")
        });
    }
    let kept: Vec<usize> = (0..hues.len())
        .filter(|&r| hues[r].is_some_and(|j| color_of[j].is_some()))
        .collect();
    let drawn: Vec<usize> = thin(&kept, style.sample.unwrap_or(PAIR_SAMPLE));
    let items: Vec<(String, String)> = if grouped {
        cols.iter()
            .filter_map(|&j| Some((hue_labels[j].clone(), color_of[j].clone()?)))
            .collect()
    } else {
        Vec::new()
    };

    // Layout: square panels, at least MIN_PANEL wide; the card grows to fit them.
    let (card_w, card_h): (f64, f64) = style.size.unwrap_or((WIDTH, HEIGHT));
    let at: Legend = legend_at(style, &items);
    let subtitle: String = match &style.subtitle {
        Some(text) => text.clone(),
        None => {
            let sampled: String = if drawn.len() < rows {
                format!("{} of {rows} rows", drawn.len())
            } else {
                String::new()
            };
            [texts.subtitle.as_str(), sampled.as_str()]
                .into_iter()
                .filter(|part| !part.is_empty())
                .collect::<Vec<&str>>()
                .join(",  ")
        }
    };
    let (left, top): (f64, f64) = (72.0, if subtitle.is_empty() { 92.0 } else { 110.0 });
    let right: f64 = if at == Legend::Right {
        svg::legend_width(&items)
    } else {
        24.0
    };
    let bottom: f64 = 58.0;
    let panel: f64 = ((card_w - left - right) / k as f64)
        .min((card_h - top - bottom) / k as f64)
        .max(MIN_PANEL);
    let width: f64 = (left + panel * k as f64 + right).max(card_w);
    let legend_h: f64 = svg::bottom_legend_h(at, &items, width);
    let height: f64 = top + panel * k as f64 + bottom + legend_h;
    let gap: f64 = 8.0;
    let inner: f64 = panel - gap;

    // Each column's range, padded by 5% like matplotlib's margins.
    let ranges: Vec<(f64, f64)> = values
        .iter()
        .map(|v| {
            let (lo, hi): (f64, f64) = extent(kept.iter().map(|&r| v[r]));
            // A constant column gets ±1, or 5% of a value so large that +1 wouldn't move it.
            let pad: f64 = if hi > lo {
                (hi - lo) * 0.05
            } else {
                lo.abs().max(20.0) * 0.05
            };
            (lo - pad, hi + pad)
        })
        .collect();
    let x0 = |c: usize| left + c as f64 * panel + gap / 2.0;
    let y0 = |r: usize| top + r as f64 * panel + gap / 2.0;
    let px = |c: usize, v: f64| x0(c) + (v - ranges[c].0) / (ranges[c].1 - ranges[c].0) * inner;
    let py =
        |r: usize, v: f64| y0(r) + inner - (v - ranges[r].0) / (ranges[r].1 - ranges[r].0) * inner;

    let mut out: String = svg::card(&theme, &texts.title(style), &subtitle, width, height);
    for r in 0..k {
        for c in 0..k {
            out.push_str(&format!(
                r##"<rect x="{:.2}" y="{:.2}" width="{inner:.2}" height="{inner:.2}" fill="none" stroke="{}" stroke-width="1"/>"##,
                x0(c),
                y0(r),
                theme.grid
            ));
            if r == c {
                diagonal(
                    &mut out,
                    &theme,
                    texts,
                    &columns[c],
                    &values[c],
                    ranges[c],
                    (x0(c), y0(r), inner),
                    &kept,
                    hues,
                    hue_labels,
                    &cols,
                    &color_of,
                );
                continue;
            }
            // One path of round dots per color: a zero-length segment with round caps is a dot,
            // and much smaller than one <circle> per point.
            for (j, color) in color_of.iter().enumerate() {
                let Some(color) = color else { continue };
                let d: String = drawn
                    .iter()
                    .filter(|&&row| hues[row] == Some(j))
                    .map(|&row| {
                        format!(
                            "M{:.1} {:.1}h0",
                            px(c, values[c][row]),
                            py(r, values[r][row])
                        )
                    })
                    .collect();
                out.push_str(&format!(
                    r##"<path d="{d}" fill="none" stroke="{color}" stroke-width="3.4" stroke-linecap="round" stroke-opacity="0.55"/>"##
                ));
            }
            let hover: String = format!(
                "x = {}\ny = {}\nr = {} (sampled rows)",
                columns[c],
                columns[r],
                svg::format_value(pearson(&values[c], &values[r], &kept))
            );
            out.push_str(&format!(
                r##"<rect x="{:.2}" y="{:.2}" width="{inner:.2}" height="{inner:.2}" fill="{}" fill-opacity="0"><title>{}</title></rect>"##,
                x0(c),
                y0(r),
                theme.background,
                escape(&hover)
            ));
        }
    }
    // Ticks on the outer axes, names under the bottom row and left of the first column.
    let base: f64 = top + panel * k as f64;
    for (c, name) in columns.iter().enumerate() {
        for t in svg::nice_ticks(ranges[c].0, ranges[c].1, 3) {
            out.push_str(&format!(
                r##"<text x="{:.2}" y="{:.2}" font-family="sans-serif" font-size="11" fill="{}" text-anchor="middle">{}</text>"##,
                px(c, t),
                base + 10.0,
                theme.tick,
                svg::format_tick(t)
            ));
            out.push_str(&format!(
                r##"<text x="{:.2}" y="{:.2}" font-family="sans-serif" font-size="11" fill="{}" text-anchor="end">{}</text>"##,
                left - 6.0,
                py(c, t) + 4.0,
                theme.tick,
                svg::format_tick(t)
            ));
        }
        svg::x_tick(
            &mut out,
            &theme,
            name,
            x0(c) + inner / 2.0,
            base + 30.0,
            panel,
            Tick::Flat,
        );
        let cy: f64 = y0(c) + inner / 2.0;
        out.push_str(&format!(
            r##"<text x="16" y="{cy:.2}" font-family="sans-serif" font-size="12" fill="{}" text-anchor="middle" transform="rotate(-90 16 {cy:.2})"><title>{}</title>{}</text>"##,
            theme.tick,
            escape(name),
            escape(&svg::shorten(name, panel - 6.0))
        ));
    }
    match at {
        Legend::Right => svg::legend(&mut out, &theme, &items, width - right + 16.0, top + 8.0),
        Legend::Bottom => svg::legend_flow(
            &mut out,
            &theme,
            &items,
            36.0,
            height - legend_h + 12.0,
            width - 72.0,
        ),
        Legend::None => {}
    }
    out.push_str("</svg>");
    out
}

/// One diagonal panel: a histogram of the sampled rows, or one density per hue group (scaled
/// by its share of the rows, like seaborn's pairplot).
#[allow(clippy::too_many_arguments)]
fn diagonal(
    out: &mut String,
    theme: &Theme,
    texts: &Texts,
    name: &str,
    values: &[f64],
    (lo, hi): (f64, f64),
    (x0, y0, inner): (f64, f64, f64),
    kept: &[usize],
    hues: &[Option<usize>],
    hue_labels: &[String],
    cols: &[usize],
    color_of: &[Option<String>],
) {
    let to_x = |v: f64| x0 + (v - lo) / (hi - lo) * inner;
    if hue_labels.is_empty() {
        let counts: Vec<f64> = density::bin(kept.iter().map(|&r| values[r]), (lo, hi), BINS);
        let (peak, total): (f64, f64) = (
            counts.iter().fold(0.0, |a, &b| a.max(b)),
            counts.iter().sum(),
        );
        let w: f64 = inner / BINS as f64;
        let color: &str = color_of[0].as_deref().unwrap_or(theme.tick);
        for (b, &n) in counts.iter().enumerate().filter(|(_, n)| **n > 0.0) {
            let h: f64 = n / peak * inner * 0.85;
            let edge = |b: usize| lo + (hi - lo) * b as f64 / BINS as f64;
            let hover: String = format!(
                "{name} = {} to {}\nshare = {}%",
                svg::format_value(edge(b)),
                svg::format_value(edge(b + 1)),
                svg::format_value(n / total * 100.0)
            );
            out.push_str(&svg::rect(
                x0 + b as f64 * w + 0.5,
                y0 + inner - h,
                (w - 1.0).max(0.5),
                h,
                color,
                &hover,
            ));
        }
        return;
    }
    let total: f64 = kept.len() as f64;
    let xs: Vec<f64> = density::linspace(lo, hi);
    let curves: Vec<(usize, Vec<f64>)> = cols
        .iter()
        .filter_map(|&j| {
            let group = kept
                .iter()
                .filter(|&&r| hues[r] == Some(j))
                .map(|&r| values[r]);
            let counts: Vec<f64> = density::bin(group, (lo, hi), 256);
            let h: f64 = density::scott(&counts, (lo, hi))?;
            let share: f64 = counts.iter().sum::<f64>() / total;
            let ds: Vec<f64> = density::kde(&counts, (lo, hi), h, &xs);
            Some((j, ds.into_iter().map(|d| d * share).collect()))
        })
        .collect();
    let peak: f64 = curves
        .iter()
        .flat_map(|c| c.1.iter())
        .fold(0.0, |a: f64, &b| a.max(b))
        .max(1e-300);
    for (j, ds) in &curves {
        let Some(color) = &color_of[*j] else { continue };
        let d: String = svg::polyline(
            xs.iter()
                .zip(ds)
                .map(|(&x, &dv)| (to_x(x), y0 + inner - dv / peak * inner * 0.85)),
        );
        let hover: String = format!("density of {name}\n{} = {}", texts.hue, hue_labels[*j]);
        halo_path(out, theme, &d, color, &hover);
    }
}

/// Pearson r of two columns over the kept rows.
fn pearson(a: &[f64], b: &[f64], rows: &[usize]) -> f64 {
    let n: f64 = rows.len() as f64;
    let (ma, mb): (f64, f64) = (
        rows.iter().map(|&r| a[r]).sum::<f64>() / n,
        rows.iter().map(|&r| b[r]).sum::<f64>() / n,
    );
    let (mut sab, mut saa, mut sbb): (f64, f64, f64) = (0.0, 0.0, 0.0);
    for &r in rows {
        let (da, db): (f64, f64) = (a[r] - ma, b[r] - mb);
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    sab / (saa * sbb).sqrt()
}
