//! Shared drawing code. Every chart builds its SVG from these pieces, with the colors of a `Theme`.

use std::cmp::Ordering;

use unicode_width::UnicodeWidthChar;

use crate::{
    display::escape,
    plot::{
        BOUND,
        style::{Legend, Sort, Style, Theme},
    },
};

pub(crate) const WIDTH: f64 = 720.0;
pub(crate) const HEIGHT: f64 = 469.0;
pub(crate) const MIN_CLUSTER_BAR: f64 = 16.0;
pub(crate) const TOP: usize = 30; // categories drawn by default
const BAR_RADIUS: f64 = 6.0;
const HEADROOM: f64 = 22.0;
/// Advance widths of printable ASCII (space to `~`) in thousandths of an em. Each is the widest
/// of Noto Sans, DejaVu Sans and Liberation Sans (Arial's metrics), the fonts `sans-serif`
/// usually is, so a name sized with them never clips in any of these.
#[rustfmt::skip]
const ASCII_EM: [u16; 95] = [
    318, 401, 460, 838, 636, 950, 780, 275, 390, 390, 551, 838, 318, 361, 318, 372,
    636, 636, 636, 636, 636, 636, 636, 636, 636, 636, 337, 337, 838, 838, 838, 556,
    1015, 684, 686, 722, 770, 667, 611, 778, 752, 339, 500, 667, 557, 907, 760, 787,
    667, 787, 722, 667, 611, 732, 684, 989, 685, 667, 685, 390, 372, 390, 838, 556,
    500, 613, 635, 550, 635, 615, 352, 635, 634, 278, 278, 579, 278, 974, 634, 612,
    635, 635, 413, 521, 392, 634, 592, 818, 592, 592, 525, 636, 551, 636, 838,
];
const CHAR_WIDTH: f64 = 7.0; // one character cell outside ASCII at 12 px (wide CJK = 2 cells)
const ELLIPSIS: f64 = 12.0; // "…" is 1 em in these fonts

/// Opens the card: background, title and subtitle. Square corners: Fedora's viewer paints
/// transparent corners white.
pub(crate) fn card(theme: &Theme, title: &str, subtitle: &str, width: f64, height: f64) -> String {
    let (w, h, bg): (f64, f64, &str) = (width.round(), height.round(), theme.background);
    let mut svg: String = format!(
        r##"<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="background-color:{bg}"><rect x="0" y="0" width="{w}" height="{h}" fill="{bg}"/>"##
    );
    // Text wider than the card ends in "…" (the full text is on hover). `shorten` counts 12 px
    // characters; the title is 22 px bold (about 8% wider), the subtitle 13 px.
    let fit = |text: &str, scale: f64| {
        let shown: String = shorten(text, (width - 72.0) / scale);
        format!("<title>{}</title>{}", escape(text), escape(&shown))
    };
    svg.push_str(&format!(
        r##"<text x="36" y="48" font-family="sans-serif" font-size="22" font-weight="bold" fill="{}">{}</text>"##,
        theme.title,
        fit(title, 22.0 / 12.0 * 1.08)
    ));
    if !subtitle.is_empty() {
        svg.push_str(&format!(
            r##"<text x="36" y="70" font-family="sans-serif" font-size="13" fill="{}">{}</text>"##,
            theme.tick,
            fit(subtitle, 13.0 / 12.0)
        ));
    }
    svg
}

/// The x axis name under the chart and the y axis name turned on its side.
pub(crate) fn axis_names(
    svg: &mut String,
    theme: &Theme,
    x: &str,
    y: &str,
    x_center: f64,
    y_center: f64,
    height: f64,
) {
    let tick: &str = theme.tick;
    svg.push_str(&format!(
        r##"<text x="{x_center}" y="{}" font-family="sans-serif" font-size="13" fill="{tick}" text-anchor="middle">{}</text>"##,
        height - 18.0,
        escape(x)
    ));
    svg.push_str(&format!(
        r##"<text x="16" y="{y_center}" font-family="sans-serif" font-size="13" fill="{tick}" text-anchor="middle" transform="rotate(-90 16 {y_center})">{}</text>"##,
        escape(y)
    ));
}

/// The longest name drawn in full; a longer one gets "…" (the full name stays on hover).
pub(crate) const MAX_LABEL: f64 = 220.0;

/// How category names sit: flat in their slot, or tilted 45° hanging below the axis (`Down`)
/// or rising above a heatmap (`Up`), like Plotly does when names don't fit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Tick {
    Flat,
    Down,
    Up,
}

/// Width of a 12 px name, in pixels.
pub(crate) fn text_px(label: &str) -> f64 {
    label.chars().map(char_px).sum()
}

fn char_px(c: char) -> f64 {
    match c {
        ' '..='~' => f64::from(ASCII_EM[c as usize - 32]) * 0.012,
        _ => c.width().unwrap_or(0) as f64 * CHAR_WIDTH,
    }
}

/// Card width that fits a 13 px subtitle drawn from x = 36.
pub(crate) fn subtitle_width(subtitle: &str) -> f64 {
    text_px(subtitle) * 13.0 / 12.0 + 72.0
}

/// True when a name is wider than its slot: then every name of the chart tilts.
pub(crate) fn tilts(labels: &[impl AsRef<str>], slot: f64) -> bool {
    labels
        .iter()
        .any(|label| text_px(label.as_ref()) > slot - 6.0)
}

/// How far one tilted name reaches down (or up), and sideways.
pub(crate) fn tilt_reach(label: &str) -> f64 {
    text_px(label).min(MAX_LABEL) * std::f64::consts::FRAC_1_SQRT_2
}

/// How far tilted names reach below (or above) their row.
pub(crate) fn tilt_room(labels: &[impl AsRef<str>]) -> f64 {
    labels
        .iter()
        .map(|label| tilt_reach(label.as_ref()))
        .fold(0.0, f64::max)
}

/// Room for a legend right of the plot: the longest name and its swatch, at least 150 px.
pub(crate) fn legend_width(items: &[(String, String)]) -> f64 {
    items
        .iter()
        .map(|(name, _)| text_px(name).min(MAX_LABEL) + 46.0)
        .fold(150.0, f64::max)
}

/// A category name at its slot. Flat, it is shortened to the slot; tilted, only past
/// `MAX_LABEL`. The full name is in the hover title.
#[allow(clippy::too_many_arguments)]
pub(crate) fn x_tick(
    svg: &mut String,
    theme: &Theme,
    label: &str,
    cx: f64,
    y: f64,
    slot: f64,
    tick: Tick,
) {
    let (shown, y, place): (String, f64, String) = match tick {
        Tick::Flat => (
            shorten(label, slot - 6.0),
            y,
            r##"text-anchor="middle""##.to_string(),
        ),
        Tick::Down => (
            shorten(label, MAX_LABEL),
            y - 8.0,
            format!(
                r##"text-anchor="end" transform="rotate(-45 {cx} {})""##,
                y - 8.0
            ),
        ),
        Tick::Up => (
            shorten(label, MAX_LABEL),
            y,
            format!(r##"text-anchor="start" transform="rotate(-45 {cx} {y})""##),
        ),
    };
    svg.push_str(&format!(
        r##"<text x="{cx}" y="{y}" font-family="sans-serif" font-size="12" fill="{}" {place}><title>{}</title>{}</text>"##,
        theme.tick,
        escape(label),
        escape(&shown)
    ));
}

/// `label` cut with "…" to fit `max_px`. Wide characters (Chinese, Japanese, Korean) count twice.
pub(crate) fn shorten(label: &str, max_px: f64) -> String {
    if text_px(label) <= max_px {
        return label.to_string();
    }
    let mut used: f64 = ELLIPSIS;
    label
        .chars()
        .take_while(|c| {
            used += char_px(*c);
            used <= max_px
        })
        .collect::<String>()
        + "…"
}

/// Slot width that fits the longest label, at most 140 px.
pub(crate) fn label_width(labels: &[impl AsRef<str>]) -> f64 {
    let longest: f64 = labels
        .iter()
        .map(|label| text_px(label.as_ref()))
        .fold(0.0, f64::max);
    (longest + 8.0).min(140.0)
}

/// Legend rows for the drawn hue groups `cols` of `names`, in the group colors.
pub(crate) fn legend_items(
    style: &Style,
    names: &[String],
    cols: &[usize],
) -> Vec<(String, String)> {
    cols.iter()
        .enumerate()
        .map(|(s, &j)| (names[j].clone(), style.group_color(s, &names[j])))
        .collect()
}

/// Color squares with names, one per row.
pub(crate) fn legend(svg: &mut String, theme: &Theme, items: &[(String, String)], x: f64, y: f64) {
    for (i, (name, color)) in items.iter().enumerate() {
        let row: f64 = y + i as f64 * 22.0;
        svg.push_str(&format!(
            r##"<rect x="{x}" y="{}" width="12" height="12" rx="2" fill="{color}"/><text x="{}" y="{row}" font-family="sans-serif" font-size="12" fill="{}">{}</text>"##,
            row - 10.0,
            x + 18.0,
            theme.tick,
            escape(&shorten(name, MAX_LABEL))
        ));
    }
}

/// The legend under the plot: items left to right, wrapped at `width`.
pub(crate) fn legend_flow(
    svg: &mut String,
    theme: &Theme,
    items: &[(String, String)],
    x: f64,
    y: f64,
    width: f64,
) {
    for ((name, color), (dx, row)) in items.iter().zip(flow(items, width)) {
        let (ix, iy): (f64, f64) = (x + dx, y + row as f64 * 20.0);
        svg.push_str(&format!(
            r##"<rect x="{ix}" y="{}" width="12" height="12" rx="2" fill="{color}"/><text x="{}" y="{iy}" font-family="sans-serif" font-size="12" fill="{}">{}</text>"##,
            iy - 10.0,
            ix + 18.0,
            theme.tick,
            escape(name)
        ));
    }
}

/// How many rows `legend_flow` needs.
pub(crate) fn legend_rows(items: &[(String, String)], width: f64) -> usize {
    flow(items, width).last().map_or(0, |&(_, row)| row + 1)
}

/// (x offset, row) of each legend item.
fn flow(items: &[(String, String)], width: f64) -> Vec<(f64, usize)> {
    let (mut x, mut row): (f64, usize) = (0.0, 0);
    items
        .iter()
        .map(|(name, _)| {
            let w: f64 = 18.0 + text_px(name) + 20.0;
            if x > 0.0 && x + w > width {
                (x, row) = (0.0, row + 1);
            }
            x += w;
            (x - w, row)
        })
        .collect()
}

/// A dashed reference line: across at row `at` from `a` to `b`, or down at column `at`. It is
/// drawn in the text color over a halo in the card color, so it shows on any mark; the label
/// sits on a box in the card color. `value` goes in the hover title.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ref_line(
    svg: &mut String,
    theme: &Theme,
    across: bool,
    at: f64,
    a: f64,
    b: f64,
    label: &str,
    value: f64,
) {
    let (x1, y1, x2, y2): (f64, f64, f64, f64) = if across {
        (a, at, b, at)
    } else {
        (at, a, at, b)
    };
    let (bg, fg): (&str, &str) = (theme.background, theme.title);
    let hover: String = if label.is_empty() {
        format_value(value)
    } else {
        format!("{label} = {}", format_value(value))
    };
    let ends: String = format!(r##"x1="{x1:.2}" y1="{y1:.2}" x2="{x2:.2}" y2="{y2:.2}""##);
    svg.push_str(&format!(
        r##"<g><title>{}</title><line {ends} stroke="{bg}" stroke-width="4" stroke-opacity="0.8"/><line {ends} stroke="{fg}" stroke-width="1.5" stroke-dasharray="6 4"/></g>"##,
        escape(&hover)
    ));
    if !label.is_empty() {
        let (tx, ty, anchor): (f64, f64, &str) = if across {
            (b - 6.0, at - 8.0, "end")
        } else {
            (at + 8.0, a + 14.0, "start")
        };
        // textLength makes the text exactly `text_w` wide in any font, so the box fits it.
        let text_w: f64 = text_px(label);
        let bx: f64 = if across { tx - text_w - 4.0 } else { tx - 4.0 };
        svg.push_str(&format!(
            r##"<rect x="{bx:.2}" y="{:.2}" width="{:.2}" height="16" rx="3" fill="{bg}" fill-opacity="0.85"/><text x="{tx:.2}" y="{ty:.2}" font-family="sans-serif" font-size="12" fill="{fg}" text-anchor="{anchor}" textLength="{text_w:.2}" lengthAdjust="spacing">{}</text>"##,
            ty - 12.0,
            text_w + 8.0,
            escape(label)
        ));
    }
}

/// A plain rectangle with a hover title: one segment of a stacked bar.
pub(crate) fn rect(x: f64, y: f64, w: f64, h: f64, fill: &str, hover: &str) -> String {
    if !(w > 0.0 && h > 0.0) {
        return String::new();
    }
    format!(
        r##"<rect x="{x:.2}" y="{y:.2}" width="{w:.2}" height="{h:.2}" fill="{fill}"><title>{}</title></rect>"##,
        escape(hover)
    )
}

/// A category name right-aligned before column `x`, shortened to `max_px`, full name on hover.
pub(crate) fn y_tick(svg: &mut String, theme: &Theme, label: &str, x: f64, y: f64, max_px: f64) {
    svg.push_str(&format!(
        r##"<text x="{x}" y="{y:.2}" font-family="sans-serif" font-size="12" fill="{}" text-anchor="end"><title>{}</title>{}</text>"##,
        theme.tick,
        escape(label),
        escape(&shorten(label, max_px))
    ));
}

/// Numbers on marks: the old two decimals, scientific only where two decimals would show 0.00.
pub(crate) fn format_value(v: f64) -> String {
    if v.is_nan() {
        "nan".to_string()
    } else if v.is_infinite() || v.abs() >= 1e15 {
        format!("{v:.2e}") // `as i64` below saturates past 9.2e18
    } else if (v - v.round()).abs() < 1e-9 {
        format!("{}", v.round() as i64)
    } else if v.abs() < 0.01 {
        format!("{v:.2e}")
    } else {
        format!("{v:.2}")
    }
}

/// Axis ticks: like `format_value`, with one decimal from 100 up.
pub(crate) fn format_tick(v: f64) -> String {
    if v.abs() >= 100.0 && (v - v.round()).abs() >= 1e-9 {
        format!("{v:.1}")
    } else {
        format_value(v)
    }
}

fn nice_step(span: f64, target_ticks: usize) -> f64 {
    if !span.is_finite() || span <= 0.0 {
        return 1.0;
    }
    let rough: f64 = span / target_ticks as f64;
    let magnitude: f64 = 10f64.powi(rough.log10().floor() as i32);
    let nice: f64 = match rough / magnitude {
        r if r > 5.0 => 10.0,
        r if r > 2.0 => 5.0,
        r if r > 1.0 => 2.0,
        _ => 1.0,
    };
    nice * magnitude
}

/// About `target` round tick values inside [lo, hi] (for small panels that draw their own axes).
pub(crate) fn nice_ticks(lo: f64, hi: f64, target: usize) -> Vec<f64> {
    multiples(lo, hi, nice_step(hi - lo, target))
}

/// The multiples of `step` in [lo, hi], at most 50. Counting with `k += 1.0` never ends once
/// `lo / step` passes 2^53 or the span is infinite: `k` stops changing and memory fills up.
fn multiples(lo: f64, hi: f64, step: f64) -> Vec<f64> {
    let first: f64 = (lo / step).ceil();
    let mut ticks: Vec<f64> = (0..50)
        .map(|k| (first + f64::from(k)) * step)
        .take_while(|t| *t <= hi + step * 1e-9)
        .collect();
    ticks.dedup(); // equal when the step is below the precision of lo
    ticks
}

/// `hi`, or a little above `lo` for an empty range (a constant column), so no scale divides by 0.
/// `lo + 1.0` alone equals `lo` past 2^53.
fn top_of(lo: f64, hi: f64) -> f64 {
    if hi - lo > 1e-9 {
        hi
    } else {
        lo + lo.abs().max(1e9) * 1e-9
    }
}

/// An SVG path through `points` ("M x,y L x,y ...").
pub(crate) fn polyline(points: impl Iterator<Item = (f64, f64)>) -> String {
    points
        .enumerate()
        .map(|(k, (x, y))| format!("{}{x:.2},{y:.2}", if k == 0 { "M" } else { " L" }))
        .collect()
}

/// Linear value axis over [lo, hi], negative values included.
pub(crate) struct YScale {
    lo: f64,
    hi: f64,
    step: f64,
    top: f64,
    height: f64,
}

impl YScale {
    pub(crate) fn new(lo: f64, hi: f64, top: f64, height: f64) -> Self {
        let hi: f64 = top_of(lo, hi);
        Self {
            lo,
            hi,
            step: nice_step(hi - lo, 4),
            top,
            height,
        }
    }

    /// Pixel row of a value. Values outside `.ylim` stick to the edge.
    pub(crate) fn y(&self, v: f64) -> f64 {
        let usable: f64 = (self.height - HEADROOM).max(1.0);
        self.top + HEADROOM + usable * (self.hi - v.clamp(self.lo, self.hi)) / (self.hi - self.lo)
    }

    /// Where bars start: 0, or the nearest edge when `.ylim` leaves 0 out.
    fn base(&self) -> f64 {
        0f64.clamp(self.lo, self.hi)
    }

    pub(crate) fn bottom(&self) -> f64 {
        self.y(self.lo)
    }

    /// Grid lines and tick labels; the line at the base is the axis.
    pub(crate) fn draw(&self, svg: &mut String, theme: &Theme, left: f64, right: f64) {
        let span: f64 = self.hi - self.lo;
        // The ends get their own label only with 16 px (a 12 px label and a gap) to the next
        // tick; a share of the span alone let -2.17e-6 sit on top of -2.00e-6 in a short chart.
        let usable: f64 = (self.height - HEADROOM).max(1.0);
        let room: f64 = span * (16.0 / usable).max(0.02);
        let mut ticks: Vec<f64> = Vec::new();
        if (self.lo / self.step).ceil() * self.step - self.lo > room {
            ticks.push(self.lo);
        }
        ticks.extend(
            multiples(self.lo, self.hi, self.step)
                .into_iter()
                .take_while(|t| *t < self.lo + span * 0.92),
        );
        if ticks.last().is_none_or(|last| self.hi - last > room) {
            ticks.push(self.hi);
        }
        for t in ticks {
            let y: f64 = self.y(t);
            let (stroke, width): (&str, &str) = if t == self.base() {
                (theme.axis, "1.5")
            } else {
                (theme.grid, "1")
            };
            let label: String = if t == self.lo || t == self.hi {
                format_value(t)
            } else {
                format_tick(t)
            };
            svg.push_str(&format!(
                r##"<line x1="{left}" y1="{y}" x2="{right}" y2="{y}" stroke="{stroke}" stroke-width="{width}"/><text x="{}" y="{}" font-family="sans-serif" font-size="12" fill="{}" text-anchor="end">{label}</text>"##,
                left - 10.0,
                y + 4.0,
                theme.tick
            ));
        }
    }
}

/// One bar from the base to `value`, rounded at the end away from the base, with a hover
/// title. A NaN value (a missing group) draws nothing.
pub(crate) fn bar(scale: &YScale, x: f64, w: f64, value: f64, fill: &str, hover: &str) -> String {
    if !value.is_finite() || w <= 0.0 {
        return String::new();
    }
    let (tip, base): (f64, f64) = (scale.y(value), scale.y(scale.base()));
    let h: f64 = (base - tip).abs();
    if h <= 0.0 {
        return String::new();
    }
    let r: f64 = BAR_RADIUS.min(w / 2.0).min(h);
    let corner: f64 = if tip < base { tip + r } else { tip - r };
    let (x0, x1): (f64, f64) = (x, x + w);
    format!(
        r##"<path d="M {x0:.2},{base:.2} L {x0:.2},{corner:.2} Q {x0:.2},{tip:.2} {xl:.2},{tip:.2} L {xr:.2},{tip:.2} Q {x1:.2},{tip:.2} {x1:.2},{corner:.2} L {x1:.2},{base:.2} Z" fill="{fill}"><title>{hover}</title></path>"##,
        xl = x0 + r,
        xr = x1 - r,
        hover = escape(hover)
    )
}

/// A bar from 0 to `value` lying down at row `y`, rounded at the end away from 0.
pub(crate) fn hbar(scale: &XScale, y: f64, h: f64, value: f64, fill: &str, hover: &str) -> String {
    if !value.is_finite() || h <= 0.0 {
        return String::new();
    }
    let (tip, base): (f64, f64) = (scale.x(value), scale.x(0f64.clamp(scale.lo, scale.hi)));
    let w: f64 = (tip - base).abs();
    if w <= 0.0 {
        return String::new();
    }
    let r: f64 = BAR_RADIUS.min(h / 2.0).min(w);
    let corner: f64 = if tip > base { tip - r } else { tip + r };
    let (y0, y1): (f64, f64) = (y, y + h);
    format!(
        r##"<path d="M {base:.2},{y0:.2} L {corner:.2},{y0:.2} Q {tip:.2},{y0:.2} {tip:.2},{yt:.2} L {tip:.2},{yb:.2} Q {tip:.2},{y1:.2} {corner:.2},{y1:.2} L {base:.2},{y1:.2} Z" fill="{fill}"><title>{hover}</title></path>"##,
        yt = y0 + r,
        yb = y1 - r,
        hover = escape(hover)
    )
}

/// Linear x axis for scatter, line and regplot, with nice ticks.
pub(crate) struct XScale {
    lo: f64,
    hi: f64,
    step: f64,
    left: f64,
    width: f64,
}

impl XScale {
    pub(crate) fn new(lo: f64, hi: f64, left: f64, width: f64) -> Self {
        let hi: f64 = top_of(lo, hi);
        Self {
            lo,
            hi,
            step: nice_step(hi - lo, 5),
            left,
            width,
        }
    }

    /// Pixel column of a value. Values outside `.xlim` stick to the edge.
    pub(crate) fn x(&self, v: f64) -> f64 {
        self.left + self.width * (v.clamp(self.lo, self.hi) - self.lo) / (self.hi - self.lo)
    }

    fn ticks(&self) -> Vec<f64> {
        multiples(self.lo, self.hi, self.step)
    }

    /// Tick labels under the plot at row `y`.
    pub(crate) fn draw(&self, svg: &mut String, theme: &Theme, y: f64) {
        for t in self.ticks() {
            svg.push_str(&format!(
                r##"<text x="{}" y="{y}" font-family="sans-serif" font-size="12" fill="{}" text-anchor="middle">{}</text>"##,
                self.x(t),
                theme.tick,
                format_tick(t)
            ));
        }
    }

    /// Grid lines from `top` to `bottom` at each tick; the line at 0 is the axis.
    pub(crate) fn grid(&self, svg: &mut String, theme: &Theme, top: f64, bottom: f64) {
        for t in self.ticks() {
            let (stroke, width): (&str, &str) = if t == 0.0 {
                (theme.axis, "1.5")
            } else {
                (theme.grid, "1")
            };
            let x: f64 = self.x(t);
            svg.push_str(&format!(
                r##"<line x1="{x:.2}" y1="{top}" x2="{x:.2}" y2="{bottom}" stroke="{stroke}" stroke-width="{width}"/>"##
            ));
        }
    }
}

/// The color at `t` in [0, 1] along `stops`, mixed linearly between neighbours.
pub(crate) fn mix(stops: &[&str], t: f64) -> String {
    let t: f64 = if t.is_finite() {
        t.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let last: usize = stops.len().saturating_sub(1);
    let pos: f64 = t * last as f64;
    let i: usize = (pos.floor() as usize).min(last.saturating_sub(1));
    let (a, b): ([f64; 3], [f64; 3]) = (rgb(stops[i]), rgb(stops[(i + 1).min(last)]));
    let f: f64 = pos - i as f64;
    let c = |k: usize| (a[k] + (b[k] - a[k]) * f).round() as u8;
    format!("#{:02X}{:02X}{:02X}", c(0), c(1), c(2))
}

/// A vertical color scale: `stops` from `lo` (bottom) to `hi` (top), with nice ticks on the
/// right. Thin slices, not an SVG gradient, so no ids clash when a notebook shows many charts.
#[allow(clippy::too_many_arguments)]
pub(crate) fn colorbar(
    svg: &mut String,
    theme: &Theme,
    stops: &[&str],
    lo: f64,
    hi: f64,
    x: f64,
    top: f64,
    height: f64,
) {
    const SLICES: usize = 64;
    let (w, slice): (f64, f64) = (14.0, height / SLICES as f64);
    for s in 0..SLICES {
        let fill: String = mix(stops, 1.0 - (s as f64 + 0.5) / SLICES as f64);
        svg.push_str(&format!(
            r##"<rect x="{x}" y="{:.2}" width="{w}" height="{:.2}" fill="{fill}"/>"##,
            top + s as f64 * slice,
            slice + 0.5
        ));
    }
    let span: f64 = hi - lo;
    let mut ticks: Vec<f64> = vec![lo];
    if span > 1e-12 {
        let inside = |t: &f64| *t > lo + span * 0.06 && *t < hi - span * 0.06;
        ticks.extend(
            multiples(lo, hi, nice_step(span, 4))
                .into_iter()
                .filter(inside),
        );
        ticks.push(hi);
    }
    for t in ticks {
        let y: f64 = if span > 1e-12 {
            top + height * (hi - t) / span
        } else {
            top + height / 2.0
        };
        svg.push_str(&format!(
            r##"<line x1="{}" y1="{y:.2}" x2="{}" y2="{y:.2}" stroke="{}" stroke-width="1"/><text x="{}" y="{:.2}" font-family="sans-serif" font-size="11" fill="{}">{}</text>"##,
            x + w,
            x + w + 4.0,
            theme.tick,
            x + w + 7.0,
            y + 4.0,
            theme.tick,
            format_tick(t)
        ));
    }
}

/// Dark text on light fills, white text on dark fills.
pub(crate) fn text_on(fill: &str) -> &'static str {
    let [r, g, b] = rgb(fill);
    if 0.2126 * r + 0.7152 * g + 0.0722 * b > 150.0 {
        "#16181D"
    } else {
        "#FFFFFF"
    }
}

fn rgb(hex: &str) -> [f64; 3] {
    let h: &str = hex.trim_start_matches('#');
    let full: String = if h.len() == 3 {
        h.chars().flat_map(|c| [c, c]).collect()
    } else {
        h.to_string()
    };
    let n: u32 = u32::from_str_radix(&full, 16).unwrap_or(0);
    [(n >> 16) & 0xFF, (n >> 8) & 0xFF, n & 0xFF].map(|v| v as f64)
}

/// The lowest and highest finite value; (inf, -inf) when there is none.
pub(crate) fn extent(values: impl Iterator<Item = f64>) -> (f64, f64) {
    values
        .filter(|v| v.is_finite())
        .map(|v| v.clamp(-BOUND, BOUND))
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(v), hi.max(v))
        })
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
    let right: f64 = if at == Legend::Right {
        legend_width(items)
    } else {
        36.0
    };
    let plot_w: f64 =
        (card_w.max(subtitle_width(subtitle)) - left - right).max(slots.max(1) as f64 * min_slot);
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
    if !tilts(labels, slot) {
        return Tick::Flat;
    }
    l.height += (tilt_room(labels) - 18.0).max(0.0);
    let shift: f64 = labels
        .iter()
        .enumerate()
        .map(|(i, label)| tilt_reach(label.as_ref()) - (i as f64 + 0.5) * slot + 8.0 - l.left)
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
        legend_rows(items, width - 72.0) as f64 * 20.0 + 8.0
    } else {
        0.0
    }
}

/// The value axis: `.ylim`, else the data range with 0 and every `.hline` included.
pub(crate) fn range(style: &Style, values: impl Iterator<Item = f64>) -> (f64, f64) {
    let lines = style.hlines.iter().map(|line| line.0);
    style
        .ylim
        .unwrap_or_else(|| extent(values.chain(lines).chain([0.0])))
}

/// `.hline` lines across the plot.
pub(crate) fn hlines(out: &mut String, theme: &Theme, style: &Style, scale: &YScale, l: &Layout) {
    for (v, label) in &style.hlines {
        ref_line(
            out,
            theme,
            true,
            scale.y(*v),
            l.left,
            l.width - l.right,
            label,
            *v,
        );
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
        format!("{}%", format_value(v))
    } else {
        format_value(v)
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
        Legend::Right => legend(out, theme, items, l.width - l.right + 16.0, l.top + 8.0),
        Legend::Bottom => legend_flow(
            out,
            theme,
            items,
            36.0,
            l.height - l.legend_h + 12.0,
            l.width - 72.0,
        ),
        Legend::None => {}
    }
    axis_names(
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

pub(crate) fn value_label(out: &mut String, theme: &Theme, x: f64, y: f64, text: &str) {
    out.push_str(&format!(
        r##"<text x="{x}" y="{y}" font-family="sans-serif" font-size="12" fill="{}" text-anchor="middle">{text}</text>"##,
        theme.value
    ));
}

/// Indices of the categories to draw: `.order` if given, else the stored order re-sorted by
/// `.sort` (`keys` are what "value" sorts by), then at most `top`.
pub(crate) fn pick(labels: &[String], style: &Style, keys: &[f64], top: usize) -> Vec<usize> {
    let mut rows: Vec<usize> = ordered(labels, style.order.as_deref());
    if style.order.is_none() {
        match style.sort {
            Some(Sort::Value) => {
                let key = |i: usize| keys.get(i).copied().filter(|v| v.is_finite());
                let key = |i: usize| key(i).unwrap_or(f64::NEG_INFINITY);
                rows.sort_by(|&a, &b| key(b).total_cmp(&key(a)));
            }
            Some(Sort::Name) => rows.sort_by(|&a, &b| by_name(&labels[a], &labels[b])),
            Some(Sort::Count) | None => {}
        }
    }
    rows.truncate(top);
    rows
}

/// `order` (names missing from `labels` are skipped), else every label in the stored order.
pub(crate) fn ordered(labels: &[String], order: Option<&[String]>) -> Vec<usize> {
    match order {
        Some(order) => order
            .iter()
            .filter_map(|name| labels.iter().position(|label| label == name))
            .collect(),
        None => (0..labels.len()).collect(),
    }
}

/// Numbers first, by size ("2" before "10"), then other names A to Z. A total order, so
/// `sort_by` never sees an inconsistent comparison.
fn by_name(a: &str, b: &str) -> Ordering {
    match (a.parse::<f64>().ok(), b.parse::<f64>().ok()) {
        (Some(x), Some(y)) => x.total_cmp(&y),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => a.cmp(b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plot::style::CLASSIC;

    #[test]
    fn long_names_tilt_instead_of_shortening() {
        let names: [&str; 2] = ["b.tech", "Number of vessels fluro"];
        assert!(!tilts(&names[..1], 60.0) && tilts(&names, 60.0));
        let mut out: String = String::new();
        x_tick(&mut out, &CLASSIC, names[1], 100.0, 400.0, 60.0, Tick::Down);
        assert!(out.contains("rotate(-45") && out.contains(">Number of vessels fluro</text>"));
    }

    #[test]
    fn shorten_counts_wide_characters_twice() {
        assert_eq!(shorten("short", 100.0), "short");
        assert_eq!(shorten("abcdefghij", 42.0), "abcd…");
        assert_eq!(shorten("中文中文中文", 42.0), "中文…");
        assert!(text_px("MMMM") > 3.0 * text_px("llll")); // per-glyph widths, not one average
    }

    #[test]
    fn pick_sorts_by_value_or_name_unless_ordered() {
        let labels: Vec<String> = ["10", "2", "b", "a"].map(String::from).to_vec();
        let keys: [f64; 4] = [1.0, f64::NAN, 3.0, 2.0];
        let mut style: Style = Style::default();
        assert_eq!(pick(&labels, &style, &keys, 30), [0, 1, 2, 3]);
        style.sort = Some(Sort::Value);
        assert_eq!(pick(&labels, &style, &keys, 3), [2, 3, 0]); // NaN last, then cut to 3
        style.sort = Some(Sort::Name);
        assert_eq!(pick(&labels, &style, &keys, 30), [1, 0, 3, 2]); // 2 before 10
        style.order = Some(vec!["a".to_string(), "10".to_string()]);
        assert_eq!(pick(&labels, &style, &keys, 30), [3, 0]);
    }
}
