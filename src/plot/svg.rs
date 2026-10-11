//! Shared drawing code. Every chart builds its SVG from these pieces, with the colors of a `Theme`.

use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::{display::escape, plot::style::Theme};

pub(crate) const WIDTH: f64 = 720.0;
pub(crate) const HEIGHT: f64 = 469.0;
pub(crate) const MIN_CLUSTER_BAR: f64 = 16.0;
const BAR_RADIUS: f64 = 6.0;
const HEADROOM: f64 = 22.0;
// ponytail: one average width per character cell (wide CJK = 2 cells); measure real glyphs if
// labels still clip.
pub(crate) const CHAR_WIDTH: f64 = 7.0; // average width of a 12 px sans-serif character

/// Opens the card: background, title and subtitle. Square corners: Fedora's viewer paints
/// transparent corners white.
pub(crate) fn card(theme: &Theme, title: &str, subtitle: &str, width: f64, height: f64) -> String {
    let (w, h, bg): (f64, f64, &str) = (width.round(), height.round(), theme.background);
    let mut svg: String = format!(
        r##"<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="background-color:{bg}"><rect x="0" y="0" width="{w}" height="{h}" fill="{bg}"/>"##
    );
    svg.push_str(&format!(
        r##"<text x="36" y="48" font-family="sans-serif" font-size="22" font-weight="bold" fill="{}">{}</text>"##,
        theme.title,
        escape(title)
    ));
    if !subtitle.is_empty() {
        svg.push_str(&format!(
            r##"<text x="36" y="70" font-family="sans-serif" font-size="13" fill="{}">{}</text>"##,
            theme.tick,
            escape(subtitle)
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
    label.width() as f64 * CHAR_WIDTH
}

/// True when a name is wider than its slot: then every name of the chart tilts.
pub(crate) fn tilts(labels: &[impl AsRef<str>], slot: f64) -> bool {
    labels.iter().any(|label| text_px(label.as_ref()) > slot - 6.0)
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
        Tick::Flat => (shorten(label, slot - 6.0), y, r##"text-anchor="middle""##.to_string()),
        Tick::Down => (
            shorten(label, MAX_LABEL),
            y - 8.0,
            format!(r##"text-anchor="end" transform="rotate(-45 {cx} {})""##, y - 8.0),
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
    if label.width() as f64 * CHAR_WIDTH <= max_px {
        return label.to_string();
    }
    let mut used: f64 = CHAR_WIDTH; // the "…"
    label
        .chars()
        .take_while(|c| {
            used += c.width().unwrap_or(0) as f64 * CHAR_WIDTH;
            used <= max_px
        })
        .collect::<String>()
        + "…"
}

/// Slot width that fits the longest label, at most 140 px.
pub(crate) fn label_width(labels: &[impl AsRef<str>]) -> f64 {
    let longest: usize = labels
        .iter()
        .map(|label| label.as_ref().width())
        .max()
        .unwrap_or(0);
    (longest as f64 * CHAR_WIDTH + 8.0).min(140.0)
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
            let w: f64 = 18.0 + name.width() as f64 * CHAR_WIDTH + 20.0;
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
    let (x1, y1, x2, y2): (f64, f64, f64, f64) = if across { (a, at, b, at) } else { (at, a, at, b) };
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
        let text_w: f64 = label.width() as f64 * 6.2;
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
    if !v.is_finite() {
        "nan".to_string()
    } else if (v - v.round()).abs() < 1e-9 {
        format!("{}", v.round() as i64)
    } else if v.abs() < 0.01 {
        format!("{v:.2e}")
    } else {
        format!("{v:.2}")
    }
}

/// Axis ticks: like `format_value`, with one decimal from 100 up.
fn format_tick(v: f64) -> String {
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
        let hi: f64 = if hi - lo > 1e-9 { hi } else { lo + 1.0 };
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
        let mut ticks: Vec<f64> = Vec::new();
        let mut k: f64 = (self.lo / self.step).ceil();
        if k * self.step - self.lo > span * 0.02 {
            ticks.push(self.lo);
        }
        while k * self.step < self.lo + span * 0.92 {
            ticks.push(k * self.step);
            k += 1.0;
        }
        if ticks.last().is_none_or(|last| self.hi - last > span * 0.02) {
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
        let hi: f64 = if hi - lo > 1e-9 { hi } else { lo + 1.0 };
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
        let mut ticks: Vec<f64> = Vec::new();
        let mut k: f64 = (self.lo / self.step).ceil();
        while k * self.step <= self.hi + self.step * 1e-9 {
            ticks.push(k * self.step);
            k += 1.0;
        }
        ticks
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
        let step: f64 = nice_step(span, 4);
        let mut k: f64 = (lo / step).ceil();
        while k * step < hi - span * 0.06 {
            if k * step > lo + span * 0.06 {
                ticks.push(k * step);
            }
            k += 1.0;
        }
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
        assert_eq!(shorten("abcdefghij", 6.0 * CHAR_WIDTH), "abcde…");
        assert_eq!(shorten("中文中文中文", 6.0 * CHAR_WIDTH), "中文…");
    }
}
