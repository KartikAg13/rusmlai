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
            r##"<text x="36" y="70" font-family="sans-serif" font-size="14" fill="{}">{}</text>"##,
            theme.subtitle,
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

/// A category name under its slot. Shortened with "…" when wider than the slot; the full name
/// is in the hover title.
pub(crate) fn x_tick(svg: &mut String, theme: &Theme, label: &str, cx: f64, y: f64, slot: f64) {
    let shown: String = shorten(label, slot - 6.0);
    svg.push_str(&format!(
        r##"<text x="{cx}" y="{y}" font-family="sans-serif" font-size="12" fill="{}" text-anchor="middle"><title>{}</title>{}</text>"##,
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
            escape(name)
        ));
    }
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

    /// Tick labels under the plot at row `y`.
    pub(crate) fn draw(&self, svg: &mut String, theme: &Theme, y: f64) {
        let mut k: f64 = (self.lo / self.step).ceil();
        while k * self.step <= self.hi + self.step * 1e-9 {
            let t: f64 = k * self.step;
            svg.push_str(&format!(
                r##"<text x="{}" y="{y}" font-family="sans-serif" font-size="12" fill="{}" text-anchor="middle">{}</text>"##,
                self.x(t),
                theme.tick,
                format_tick(t)
            ));
            k += 1.0;
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

    #[test]
    fn shorten_counts_wide_characters_twice() {
        assert_eq!(shorten("short", 100.0), "short");
        assert_eq!(shorten("abcdefghij", 6.0 * CHAR_WIDTH), "abcde…");
        assert_eq!(shorten("中文中文中文", 6.0 * CHAR_WIDTH), "中文…");
    }
}
