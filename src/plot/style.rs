//! Chart options: the Rust form of Python's keyword arguments, plus notebook-wide defaults.

use std::sync::{LazyLock, PoisonError, RwLock};

use crate::{
    display::Figure,
    error::OrFatal,
    plot::{bar::MAX_GROUPS, chart::Chart, error::PlotError},
};

/// Colors of a card. `CLASSIC` is the original rusmlai look.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Theme {
    pub background: &'static str,
    pub bar: &'static str,
    pub grid: &'static str,
    pub axis: &'static str,
    pub tick: &'static str,
    pub title: &'static str,
    pub subtitle: &'static str,
    pub value: &'static str,
    pub palette: &'static [&'static str],
    /// Heatmap colors: the low end, the middle of a diverging range, and negative values.
    pub cell_low: &'static str,
    pub cell_mid: &'static str,
    pub negative: &'static str,
    /// Regression line.
    pub fit: &'static str,
}

pub(crate) const CLASSIC: Theme = Theme {
    background: "#15171C",
    bar: "#199E70",
    grid: "#24272E",
    axis: "#353942",
    tick: "#8B8F9A",
    title: "#F2F3F5",
    subtitle: "#E4E7EE",
    value: "#FFFFFF",
    palette: GREENS,
    cell_low: "#0E2A20",
    cell_mid: "#1A2A24",
    negative: "#C44B4B",
    fit: "#E8C547",
};

const LIGHT: Theme = Theme {
    background: "#FFFFFF",
    bar: "#199E70",
    grid: "#ECEEF1",
    axis: "#C9CDD4",
    tick: "#5B606B",
    title: "#16181D",
    subtitle: "#3A3F48",
    value: "#16181D",
    palette: GREENS,
    cell_low: "#E3F2EB",
    cell_mid: "#F1F3F2",
    negative: "#C44B4B",
    fit: "#B8860B",
};

const THEMES: &[(&str, Theme)] = &[("dark", CLASSIC), ("classic", CLASSIC), ("light", LIGHT)];

const GREENS: &[&str] = &[
    "#199E70", "#3DCF9A", "#0E6B4A", "#7EE0C0", "#5BA88A", "#C4F0DE", "#2A8F6A",
];

const PALETTES: &[(&str, &[&str])] = &[
    ("greens", GREENS),
    (
        "colorblind",
        &[
            "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999",
        ],
    ),
    (
        "tab10",
        &[
            "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F",
            "#BCBD22", "#17BECF",
        ],
    ),
    (
        "deep",
        &[
            "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
            "#CCB974", "#64B5CD",
        ],
    ),
    (
        "muted",
        &[
            "#4878D0", "#EE854A", "#6ACC64", "#D65F5F", "#956CB4", "#8C613C", "#DC7EC0", "#797979",
            "#D5BB67", "#82C6E2",
        ],
    ),
    (
        "pastel",
        &[
            "#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B", "#D0BBFF", "#DEBB9B", "#FAB0E4", "#CFCFCF",
            "#FFFEA3", "#B9F2F0",
        ],
    ),
    (
        "bright",
        &[
            "#023EFF", "#FF7C00", "#1AC938", "#E8000B", "#8B2BE2", "#9F4800", "#F14CC1", "#A3A3A3",
            "#FFC400", "#00D7FF",
        ],
    ),
    (
        "dark",
        &[
            "#001C7F", "#B1400D", "#12711C", "#8C0800", "#591E71", "#592F0D", "#A23582", "#3C3C3C",
            "#B8850A", "#006374",
        ],
    ),
];

/// A heatmap color range: stops from low to high. A diverging range has its middle stop at 0.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Cmap {
    pub stops: &'static [&'static str],
    pub diverging: bool,
}

const CMAPS: &[(&str, Cmap)] = &[
    (
        "greens",
        Cmap {
            stops: &["#0E2A20", "#199E70"],
            diverging: false,
        },
    ),
    (
        "viridis",
        Cmap {
            stops: &["#440154", "#3B528B", "#21908C", "#5DC863", "#FDE725"],
            diverging: false,
        },
    ),
    (
        "magma",
        Cmap {
            stops: &["#000004", "#51127C", "#B73779", "#FC8961", "#FCFDBF"],
            diverging: false,
        },
    ),
    (
        "blues",
        Cmap {
            stops: &["#F7FBFF", "#6BAED6", "#08306B"],
            diverging: false,
        },
    ),
    (
        "redgreen",
        Cmap {
            stops: &["#C44B4B", "#1A2A24", "#199E70"],
            diverging: true,
        },
    ),
    (
        "coolwarm",
        Cmap {
            stops: &["#3B4CC0", "#DDDDDD", "#B40426"],
            diverging: true,
        },
    ),
    (
        "rdbu",
        Cmap {
            stops: &["#B2182B", "#F7F7F7", "#2166AC"],
            diverging: true,
        },
    ),
];

/// Every chart option. `None` means "use the default".
#[derive(Debug, Clone, Default)]
pub(crate) struct Style {
    pub title: Option<String>,
    pub subtitle: Option<String>,
    pub xlabel: Option<String>,
    pub ylabel: Option<String>,
    pub theme: Option<Theme>,
    pub palette: Option<Vec<String>>,
    pub color: Option<String>,
    pub color_map: Vec<(String, String)>,
    pub size: Option<(f64, f64)>,
    pub order: Option<Vec<String>>,
    pub hue_order: Option<Vec<String>>,
    pub top: Option<usize>,
    pub values: Option<bool>,
    pub ylim: Option<(f64, f64)>,
    pub xlim: Option<(f64, f64)>,
    pub cmap: Option<Cmap>,
}

static DEFAULTS: LazyLock<RwLock<Style>> = LazyLock::new(|| RwLock::new(Style::default()));

/// Card style for every chart that follows, like `sns.set_theme`: `"dark"` (default) or `"light"`.
#[track_caller]
pub fn set_theme(name: &str) {
    let theme: Theme = lookup(THEMES, "set_theme", name).or_fatal();
    DEFAULTS
        .write()
        .unwrap_or_else(PoisonError::into_inner)
        .theme = Some(theme);
}

/// Colors of groups for every chart that follows, like `sns.set_palette`.
#[track_caller]
pub fn set_palette(name: &str) {
    let palette: &[&str] = lookup(PALETTES, "set_palette", name).or_fatal();
    DEFAULTS
        .write()
        .unwrap_or_else(PoisonError::into_inner)
        .palette = Some(strings(palette));
}

impl Style {
    /// The options of one chart on top of `set_theme` and `set_palette`.
    pub(crate) fn resolved(&self) -> Style {
        let defaults = DEFAULTS.read().unwrap_or_else(PoisonError::into_inner);
        Style {
            theme: self.theme.or(defaults.theme),
            palette: self.palette.clone().or_else(|| defaults.palette.clone()),
            ..self.clone()
        }
    }

    pub(crate) fn theme(&self) -> Theme {
        self.theme.unwrap_or(CLASSIC)
    }

    /// A single-series mark: `.color_map` for its name, then `.color`, then the theme.
    pub(crate) fn single_color(&self, name: &str) -> String {
        self.mapped(name)
            .or_else(|| self.color.clone())
            .unwrap_or_else(|| self.theme().bar.to_string())
    }

    /// Group `i`: `.color_map` for its name, then the palette.
    pub(crate) fn group_color(&self, i: usize, name: &str) -> String {
        self.mapped(name).unwrap_or_else(|| match &self.palette {
            Some(palette) => palette[i % palette.len()].clone(),
            None => {
                let palette: &[&str] = self.theme().palette;
                palette[i % palette.len()].to_string()
            }
        })
    }

    fn mapped(&self, name: &str) -> Option<String> {
        self.color_map
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, color)| color.clone())
    }
}

impl Figure {
    /// Checks the option against the chart, then applies it. Errors panic at the caller's line.
    #[track_caller]
    fn set(
        mut self,
        option: &'static str,
        apply: impl FnOnce(&mut Style, &Chart) -> Result<(), PlotError>,
    ) -> Self {
        self.chart
            .check(option)
            .and_then(|()| apply(&mut self.style, &self.chart))
            .or_fatal();
        self
    }

    /// Replaces the title. (matplotlib: `ax.set_title`)
    #[track_caller]
    pub fn title(self, text: &str) -> Self {
        self.set(".title", |s, _| {
            s.title = Some(text.to_string());
            Ok(())
        })
    }

    /// Replaces the line under the title.
    #[track_caller]
    pub fn subtitle(self, text: &str) -> Self {
        self.set(".subtitle", |s, _| {
            s.subtitle = Some(text.to_string());
            Ok(())
        })
    }

    /// Replaces the x axis name. (matplotlib: `ax.set_xlabel`)
    #[track_caller]
    pub fn xlabel(self, text: &str) -> Self {
        self.set(".xlabel", |s, _| {
            s.xlabel = Some(text.to_string());
            Ok(())
        })
    }

    /// Replaces the y axis name. (matplotlib: `ax.set_ylabel`)
    #[track_caller]
    pub fn ylabel(self, text: &str) -> Self {
        self.set(".ylabel", |s, _| {
            s.ylabel = Some(text.to_string());
            Ok(())
        })
    }

    /// `"dark"` (default) or `"light"`. (plotly: `template=`)
    #[track_caller]
    pub fn theme(self, name: &str) -> Self {
        self.set(".theme", |s, _| {
            s.theme = Some(lookup(THEMES, ".theme", name)?);
            Ok(())
        })
    }

    /// Colors of the groups: greens (default), colorblind, tab10, deep, muted, pastel, bright, dark.
    /// (seaborn: `palette=`)
    #[track_caller]
    pub fn palette(self, name: &str) -> Self {
        self.set(".palette", |s, _| {
            s.palette = Some(strings(lookup(PALETTES, ".palette", name)?));
            Ok(())
        })
    }

    /// Your own palette, as hex colors. (plotly: `color_discrete_sequence=`)
    #[track_caller]
    pub fn colors(self, colors: &[&str]) -> Self {
        self.set(".colors", |s, _| {
            if colors.is_empty() {
                return Err(invalid(".colors", "the list is empty".to_string()));
            }
            s.palette = Some(
                colors
                    .iter()
                    .map(|color| hex(".colors", color))
                    .collect::<Result<Vec<String>, PlotError>>()?,
            );
            Ok(())
        })
    }

    /// One color for a single series. (seaborn: `color=`)
    #[track_caller]
    pub fn color(self, color: &str) -> Self {
        self.set(".color", |s, _| {
            s.color = Some(hex(".color", color)?);
            Ok(())
        })
    }

    /// A fixed color for named categories. (plotly: `color_discrete_map=`)
    #[track_caller]
    pub fn color_map(self, map: &[(&str, &str)]) -> Self {
        self.set(".color_map", |s, chart| {
            let names: &[String] = match chart.categories(true) {
                [] => chart.categories(false),
                hue => hue,
            };
            for &(name, color) in map {
                known(".color_map", name, names)?;
                s.color_map
                    .push((name.to_string(), hex(".color_map", color)?));
            }
            Ok(())
        })
    }

    /// The card size in pixels. The card still gets wider when categories need room.
    /// (matplotlib: `figsize=`)
    #[track_caller]
    pub fn size(self, width: f64, height: f64) -> Self {
        self.set(".size", |s, _| {
            let ok: bool =
                width.is_finite() && height.is_finite() && width >= 200.0 && height >= 150.0;
            if !ok {
                return Err(invalid(
                    ".size",
                    format!("{width} x {height} is too small; use at least 200 x 150"),
                ));
            }
            s.size = Some((width, height));
            Ok(())
        })
    }

    /// Which x categories to draw, in this order. (seaborn: `order=`)
    #[track_caller]
    pub fn order(self, names: &[&str]) -> Self {
        self.set(".order", |s, chart| {
            for name in names {
                known(".order", name, chart.categories(false))?;
            }
            s.order = Some(strings(names));
            Ok(())
        })
    }

    /// Which hue groups to draw, in this order. (seaborn: `hue_order=`)
    #[track_caller]
    pub fn hue_order(self, names: &[&str]) -> Self {
        self.set(".hue_order", |s, chart| {
            for name in names {
                known(".hue_order", name, chart.categories(true))?;
            }
            s.hue_order = Some(strings(names));
            Ok(())
        })
    }

    /// Draws at most `n` categories (30 by default).
    #[track_caller]
    pub fn top(self, n: usize) -> Self {
        self.set(".top", |s, _| {
            if n == 0 {
                return Err(invalid(".top", "n must be at least 1".to_string()));
            }
            s.top = Some(n);
            Ok(())
        })
    }

    /// Numbers on the marks on or off. (plotly: `text_auto=`)
    #[track_caller]
    pub fn values(self, show: bool) -> Self {
        self.set(".values", |s, _| {
            s.values = Some(show);
            Ok(())
        })
    }

    /// The range of the value axis. (matplotlib: `ax.set_ylim`)
    #[track_caller]
    pub fn ylim(self, low: f64, high: f64) -> Self {
        self.set(".ylim", |s, _| {
            let ok: bool = low.is_finite() && high.is_finite() && low < high;
            if !ok {
                return Err(invalid(
                    ".ylim",
                    format!("({low}, {high}) is not a range; low must be below high"),
                ));
            }
            s.ylim = Some((low, high));
            Ok(())
        })
    }

    /// The range of the x axis on scatter, line and regplot charts. (matplotlib: `ax.set_xlim`)
    #[track_caller]
    pub fn xlim(self, low: f64, high: f64) -> Self {
        self.set(".xlim", |s, _| {
            let ok: bool = low.is_finite() && high.is_finite() && low < high;
            if !ok {
                return Err(invalid(
                    ".xlim",
                    format!("({low}, {high}) is not a range; low must be below high"),
                ));
            }
            s.xlim = Some((low, high));
            Ok(())
        })
    }

    /// The color range of a heatmap: greens, viridis, magma, blues (low to high), or
    /// redgreen, coolwarm, rdbu (negative to positive). (seaborn: `cmap=`)
    #[track_caller]
    pub fn cmap(self, name: &str) -> Self {
        self.set(".cmap", |s, _| {
            s.cmap = Some(lookup(CMAPS, ".cmap", name)?);
            Ok(())
        })
    }
}

fn invalid(option: &'static str, reason: String) -> PlotError {
    PlotError::InvalidOption { option, reason }
}

fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|item| item.to_string()).collect()
}

fn lookup<T: Copy>(table: &[(&str, T)], option: &'static str, name: &str) -> Result<T, PlotError> {
    match table.iter().find(|(n, _)| *n == name) {
        Some((_, value)) => Ok(*value),
        None => {
            let names: Vec<&str> = table.iter().map(|(n, _)| *n).collect();
            Err(invalid(
                option,
                format!("unknown name '{name}'. Use one of: {}", names.join(", ")),
            ))
        }
    }
}

fn hex(option: &'static str, color: &str) -> Result<String, PlotError> {
    let digits: &str = color.strip_prefix('#').unwrap_or("");
    if matches!(digits.len(), 3 | 6) && digits.chars().all(|c| c.is_ascii_hexdigit()) {
        Ok(color.to_string())
    } else {
        Err(invalid(
            option,
            format!("'{color}' is not a hex color like \"#199E70\""),
        ))
    }
}

fn known(option: &'static str, name: &str, categories: &[String]) -> Result<(), PlotError> {
    if categories.iter().any(|c| c == name) {
        return Ok(());
    }
    let mut list: String = match categories {
        [] => "none".to_string(),
        _ => categories[..categories.len().min(20)].join(", "),
    };
    if categories.len() > 20 {
        list += &format!(" and {} more", categories.len() - 20);
    }
    if categories.len() >= MAX_GROUPS {
        list += &format!(". Charts keep only the first {MAX_GROUPS} categories");
    }
    Err(invalid(
        option,
        format!("'{name}' is not a category of this chart. Categories: {list}"),
    ))
}
