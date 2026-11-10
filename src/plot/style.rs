//! Chart options: the Rust form of Python's keyword arguments, plus notebook-wide defaults.

use crate::{
    display::Figure,
    error::OrFatal,
    plot::{MAX_GROUPS, MAX_SAMPLE, chart::Chart, error::PlotError},
};

/// Colors of a card. `CLASSIC` is the original rusmlai look.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Theme {
    pub background: &'static str,
    /// The high end of the default heatmap color scale (single-series charts use the palette).
    pub bar: &'static str,
    pub grid: &'static str,
    pub axis: &'static str,
    pub tick: &'static str,
    pub title: &'static str,
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
    value: "#FFFFFF",
    palette: RUSMLAI,
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
    value: "#16181D",
    palette: RUSMLAI_LIGHT,
    cell_low: "#E3F2EB",
    cell_mid: "#F1F3F2",
    negative: "#C44B4B",
    fit: "#B8860B",
};

const THEMES: &[(&str, Theme)] = &[("dark", CLASSIC), ("classic", CLASSIC), ("light", LIGHT)];

/// The default group colors on the dark card (chosen 2026-10-10): olive, cream, rust, navy, sky,
/// then gold, plum and sage. Navy is one step lighter than the chosen `#1F487E`, which almost
/// disappears on the dark card. Neighbours pass the color-blindness and normal-vision checks of
/// the dataviz validator; the muted olive and cream fail its chroma check on purpose.
const RUSMLAI: &[&str] = &[
    "#566E3D", "#F5DFBB", "#B7410E", "#436DA6", "#39A9DB", "#D79B3C", "#955A89", "#8ED09C",
];

/// The same hues for the white card: cream, sky, gold and sage are darker steps (cream `#F5DFBB`
/// would almost disappear on white), and navy is the chosen `#1F487E`.
const RUSMLAI_LIGHT: &[&str] = &[
    "#566E3D", "#BDA885", "#B7410E", "#1F487E", "#018CBC", "#C0851F", "#864C7A", "#519160",
];

const GREENS: &[&str] = &[
    "#199E70", "#3DCF9A", "#0E6B4A", "#7EE0C0", "#5BA88A", "#C4F0DE", "#2A8F6A",
];

/// `rusmlai` is empty: it means "the theme's own palette", so it follows `.theme("light")`.
const PALETTES: &[(&str, &[&str])] = &[
    ("rusmlai", &[]),
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
    pub sample: Option<usize>,
    pub sort: Option<Sort>,
    pub horizontal: bool,
    pub percent: bool,
    pub stacked: bool,
    /// Reference lines: (value, label).
    pub hlines: Vec<(f64, String)>,
    pub vlines: Vec<(f64, String)>,
    pub legend: Option<Legend>,
    /// A density line over a histogram.
    pub kde: bool,
}

/// `.legend(..)`: right of the plot (default), under it, or hidden.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Legend {
    Right,
    Bottom,
    None,
}

/// `.sort(..)`: tallest first, A to Z, or the stored order (most rows first).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Sort {
    Value,
    Name,
    Count,
}

// The notebook-wide defaults live in environment variables, not in a static: evcxr builds each
// notebook cell with its own copy of rusmlai, so a static set in one cell is unseen by the next.
// The process environment is shared by every copy. A shell can set them too.
const THEME_VAR: &str = "RUSMLAI_THEME";
const PALETTE_VAR: &str = "RUSMLAI_PALETTE";

/// The card style of every chart that follows: `"dark"` (the default) or `"light"`.
/// (seaborn: `sns.set_theme`)
///
/// The choice lives in the `RUSMLAI_THEME` environment variable, so every notebook cell sees
/// it. A chart's own `.theme` wins over it. Call it before drawing, not while another thread
/// runs a query.
///
/// # Panics
///
/// If the name is unknown. The message lists the names.
///
/// # Examples
///
/// White cards from here on:
/// ```standalone_crate
/// use rusmlai::prelude::*;
///
/// set_theme("light");
/// let ds = read_csv("tests/fixtures/data/students.csv");
/// let svg = ds.countplot("batch", "").svg();
/// assert!(svg.contains("background-color:#FFFFFF"));
/// ```
///
/// A chart's `.theme` wins over the default:
/// ```standalone_crate
/// # use rusmlai::prelude::*;
/// set_theme("light");
/// let ds = read_csv("tests/fixtures/data/students.csv");
/// let svg = ds.countplot("batch", "").theme("dark").svg();
/// assert!(svg.contains("background-color:#15171C"));
/// ```
///
/// Edge case: an unknown name:
/// ```should_panic
/// rusmlai::prelude::set_theme("solarized"); // panics: Use one of: dark, classic, light
/// ```
#[track_caller]
pub fn set_theme(name: &str) {
    lookup(THEMES, "set_theme", name).or_fatal();
    set_default(THEME_VAR, name);
}

/// The group colors of every chart that follows. (seaborn: `sns.set_palette`)
///
/// The names are the ones of [`Figure::palette`]. The choice lives in the `RUSMLAI_PALETTE`
/// environment variable, so every notebook cell sees it. A chart's own `.palette` or
/// `.colors` wins over it. Call it before drawing, not while another thread runs a query.
///
/// # Panics
///
/// If the name is unknown. The message lists the names.
///
/// # Examples
///
/// Tableau colors from here on:
/// ```standalone_crate
/// use rusmlai::prelude::*;
///
/// set_palette("tab10");
/// let ds = read_csv("tests/fixtures/data/students.csv");
/// let svg = ds.countplot("batch", "").svg();
/// assert!(svg.contains("#1F77B4")); // the first tab10 color
/// ```
///
/// A chart's `.colors` wins over the default:
/// ```standalone_crate
/// # use rusmlai::prelude::*;
/// set_palette("tab10");
/// let ds = read_csv("tests/fixtures/data/students.csv");
/// let svg = ds.countplot("batch", "").colors(&["gold"]).svg();
/// assert!(svg.contains("#FFD700") && !svg.contains("#1F77B4"));
/// ```
///
/// Edge case: an unknown name:
/// ```should_panic
/// rusmlai::prelude::set_palette("viridis"); // panics: Use one of: rusmlai, greens, ...
/// ```
#[track_caller]
pub fn set_palette(name: &str) {
    lookup(PALETTES, "set_palette", name).or_fatal();
    set_default(PALETTE_VAR, name);
}

fn set_default(var: &str, name: &str) {
    // SAFETY: changing the environment is unsound only while another thread reads it. rusmlai
    // starts no threads, and Polars' threads read it only during a query; set_theme and
    // set_palette run between queries (in a notebook, in a cell of their own).
    unsafe { std::env::set_var(var, name) };
}

/// A default set by `set_theme` / `set_palette` (or the shell), if it names a known entry.
fn default<T: Copy>(table: &[(&str, T)], var: &'static str) -> Option<T> {
    let name: String = std::env::var(var).ok()?;
    lookup(table, var, &name).ok()
}

impl Style {
    /// The options of one chart on top of `set_theme` and `set_palette`.
    pub(crate) fn resolved(&self) -> Style {
        Style {
            theme: self.theme.or_else(|| default(THEMES, THEME_VAR)),
            palette: self
                .palette
                .clone()
                .or_else(|| default(PALETTES, PALETTE_VAR).map(strings)),
            ..self.clone()
        }
    }

    pub(crate) fn theme(&self) -> Theme {
        self.theme.unwrap_or(CLASSIC)
    }

    /// A single-series mark: `.color_map` for its name, then `.color`, then the palette's first
    /// color (seaborn's "C0").
    pub(crate) fn single_color(&self, name: &str) -> String {
        self.mapped(name)
            .or_else(|| self.color.clone())
            .unwrap_or_else(|| self.palette_color(0))
    }

    /// Group `i`: `.color_map` for its name, then the palette.
    pub(crate) fn group_color(&self, i: usize, name: &str) -> String {
        self.mapped(name).unwrap_or_else(|| self.palette_color(i))
    }

    /// Category `i` of a chart with no hue: `.color_map`, then `.color`, then the palette.
    pub(crate) fn category_color(&self, i: usize, name: &str) -> String {
        self.mapped(name)
            .or_else(|| self.color.clone())
            .unwrap_or_else(|| self.palette_color(i))
    }

    /// The palette in use: `.palette` / `.colors`, else the theme's.
    fn palette_colors(&self) -> Vec<String> {
        match &self.palette {
            Some(palette) if !palette.is_empty() => palette.clone(),
            _ => strings(self.theme().palette),
        }
    }

    fn palette_color(&self, i: usize) -> String {
        let palette: Vec<String> = self.palette_colors();
        palette[i % palette.len()].clone()
    }

    fn mapped(&self, name: &str) -> Option<String> {
        self.color_map
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, color)| color.clone())
    }
}

/// Chart options, the Rust form of Python's keyword arguments. Each one takes the figure and
/// returns it, so they chain: `ds.countplot("batch", "").title("Batches").palette("colorblind")`.
///
/// Each chart takes the options that fit it, and its method lists them. Any other option
/// panics at your line, for example `.cmap() does not work on a countplot chart`. A bad value
/// panics too, with the reason and the names you can use.
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

    /// `set` for an option that can't be wrong.
    #[track_caller]
    fn put(self, option: &'static str, apply: impl FnOnce(&mut Style)) -> Self {
        self.set(option, |s, _| {
            apply(s);
            Ok(())
        })
    }

    /// Replaces the title. (matplotlib: `ax.set_title`)
    ///
    /// The title also names the auto-saved PNG, minus the characters a file name can't hold.
    /// Every chart takes it.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.countplot("batch", "").title("Students per batch").svg();
    /// assert!(svg.contains("Students per batch"));
    /// ```
    ///
    /// With the other text options:
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds
    ///     .bar("batch", "cgpa", "", "")
    ///     .title("Mean CGPA")
    ///     .subtitle("by batch")
    ///     .xlabel("batch")
    ///     .ylabel("CGPA")
    ///     .svg();
    /// assert!(svg.contains("Mean CGPA") && svg.contains("by batch"));
    /// ```
    ///
    /// Edge case: `<`, `>` and `&` are escaped in the SVG:
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.countplot("batch", "").title("CGPA < 6 & failed").svg();
    /// assert!(svg.contains("CGPA &lt; 6 &amp; failed"));
    /// ```
    #[track_caller]
    pub fn title(self, text: &str) -> Self {
        self.put(".title", |s| s.title = Some(text.to_string()))
    }

    /// Replaces the line under the title. Every chart takes it.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.countplot("batch", "").subtitle("spring term").svg();
    /// assert!(svg.contains("spring term"));
    /// ```
    ///
    /// Edge case: `""` removes the default line (here the counts' unique and mode facts):
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.countplot("batch", "").subtitle("").svg();
    /// assert!(!svg.contains("unique=3"));
    /// ```
    #[track_caller]
    pub fn subtitle(self, text: &str) -> Self {
        self.put(".subtitle", |s| s.subtitle = Some(text.to_string()))
    }

    /// Replaces the x axis name, which is the column name by default. (matplotlib:
    /// `ax.set_xlabel`)
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.hist("cgpa", "", 10).xlabel("CGPA (out of 10)").svg();
    /// assert!(svg.contains("CGPA (out of 10)"));
    /// ```
    ///
    /// On sideways bars `.xlabel` still names the categories, now down the left:
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.countplot("batch", "").horizontal().xlabel("group").svg();
    /// assert!(svg.contains("group"));
    /// ```
    ///
    /// Edge case: a pie has no axes:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.pie("batch").xlabel("batch"); // panics: .xlabel() does not work on a pie chart
    /// ```
    #[track_caller]
    pub fn xlabel(self, text: &str) -> Self {
        self.put(".xlabel", |s| s.xlabel = Some(text.to_string()))
    }

    /// Replaces the y axis name. (matplotlib: `ax.set_ylabel`)
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.countplot("batch", "").ylabel("students").svg();
    /// assert!(svg.contains("students"));
    /// ```
    ///
    /// Edge case: the missing matrix has no y axis name to replace:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// // panics: .ylabel() does not work on a missing_matrix chart
    /// ds.missing_matrix().ylabel("row");
    /// ```
    #[track_caller]
    pub fn ylabel(self, text: &str) -> Self {
        self.put(".ylabel", |s| s.ylabel = Some(text.to_string()))
    }

    /// The card style: `"dark"` (the default) or `"light"`. It wins over [`set_theme`].
    /// Every chart takes it. (plotly: `template=`)
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.countplot("batch", "").theme("light").svg();
    /// assert!(svg.contains("background-color:#FFFFFF"));
    /// ```
    ///
    /// The light card also switches the default palette to darker steps, so pale colors stay
    /// visible on white:
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.countplot("batch", "").theme("light").svg();
    /// assert!(svg.contains("#BDA885")); // the light card's cream
    /// ```
    ///
    /// Edge case: an unknown name:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.countplot("batch", "").theme("solarized"); // panics: Use one of: dark, classic, light
    /// ```
    #[track_caller]
    pub fn theme(self, name: &str) -> Self {
        self.set(".theme", |s, _| {
            s.theme = Some(lookup(THEMES, ".theme", name)?);
            Ok(())
        })
    }

    /// The group colors: rusmlai (the default), greens, colorblind, tab10, deep, muted, pastel,
    /// bright or dark. It wins over [`set_palette`]. (seaborn: `palette=`)
    ///
    /// Bars, boxes and slices without a hue take one color per category. Hue groups take one
    /// color per group. A single-color chart (a histogram without hue) takes the first color.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.countplot("batch", "").palette("colorblind").svg();
    /// assert!(svg.contains("#E69F00")); // the first colorblind color
    /// ```
    ///
    /// `"C0"` to `"C9"` in a later `.color` pick from this palette, like matplotlib:
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.hist("cgpa", "", 10).palette("tab10").color("C3").svg();
    /// assert!(svg.contains("#D62728")); // tab10's red
    /// ```
    ///
    /// Edge case: an unknown name:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.countplot("batch", "").palette("viridis"); // panics: Use one of: rusmlai, greens, ...
    /// ```
    #[track_caller]
    pub fn palette(self, name: &str) -> Self {
        self.set(".palette", |s, _| {
            s.palette = Some(strings(lookup(PALETTES, ".palette", name)?));
            Ok(())
        })
    }

    /// Your own palette: hex codes or color names (see [`Figure::color`]).
    /// (plotly: `color_discrete_sequence=`)
    ///
    /// # Panics
    ///
    /// If the list is empty or a color is unknown.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.countplot("batch", "").colors(&["#199E70", "tomato", "tab:blue"]).svg();
    /// assert!(svg.contains("#FF6347") && svg.contains("#1F77B4"));
    /// ```
    ///
    /// A short list starts over: the third batch gets gold again.
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.countplot("batch", "").colors(&["gold", "navy"]).svg();
    /// assert_eq!(svg.matches("fill=\"#FFD700\"").count(), 2);
    /// ```
    ///
    /// Edge case: an empty list:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.countplot("batch", "").colors(&[]); // panics: .colors(): the list is empty
    /// ```
    #[track_caller]
    pub fn colors(self, colors: &[&str]) -> Self {
        self.set(".colors", |s, _| {
            if colors.is_empty() {
                return Err(invalid(".colors", "the list is empty".to_string()));
            }
            let current: Vec<String> = s.resolved().palette_colors();
            s.palette = Some(
                colors
                    .iter()
                    .map(|color| paint(".colors", color, &current))
                    .collect::<Result<Vec<String>, PlotError>>()?,
            );
            Ok(())
        })
    }

    /// One color for the whole chart: a hex code, a matplotlib / CSS name ("teal", "tomato"), a
    /// Tableau name ("tab:blue"), or "C0" to "C9" for a color of the current palette.
    /// (seaborn: `color=`)
    ///
    /// Hue groups keep their palette colors; `.color_map` sets those.
    ///
    /// # Panics
    ///
    /// If the color is unknown. The message suggests close names.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.hist("cgpa", "", 10).color("teal").svg();
    /// assert!(svg.contains("#008080"));
    /// ```
    ///
    /// The four ways to name a color:
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// for (color, hex) in [
    ///     ("#199E70", "#199E70"),  // hex
    ///     ("Tomato", "#FF6347"),   // CSS name, any case
    ///     ("tab:blue", "#1F77B4"), // Tableau
    ///     ("C1", "#FF7F0E"),       // the 2nd color of the palette, here tab10
    /// ] {
    ///     let svg = ds.hist("cgpa", "", 10).palette("tab10").color(color).svg();
    ///     assert!(svg.contains(hex), "{color}");
    /// }
    /// ```
    ///
    /// Edge case: a typo:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.hist("cgpa", "", 10).color("tomatoe"); // panics: ... Close names: tomato
    /// ```
    #[track_caller]
    pub fn color(self, color: &str) -> Self {
        self.set(".color", |s, _| {
            s.color = Some(paint(".color", color, &s.resolved().palette_colors())?);
            Ok(())
        })
    }

    /// A fixed color for named categories, or for hue groups when the chart has a hue. Other
    /// names keep their palette colors. (plotly: `color_discrete_map=`)
    ///
    /// # Panics
    ///
    /// If a name isn't a category of the chart, or a color is unknown.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.countplot("batch", "").color_map(&[("B13", "gold")]).svg();
    /// assert!(svg.contains("#FFD700"));
    /// ```
    ///
    /// With a hue, the names are the hue groups:
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds
    ///     .countplot("batch", "passed")
    ///     .color_map(&[("true", "seagreen"), ("false", "tomato")])
    ///     .svg();
    /// assert!(svg.contains("#2E8B57") && svg.contains("#FF6347"));
    /// ```
    ///
    /// Edge case: a name that isn't a category:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.countplot("batch", "").color_map(&[("B99", "gold")]); // panics: Categories: B13, B14, B15
    /// ```
    #[track_caller]
    pub fn color_map(self, map: &[(&str, &str)]) -> Self {
        self.set(".color_map", |s, chart| {
            let current: Vec<String> = s.resolved().palette_colors();
            let names: &[String] = match chart.categories(true) {
                [] => chart.categories(false),
                hue => hue,
            };
            for &(name, color) in map {
                known(".color_map", name, names)?;
                s.color_map
                    .push((name.to_string(), paint(".color_map", color, &current)?));
            }
            Ok(())
        })
    }

    /// The card size in pixels (720 × 469 by default). The card still gets wider when
    /// categories need room. Every chart takes it. (matplotlib: `figsize=`)
    ///
    /// # Panics
    ///
    /// If the size is below 200 × 150 or not finite.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.countplot("batch", "").size(600.0, 400.0).svg();
    /// assert!(svg.starts_with(r#"<svg width="600" height="400""#));
    /// ```
    ///
    /// Eleven names don't fit in 200 pixels, so the card grows wider:
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.countplot("name", "").size(200.0, 150.0).svg();
    /// assert!(!svg.starts_with(r#"<svg width="200""#));
    /// ```
    ///
    /// Edge case: too small:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.countplot("batch", "").size(100.0, 100.0); // panics: use at least 200 x 150
    /// ```
    #[track_caller]
    pub fn size(self, width: f64, height: f64) -> Self {
        self.set(".size", |s, _| {
            let ok: bool =
                width.is_finite() && height.is_finite() && width >= 200.0 && height >= 150.0;
            if !ok {
                return Err(invalid(
                    ".size",
                    format!("{width:?} x {height:?} is too small; use at least 200 x 150"),
                ));
            }
            s.size = Some((width, height));
            Ok(())
        })
    }

    /// Which x categories to draw, in this order. It wins over `.sort`. (seaborn: `order=`)
    ///
    /// # Panics
    ///
    /// If a name isn't a category of the chart. The message lists the categories.
    ///
    /// # Examples
    ///
    /// Two batches, B15 first:
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.countplot("batch", "").order(&["B15", "B13"]).svg();
    /// assert!(!svg.contains("batch = B14"));
    /// assert!(svg.find("batch = B15") < svg.find("batch = B13"));
    /// ```
    ///
    /// The categories of a number column are its values as text:
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.countplot("age", "").order(&["24", "19"]).svg();
    /// assert!(svg.find("age = 24") < svg.find("age = 19"));
    /// ```
    ///
    /// Edge case: a name that isn't a category:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.countplot("batch", "").order(&["B99"]); // panics: Categories: B13, B14, B15
    /// ```
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
    ///
    /// # Panics
    ///
    /// If the chart has no hue, or a name isn't one of its groups.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.countplot("batch", "passed").hue_order(&["false", "true"]).svg();
    /// assert!(svg.find("passed = false") < svg.find("passed = true"));
    /// ```
    ///
    /// A shorter list draws only those groups:
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.countplot("batch", "passed").hue_order(&["false"]).svg();
    /// assert!(!svg.contains("passed = true"));
    /// ```
    ///
    /// Edge case: a chart without a hue:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.countplot("batch", "").hue_order(&["true"]); // panics: this chart has no hue groups
    /// ```
    #[track_caller]
    pub fn hue_order(self, names: &[&str]) -> Self {
        self.set(".hue_order", |s, chart| {
            if chart.categories(true).is_empty() {
                return Err(invalid(
                    ".hue_order",
                    "this chart has no hue groups".to_string(),
                ));
            }
            for name in names {
                known(".hue_order", name, chart.categories(true))?;
            }
            s.hue_order = Some(strings(names));
            Ok(())
        })
    }

    /// Draws at most `n` categories (30 by default, 8 slices on a pie). The subtitle says how
    /// many are shown.
    ///
    /// # Panics
    ///
    /// If `n` is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.countplot("name", "").top(5).svg();
    /// assert!(svg.contains("showing 5 of 11"));
    /// ```
    ///
    /// On a pie, the rest join one "Other" slice:
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.pie("batch").top(1).svg();
    /// assert!(svg.contains("Other  8  (67%)"));
    /// ```
    ///
    /// Edge case: zero:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.countplot("name", "").top(0); // panics: n must be at least 1
    /// ```
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

    /// Turns the numbers on the bars, cells or slices on or off. They are on by default,
    /// except on histograms and on bars split by a hue. (plotly: `text_auto=`)
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let with: String = ds.countplot("batch", "").svg();
    /// let without: String = ds.countplot("batch", "").values(false).svg();
    /// assert!(without.len() < with.len());
    /// ```
    ///
    /// Numbers on bars split by a hue:
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let plain: String = ds.countplot("batch", "passed").svg();
    /// let numbered: String = ds.countplot("batch", "passed").values(true).svg();
    /// assert!(numbered.len() > plain.len());
    /// ```
    ///
    /// Edge case: a scatter has no numbers to show:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// // panics: .values() does not work on a scatter chart
    /// ds.scatter("age", "cgpa", "").values(true);
    /// ```
    #[track_caller]
    pub fn values(self, show: bool) -> Self {
        self.put(".values", |s| s.values = Some(show))
    }

    /// The range of the value axis. (matplotlib: `ax.set_ylim`)
    ///
    /// # Panics
    ///
    /// If `low` isn't below `high`, or either isn't finite.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.bar("batch", "cgpa", "", "").ylim(0.0, 10.0).svg();
    /// assert!(svg.contains(">10<")); // the top tick
    /// ```
    ///
    /// On sideways bars the value axis runs along the bottom, and `.ylim` still sets it:
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.bar("batch", "cgpa", "", "").horizontal().ylim(0.0, 10.0).svg();
    /// assert!(svg.contains(">10<"));
    /// ```
    ///
    /// Edge case: a range upside down:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.bar("batch", "cgpa", "", "").ylim(10.0, 0.0); // panics: (10.0, 0.0) is not a range
    /// ```
    #[track_caller]
    pub fn ylim(self, low: f64, high: f64) -> Self {
        self.set(".ylim", |s, _| {
            s.ylim = Some(limits(".ylim", low, high)?);
            Ok(())
        })
    }

    /// The range of the x axis on scatter, line, regplot, histogram, density and Q-Q charts. A
    /// histogram re-bins inside it and leaves out the rows outside. (matplotlib: `ax.set_xlim`,
    /// seaborn: `binrange=`)
    ///
    /// # Panics
    ///
    /// If `low` isn't below `high` or either isn't finite, or the chart has no number x axis.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.scatter("age", "cgpa", "").xlim(15.0, 30.0).svg();
    /// assert!(svg.contains(">30<")); // the last tick
    /// ```
    ///
    /// A histogram makes its bins inside the range and says how many rows it left out:
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.hist("cgpa", "", 4).xlim(6.0, 8.0).svg();
    /// assert!(svg.contains("cgpa = 6 to 6.50"));
    /// assert!(svg.contains("6 rows outside the range"));
    /// ```
    ///
    /// Edge case: bars have categories, not a number x axis:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.countplot("batch", "").xlim(0.0, 1.0); // panics: .xlim() does not work on a countplot chart
    /// ```
    #[track_caller]
    pub fn xlim(self, low: f64, high: f64) -> Self {
        self.set(".xlim", |s, _| {
            s.xlim = Some(limits(".xlim", low, high)?);
            Ok(())
        })
    }

    /// How many sampled points a strip, scatter, regplot, Q-Q or pair chart draws (2,500 by
    /// default, 1,000 on Q-Q and pair charts). At most 50,000, because every point is text in
    /// the SVG. The subtitle says how many rows the points stand for.
    ///
    /// # Panics
    ///
    /// If `n` is 0 or above 50,000.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.scatter("age", "cgpa", "").sample(5).svg();
    /// assert_eq!(svg.matches("<circle").count(), 5);
    /// assert!(svg.contains("5 of 10 rows"));
    /// ```
    ///
    /// A number above the row count draws every row:
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.scatter("age", "cgpa", "").sample(50_000).svg();
    /// assert_eq!(svg.matches("<circle").count(), 10);
    /// ```
    ///
    /// Edge case: zero:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.scatter("age", "cgpa", "").sample(0); // panics: n must be between 1 and 50000, got 0
    /// ```
    #[track_caller]
    pub fn sample(self, n: usize) -> Self {
        self.set(".sample", |s, _| {
            if !(1..=MAX_SAMPLE).contains(&n) {
                return Err(invalid(
                    ".sample",
                    format!("n must be between 1 and {MAX_SAMPLE}, got {n}"),
                ));
            }
            s.sample = Some(n);
            Ok(())
        })
    }

    /// Category order: "value" (tallest first), "name" (A to Z, numbers by size), or "count"
    /// (the default: most rows first, numbers ascending). `.order` wins over it.
    /// (seaborn: `order=` from a sorted index)
    ///
    /// # Panics
    ///
    /// If `by` is not one of the three names.
    ///
    /// # Examples
    ///
    /// The batch with the highest mean CGPA first:
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.bar("batch", "cgpa", "", "").sort("value").svg();
    /// assert!(svg.find("batch = B13") < svg.find("batch = B15")); // 7.92, then 7.87
    /// assert!(svg.find("batch = B15") < svg.find("batch = B14")); // then 6.38
    /// ```
    ///
    /// Names from A to Z:
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.countplot("name", "").sort("name").svg();
    /// assert!(svg.find("name = Aarav Shah") < svg.find("name = Zoë Müller"));
    /// ```
    ///
    /// Edge case: an unknown name:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.countplot("batch", "").sort("size"); // panics: Use one of: value, name, count
    /// ```
    #[track_caller]
    pub fn sort(self, by: &str) -> Self {
        self.set(".sort", |s, _| {
            s.sort = Some(lookup(SORTS, ".sort", by)?);
            Ok(())
        })
    }

    /// Bars sideways: categories down the left, values along the bottom. `.xlabel` still names
    /// the categories and `.ylabel` the values. Bar charts only. (seaborn: `orient="h"`)
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let up: String = ds.countplot("batch", "").svg();
    /// let sideways: String = ds.countplot("batch", "").horizontal().svg();
    /// assert_ne!(up, sideways);
    /// ```
    ///
    /// Long names read better down the left. `.hline` then marks the value axis:
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds
    ///     .bar("name", "cgpa", "", "")
    ///     .horizontal()
    ///     .sort("value")
    ///     .hline(6.0, "pass mark")
    ///     .svg();
    /// assert!(svg.contains("pass mark"));
    /// ```
    ///
    /// Edge case: a histogram's bins always stand up:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.hist("cgpa", "", 10).horizontal(); // panics: .horizontal() does not work on a hist chart
    /// ```
    #[track_caller]
    pub fn horizontal(self) -> Self {
        self.put(".horizontal", |s| s.horizontal = true)
    }

    /// Counts as a share of all rows, in percent. On `count` and `hist` only.
    /// (seaborn: `stat="percent"`)
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.countplot("batch", "").percent().svg();
    /// assert!(svg.contains("percent = 33.33%"));
    /// ```
    ///
    /// With a hue, each bar is a share of all rows, so all the bars add up to 100%:
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.countplot("batch", "passed").percent().svg();
    /// assert!(svg.contains("passed = true\npercent = 25%")); // 3 of 12 rows
    /// ```
    ///
    /// Edge case: a mean is not a count, so it has no share:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// // panics: .percent() does not work on a bar chart
    /// ds.bar("batch", "cgpa", "", "").percent();
    /// ```
    #[track_caller]
    pub fn percent(self) -> Self {
        self.put(".percent", |s| s.percent = true)
    }

    /// A smooth density line over a histogram, scaled to the counts. On `hist` only.
    /// (seaborn: `histplot(kde=True)`)
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.hist("cgpa", "", 10).kde().svg();
    /// assert!(svg.contains("density of cgpa"));
    /// ```
    ///
    /// One line per hue group, in the group's color:
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.hist("cgpa", "passed", 6).kde().colors(&["seagreen", "tomato"]).svg();
    /// assert!(svg.contains("#2E8B57") && svg.contains("#FF6347"));
    /// ```
    ///
    /// Edge case: bars of categories have no density:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.countplot("batch", "").kde(); // panics: .kde() does not work on a countplot chart
    /// ```
    #[track_caller]
    pub fn kde(self) -> Self {
        self.put(".kde", |s| s.kde = true)
    }

    /// Hue groups on top of each other instead of side by side. On `count` and `hist` only.
    /// Without a hue there is one group, so nothing changes. (plotly: `barmode="stack"`)
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let side: String = ds.countplot("batch", "passed").svg();
    /// let stacked: String = ds.countplot("batch", "passed").stacked().svg();
    /// assert_ne!(side, stacked);
    /// ```
    ///
    /// Stacked histograms in percent:
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.hist("cgpa", "passed", 6).stacked().percent().svg();
    /// assert!(svg.contains("percent"));
    /// ```
    ///
    /// Edge case: means don't add up, so they don't stack:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// // panics: .stacked() does not work on a bar chart
    /// ds.bar("batch", "cgpa", "passed", "").stacked();
    /// ```
    #[track_caller]
    pub fn stacked(self) -> Self {
        self.put(".stacked", |s| s.stacked = true)
    }

    /// A dashed line at value `y`, with a label (`""` for none). On horizontal bars it runs along
    /// the value axis. Call it again for more lines. (matplotlib: `ax.axhline`)
    ///
    /// # Panics
    ///
    /// If `y` isn't finite, or the chart has no value axis (pie, heatmap, pair and missing
    /// matrix charts).
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.bar("batch", "cgpa", "", "").hline(6.0, "pass mark").svg();
    /// assert!(svg.contains("pass mark"));
    /// ```
    ///
    /// Two lines:
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds
    ///     .boxplot("batch", "cgpa", "")
    ///     .hline(6.0, "pass mark")
    ///     .hline(8.5, "distinction")
    ///     .svg();
    /// assert!(svg.contains("pass mark") && svg.contains("distinction"));
    /// ```
    ///
    /// Edge case: NaN is not a place on the axis:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.bar("batch", "cgpa", "", "").hline(f64::NAN, ""); // panics: NaN is not a finite number
    /// ```
    #[track_caller]
    pub fn hline(self, y: f64, label: &str) -> Self {
        self.set(".hline", |s, _| {
            s.hlines.push((finite(".hline", y)?, label.to_string()));
            Ok(())
        })
    }

    /// A dashed line at `x`, with a label (`""` for none), on charts with a number x axis:
    /// scatter, line, regplot, histogram, density and Q-Q. Call it again for more lines.
    /// (matplotlib: `ax.axvline`)
    ///
    /// # Panics
    ///
    /// If `x` isn't finite, or the chart has no number x axis.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.scatter("age", "cgpa", "").vline(21.0, "median age").svg();
    /// assert!(svg.contains("median age"));
    /// ```
    ///
    /// Two lines on a histogram:
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds
    ///     .hist("cgpa", "", 10)
    ///     .vline(6.0, "pass mark")
    ///     .vline(8.5, "distinction")
    ///     .svg();
    /// assert!(svg.contains("pass mark") && svg.contains("distinction"));
    /// ```
    ///
    /// Edge case: bars have categories, not a number x axis:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.countplot("batch", "").vline(1.0, ""); // panics: .vline() does not work on a countplot chart
    /// ```
    #[track_caller]
    pub fn vline(self, x: f64, label: &str) -> Self {
        self.set(".vline", |s, _| {
            s.vlines.push((finite(".vline", x)?, label.to_string()));
            Ok(())
        })
    }

    /// Where the legend goes: "right" (the default), "bottom", or "none".
    /// (seaborn: `legend=False`)
    ///
    /// A legend lists the hue groups, so charts without a hue draw none.
    ///
    /// # Panics
    ///
    /// If `at` is not one of the three names.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let right: String = ds.countplot("batch", "passed").svg();
    /// let bottom: String = ds.countplot("batch", "passed").legend("bottom").svg();
    /// assert_ne!(right, bottom);
    /// ```
    ///
    /// No legend:
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let with: String = ds.countplot("batch", "passed").svg();
    /// let without: String = ds.countplot("batch", "passed").legend("none").svg();
    /// assert!(without.len() < with.len());
    /// ```
    ///
    /// Edge case: an unknown place:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.countplot("batch", "passed").legend("left"); // panics: Use one of: right, bottom, none
    /// ```
    #[track_caller]
    pub fn legend(self, at: &str) -> Self {
        self.set(".legend", |s, _| {
            s.legend = Some(lookup(LEGENDS, ".legend", at)?);
            Ok(())
        })
    }

    /// The color range of a heatmap: greens, viridis, magma, blues (low to high), or
    /// redgreen, coolwarm, rdbu (negative to positive). On `heatmap` and `corrplot` only.
    /// (seaborn: `cmap=`)
    ///
    /// A diverging range (negative to positive) puts its middle color at 0.
    ///
    /// # Panics
    ///
    /// If the name is unknown, or the chart isn't a heatmap.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.heatmap("batch", "passed", "cgpa", "mean").cmap("viridis").svg();
    /// assert!(svg.contains("Heatmap: batch × passed (mean cgpa)"));
    /// ```
    ///
    /// A diverging range on correlations, where 0 is the middle:
    /// ```
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// let svg = ds.corrplot().cmap("coolwarm").svg();
    /// assert!(svg.contains("#B40426")); // coolwarm's red, at a correlation of 1
    /// ```
    ///
    /// Edge case: bars have no color range:
    /// ```should_panic
    /// # use rusmlai::prelude::*;
    /// # let ds = read_csv("tests/fixtures/data/students.csv");
    /// ds.countplot("batch", "").cmap("viridis"); // panics: .cmap() does not work on a countplot chart
    /// ```
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

const LEGENDS: &[(&str, Legend)] = &[
    ("right", Legend::Right),
    ("bottom", Legend::Bottom),
    ("none", Legend::None),
];

fn limits(option: &'static str, low: f64, high: f64) -> Result<(f64, f64), PlotError> {
    if (high - low).is_finite() && low < high {
        Ok((low, high))
    } else {
        Err(invalid(
            option,
            format!("({low:?}, {high:?}) is not a range: use finite numbers with low below high"),
        ))
    }
}

fn finite(option: &'static str, v: f64) -> Result<f64, PlotError> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(invalid(option, format!("{v:?} is not a finite number")))
    }
}

const SORTS: &[(&str, Sort)] = &[
    ("value", Sort::Value),
    ("name", Sort::Name),
    ("count", Sort::Count),
];

/// A color as hex: a hex code, a matplotlib / CSS name, a Tableau name, or "C0".."C9" from
/// `palette` (the palette in effect at that point of the chain, like matplotlib's "C0").
fn paint(option: &'static str, color: &str, palette: &[String]) -> Result<String, PlotError> {
    let name: String = color.trim().to_ascii_lowercase();
    if let Some(digits) = name.strip_prefix('#') {
        if matches!(digits.len(), 3 | 6) && digits.chars().all(|c| c.is_ascii_hexdigit()) {
            return Ok(color.trim().to_string());
        }
    } else if let Some(i) = name.strip_prefix('c').and_then(|d| d.parse::<usize>().ok()) {
        return palette.get(i).cloned().ok_or_else(|| {
            invalid(
                option,
                format!("'{color}': the palette has only {} colors", palette.len()),
            )
        });
    } else if let Some(hex) = TABLEAU
        .iter()
        .chain(&NAMED)
        .find_map(|&(n, hex)| (n == name).then_some(hex))
    {
        return Ok(format!("#{hex}"));
    }
    let close: Vec<&str> = TABLEAU
        .iter()
        .chain(&NAMED)
        .map(|&(n, _)| n)
        .filter(|n| {
            name.len() >= 3
                && (n.contains(name.as_str())
                    || name.contains(n)
                    || name.get(..3).is_some_and(|p| n.starts_with(p)))
        })
        .take(6)
        .collect();
    let hint: String = if close.is_empty() {
        String::new()
    } else {
        format!(" Close names: {}.", close.join(", "))
    };
    Err(invalid(
        option,
        format!(
            "'{color}' is not a color. Use a hex code (\"#199E70\"), a name (\"teal\", \"tab:blue\") or \"C0\".{hint}"
        ),
    ))
}

const TABLEAU: [(&str, &str); 11] = [
    ("tab:blue", "1F77B4"),
    ("tab:orange", "FF7F0E"),
    ("tab:green", "2CA02C"),
    ("tab:red", "D62728"),
    ("tab:purple", "9467BD"),
    ("tab:brown", "8C564B"),
    ("tab:pink", "E377C2"),
    ("tab:gray", "7F7F7F"),
    ("tab:grey", "7F7F7F"),
    ("tab:olive", "BCBD22"),
    ("tab:cyan", "17BECF"),
];

/// The 148 CSS color names, which matplotlib also uses.
const NAMED: [(&str, &str); 148] = [
    ("aliceblue", "F0F8FF"),
    ("antiquewhite", "FAEBD7"),
    ("aqua", "00FFFF"),
    ("aquamarine", "7FFFD4"),
    ("azure", "F0FFFF"),
    ("beige", "F5F5DC"),
    ("bisque", "FFE4C4"),
    ("black", "000000"),
    ("blanchedalmond", "FFEBCD"),
    ("blue", "0000FF"),
    ("blueviolet", "8A2BE2"),
    ("brown", "A52A2A"),
    ("burlywood", "DEB887"),
    ("cadetblue", "5F9EA0"),
    ("chartreuse", "7FFF00"),
    ("chocolate", "D2691E"),
    ("coral", "FF7F50"),
    ("cornflowerblue", "6495ED"),
    ("cornsilk", "FFF8DC"),
    ("crimson", "DC143C"),
    ("cyan", "00FFFF"),
    ("darkblue", "00008B"),
    ("darkcyan", "008B8B"),
    ("darkgoldenrod", "B8860B"),
    ("darkgray", "A9A9A9"),
    ("darkgreen", "006400"),
    ("darkgrey", "A9A9A9"),
    ("darkkhaki", "BDB76B"),
    ("darkmagenta", "8B008B"),
    ("darkolivegreen", "556B2F"),
    ("darkorange", "FF8C00"),
    ("darkorchid", "9932CC"),
    ("darkred", "8B0000"),
    ("darksalmon", "E9967A"),
    ("darkseagreen", "8FBC8F"),
    ("darkslateblue", "483D8B"),
    ("darkslategray", "2F4F4F"),
    ("darkslategrey", "2F4F4F"),
    ("darkturquoise", "00CED1"),
    ("darkviolet", "9400D3"),
    ("deeppink", "FF1493"),
    ("deepskyblue", "00BFFF"),
    ("dimgray", "696969"),
    ("dimgrey", "696969"),
    ("dodgerblue", "1E90FF"),
    ("firebrick", "B22222"),
    ("floralwhite", "FFFAF0"),
    ("forestgreen", "228B22"),
    ("fuchsia", "FF00FF"),
    ("gainsboro", "DCDCDC"),
    ("ghostwhite", "F8F8FF"),
    ("gold", "FFD700"),
    ("goldenrod", "DAA520"),
    ("gray", "808080"),
    ("green", "008000"),
    ("greenyellow", "ADFF2F"),
    ("grey", "808080"),
    ("honeydew", "F0FFF0"),
    ("hotpink", "FF69B4"),
    ("indianred", "CD5C5C"),
    ("indigo", "4B0082"),
    ("ivory", "FFFFF0"),
    ("khaki", "F0E68C"),
    ("lavender", "E6E6FA"),
    ("lavenderblush", "FFF0F5"),
    ("lawngreen", "7CFC00"),
    ("lemonchiffon", "FFFACD"),
    ("lightblue", "ADD8E6"),
    ("lightcoral", "F08080"),
    ("lightcyan", "E0FFFF"),
    ("lightgoldenrodyellow", "FAFAD2"),
    ("lightgray", "D3D3D3"),
    ("lightgreen", "90EE90"),
    ("lightgrey", "D3D3D3"),
    ("lightpink", "FFB6C1"),
    ("lightsalmon", "FFA07A"),
    ("lightseagreen", "20B2AA"),
    ("lightskyblue", "87CEFA"),
    ("lightslategray", "778899"),
    ("lightslategrey", "778899"),
    ("lightsteelblue", "B0C4DE"),
    ("lightyellow", "FFFFE0"),
    ("lime", "00FF00"),
    ("limegreen", "32CD32"),
    ("linen", "FAF0E6"),
    ("magenta", "FF00FF"),
    ("maroon", "800000"),
    ("mediumaquamarine", "66CDAA"),
    ("mediumblue", "0000CD"),
    ("mediumorchid", "BA55D3"),
    ("mediumpurple", "9370DB"),
    ("mediumseagreen", "3CB371"),
    ("mediumslateblue", "7B68EE"),
    ("mediumspringgreen", "00FA9A"),
    ("mediumturquoise", "48D1CC"),
    ("mediumvioletred", "C71585"),
    ("midnightblue", "191970"),
    ("mintcream", "F5FFFA"),
    ("mistyrose", "FFE4E1"),
    ("moccasin", "FFE4B5"),
    ("navajowhite", "FFDEAD"),
    ("navy", "000080"),
    ("oldlace", "FDF5E6"),
    ("olive", "808000"),
    ("olivedrab", "6B8E23"),
    ("orange", "FFA500"),
    ("orangered", "FF4500"),
    ("orchid", "DA70D6"),
    ("palegoldenrod", "EEE8AA"),
    ("palegreen", "98FB98"),
    ("paleturquoise", "AFEEEE"),
    ("palevioletred", "DB7093"),
    ("papayawhip", "FFEFD5"),
    ("peachpuff", "FFDAB9"),
    ("peru", "CD853F"),
    ("pink", "FFC0CB"),
    ("plum", "DDA0DD"),
    ("powderblue", "B0E0E6"),
    ("purple", "800080"),
    ("rebeccapurple", "663399"),
    ("red", "FF0000"),
    ("rosybrown", "BC8F8F"),
    ("royalblue", "4169E1"),
    ("saddlebrown", "8B4513"),
    ("salmon", "FA8072"),
    ("sandybrown", "F4A460"),
    ("seagreen", "2E8B57"),
    ("seashell", "FFF5EE"),
    ("sienna", "A0522D"),
    ("silver", "C0C0C0"),
    ("skyblue", "87CEEB"),
    ("slateblue", "6A5ACD"),
    ("slategray", "708090"),
    ("slategrey", "708090"),
    ("snow", "FFFAFA"),
    ("springgreen", "00FF7F"),
    ("steelblue", "4682B4"),
    ("tan", "D2B48C"),
    ("teal", "008080"),
    ("thistle", "D8BFD8"),
    ("tomato", "FF6347"),
    ("turquoise", "40E0D0"),
    ("violet", "EE82EE"),
    ("wheat", "F5DEB3"),
    ("white", "FFFFFF"),
    ("whitesmoke", "F5F5F5"),
    ("yellow", "FFFF00"),
    ("yellowgreen", "9ACD32"),
];

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colors_by_hex_name_tableau_and_palette_index() {
        let palette: Vec<String> = strings(RUSMLAI);
        assert_eq!(paint(".color", "#199E70", &palette).unwrap(), "#199E70");
        assert_eq!(paint(".color", "Teal", &palette).unwrap(), "#008080");
        assert_eq!(paint(".color", "tab:blue", &palette).unwrap(), "#1F77B4");
        assert_eq!(paint(".color", "C1", &palette).unwrap(), RUSMLAI[1]);
        let typo: String = paint(".color", "tomatoe", &palette)
            .unwrap_err()
            .to_string();
        assert!(typo.contains("tomato"), "{typo}");
    }
}
