//! What a `Figure` draws: the chart's name, its default texts, and its numbers.

use crate::plot::{
    bar, density, dist, error::PlotError, heat, pair, result::PlotResult, style::Style, xy,
};

/// Default texts of a chart. `.title()`, `.subtitle()`, `.xlabel()` and `.ylabel()` replace them.
#[derive(Default)]
pub struct Texts {
    pub title: String,
    pub subtitle: String,
    /// x axis name, also the first line of each hover title.
    pub x: String,
    /// y axis name.
    pub y: String,
    /// What a value is, in hover titles: `mean exam_score`, `count`.
    pub value: String,
    pub hue: String,
}

impl Texts {
    /// `.title`, else the default.
    pub(crate) fn title(&self, style: &Style) -> String {
        style.title.clone().unwrap_or_else(|| self.title.clone())
    }

    /// `.subtitle`, else the default.
    pub(crate) fn subtitle<'a>(&'a self, style: &'a Style) -> &'a str {
        style.subtitle.as_deref().unwrap_or(&self.subtitle)
    }

    /// The axis names: `.xlabel` / `.ylabel`, else the column names.
    pub(crate) fn names<'a>(&'a self, style: &'a Style) -> (&'a str, &'a str) {
        (
            style.xlabel.as_deref().unwrap_or(&self.x),
            style.ylabel.as_deref().unwrap_or(&self.y),
        )
    }
}

/// A chart without its options: what `Figure` draws.
pub struct Chart {
    /// The method that made it (`bar`, `count`, `corrplot`, ...), for error messages.
    pub name: &'static str,
    pub texts: Texts,
    pub body: Body,
}

/// The numbers, one variant per renderer.
pub enum Body {
    /// `bar`, `count`, `describe_bar` and `missing`.
    Bar(PlotResult),
    Pie(PlotResult),
    /// `result` holds fine bins; `bins` is how many the chart draws.
    Hist {
        result: PlotResult,
        bins: usize,
    },
    Box(PlotResult),
    /// Sampled rows: `points` are (x index, hue index, y), up to `MAX_SAMPLE` of the `rows`
    /// rows with values; `.sample(n)` draws n of them.
    Strip {
        x_labels: Vec<String>,
        hue_labels: Vec<String>,
        points: Vec<(usize, usize, f64)>,
        rows: usize,
    },
    /// `heatmap` and `corrplot`. `fixed` pins the color range (corrplot: -1 to 1).
    Matrix {
        result: PlotResult,
        fixed: Option<(f64, f64)>,
    },
    /// `scatter`, `line` and `regplot`: `points` are (x, y, hue index); `fit` is (a, b) of
    /// y = a + b·x. Scatter and regplot points are sampled like `Strip` (`rows` > 0).
    Xy {
        hue_labels: Vec<String>,
        points: Vec<(f64, f64, usize)>,
        rows: usize,
        fit: Option<(f64, f64)>,
    },
    /// `kdeplot`: the fine bins of `hist`, smoothed at render.
    Kde(PlotResult),
    /// `violinplot`: the cached boxes, and per `[x][hue]` a sampled density as (y, density).
    Violin {
        result: PlotResult,
        shapes: Vec<Vec<Vec<(f64, f64)>>>,
        shown: usize,
        rows: usize,
    },
    /// `qqplot`: (normal quantile, value) pairs, and the normal line (mean, std).
    Qq {
        points: Vec<(f64, f64)>,
        line: (f64, f64),
        rows: usize,
    },
    /// `pairplot`: sampled values per column, and each row's hue group.
    Pair {
        columns: Vec<String>,
        values: Vec<Vec<f64>>,
        hue_labels: Vec<String>,
        hues: Vec<Option<usize>>,
        rows: usize,
    },
    /// `missing_matrix`: null shares per band of rows, and each column's null count.
    Missing {
        result: PlotResult,
        nulls: Vec<usize>,
        rows: usize,
    },
}

impl Chart {
    pub(crate) fn render(&self, style: &Style) -> String {
        let texts: &Texts = &self.texts;
        match &self.body {
            Body::Bar(result) => {
                let per_category: bool = !matches!(self.name, "describe_bar" | "missing");
                bar::render(texts, result, per_category, style)
            }
            Body::Pie(result) => bar::render_pie(texts, result, style),
            Body::Hist { result, bins } => dist::render_hist(texts, result, *bins, style),
            Body::Box(result) => dist::render_box(texts, result, style),
            Body::Strip {
                x_labels,
                hue_labels,
                points,
                rows,
            } => dist::render_strip(texts, x_labels, hue_labels, points, *rows, style),
            Body::Matrix { result, fixed } => heat::render_matrix(texts, result, *fixed, style),
            Body::Xy {
                hue_labels,
                points,
                rows,
                fit,
            } => xy::render_xy(self.name, texts, hue_labels, points, *rows, *fit, style),
            Body::Kde(result) => density::render_kde(texts, result, style),
            Body::Violin {
                result,
                shapes,
                shown,
                rows,
            } => density::render_violin(texts, result, shapes, *shown, *rows, style),
            Body::Qq { points, line, rows } => {
                density::render_qq(texts, points, *line, *rows, style)
            }
            Body::Pair {
                columns,
                values,
                hue_labels,
                hues,
                rows,
            } => pair::render_pair(texts, columns, values, hue_labels, hues, *rows, style),
            Body::Missing {
                result,
                nulls,
                rows,
            } => heat::render_missing(texts, result, nulls, *rows, style),
        }
    }

    /// The options each chart accepts (the table in docs/plots.md).
    pub(crate) fn check(&self, option: &'static str) -> Result<(), PlotError> {
        let allowed: bool = match &self.body {
            // .percent and .stacked need counts.
            Body::Bar(_) => {
                !matches!(option, ".cmap" | ".xlim" | ".sample" | ".vline" | ".kde")
                    && (self.name == "count" || !matches!(option, ".percent" | ".stacked"))
            }
            Body::Pie(_) => matches!(
                option,
                ".title"
                    | ".subtitle"
                    | ".theme"
                    | ".size"
                    | ".palette"
                    | ".colors"
                    | ".color_map"
                    | ".order"
                    | ".top"
                    | ".values"
                    | ".sort"
            ),
            Body::Hist { .. } => !matches!(
                option,
                ".order" | ".top" | ".cmap" | ".sort" | ".sample" | ".horizontal"
            ),
            Body::Box(_) | Body::Violin { .. } | Body::Strip { .. } => {
                let sampled: bool = matches!(self.body, Body::Strip { .. });
                !matches!(
                    option,
                    ".values"
                        | ".cmap"
                        | ".xlim"
                        | ".horizontal"
                        | ".percent"
                        | ".stacked"
                        | ".vline"
                        | ".kde"
                ) && (sampled || option != ".sample")
            }
            Body::Matrix { .. } => matches!(
                option,
                ".title"
                    | ".subtitle"
                    | ".theme"
                    | ".size"
                    | ".xlabel"
                    | ".ylabel"
                    | ".cmap"
                    | ".values"
            ),
            Body::Xy { .. } => {
                !matches!(
                    option,
                    ".order"
                        | ".top"
                        | ".values"
                        | ".cmap"
                        | ".sort"
                        | ".horizontal"
                        | ".percent"
                        | ".stacked"
                        | ".kde"
                ) && !(option == ".sample" && self.name == "line")
            }
            Body::Kde(_) => matches!(
                option,
                ".title"
                    | ".subtitle"
                    | ".theme"
                    | ".size"
                    | ".xlabel"
                    | ".ylabel"
                    | ".palette"
                    | ".colors"
                    | ".color"
                    | ".color_map"
                    | ".hue_order"
                    | ".xlim"
                    | ".ylim"
                    | ".hline"
                    | ".vline"
                    | ".legend"
            ),
            Body::Qq { .. } => matches!(
                option,
                ".title"
                    | ".subtitle"
                    | ".theme"
                    | ".size"
                    | ".xlabel"
                    | ".ylabel"
                    | ".palette"
                    | ".colors"
                    | ".color"
                    | ".xlim"
                    | ".ylim"
                    | ".hline"
                    | ".vline"
                    | ".sample"
            ),
            Body::Pair { .. } => matches!(
                option,
                ".title"
                    | ".subtitle"
                    | ".theme"
                    | ".size"
                    | ".palette"
                    | ".colors"
                    | ".color"
                    | ".color_map"
                    | ".hue_order"
                    | ".sample"
                    | ".legend"
            ),
            Body::Missing { .. } => matches!(
                option,
                ".title" | ".subtitle" | ".theme" | ".size" | ".palette" | ".colors" | ".color"
            ),
        };
        if allowed {
            Ok(())
        } else {
            Err(PlotError::OptionNotAllowed {
                option,
                chart: self.name,
            })
        }
    }

    /// Category names: the x labels (pie slices), or the hue labels (empty without a hue).
    pub(crate) fn categories(&self, hue: bool) -> &[String] {
        match (&self.body, hue) {
            (Body::Bar(result) | Body::Pie(result), _) => match (result, hue) {
                (PlotResult::Series { labels, .. }, false) => labels,
                (PlotResult::Clustered { x_labels, .. }, false) => x_labels,
                (PlotResult::Clustered { hue_labels, .. }, true) => hue_labels,
                _ => &[],
            },
            (
                Body::Hist {
                    result: PlotResult::Hist { names, .. },
                    ..
                }
                | Body::Kde(PlotResult::Hist { names, .. }),
                true,
            ) if !self.texts.hue.is_empty() => names,
            (
                Body::Box(PlotResult::Boxes { x_labels, .. })
                | Body::Violin {
                    result: PlotResult::Boxes { x_labels, .. },
                    ..
                },
                false,
            ) => x_labels,
            (
                Body::Box(PlotResult::Boxes { hue_labels, .. })
                | Body::Violin {
                    result: PlotResult::Boxes { hue_labels, .. },
                    ..
                },
                true,
            ) => hue_labels,
            (Body::Strip { x_labels, .. }, false) => x_labels,
            (
                Body::Strip { hue_labels, .. }
                | Body::Xy { hue_labels, .. }
                | Body::Pair { hue_labels, .. },
                true,
            ) => hue_labels,
            _ => &[],
        }
    }
}
