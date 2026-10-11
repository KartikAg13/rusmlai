//! What a `Figure` draws: one variant per chart, with its texts, its numbers and its subtitle.

use crate::plot::{bar, dist, error::PlotError, heat, result::PlotResult, style::Style, xy};

/// Default texts of a chart. `.title()`, `.xlabel()` and `.ylabel()` replace them.
pub struct Texts {
    pub title: String,
    /// x axis name, also the first line of each hover title.
    pub x: String,
    /// y axis name.
    pub y: String,
    /// What a value is, in hover titles: `mean exam_score`, `count`.
    pub value: String,
    pub hue: String,
}

pub enum Chart {
    /// `bar`, `count` and `describe_bar`.
    Bar {
        name: &'static str,
        texts: Texts,
        result: PlotResult,
        subtitle: String,
    },
    Pie {
        texts: Texts,
        result: PlotResult,
        subtitle: String,
    },
    /// `result` holds fine bins; `bins` is how many the chart draws.
    Hist {
        texts: Texts,
        result: PlotResult,
        bins: usize,
        subtitle: String,
    },
    Box {
        texts: Texts,
        result: PlotResult,
        subtitle: String,
    },
    /// Sampled rows: `points` are (x index, hue index, y), up to `MAX_SAMPLE` of the `rows`
    /// rows with values; `.sample(n)` draws n of them.
    Strip {
        texts: Texts,
        x_labels: Vec<String>,
        hue_labels: Vec<String>,
        points: Vec<(usize, usize, f64)>,
        rows: usize,
        subtitle: String,
    },
    /// `heatmap` and `corrplot`. `fixed` pins the color range (corrplot: -1 to 1).
    Matrix {
        name: &'static str,
        texts: Texts,
        result: PlotResult,
        fixed: Option<(f64, f64)>,
        subtitle: String,
    },
    /// `scatter`, `line` and `regplot`: `points` are (x, y, hue index); `fit` is (a, b) of
    /// y = a + b·x. Scatter and regplot points are sampled like `Strip` (`rows` > 0).
    Xy {
        name: &'static str,
        texts: Texts,
        hue_labels: Vec<String>,
        points: Vec<(f64, f64, usize)>,
        rows: usize,
        fit: Option<(f64, f64)>,
        subtitle: String,
    },
}

impl Chart {
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Chart::Bar { name, .. } => name,
            Chart::Pie { .. } => "pie",
            Chart::Hist { .. } => "hist",
            Chart::Box { .. } => "boxplot",
            Chart::Strip { .. } => "stripplot",
            Chart::Matrix { name, .. } | Chart::Xy { name, .. } => name,
        }
    }

    pub(crate) fn title(&self, style: &Style) -> String {
        let (Chart::Bar { texts, .. }
        | Chart::Pie { texts, .. }
        | Chart::Hist { texts, .. }
        | Chart::Box { texts, .. }
        | Chart::Strip { texts, .. }
        | Chart::Matrix { texts, .. }
        | Chart::Xy { texts, .. }) = self;
        style.title.clone().unwrap_or_else(|| texts.title.clone())
    }

    pub(crate) fn render(&self, style: &Style) -> String {
        match self {
            Chart::Bar {
                name,
                texts,
                result,
                subtitle,
            } => bar::render(texts, result, subtitle, *name != "describe_bar", style),
            Chart::Pie {
                texts,
                result,
                subtitle,
            } => bar::render_pie(texts, result, subtitle, style),
            Chart::Hist {
                texts,
                result,
                bins,
                subtitle,
            } => dist::render_hist(texts, result, *bins, subtitle, style),
            Chart::Box {
                texts,
                result,
                subtitle,
            } => dist::render_box(texts, result, subtitle, style),
            Chart::Strip {
                texts,
                x_labels,
                hue_labels,
                points,
                rows,
                subtitle,
            } => dist::render_strip(texts, x_labels, hue_labels, points, *rows, subtitle, style),
            Chart::Matrix {
                texts,
                result,
                fixed,
                subtitle,
                ..
            } => heat::render_matrix(texts, result, *fixed, subtitle, style),
            Chart::Xy {
                name,
                texts,
                hue_labels,
                points,
                rows,
                fit,
                subtitle,
            } => xy::render_xy(name, texts, hue_labels, points, *rows, *fit, subtitle, style),
        }
    }

    /// The options each chart accepts (the table in docs/plots.md).
    pub(crate) fn check(&self, option: &'static str) -> Result<(), PlotError> {
        let allowed: bool = match self {
            // .percent and .stacked need counts.
            Chart::Bar { name, .. } => {
                !matches!(option, ".cmap" | ".xlim" | ".sample" | ".vline")
                    && (*name == "count" || !matches!(option, ".percent" | ".stacked"))
            }
            Chart::Pie { .. } => matches!(
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
            Chart::Hist { .. } => !matches!(
                option,
                ".order" | ".top" | ".cmap" | ".sort" | ".sample" | ".horizontal"
            ),
            Chart::Box { .. } | Chart::Strip { .. } => {
                let sampled: bool = matches!(self, Chart::Strip { .. });
                !matches!(
                    option,
                    ".values"
                        | ".cmap"
                        | ".xlim"
                        | ".horizontal"
                        | ".percent"
                        | ".stacked"
                        | ".vline"
                ) && (sampled || option != ".sample")
            }
            Chart::Matrix { .. } => matches!(
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
            Chart::Xy { name, .. } => {
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
                ) && !(option == ".sample" && *name == "line")
            }
        };
        if allowed {
            Ok(())
        } else {
            Err(PlotError::OptionNotAllowed {
                option,
                chart: self.name(),
            })
        }
    }

    /// Category names: the x labels (pie slices), or the hue labels (empty without a hue).
    pub(crate) fn categories(&self, hue: bool) -> &[String] {
        match (self, hue) {
            (Chart::Bar { result, .. } | Chart::Pie { result, .. }, _) => match (result, hue) {
                (PlotResult::Series { labels, .. }, false) => labels,
                (PlotResult::Clustered { x_labels, .. }, false) => x_labels,
                (PlotResult::Clustered { hue_labels, .. }, true) => hue_labels,
                _ => &[],
            },
            (
                Chart::Hist {
                    texts,
                    result: PlotResult::Hist { names, .. },
                    ..
                },
                true,
            ) if !texts.hue.is_empty() => names,
            (
                Chart::Box {
                    result: PlotResult::Boxes { x_labels, .. },
                    ..
                },
                false,
            ) => x_labels,
            (
                Chart::Box {
                    result: PlotResult::Boxes { hue_labels, .. },
                    ..
                },
                true,
            ) => hue_labels,
            (Chart::Strip { x_labels, .. }, false) => x_labels,
            (Chart::Strip { hue_labels, .. }, true) => hue_labels,
            (Chart::Xy { hue_labels, .. }, true) => hue_labels,
            _ => &[],
        }
    }
}
