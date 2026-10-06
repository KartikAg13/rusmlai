//! What a `Figure` draws: one variant per chart, with its texts, its numbers and its subtitle.

use crate::plot::{bar, error::PlotError, result::PlotResult, style::Style};

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
}

impl Chart {
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Chart::Bar { name, .. } => name,
            Chart::Pie { .. } => "pie",
        }
    }

    pub(crate) fn title(&self, style: &Style) -> String {
        let (Chart::Bar { texts, .. } | Chart::Pie { texts, .. }) = self;
        style.title.clone().unwrap_or_else(|| texts.title.clone())
    }

    pub(crate) fn render(&self, style: &Style) -> String {
        match self {
            Chart::Bar {
                texts,
                result,
                subtitle,
                ..
            } => bar::render(texts, result, subtitle, style),
            Chart::Pie {
                texts,
                result,
                subtitle,
            } => bar::render_pie(texts, result, subtitle, style),
        }
    }

    /// The options each chart accepts (the table in docs/plots.md).
    pub(crate) fn check(&self, option: &'static str) -> Result<(), PlotError> {
        let allowed: bool = match self {
            Chart::Bar { .. } => !matches!(option, ".cmap" | ".xlim"),
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
            ),
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
        let (Chart::Bar { result, .. } | Chart::Pie { result, .. }) = self;
        match (result, hue) {
            (PlotResult::Series { labels, .. }, false) => labels,
            (PlotResult::Clustered { x_labels, .. }, false) => x_labels,
            (PlotResult::Clustered { hue_labels, .. }, true) => hue_labels,
            _ => &[],
        }
    }
}
