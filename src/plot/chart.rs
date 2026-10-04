//! What a `Figure` draws: one variant per chart, with its spec, its numbers and its subtitle.

use crate::plot::{bar, error::PlotError, result::PlotResult, spec::BarSpec, style::Style};

pub enum Chart {
    Bar {
        spec: BarSpec,
        result: PlotResult,
        subtitle: String,
    },
}

impl Chart {
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Chart::Bar { spec, .. } if spec.y.is_none() => "count",
            Chart::Bar { .. } => "bar",
        }
    }

    pub(crate) fn title(&self, style: &Style) -> String {
        match self {
            Chart::Bar { spec, .. } => bar::title(spec, style),
        }
    }

    pub(crate) fn render(&self, style: &Style) -> String {
        match self {
            Chart::Bar {
                spec,
                result,
                subtitle,
            } => bar::render(spec, result, subtitle, style),
        }
    }

    /// The options each chart accepts (the table in docs/plots.md).
    pub(crate) fn check(&self, option: &'static str) -> Result<(), PlotError> {
        let allowed: bool = match self {
            Chart::Bar { .. } => !matches!(option, ".cmap" | ".xlim"),
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

    /// Category names: the x labels, or the hue labels (empty without a hue).
    pub(crate) fn categories(&self, hue: bool) -> &[String] {
        let Chart::Bar { result, .. } = self;
        match (result, hue) {
            (PlotResult::Series { labels, .. }, false) => labels,
            (PlotResult::Clustered { x_labels, .. }, false) => x_labels,
            (PlotResult::Clustered { hue_labels, .. }, true) => hue_labels,
            _ => &[],
        }
    }
}
