pub(crate) mod bar;
pub mod chart;
pub mod error;
pub mod result;
pub mod spec;
pub mod style;
pub(crate) mod svg;

use crate::{dataset::Dataset, display::Figure, error::OrFatal, plot::error::PlotError};

impl Dataset {
    /// `estimator(y)` per `x` category, split by `hue`. (seaborn: `barplot`)
    ///
    /// `y = ""` counts rows, `hue = ""` draws one series, `estimator = ""` means "mean".
    #[track_caller]
    pub fn bar(&self, x: &str, y: &str, hue: &str, estimator: &str) -> Figure {
        self.try_bar(x, y, hue, estimator).or_fatal()
    }

    pub fn try_bar(
        &self,
        x: &str,
        y: &str,
        hue: &str,
        estimator: &str,
    ) -> Result<Figure, PlotError> {
        bar::try_bar(self, x, y, hue, estimator)
    }

    /// Rows per `x` category, split by `hue`. (seaborn: `countplot`)
    #[track_caller]
    pub fn count(&self, x: &str, hue: &str) -> Figure {
        self.try_bar(x, "", hue, "").or_fatal()
    }

    /// One bar per numeric column (`column = ""`, `estimator` default "mean"), or min, q25,
    /// median, mean, q75, max and std of one column. From the cached stats.
    #[track_caller]
    pub fn describe_bar(&self, column: &str, estimator: &str) -> Figure {
        self.try_describe_bar(column, estimator).or_fatal()
    }

    pub fn try_describe_bar(&self, column: &str, estimator: &str) -> Result<Figure, PlotError> {
        bar::try_describe_bar(self, column, estimator)
    }

    /// The top 8 categories of `x` and "Other", from the cached counts. (matplotlib: `plt.pie`)
    #[track_caller]
    pub fn pie(&self, x: &str) -> Figure {
        self.try_pie(x).or_fatal()
    }

    pub fn try_pie(&self, x: &str) -> Result<Figure, PlotError> {
        bar::try_pie(self, x)
    }
}
