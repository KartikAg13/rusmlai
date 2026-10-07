pub(crate) mod bar;
pub mod chart;
pub(crate) mod dist;
pub mod error;
pub(crate) mod heat;
pub mod result;
pub mod spec;
pub mod style;
pub(crate) mod svg;
pub(crate) mod xy;

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

    /// Counts of `column` in `bins` equal bins, split by `hue`. (seaborn: `histplot`)
    #[track_caller]
    pub fn hist(&self, column: &str, hue: &str, bins: usize) -> Figure {
        self.try_hist(column, hue, bins).or_fatal()
    }

    pub fn try_hist(&self, column: &str, hue: &str, bins: usize) -> Result<Figure, PlotError> {
        dist::try_hist(self, column, hue, bins)
    }

    /// Min, q25, median, q75 and max of `y` per `x` category, split by `hue`.
    /// `x = ""` draws one box from the cached stats. (seaborn: `boxplot`)
    #[track_caller]
    pub fn boxplot(&self, x: &str, y: &str, hue: &str) -> Figure {
        self.try_boxplot(x, y, hue).or_fatal()
    }

    pub fn try_boxplot(&self, x: &str, y: &str, hue: &str) -> Result<Figure, PlotError> {
        dist::try_boxplot(self, x, y, hue)
    }

    /// About 2,000 sampled values of `y`, spread inside each `x` category and colored by
    /// `hue`. (seaborn: `stripplot`)
    #[track_caller]
    pub fn stripplot(&self, x: &str, y: &str, hue: &str) -> Figure {
        self.try_stripplot(x, y, hue).or_fatal()
    }

    pub fn try_stripplot(&self, x: &str, y: &str, hue: &str) -> Result<Figure, PlotError> {
        dist::try_stripplot(self, x, y, hue)
    }

    /// `estimator(value)` for each pair of `x` and `y` categories; `value = ""` counts rows.
    /// (seaborn: `heatmap` of a pivot table)
    #[track_caller]
    pub fn heatmap(&self, x: &str, y: &str, value: &str, estimator: &str) -> Figure {
        self.try_heatmap(x, y, value, estimator).or_fatal()
    }

    pub fn try_heatmap(
        &self,
        x: &str,
        y: &str,
        value: &str,
        estimator: &str,
    ) -> Result<Figure, PlotError> {
        heat::try_heatmap(self, x, y, value, estimator)
    }

    /// Pearson correlation of every pair of numeric columns. (seaborn: `heatmap(df.corr())`)
    #[track_caller]
    pub fn corrplot(&self) -> Figure {
        self.try_corrplot().or_fatal()
    }

    pub fn try_corrplot(&self) -> Result<Figure, PlotError> {
        heat::try_corrplot(self)
    }

    /// About 2,000 sampled rows of `y` against `x`, colored by `hue`. (seaborn: `scatterplot`)
    #[track_caller]
    pub fn scatter(&self, x: &str, y: &str, hue: &str) -> Figure {
        self.try_scatter(x, y, hue).or_fatal()
    }

    pub fn try_scatter(&self, x: &str, y: &str, hue: &str) -> Result<Figure, PlotError> {
        xy::try_scatter(self, x, y, hue)
    }

    /// The mean of `y` at each `x`, sorted by `x`, one line per `hue` group.
    /// (seaborn: `lineplot`)
    #[track_caller]
    pub fn line(&self, x: &str, y: &str, hue: &str) -> Figure {
        self.try_line(x, y, hue).or_fatal()
    }

    pub fn try_line(&self, x: &str, y: &str, hue: &str) -> Result<Figure, PlotError> {
        xy::try_line(self, x, y, hue)
    }

    /// A scatter plus a least-squares line fitted on all rows. (seaborn: `regplot`)
    #[track_caller]
    pub fn regplot(&self, x: &str, y: &str, hue: &str) -> Figure {
        self.try_regplot(x, y, hue).or_fatal()
    }

    pub fn try_regplot(&self, x: &str, y: &str, hue: &str) -> Result<Figure, PlotError> {
        xy::try_regplot(self, x, y, hue)
    }
}
