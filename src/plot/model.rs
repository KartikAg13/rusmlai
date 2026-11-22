//! Charts of a model's predictions, named after scikit-learn's display classes:
//! [`ConfusionMatrixDisplay`], [`RocCurveDisplay`], [`PrecisionRecallDisplay`],
//! [`PredictionErrorDisplay`] and [`CalibrationDisplay`]. Each has `from_predictions` (the
//! true and the predicted values) and `from_estimator` (a fitted model, the rows and the
//! target column), and draws with the renderers of the data charts: the confusion matrix is a
//! heatmap, the curves are line charts and the prediction errors a scatter chart. Options
//! chain on the [`Figure`] as on any chart. A figure from predictions has no CSV, so a
//! program saves it to `metadata/plots/` in the current folder; one from an estimator saves it
//! next to the rows' CSV.

use std::ffi::OsStr;

use polars::prelude::{DataType, Series};

use crate::{
    dataset::Dataset,
    display::Figure,
    error::OrFatal,
    metrics::{average_precision, labeled_confusion_matrix, try_calibration_curve, try_precision_recall_curve, try_roc_curve, unique_labels},
    models::{Classifier, ModelError, Regressor},
    plot::{
        chart::{Body, Chart, Texts},
        error::PlotError,
        result::PlotResult,
        xy::thin_line,
    },
};

/// The most classes [`ConfusionMatrixDisplay`] draws.
const MAX_CLASSES: usize = 100;

/// The target column of `dataset`.
fn target(dataset: &Dataset, target: &str) -> Result<Series, PlotError> {
    Ok(dataset.column(target)?.as_materialized_series().clone())
}

/// The CSV a chart of `dataset` saves next to.
fn csv(dataset: &Dataset) -> Option<&OsStr> {
    dataset.metadata.filepath.as_deref()
}

/// The score of the positive class from `predict_proba`: the column of `pos_label`, or of
/// the greater of two classes (the last column) for `""`.
fn positive_score(model: &impl Classifier, dataset: &Dataset, pos_label: &str) -> Result<Series, PlotError> {
    let proba: Dataset = model.try_predict_proba(dataset)?;
    let columns = proba.dataframe().columns();
    let column = match pos_label {
        "" if columns.len() == 2 => &columns[1],
        "" => {
            return Err(PlotError::from(ModelError::InvalidSetting {
                setting: "pos_label",
                value: format!("\"\" with {} classes", columns.len()),
                expected: "a class: the curve is of one class against the rest",
            }));
        }
        label => proba.column(label)?,
    };
    Ok(column.as_materialized_series().clone())
}

/// A line chart of `points` (x ascending), with a reference line y = a + b·x when `fit` is set.
fn line(name: &'static str, texts: Texts, points: &[(f64, f64)], fit: Option<(f64, f64)>, csv: Option<&OsStr>) -> Figure {
    let points: Vec<(f64, f64, usize)> = thin_line(points).into_iter().map(|(x, y)| (x, y, 0)).collect();
    let body: Body = Body::Xy { hue_labels: Vec::new(), points, rows: 0, fit };
    Figure::new(Chart { name, texts, body }, csv)
}

/// The area under a curve of points in x order, by trapezoids (scikit-learn's `auc`).
fn area(xs: &[f64], ys: &[f64]) -> f64 {
    let pairs = xs.windows(2).zip(ys.windows(2));
    pairs.map(|(x, y)| (x[1] - x[0]) * (y[0] + y[1]) / 2.0).sum()
}

/// The confusion matrix as a heatmap with the count (or share) in each cell: true labels down,
/// predicted labels across (scikit-learn's `ConfusionMatrixDisplay`).
pub struct ConfusionMatrixDisplay;

impl ConfusionMatrixDisplay {
    /// The confusion matrix of `y_pred` against `y_true`. `normalize` is that of
    /// [`confusion_matrix`](crate::metrics::confusion_matrix): `""`, `"true"`, `"pred"` or
    /// `"all"`. `.values(false)` hides the numbers in the cells.
    ///
    /// ```
    /// use rusmlai::polars::prelude::*;
    /// use rusmlai::prelude::*;
    /// let y_true = Series::new("y".into(), ["cat", "dog", "cat", "bird"]);
    /// let y_pred = Series::new("p".into(), ["cat", "cat", "cat", "bird"]);
    /// let svg = ConfusionMatrixDisplay::from_predictions(&y_true, &y_pred, "").svg();
    /// assert!(svg.contains("True label = dog"));
    /// ```
    ///
    /// Normalized over the true labels, each row sums to 1:
    /// ```
    /// use rusmlai::polars::prelude::*;
    /// use rusmlai::prelude::*;
    /// let y_true = Series::new("y".into(), [0, 0, 1, 1]);
    /// let y_pred = Series::new("p".into(), [0, 1, 1, 1]);
    /// let svg = ConfusionMatrixDisplay::from_predictions(&y_true, &y_pred, "true").title("Shares").svg();
    /// assert!(svg.contains("Shares"));
    /// ```
    ///
    /// Edge case: an unknown `normalize` panics:
    /// ```should_panic
    /// use rusmlai::polars::prelude::*;
    /// use rusmlai::prelude::*;
    /// let y = Series::new("y".into(), [0, 1]);
    /// ConfusionMatrixDisplay::from_predictions(&y, &y, "rows"); // panics: normalize
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import matplotlib.pyplot as plt
    /// from sklearn.metrics import ConfusionMatrixDisplay
    /// y_true, y_pred = ["cat", "dog", "cat", "bird"], ["cat", "cat", "cat", "bird"]
    /// display = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    /// assert display.confusion_matrix.tolist() == [[1, 0, 0], [0, 2, 0], [0, 1, 0]]
    /// plt.show()
    /// ```
    #[track_caller]
    pub fn from_predictions(y_true: &Series, y_pred: &Series, normalize: &str) -> Figure {
        Self::try_from_predictions(y_true, y_pred, normalize).or_fatal()
    }

    /// [`ConfusionMatrixDisplay::from_predictions`] that returns an error instead of panicking.
    ///
    /// # Errors
    ///
    /// Those of [`try_confusion_matrix`](crate::metrics::try_confusion_matrix), as
    /// `PlotError::Model`.
    pub fn try_from_predictions(y_true: &Series, y_pred: &Series, normalize: &str) -> Result<Figure, PlotError> {
        Self::figure(y_true, y_pred, normalize, None)
    }

    /// [`ConfusionMatrixDisplay::from_predictions`] of `model`'s predictions on `dataset`
    /// against its column `target`.
    ///
    /// ```
    /// use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv").drop(&["id", "name", "batch"]).dropna(&[]);
    /// let mut model = DecisionTreeClassifier::new().max_depth(2);
    /// model.fit(&ds, "passed");
    /// let svg = ConfusionMatrixDisplay::from_estimator(&model, &ds, "passed", "").svg();
    /// assert!(svg.contains("Predicted label = true"));
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import matplotlib.pyplot as plt
    /// import pandas as pd
    /// from sklearn.metrics import ConfusionMatrixDisplay
    /// from sklearn.tree import DecisionTreeClassifier
    /// ds = pd.read_csv("tests/fixtures/data/students.csv").drop(columns=["id", "name", "batch"]).dropna()
    /// x, y = ds.drop(columns=["passed"]), ds["passed"]
    /// model = DecisionTreeClassifier(max_depth=2, random_state=42).fit(x, y)
    /// ConfusionMatrixDisplay.from_estimator(model, x, y)
    /// plt.show()
    /// ```
    #[track_caller]
    pub fn from_estimator(model: &impl Classifier, dataset: &Dataset, target: &str, normalize: &str) -> Figure {
        Self::try_from_estimator(model, dataset, target, normalize).or_fatal()
    }

    /// [`ConfusionMatrixDisplay::from_estimator`] that returns an error instead of panicking.
    ///
    /// # Errors
    ///
    /// Those of `model.try_predict` and of [`ConfusionMatrixDisplay::try_from_predictions`];
    /// `PlotError::Dataset` when `target` is not a column.
    pub fn try_from_estimator(model: &impl Classifier, dataset: &Dataset, target: &str, normalize: &str) -> Result<Figure, PlotError> {
        let y_true: Series = self::target(dataset, target)?;
        Self::figure(&y_true, &model.try_predict(dataset)?, normalize, csv(dataset))
    }

    fn figure(y_true: &Series, y_pred: &Series, normalize: &str, csv: Option<&OsStr>) -> Result<Figure, PlotError> {
        // Counted before the matrix: 5,000 labels would be 25 million cells.
        let classes: usize = unique_labels(y_true, y_pred)?.0.len();
        if classes > MAX_CLASSES {
            return Err(PlotError::Undefined(format!(
                "{classes} classes; a confusion matrix draws at most {MAX_CLASSES} (confusion_matrix has no limit)"
            )));
        }
        let (labels, grid): (Vec<String>, Vec<Vec<f64>>) = labeled_confusion_matrix(y_true, y_pred, normalize)?;
        let texts: Texts = Texts {
            title: "Confusion Matrix".to_string(),
            x: "Predicted label".to_string(),
            y: "True label".to_string(),
            value: if normalize.is_empty() { "count" } else { "share" }.to_string(),
            ..Texts::default()
        };
        let result: PlotResult = PlotResult::Matrix { x_labels: labels.clone(), y_labels: labels, grid };
        let fixed: Option<(f64, f64)> = (!normalize.is_empty()).then_some((0.0, 1.0));
        let chart: Chart = Chart { name: "ConfusionMatrixDisplay", texts, body: Body::Matrix { result, fixed } };
        Ok(Figure::new(chart, csv).values(true))
    }
}

/// The ROC curve: the true positive rate against the false positive rate, with the area under
/// it in the subtitle (scikit-learn's `RocCurveDisplay`).
pub struct RocCurveDisplay;

impl RocCurveDisplay {
    /// The ROC curve of `y_score`, the positive class's score, against `y_true`, from
    /// [`roc_curve`](crate::metrics::roc_curve); `pos_label` is that of `roc_curve`.
    ///
    /// ```
    /// use rusmlai::polars::prelude::*;
    /// use rusmlai::prelude::*;
    /// let y = Series::new("y".into(), [0, 0, 1, 1]);
    /// let score = Series::new("s".into(), [0.1, 0.4, 0.35, 0.8]);
    /// let svg = RocCurveDisplay::from_predictions(&y, &score, "").svg();
    /// assert!(svg.contains("AUC = 0.75"));
    /// ```
    ///
    /// Edge case: three classes have no ROC curve without a positive class:
    /// ```should_panic
    /// use rusmlai::polars::prelude::*;
    /// use rusmlai::prelude::*;
    /// let y = Series::new("y".into(), [0, 1, 2]);
    /// let score = Series::new("s".into(), [0.1, 0.5, 0.9]);
    /// RocCurveDisplay::from_predictions(&y, &score, ""); // panics: two classes
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import matplotlib.pyplot as plt
    /// from sklearn.metrics import RocCurveDisplay
    /// display = RocCurveDisplay.from_predictions([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    /// assert display.roc_auc == 0.75
    /// plt.show()
    /// ```
    #[track_caller]
    pub fn from_predictions(y_true: &Series, y_score: &Series, pos_label: &str) -> Figure {
        Self::try_from_predictions(y_true, y_score, pos_label).or_fatal()
    }

    /// [`RocCurveDisplay::from_predictions`] that returns an error instead of panicking.
    ///
    /// # Errors
    ///
    /// Those of [`try_roc_curve`](crate::metrics::try_roc_curve), as `PlotError::Model`.
    pub fn try_from_predictions(y_true: &Series, y_score: &Series, pos_label: &str) -> Result<Figure, PlotError> {
        Self::figure(y_true, y_score, pos_label, None)
    }

    /// [`RocCurveDisplay::from_predictions`] of `model`'s `predict_proba` on `dataset`: the
    /// column of `pos_label`, or of the greater class for `""` with two classes.
    ///
    /// ```
    /// use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv").drop(&["id", "name", "batch"]).dropna(&[]);
    /// let mut model = LogisticRegression::new();
    /// model.fit(&ds, "passed");
    /// let svg = RocCurveDisplay::from_estimator(&model, &ds, "passed", "").svg();
    /// assert!(svg.contains("AUC = "));
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import matplotlib.pyplot as plt
    /// import pandas as pd
    /// from sklearn.linear_model import LogisticRegression
    /// from sklearn.metrics import RocCurveDisplay
    /// ds = pd.read_csv("tests/fixtures/data/students.csv").drop(columns=["id", "name", "batch"]).dropna()
    /// x, y = ds.drop(columns=["passed"]), ds["passed"]
    /// RocCurveDisplay.from_estimator(LogisticRegression().fit(x, y), x, y)
    /// plt.show()
    /// ```
    #[track_caller]
    pub fn from_estimator(model: &impl Classifier, dataset: &Dataset, target: &str, pos_label: &str) -> Figure {
        Self::try_from_estimator(model, dataset, target, pos_label).or_fatal()
    }

    /// [`RocCurveDisplay::from_estimator`] that returns an error instead of panicking.
    ///
    /// # Errors
    ///
    /// Those of `model.try_predict_proba` and of [`RocCurveDisplay::try_from_predictions`];
    /// `PlotError::Dataset` when `target` or `pos_label` is not a column.
    pub fn try_from_estimator(model: &impl Classifier, dataset: &Dataset, target: &str, pos_label: &str) -> Result<Figure, PlotError> {
        let y_true: Series = self::target(dataset, target)?;
        Self::figure(&y_true, &positive_score(model, dataset, pos_label)?, pos_label, csv(dataset))
    }

    fn figure(y_true: &Series, y_score: &Series, pos_label: &str, csv: Option<&OsStr>) -> Result<Figure, PlotError> {
        let (fpr, tpr, _) = try_roc_curve(y_true, y_score, pos_label)?;
        let texts: Texts = Texts {
            title: "ROC Curve".to_string(),
            subtitle: format!("AUC = {}", crate::plot::svg::format_value(area(&fpr, &tpr))),
            x: "False Positive Rate".to_string(),
            y: "True Positive Rate".to_string(),
            value: "True Positive Rate".to_string(),
            ..Texts::default()
        };
        let points: Vec<(f64, f64)> = fpr.into_iter().zip(tpr).collect();
        Ok(line("RocCurveDisplay", texts, &points, None, csv))
    }
}

/// The precision against the recall at each threshold, with the average precision in the
/// subtitle (scikit-learn's `PrecisionRecallDisplay`). The points are joined by straight
/// lines; scikit-learn draws steps.
pub struct PrecisionRecallDisplay;

impl PrecisionRecallDisplay {
    /// The curve of [`precision_recall_curve`](crate::metrics::precision_recall_curve);
    /// `pos_label` is that of `roc_curve`.
    ///
    /// ```
    /// use rusmlai::polars::prelude::*;
    /// use rusmlai::prelude::*;
    /// let y = Series::new("y".into(), [0, 0, 1, 1]);
    /// let score = Series::new("s".into(), [0.1, 0.4, 0.35, 0.8]);
    /// let svg = PrecisionRecallDisplay::from_predictions(&y, &score, "").svg();
    /// assert!(svg.contains("AP = 0.83"));
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import matplotlib.pyplot as plt
    /// from sklearn.metrics import PrecisionRecallDisplay
    /// display = PrecisionRecallDisplay.from_predictions([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    /// assert abs(display.average_precision - 5 / 6) < 1e-15
    /// plt.show()
    /// ```
    #[track_caller]
    pub fn from_predictions(y_true: &Series, y_score: &Series, pos_label: &str) -> Figure {
        Self::try_from_predictions(y_true, y_score, pos_label).or_fatal()
    }

    /// [`PrecisionRecallDisplay::from_predictions`] that returns an error instead of
    /// panicking.
    ///
    /// # Errors
    ///
    /// Those of [`try_precision_recall_curve`](crate::metrics::try_precision_recall_curve),
    /// as `PlotError::Model`.
    pub fn try_from_predictions(y_true: &Series, y_score: &Series, pos_label: &str) -> Result<Figure, PlotError> {
        Self::figure(y_true, y_score, pos_label, None)
    }

    /// [`PrecisionRecallDisplay::from_predictions`] of `model`'s `predict_proba` on
    /// `dataset`, as [`RocCurveDisplay::from_estimator`] takes it.
    ///
    /// ```
    /// use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv").drop(&["id", "name", "batch"]).dropna(&[]);
    /// let mut model = LogisticRegression::new();
    /// model.fit(&ds, "passed");
    /// let svg = PrecisionRecallDisplay::from_estimator(&model, &ds, "passed", "").svg();
    /// assert!(svg.contains("AP = "));
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import matplotlib.pyplot as plt
    /// import pandas as pd
    /// from sklearn.linear_model import LogisticRegression
    /// from sklearn.metrics import PrecisionRecallDisplay
    /// ds = pd.read_csv("tests/fixtures/data/students.csv").drop(columns=["id", "name", "batch"]).dropna()
    /// x, y = ds.drop(columns=["passed"]), ds["passed"]
    /// PrecisionRecallDisplay.from_estimator(LogisticRegression().fit(x, y), x, y)
    /// plt.show()
    /// ```
    #[track_caller]
    pub fn from_estimator(model: &impl Classifier, dataset: &Dataset, target: &str, pos_label: &str) -> Figure {
        Self::try_from_estimator(model, dataset, target, pos_label).or_fatal()
    }

    /// [`PrecisionRecallDisplay::from_estimator`] that returns an error instead of panicking.
    ///
    /// # Errors
    ///
    /// Those of [`RocCurveDisplay::try_from_estimator`].
    pub fn try_from_estimator(model: &impl Classifier, dataset: &Dataset, target: &str, pos_label: &str) -> Result<Figure, PlotError> {
        let y_true: Series = self::target(dataset, target)?;
        Self::figure(&y_true, &positive_score(model, dataset, pos_label)?, pos_label, csv(dataset))
    }

    fn figure(y_true: &Series, y_score: &Series, pos_label: &str, csv: Option<&OsStr>) -> Result<Figure, PlotError> {
        let (precision, recall, _) = try_precision_recall_curve(y_true, y_score, pos_label)?;
        let ap: f64 = average_precision(&precision, &recall);
        let texts: Texts = Texts {
            title: "Precision-Recall Curve".to_string(),
            subtitle: format!("AP = {}", crate::plot::svg::format_value(ap)),
            x: "Recall".to_string(),
            y: "Precision".to_string(),
            value: "Precision".to_string(),
            ..Texts::default()
        };
        let points: Vec<(f64, f64)> = recall.into_iter().zip(precision).rev().collect();
        Ok(line("PrecisionRecallDisplay", texts, &points, None, csv))
    }
}

/// A regressor's errors: each row's residual (true − predicted) against its prediction, or
/// the true value against the prediction with the line y = x (scikit-learn's
/// `PredictionErrorDisplay`). It draws 2,500 sampled rows, like `scatter`; `.sample(n)`
/// changes it (scikit-learn: 1,000).
pub struct PredictionErrorDisplay;

impl PredictionErrorDisplay {
    /// The errors of `y_pred` against `y_true`. `kind` is `"residual_vs_predicted"` or
    /// `"actual_vs_predicted"`.
    ///
    /// ```
    /// use rusmlai::polars::prelude::*;
    /// use rusmlai::prelude::*;
    /// let y_true = Series::new("y".into(), [1.0, 2.0, 3.0, 4.0]);
    /// let y_pred = Series::new("p".into(), [1.5, 1.5, 3.5, 3.5]);
    /// let svg = PredictionErrorDisplay::from_predictions(&y_true, &y_pred, "residual_vs_predicted").svg();
    /// assert!(svg.contains("Residuals (actual - predicted) = -0.5"));
    /// ```
    ///
    /// The true values with the line of a perfect model:
    /// ```
    /// use rusmlai::polars::prelude::*;
    /// use rusmlai::prelude::*;
    /// let y_true = Series::new("y".into(), [1.0, 2.0, 3.0]);
    /// let y_pred = Series::new("p".into(), [1.0, 2.5, 2.5]);
    /// let svg = PredictionErrorDisplay::from_predictions(&y_true, &y_pred, "actual_vs_predicted").svg();
    /// assert!(svg.contains("Actual values"));
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import matplotlib.pyplot as plt
    /// import numpy as np
    /// from sklearn.metrics import PredictionErrorDisplay
    /// y_true, y_pred = np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.5, 1.5, 3.5, 3.5])
    /// display = PredictionErrorDisplay.from_predictions(y_true, y_pred)
    /// assert display.y_true.tolist() == [1.0, 2.0, 3.0, 4.0]
    /// plt.show()
    /// ```
    #[track_caller]
    pub fn from_predictions(y_true: &Series, y_pred: &Series, kind: &str) -> Figure {
        Self::try_from_predictions(y_true, y_pred, kind).or_fatal()
    }

    /// [`PredictionErrorDisplay::from_predictions`] that returns an error instead of
    /// panicking.
    ///
    /// # Errors
    ///
    /// `PlotError::Model` for inputs of different lengths, nulls, or another `kind`.
    pub fn try_from_predictions(y_true: &Series, y_pred: &Series, kind: &str) -> Result<Figure, PlotError> {
        Self::figure(y_true, y_pred, kind, None)
    }

    /// [`PredictionErrorDisplay::from_predictions`] of `model`'s predictions on `dataset`
    /// against its column `target`.
    ///
    /// ```
    /// use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv").select(&["age", "cgpa"]).dropna(&[]);
    /// let mut model = LinearRegression::new();
    /// model.fit(&ds, "cgpa");
    /// let svg = PredictionErrorDisplay::from_estimator(&model, &ds, "cgpa", "residual_vs_predicted").svg();
    /// assert!(svg.contains("Predicted values"));
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import matplotlib.pyplot as plt
    /// import pandas as pd
    /// from sklearn.linear_model import LinearRegression
    /// from sklearn.metrics import PredictionErrorDisplay
    /// ds = pd.read_csv("tests/fixtures/data/students.csv")[["age", "cgpa"]].dropna()
    /// x, y = ds[["age"]], ds["cgpa"]
    /// PredictionErrorDisplay.from_estimator(LinearRegression().fit(x, y), x, y)
    /// plt.show()
    /// ```
    #[track_caller]
    pub fn from_estimator(model: &impl Regressor, dataset: &Dataset, target: &str, kind: &str) -> Figure {
        Self::try_from_estimator(model, dataset, target, kind).or_fatal()
    }

    /// [`PredictionErrorDisplay::from_estimator`] that returns an error instead of panicking.
    ///
    /// # Errors
    ///
    /// Those of `model.try_predict` and of [`PredictionErrorDisplay::try_from_predictions`];
    /// `PlotError::Dataset` when `target` is not a column.
    pub fn try_from_estimator(model: &impl Regressor, dataset: &Dataset, target: &str, kind: &str) -> Result<Figure, PlotError> {
        let y_true: Series = self::target(dataset, target)?;
        Self::figure(&y_true, &model.try_predict(dataset)?, kind, csv(dataset))
    }

    fn figure(y_true: &Series, y_pred: &Series, kind: &str, csv: Option<&OsStr>) -> Result<Figure, PlotError> {
        let residuals: bool = match kind {
            "residual_vs_predicted" => true,
            "actual_vs_predicted" => false,
            _ => {
                return Err(PlotError::from(ModelError::InvalidSetting {
                    setting: "kind",
                    value: kind.into(),
                    expected: "\"residual_vs_predicted\" or \"actual_vs_predicted\"",
                }));
            }
        };
        if y_true.len() != y_pred.len() {
            return Err(PlotError::from(ModelError::LengthMismatch { y_true: y_true.len(), y_pred: y_pred.len() }));
        }
        let (t, p): (Series, Series) = (y_true.cast(&DataType::Float64)?, y_pred.cast(&DataType::Float64)?);
        let points: Vec<(f64, f64, usize)> = t
            .f64()?
            .iter()
            .zip(p.f64()?.iter())
            .filter_map(|(t, p)| Some((p?, if residuals { t? - p? } else { t? }, 0)))
            .filter(|(x, y, _)| x.is_finite() && y.is_finite())
            .collect();
        if points.is_empty() {
            return Err(PlotError::EmptyResult("y_true and y_pred".to_string()));
        }
        let y: &str = if residuals { "Residuals (actual - predicted)" } else { "Actual values" };
        let texts: Texts = Texts {
            title: "Prediction Errors".to_string(),
            x: "Predicted values".to_string(),
            y: y.to_string(),
            value: y.to_string(),
            ..Texts::default()
        };
        let rows: usize = points.len();
        // The line of a perfect model: y = 0 for residuals, y = x for actual values.
        let fit: (f64, f64) = if residuals { (0.0, 0.0) } else { (0.0, 1.0) };
        let body: Body = Body::Xy { hue_labels: Vec::new(), points, rows, fit: Some(fit) };
        Ok(Figure::new(Chart { name: "PredictionErrorDisplay", texts, body }, csv))
    }
}

/// The share of positive rows against the mean predicted probability, per bin of the
/// probability, with the line of a perfectly calibrated model (scikit-learn's
/// `CalibrationDisplay`).
pub struct CalibrationDisplay;

impl CalibrationDisplay {
    /// The curve of [`calibration_curve`](crate::metrics::calibration_curve) with its
    /// `n_bins` (scikit-learn: 5), `strategy` (`"uniform"` or `"quantile"`) and `pos_label`.
    ///
    /// ```
    /// use rusmlai::polars::prelude::*;
    /// use rusmlai::prelude::*;
    /// let y = Series::new("y".into(), [0, 0, 1, 1, 1]);
    /// let p = Series::new("p".into(), [0.1, 0.3, 0.6, 0.7, 0.9]);
    /// let svg = CalibrationDisplay::from_predictions(&y, &p, 3, "uniform", "").svg();
    /// assert!(svg.contains("Mean predicted probability = 0.6"));
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import matplotlib.pyplot as plt
    /// from sklearn.calibration import CalibrationDisplay
    /// display = CalibrationDisplay.from_predictions([0, 0, 1, 1, 1], [0.1, 0.3, 0.6, 0.7, 0.9], n_bins=3)
    /// assert display.prob_true.tolist() == [0.0, 1.0, 1.0]
    /// plt.show()
    /// ```
    #[track_caller]
    pub fn from_predictions(y_true: &Series, y_prob: &Series, n_bins: usize, strategy: &str, pos_label: &str) -> Figure {
        Self::try_from_predictions(y_true, y_prob, n_bins, strategy, pos_label).or_fatal()
    }

    /// [`CalibrationDisplay::from_predictions`] that returns an error instead of panicking.
    ///
    /// # Errors
    ///
    /// Those of [`try_calibration_curve`](crate::metrics::try_calibration_curve), as
    /// `PlotError::Model`.
    pub fn try_from_predictions(y_true: &Series, y_prob: &Series, n_bins: usize, strategy: &str, pos_label: &str) -> Result<Figure, PlotError> {
        Self::figure(y_true, y_prob, n_bins, strategy, pos_label, None)
    }

    /// [`CalibrationDisplay::from_predictions`] of `model`'s `predict_proba` on `dataset`, as
    /// [`RocCurveDisplay::from_estimator`] takes it.
    ///
    /// ```
    /// use rusmlai::prelude::*;
    /// let ds = read_csv("tests/fixtures/data/students.csv").drop(&["id", "name", "batch"]).dropna(&[]);
    /// let mut model = LogisticRegression::new();
    /// model.fit(&ds, "passed");
    /// let svg = CalibrationDisplay::from_estimator(&model, &ds, "passed", 5, "uniform", "").svg();
    /// assert!(svg.contains("Fraction of positives"));
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import matplotlib.pyplot as plt
    /// import pandas as pd
    /// from sklearn.calibration import CalibrationDisplay
    /// from sklearn.linear_model import LogisticRegression
    /// ds = pd.read_csv("tests/fixtures/data/students.csv").drop(columns=["id", "name", "batch"]).dropna()
    /// x, y = ds.drop(columns=["passed"]), ds["passed"]
    /// CalibrationDisplay.from_estimator(LogisticRegression().fit(x, y), x, y, n_bins=5)
    /// plt.show()
    /// ```
    #[track_caller]
    pub fn from_estimator(
        model: &impl Classifier,
        dataset: &Dataset,
        target: &str,
        n_bins: usize,
        strategy: &str,
        pos_label: &str,
    ) -> Figure {
        Self::try_from_estimator(model, dataset, target, n_bins, strategy, pos_label).or_fatal()
    }

    /// [`CalibrationDisplay::from_estimator`] that returns an error instead of panicking.
    ///
    /// # Errors
    ///
    /// Those of [`RocCurveDisplay::try_from_estimator`] and of
    /// [`CalibrationDisplay::try_from_predictions`].
    pub fn try_from_estimator(
        model: &impl Classifier,
        dataset: &Dataset,
        target: &str,
        n_bins: usize,
        strategy: &str,
        pos_label: &str,
    ) -> Result<Figure, PlotError> {
        let y_true: Series = self::target(dataset, target)?;
        let score: Series = positive_score(model, dataset, pos_label)?;
        Self::figure(&y_true, &score, n_bins, strategy, pos_label, csv(dataset))
    }

    fn figure(
        y_true: &Series,
        y_prob: &Series,
        n_bins: usize,
        strategy: &str,
        pos_label: &str,
        csv: Option<&OsStr>,
    ) -> Result<Figure, PlotError> {
        let (prob_true, prob_pred) = try_calibration_curve(y_true, y_prob, n_bins, strategy, pos_label)?;
        let texts: Texts = Texts {
            title: "Calibration Curve".to_string(),
            subtitle: "the line is a perfectly calibrated model".to_string(),
            x: "Mean predicted probability".to_string(),
            y: "Fraction of positives".to_string(),
            value: "Fraction of positives".to_string(),
            ..Texts::default()
        };
        let points: Vec<(f64, f64)> = prob_pred.into_iter().zip(prob_true).collect();
        Ok(line("CalibrationDisplay", texts, &points, Some((0.0, 1.0)), csv))
    }
}
