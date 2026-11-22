//! Tabular machine learning with the names you know from pandas, scikit-learn and seaborn,
//! on top of [Polars](https://docs.pola.rs/). The same code runs in a `.rs` program and in a
//! Jupyter notebook through the evcxr kernel.
//!
//! ```
//! use rusmlai::prelude::*;
//!
//! let ds = read_csv("tests/fixtures/data/students.csv"); // stats cached in tests/fixtures/data/metadata/
//! ds.info();
//! ds.describe();
//! let gains = ds.drop(&["id", "name"]).information_gain("passed");
//! println!("most informative: {}", gains[0].0);
//! let svg: String = ds.countplot("batch", "passed").title("Batches").svg();
//! assert!(svg.starts_with("<svg"));
//! ```
//!
//! Each public item's docs end the same way, with its Python equivalent under "Python:".
//!
//! Public methods panic with a clear message at your line, like Python raising. Each one has a
//! `try_*` twin that returns a `Result` instead, for `?` in your own functions.
//!
//! Some examples use `?` on a `try_*` method. To copy one into your `main`, write
//! `fn main() -> Result<(), Box<dyn std::error::Error>>` and end it with `Ok(())`, or call the
//! method without `try_`.
//!
//! The cache files in `metadata/` next to a CSV are trusted like code: they set what scalers,
//! imputers and models learn. For data that came from someone else with its `metadata/` folder,
//! turn the cache off with [`set_cache(false)`](metadata::set_cache).
//!
//! Python:
//!
//! ```python
//! import matplotlib.pyplot as plt
//! import pandas as pd
//! import seaborn as sns
//! from sklearn.metrics import mutual_info_score
//! ds = pd.read_csv("tests/fixtures/data/students.csv")  # no cache: every stat is computed again
//! ds.info()
//! print(ds.describe())
//! gains = {}
//! for column in ["age", "cgpa", "batch"]:
//!     rows = ds[[column, "passed"]].dropna()  # rusmlai leaves out missing values too
//!     x = rows[column]
//!     gains[column] = mutual_info_score(pd.cut(x, 256) if x.dtype.kind in "if" else x, rows["passed"])
//! print("most informative:", max(gains, key=gains.get))
//! ax = sns.countplot(data=ds, x="batch", hue="passed")
//! ax.set_title("Batches")
//! plt.show()
//! ```

pub mod dataset;
pub mod display;
pub mod error;
pub mod metadata;
pub mod metrics;
pub mod models;
pub mod pipeline;
pub mod plot;
pub mod preprocess;
mod utils;

/// The Polars version rusmlai is built on, so your code can use the same one without a
/// dependency of its own: `use rusmlai::polars::prelude::*;`.
pub use polars;

pub mod prelude {
    pub use polars::df;

    pub use crate::{
        dataset::{
            Dataset,
            clean::Scalar,
            error::DatasetError,
            read::{
                read_excel, read_parquet, read_sql, try_read_excel, try_read_parquet, try_read_sql,
            },
            read_csv,
            split::{train_test_split, try_train_test_split},
            try_read_csv,
        },
        display::{Figure, TableView},
        error::{set_warnings, take_warnings},
        metadata::{dataset::role::ColumnRole, models::ModelCache, set_cache},
        metrics::{
            accuracy_score, average_precision_score, balanced_accuracy_score, calibration_curve,
            confusion_matrix, log_loss, mean_absolute_error, mean_squared_error,
            precision_recall_curve, r2_score, roc_auc_score, roc_curve, root_mean_squared_error,
            try_average_precision_score, try_calibration_curve, try_confusion_matrix,
            try_precision_recall_curve,
        },
        models::{
            AdaBoostClassifier, BaggingClassifier, BaggingRegressor, Classifier, DecisionTreeClassifier,
            DecisionTreeRegressor, DummyRegressor, ExtraTreesClassifier, ExtraTreesRegressor,
            GradientBoostingClassifier,
            GradientBoostingRegressor,
            ElasticNet, CatBoostClassifier, CatBoostRegressor, GridSearchCV, HillClimbing, OlsSummary, LGBMClassifier, LGBMRegressor, Lasso, LinearRegression,
            LogisticRegression, Model, ModelError, Ridge, RidgeCV,
            RandomForestClassifier, RandomForestRegressor, Regressor, StackingClassifier,
            StackingRegressor, XGBClassifier, XGBRegressor,
            Cv, KFold, StratifiedKFold, compute_sample_weight, cross_val_predict, cross_val_score, try_cross_val_predict,
            try_cross_val_score,
        },
        pipeline::Pipeline,
        plot::{
            model::{
                CalibrationDisplay, ConfusionMatrixDisplay, PrecisionRecallDisplay,
                PredictionErrorDisplay, RocCurveDisplay,
            },
            style::{set_palette, set_theme},
        },
        preprocess::{
            ColumnTransformer, CountEncoder, KBinsDiscretizer, LabelEncoder, MaxAbsScaler, MinMaxScaler, OneHotEncoder,
            OrdinalEncoder, PreprocessError, QuantileTransformer, RobustScaler, SimpleImputer, StandardScaler, TargetEncoder,
            Transform,
        },
        steps,
    };
}
