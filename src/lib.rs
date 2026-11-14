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
//! features = ds.drop(columns=["id", "name", "passed"])
//! gains = {c: mutual_info_score(features[c].map(str), ds["passed"]) for c in features}
//! print("most informative:", max(gains, key=gains.get))
//! ax = sns.countplot(data=ds, x="batch", hue="passed")
//! ax.set_title("Batches")
//! plt.show()
//! ```
//!
//! Each public item's docs show its Python equivalent the same way, under "Python:".
//!
//! Public methods panic with a clear message at your line, like Python raising. Each one has a
//! `try_*` twin that returns a `Result` instead, for `?` in your own functions.

pub mod dataset;
pub mod display;
pub mod error;
pub mod metadata;
pub mod metrics;
pub mod models;
pub mod pipeline;
pub mod plot;
pub mod preprocess;
pub mod utils;

pub mod prelude {
    pub use crate::{
        dataset::{
            Dataset,
            error::DatasetError,
            read_csv,
            split::{train_test_split, try_train_test_split},
            try_read_csv,
        },
        display::{Figure, TableView},
        metadata::models::ModelCache,
        metrics::{
            accuracy_score, log_loss, mean_absolute_error, mean_squared_error, r2_score,
            root_mean_squared_error,
        },
        models::{
            Classifier, DummyRegressor, LinearRegression, LogisticRegression, Model, ModelError,
            Regressor,
        },
        pipeline::{ModelStacker, Pipeline},
        plot::style::{set_palette, set_theme},
        preprocess::{
            ColumnTransformer, LabelEncoder, MaxAbsScaler, MinMaxScaler, OneHotEncoder,
            OrdinalEncoder, PreprocessError, RobustScaler, SimpleImputer, StandardScaler,
            Transform,
        },
        steps,
    };
}
