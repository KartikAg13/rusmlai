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
        metrics::{mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error},
        models::{DummyRegressor, LinearRegression, ModelError, Regressor, Ridge},
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
