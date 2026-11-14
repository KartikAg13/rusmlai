//! Transformers that clean and encode a [`Dataset`], and the [`ColumnTransformer`] that combines
//! them. A transformer learns in `fit` by reading the cached stats, and gives Polars expressions,
//! so a whole [`ColumnTransformer`] runs as one lazy `select`.

use std::{fmt::Debug, sync::Arc};

use indexmap::IndexMap;
use polars::prelude::{DataFrame, Expr, IntoLazy};

use crate::utils::col;
use crate::{
    dataset::Dataset,
    error::OrFatal,
    metadata::{
        Metadata,
        dataset::{DatasetCache, info::ColumnInfo},
        plot::PlotCache,
    },
};

pub mod column_transformer;
pub mod encoder;
pub mod error;
pub mod imputer;
pub mod scaler;

pub use column_transformer::ColumnTransformer;
pub use encoder::{LabelEncoder, OneHotEncoder, OrdinalEncoder};
pub use error::PreprocessError;
pub use imputer::SimpleImputer;
pub use scaler::{MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler};

/// Output column name to the expression that computes it.
pub type Exprs = IndexMap<String, Expr>;

/// A step that learns from a dataset and then rewrites it, like a scikit-learn transformer.
pub trait Transform: Debug {
    /// Learns from every column of `dataset`, normally by reading its cached stats.
    fn try_fit(&mut self, dataset: &Dataset) -> Result<(), PreprocessError>;

    /// The output columns, as expressions of the `inputs` expressions. No data is touched, so
    /// steps compose: the output of one step is the input of the next.
    fn exprs(&self, inputs: &Exprs) -> Result<Exprs, PreprocessError>;

    /// Applies the fitted step to `dataset` in one lazy `select`.
    fn try_transform(&self, dataset: &Dataset) -> Result<Dataset, PreprocessError> {
        apply(dataset, &self.exprs(&identity(dataset))?)
    }

    /// [`Transform::try_fit`], then [`Transform::try_transform`] on the same data.
    fn try_fit_transform(&mut self, dataset: &Dataset) -> Result<Dataset, PreprocessError> {
        self.try_fit(dataset)?;
        self.try_transform(dataset)
    }

    /// Learns from `dataset`. Panics with the error's message at your line.
    #[track_caller]
    fn fit(&mut self, dataset: &Dataset) {
        self.try_fit(dataset).or_fatal()
    }

    /// Applies what [`Transform::fit`] learned. Panics with the error's message at your line.
    #[track_caller]
    fn transform(&self, dataset: &Dataset) -> Dataset {
        self.try_transform(dataset).or_fatal()
    }

    /// `fit`, then `transform`. Panics with the error's message at your line.
    #[track_caller]
    fn fit_transform(&mut self, dataset: &Dataset) -> Dataset {
        self.try_fit_transform(dataset).or_fatal()
    }
}

/// A list of transformers for one [`ColumnTransformer::add`]; they run in order.
///
/// ```
/// use rusmlai::{preprocess::{SimpleImputer, StandardScaler}, steps};
/// let steps = steps![SimpleImputer::new(), StandardScaler::new()];
/// assert_eq!(steps.len(), 2);
/// ```
///
/// Python:
///
/// ```python
/// from sklearn.impute import SimpleImputer
/// from sklearn.pipeline import make_pipeline
/// from sklearn.preprocessing import StandardScaler
/// steps = make_pipeline(SimpleImputer(), StandardScaler())
/// assert len(steps.steps) == 2
/// ```
#[macro_export]
macro_rules! steps {
    ($($step:expr),* $(,)?) => {
        vec![$(Box::new($step) as Box<dyn $crate::preprocess::Transform>),*]
    };
}

/// `value` if it is one of `allowed`, else an error that lists them.
pub(crate) fn one_of(
    option: &'static str,
    value: &str,
    allowed: &[&str],
) -> Result<String, PreprocessError> {
    if allowed.contains(&value) {
        return Ok(value.into());
    }
    Err(PreprocessError::InvalidOption {
        option,
        value: value.into(),
        expected: allowed
            .iter()
            .map(|a| format!("\"{a}\""))
            .collect::<Vec<String>>()
            .join(", "),
    })
}

/// Every column of `dataset` as itself.
pub(crate) fn identity(dataset: &Dataset) -> Exprs {
    dataset
        .columns()
        .into_iter()
        .map(|name| {
            let expr: Expr = col(name.as_str());
            (name, expr)
        })
        .collect()
}

/// Runs `outputs` on `parent` and builds the new dataset's metadata without computing anything:
/// a column that passes through unchanged shares its parent's stats, and every other column
/// gets empty stats that are computed on first use.
pub(crate) fn apply(parent: &Dataset, outputs: &Exprs) -> Result<Dataset, PreprocessError> {
    let selected: Vec<Expr> = outputs
        .iter()
        .map(|(name, expr)| expr.clone().alias(name.as_str()))
        .collect();
    let dataframe: DataFrame = match parent.dataframe.clone().lazy().select(selected).collect() {
        Ok(dataframe) => dataframe,
        // `replace_strict` (the encoders) names neither the column nor the option, so find the
        // output that fails. This runs only on the error path.
        Err(error) if error.to_string().contains("replace_strict") => {
            let column: String = outputs
                .iter()
                .find(|(name, expr)| {
                    let one: Expr = (*expr).clone().alias(name.as_str());
                    parent
                        .dataframe
                        .clone()
                        .lazy()
                        .select([one])
                        .collect()
                        .is_err()
                })
                .map(|(name, _)| name.clone())
                .unwrap_or_default();
            return Err(PreprocessError::UnknownAtTransform { column });
        }
        Err(error) => return Err(error.into()),
    };
    let columns: IndexMap<String, Arc<ColumnInfo>> = dataframe
        .columns()
        .iter()
        .map(|column| {
            let name: String = column.name().to_string();
            let unchanged: bool = outputs[&name] == col(name.as_str());
            let info: Arc<ColumnInfo> = match parent.metadata.dataset_cache.columns.get(&name) {
                Some(info) if unchanged => Arc::clone(info),
                _ => Arc::new(ColumnInfo::new(column)),
            };
            (name, info)
        })
        .collect();
    // The CSV's path stays, so charts auto-save next to it and models find its model cache.
    // The stats and chart caches have no file here, so nothing writes into them.
    let metadata: Metadata = Metadata {
        filepath: parent.metadata.filepath.clone(),
        height: dataframe.height(),
        width: dataframe.width(),
        dataset_cache: DatasetCache::new(columns),
        plot_cache: PlotCache::default(),
    };
    Ok(Dataset {
        dataframe,
        metadata,
    })
}
