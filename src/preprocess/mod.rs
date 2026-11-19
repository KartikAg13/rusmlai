//! Transformers that clean and encode a [`Dataset`], and the [`ColumnTransformer`] that combines
//! them. A transformer learns in `fit` by reading the cached stats, and gives Polars expressions,
//! so a whole [`ColumnTransformer`] runs as one lazy `select`.

use std::{fmt::Debug, sync::Arc};

use indexmap::IndexMap;
use polars::prelude::{DataFrame, DataType, Expr, IntoLazy};

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

pub mod binning;
pub mod column_transformer;
pub mod encoder;
pub mod error;
pub mod imputer;
pub mod scaler;
pub mod target_encoder;

pub use binning::{KBinsDiscretizer, QuantileTransformer};
pub use column_transformer::ColumnTransformer;
pub use encoder::{CountEncoder, LabelEncoder, OneHotEncoder, OrdinalEncoder};
pub use error::PreprocessError;
pub use imputer::SimpleImputer;
pub use scaler::{MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler};
pub use target_encoder::TargetEncoder;

/// Output column name to the expression that computes it.
pub type Exprs = IndexMap<String, Expr>;

/// A step that learns from a dataset and then rewrites it, like a scikit-learn transformer.
pub trait Transform: Debug + CloneTransform + Send + Sync {
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

/// Copies a boxed step, so that a `ColumnTransformer` and a `Pipeline` can be copied: a search
/// fits a copy for each candidate, like scikit-learn's `clone`. Every `Transform + Clone` has
/// it; nothing else needs to.
#[doc(hidden)]
pub trait CloneTransform {
    /// A copy of the step, boxed.
    fn clone_box(&self) -> Box<dyn Transform>;
}

impl<T: Transform + Clone + 'static> CloneTransform for T {
    fn clone_box(&self) -> Box<dyn Transform> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Transform> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
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
        // `replace_strict` (the encoders) names neither the column, the value nor the option,
        // so find the output that fails, its input column, and the first value that fails it.
        // This runs only on the error path.
        Err(error) if error.to_string().contains("replace_strict") => {
            let fails = |frame: &DataFrame, expr: &Expr| {
                frame
                    .clone()
                    .lazy()
                    .select([expr.clone()])
                    .collect()
                    .is_err()
            };
            let Some(expr) = outputs.values().find(|expr| fails(&parent.dataframe, expr)) else {
                return Err(error.into());
            };
            let column: String = expr
                .into_iter()
                .find_map(|e| match e {
                    Expr::Column(name) => Some(name.to_string()),
                    _ => None,
                })
                .unwrap_or_default();
            let distinct: DataFrame = parent
                .dataframe
                .clone()
                .lazy()
                .select([col(column.as_str()).unique_stable()])
                .collect()?;
            let category: Option<String> = (0..distinct.height().min(10_000)).find_map(|i| {
                let one: DataFrame = distinct.slice(i as i64, 1);
                let text = one.columns()[0].cast(&DataType::String).ok()?;
                fails(&one, expr).then(|| text.str().ok()?.get(0).map(str::to_string))?
            });
            return Err(PreprocessError::UnknownAtTransform { column, category });
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
