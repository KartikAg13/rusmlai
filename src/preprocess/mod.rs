//! Transformers that clean and encode a [`Dataset`], and the [`ColumnTransformer`] that combines
//! them. A transformer learns in `fit` by reading the cached stats, and gives Polars expressions,
//! so a whole [`ColumnTransformer`] runs as one lazy `select`.

use std::{fmt::Debug, sync::Arc};

use indexmap::IndexMap;
use polars::prelude::{DataFrame, DataType, Expr, IntoLazy, Series};

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
pub trait Transform: Debug + CloneTransform + SaveTransform + Send + Sync {
    /// Learns from every column of `dataset`, normally by reading its cached stats.
    fn try_fit(&mut self, dataset: &Dataset) -> Result<(), PreprocessError>;

    /// The output columns, as expressions of the `inputs` expressions. No data is touched, so
    /// steps compose: the output of one step is the input of the next.
    fn exprs(&self, inputs: &Exprs) -> Result<Exprs, PreprocessError>;

    /// Learns from `dataset` and the target `y`. A step that needs the target
    /// ([`TargetEncoder`]) overrides it, and gives the training rows' output when that is not
    /// `transform` of them (cross fitting); `None` means it is. The default ignores `y`, so
    /// every other step fits as without it. [`ColumnTransformer`] passes the target to each of
    /// its steps, and [`Pipeline`](crate::pipeline::Pipeline) to its `ColumnTransformer`, as
    /// scikit-learn's do.
    fn try_fit_with_target(&mut self, dataset: &Dataset, y: &Series) -> Result<Option<Dataset>, PreprocessError> {
        let _ = y;
        self.try_fit(dataset)?;
        Ok(None)
    }

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

/// What a saved `Pipeline` writes of a boxed step: its type's name and its settings and
/// learned values. Every `Transform + Serialize` has it.
#[doc(hidden)]
pub trait SaveTransform {
    /// The step's type name, which [`restore_transform`] reads back.
    fn kind(&self) -> &'static str;
    /// The step's settings and learned values.
    fn state(&self) -> Result<serde_json::Value, serde_json::Error>;
}

impl<T: Transform + serde::Serialize + 'static> SaveTransform for T {
    fn kind(&self) -> &'static str {
        let full: &'static str = std::any::type_name::<T>();
        full.rsplit("::").next().unwrap_or(full)
    }

    fn state(&self) -> Result<serde_json::Value, serde_json::Error> {
        serde_json::to_value(self)
    }
}

/// The step of type `kind` from its saved `state`.
fn restore_transform(kind: &str, state: serde_json::Value) -> Result<Box<dyn Transform>, serde_json::Error> {
    use serde::de::Error;
    fn boxed<T: Transform + serde::de::DeserializeOwned + 'static>(state: serde_json::Value) -> Result<Box<dyn Transform>, serde_json::Error> {
        Ok(Box::new(serde_json::from_value::<T>(state)?))
    }
    match kind {
        "StandardScaler" => boxed::<StandardScaler>(state),
        "MinMaxScaler" => boxed::<MinMaxScaler>(state),
        "MaxAbsScaler" => boxed::<MaxAbsScaler>(state),
        "RobustScaler" => boxed::<RobustScaler>(state),
        "SimpleImputer" => boxed::<SimpleImputer>(state),
        "OneHotEncoder" => boxed::<OneHotEncoder>(state),
        "OrdinalEncoder" => boxed::<OrdinalEncoder>(state),
        "LabelEncoder" => boxed::<LabelEncoder>(state),
        "CountEncoder" => boxed::<CountEncoder>(state),
        "TargetEncoder" => boxed::<TargetEncoder>(state),
        "KBinsDiscretizer" => boxed::<KBinsDiscretizer>(state),
        "QuantileTransformer" => boxed::<QuantileTransformer>(state),
        "ColumnTransformer" => boxed::<ColumnTransformer>(state),
        other => Err(serde_json::Error::custom(format!("'{other}' is not a step rusmlai can load"))),
    }
}

impl serde::Serialize for Box<dyn Transform> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::{Error, SerializeStruct};
        let mut out = serializer.serialize_struct("Step", 2)?;
        out.serialize_field("kind", self.kind())?;
        out.serialize_field("state", &self.state().map_err(S::Error::custom)?)?;
        out.end()
    }
}

impl<'de> serde::Deserialize<'de> for Box<dyn Transform> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        #[derive(serde::Deserialize)]
        struct Step {
            kind: String,
            state: serde_json::Value,
        }
        let step: Step = Step::deserialize(deserializer)?;
        restore_transform(&step.kind, step.state).map_err(D::Error::custom)
    }
}

/// serde for the literal `Expr`s a `SimpleImputer` keeps (a fill value, a statistic): a
/// number, a whole number, text, true / false or null.
pub(crate) mod literal {
    use indexmap::IndexMap;
    use polars::prelude::{AnyValue, Expr, NULL, lit};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    enum Literal {
        Float(f64),
        Int(i64),
        Text(String),
        Bool(bool),
        Null,
    }

    fn to(expr: &Expr) -> Literal {
        let Expr::Literal(v) = expr else { return Literal::Null };
        match v.to_any_value() {
            Some(AnyValue::Float64(x)) => Literal::Float(x),
            Some(AnyValue::Float32(x)) => Literal::Float(f64::from(x)),
            Some(AnyValue::Int64(x)) => Literal::Int(x),
            Some(AnyValue::Int32(x)) => Literal::Int(i64::from(x)),
            Some(AnyValue::Boolean(x)) => Literal::Bool(x),
            Some(AnyValue::String(x)) => Literal::Text(x.into()),
            Some(AnyValue::StringOwned(x)) => Literal::Text(x.to_string()),
            Some(other) => other.extract::<f64>().map_or(Literal::Null, Literal::Float),
            None => Literal::Null,
        }
    }

    fn from(l: Literal) -> Expr {
        match l {
            Literal::Float(x) => lit(x),
            Literal::Int(x) => lit(x),
            Literal::Text(x) => lit(x),
            Literal::Bool(x) => lit(x),
            Literal::Null => lit(NULL),
        }
    }

    pub(crate) mod option {
        use super::*;
        pub(crate) fn serialize<S: Serializer>(v: &Option<Expr>, s: S) -> Result<S::Ok, S::Error> {
            v.as_ref().map(to).serialize(s)
        }
        pub(crate) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Expr>, D::Error> {
            Ok(Option::<Literal>::deserialize(d)?.map(from))
        }
    }

    pub(crate) mod map {
        use super::*;
        pub(crate) fn serialize<S: Serializer>(v: &IndexMap<String, Expr>, s: S) -> Result<S::Ok, S::Error> {
            v.iter().map(|(k, e)| (k.clone(), to(e))).collect::<IndexMap<String, Literal>>().serialize(s)
        }
        pub(crate) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<IndexMap<String, Expr>, D::Error> {
            Ok(IndexMap::<String, Literal>::deserialize(d)?.into_iter().map(|(k, l)| (k, from(l))).collect())
        }
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
