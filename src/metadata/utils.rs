use polars::{
    datatypes::DataType,
    frame::{DataFrame, column::Column},
    lazy::frame::{IntoLazy, LazyFrame},
};

use crate::metadata::error::MetadataError;

pub fn column_to_lazy(column: &Column) -> Result<(String, LazyFrame), MetadataError> {
    let name: String = column.name().clone().into_string();
    let frame: DataFrame = DataFrame::new(column.len(), vec![column.clone()])?;
    Ok((name, frame.lazy()))
}

pub fn unwrap_f64(frame: &DataFrame, name: &str) -> Result<f64, MetadataError> {
    Ok(frame
        .column(name)?
        .cast(&DataType::Float64)?
        .f64()?
        .get(0)
        .unwrap_or(f64::NAN))
}
