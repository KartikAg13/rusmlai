use std::{
    fs::{self, Metadata},
    io,
    path::Path,
    sync::OnceLock,
    time::UNIX_EPOCH,
};

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

pub fn parse_dtype(value: &str) -> Result<DataType, MetadataError> {
    match value {
        "i64" => Ok(DataType::Int64),
        "f64" => Ok(DataType::Float64),
        "str" => Ok(DataType::String),
        _ => Err(MetadataError::UnsupportedDatatype(value.to_string())),
    }
}

pub fn option_to_oncelock<T>(value: Option<T>) -> OnceLock<T> {
    let lock: OnceLock<T> = OnceLock::new();
    if let Some(v) = value {
        let _ = lock.set(v);
    }
    lock
}

pub fn file_fingerprint(path: impl AsRef<Path>) -> io::Result<(u64, u64)> {
    let meta: Metadata = fs::metadata(path)?;
    let mtime: u64 = meta
        .modified()?
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    Ok((meta.len(), mtime))
}
