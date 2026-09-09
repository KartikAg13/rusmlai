use std::{
    fs::{self, Metadata},
    io,
    path::Path,
    sync::OnceLock,
    time::UNIX_EPOCH,
};

use polars::datatypes::DataType;

use crate::dataset::error::DatasetError;

pub fn parse_dtype(value: &str) -> Result<DataType, DatasetError> {
    match value {
        "i64" => Ok(DataType::Int64),
        "f64" => Ok(DataType::Float64),
        "str" => Ok(DataType::String),
        other => Err(DatasetError::UnsupportedDatatype(other.to_string())),
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
