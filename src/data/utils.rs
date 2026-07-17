use std::{
    fs::{self, Metadata},
    io,
    path::Path,
    sync::OnceLock,
    time::UNIX_EPOCH,
};

use polars::datatypes::DataType;

use crate::data::error::DataError;

pub fn parse_dtype(s: &str) -> Result<DataType, DataError> {
    match s {
        "i64" => Ok(DataType::Int64),
        "f64" => Ok(DataType::Float64),
        "str" => Ok(DataType::String),
        other => Err(DataError::UnsupportedDatatype(other.to_string())),
    }
}

pub fn option_to_oncelock<T>(opt: Option<T>) -> OnceLock<T> {
    let lock: OnceLock<T> = OnceLock::new();
    if let Some(value) = opt {
        let _ = lock.set(value);
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
