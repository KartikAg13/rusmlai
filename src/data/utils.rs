use std::sync::OnceLock;

use polars::datatypes::DataType;

use crate::data::error::DataError;

pub fn parse_dtype(s: &str) -> Result<DataType, DataError> {
    match s {
        "Int64" => Ok(DataType::Int64),
        "Float64" => Ok(DataType::Float64),
        "String" => Ok(DataType::String),
        other => Err(DataError::UnsupportedDatatype(other.to_string())),
    }
}

pub fn option_to_oncelock<T>(opt: Option<T>) -> OnceLock<T> {
    let lock = OnceLock::new();
    if let Some(value) = opt {
        let _ = lock.set(value);
    }
    lock
}
