use std::sync::Arc;

use indexmap::IndexMap;

use crate::metadata::dataset::info::ColumnInfo;

pub mod info;
pub mod stats;

#[derive(Debug, Clone)]
pub struct DatasetCache {
    pub columns: IndexMap<String, Arc<ColumnInfo>>,
}

impl DatasetCache {
    pub fn new(columns: IndexMap<String, Arc<ColumnInfo>>) -> Self {
        Self { columns: columns }
    }
}
