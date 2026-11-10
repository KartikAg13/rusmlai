//! The in-memory chart cache.

use std::{
    path::PathBuf,
    sync::{Arc, RwLock},
};

use indexmap::IndexMap;

use crate::plot::{result::PlotResult, spec::PlotSpec};

/// The numbers behind each chart drawn so far. Clones share one map, so `select` sees the
/// charts its parent drew.
#[derive(Default, Debug, Clone)]
pub struct PlotCache {
    /// One result per chart spec, in the order they were drawn.
    pub entries: Arc<RwLock<IndexMap<PlotSpec, PlotResult>>>,
    /// The `.plots.jsonl` file a miss is appended to. Only `read_csv` sets it: a dataset made
    /// in memory must never write its charts into a CSV's cache.
    pub file: Option<PathBuf>,
}

impl PlotCache {
    /// Wraps an existing map, with no file.
    pub fn new(entries: Arc<RwLock<IndexMap<PlotSpec, PlotResult>>>) -> Self {
        Self {
            entries,
            file: None,
        }
    }
}
