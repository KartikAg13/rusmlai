//! The in-memory chart cache.

use std::sync::{Arc, RwLock};

use indexmap::IndexMap;

use crate::plot::{result::PlotResult, spec::PlotSpec};

/// The numbers behind each chart drawn so far. Clones share one map, so `select` sees the
/// charts its parent drew.
#[derive(Default, Debug, Clone)]
pub struct PlotCache {
    /// One result per chart spec, in the order they were drawn.
    pub entries: Arc<RwLock<IndexMap<PlotSpec, PlotResult>>>,
}

impl PlotCache {
    /// Wraps an existing map.
    pub fn new(entries: Arc<RwLock<IndexMap<PlotSpec, PlotResult>>>) -> Self {
        Self { entries }
    }
}
