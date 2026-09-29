use std::sync::{Arc, RwLock};

use indexmap::IndexMap;

use crate::plot::{result::PlotResult, spec::PlotSpec};

#[derive(Default, Debug, Clone)]
pub struct PlotCache {
    pub entries: Arc<RwLock<IndexMap<PlotSpec, PlotResult>>>,
}

impl PlotCache {
    pub fn new(entries: Arc<RwLock<IndexMap<PlotSpec, PlotResult>>>) -> Self {
        Self { entries }
    }
}
