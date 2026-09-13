use std::cell::RefCell;

use indexmap::IndexMap;

use crate::plot::{result::PlotResult, spec::PlotSpec};

pub struct PlotCache {
    pub entries: RefCell<IndexMap<PlotSpec, PlotResult>>,
}

impl PlotCache {
    pub fn new() -> Self {
        Self {
            entries: RefCell::new(IndexMap::new()),
        }
    }

    pub fn get_or_insert(&self, spec: PlotSpec) -> PlotResult {
        todo!()
    }
}
