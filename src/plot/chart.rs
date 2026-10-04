use crate::plot::{result::PlotResult, spec::BarSpec};

pub enum Chart {
    Bar {
        spec: BarSpec,
        result: PlotResult,
        subtitle: String,
    },
}
