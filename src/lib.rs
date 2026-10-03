pub mod dataset;
pub mod display;
pub mod error;
pub mod metadata;
pub mod pipeline;
pub mod plot;
pub mod preprocess;
pub mod utils;

pub mod prelude {
    pub use crate::{
        dataset::{Dataset, read_csv},
        display::{Figure, TableView},
    };
}
