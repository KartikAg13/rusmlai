//! Transformers that clean and encode a [`Dataset`] (scalers, encoders, imputers).
//! A placeholder: the design is in the data-processing layer of the backlog (#36).

use crate::dataset::Dataset;

/// A step that learns from a dataset and then rewrites it, like a scikit-learn transformer.
/// A placeholder: the signatures will change when the first transformer is built.
pub trait Transform {
    fn fit(&self, dataset: &Dataset);

    fn transform(&mut self, dataset: &Dataset) -> Dataset;

    fn fit_transform(&mut self, dataset: &Dataset) -> Dataset;

    fn affected_columns(&self) -> Vec<String>;

    fn changes_row_count(&self) -> bool {
        false
    }
}
