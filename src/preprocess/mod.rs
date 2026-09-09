use crate::dataset::Dataset;

pub trait Transform {
    fn fit(&self, dataset: &Dataset);

    fn transform(&mut self, dataset: &Dataset) -> Dataset;

    fn fit_transform(&mut self, dataset: &Dataset) -> Dataset;

    fn affected_columns(&self) -> Vec<String>;

    fn changes_row_count(&self) -> bool {
        false
    }
}
