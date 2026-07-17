use rusmlai::data::dataset::{Dataset, read_csv};

fn main() {
    let dataset: Dataset = read_csv(
        "/home/kartikag234/Documents/rusmlai/tests/fixtures/kaggle_playground-series-s6e1/train.csv",
    );
    dataset.metadata(&[]);
    dataset.head(5);
    dataset.describe();
    dataset.info();
}
