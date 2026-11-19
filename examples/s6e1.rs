use rusmlai::prelude::*;

fn main() {
    let train = read_csv("tests/fixtures/s6e1/train.csv");
    let test = read_csv("tests/fixtures/s6e1/train.csv");

    train.head(5);
}
