//! Quick tour on the tracked fixture: `cargo run --example students [path/to/file.csv]`.
//! The first run computes the column stats and writes them next to the CSV, in `metadata/`.
//! The second run loads them from there.

use rusmlai::prelude::*;

fn main() {
    let path: String = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "tests/fixtures/data/students.csv".to_string());
    let ds: Dataset = read_csv(&path);

    println!("shape: {:?}", ds.shape());
    ds.head(5);
    ds.info();
    ds.describe();

    let columns: Vec<String> = ds.columns();
    let rest: Dataset = ds.drop(&[columns[0].as_str()]); // for example an id column
    println!("without `{}`: {:?}", columns[0], rest.shape());

    match ds.try_numeric_stats(&columns[1]) {
        Ok(s) => println!(
            "{}: median {:.2}, q25 {:.2}, q75 {:.2}",
            columns[1], s.median, s.quantile_25, s.quantile_75
        ),
        Err(e) => println!("{e}"), // for example a text column
    }

    let target: &str = columns.last().map_or("", String::as_str);
    for (column, gain) in rest.information_gain(target).iter().take(5) {
        println!("information gain with `{target}`: {column} {gain:.3}");
    }
}
