//! `select`, `drop` and `after_transform` share column stats instead of copying them.

mod common;

use std::sync::Arc;

use polars::prelude::*;
use rusmlai::{metadata::Metadata, prelude::*};

fn same_stats(a: &Dataset, b: &Dataset, column: &str) -> bool {
    Arc::ptr_eq(
        &a.metadata.dataset_cache.columns[column],
        &b.metadata.dataset_cache.columns[column],
    )
}

#[test]
fn select_and_drop_keep_the_order_and_share_the_stats() {
    let ds: Dataset = read_csv(common::scratch_csv("ops_select"));
    let picked: Dataset = ds.select(&["passed", "age"]);
    assert_eq!(picked.columns(), ["passed", "age"]);
    assert!(same_stats(&ds, &picked, "age"));

    let dropped: Dataset = ds.drop(&["name", "batch"]);
    assert_eq!(dropped.columns(), ["id", "age", "cgpa", "passed"]);
    assert!(same_stats(&ds, &dropped, "cgpa"));
    assert_eq!(dropped.shape(), (12, 4));
}

#[test]
fn after_transform_computes_only_the_changed_columns() {
    let ds: Dataset = read_csv(common::scratch_csv("ops_transform"));
    let doubled: DataFrame = ds
        .dataframe
        .clone()
        .lazy()
        .with_column(col("cgpa") * lit(2.0))
        .collect()
        .unwrap();
    let new: Dataset = Dataset {
        metadata: ds.metadata.dataset_cache.after_transform(
            &Dataset {
                dataframe: doubled.clone(),
                metadata: ds.metadata.clone(),
            },
            vec!["cgpa".into()],
            false,
        ),
        dataframe: doubled,
    };
    assert!(same_stats(&ds, &new, "age"));
    assert!(!same_stats(&ds, &new, "cgpa"));
    let (old, now): (f64, f64) = (
        ds.try_numeric_stats("cgpa").unwrap().median,
        new.try_numeric_stats("cgpa").unwrap().median,
    );
    assert_eq!(now, 2.0 * old);
}

#[test]
fn after_transform_computes_every_column_when_rows_change() {
    let ds: Dataset = read_csv(common::scratch_csv("ops_rows"));
    let head: DataFrame = ds.dataframe.head(Some(4));
    let probe: Dataset = Dataset {
        dataframe: head.clone(),
        metadata: ds.metadata.clone(),
    };
    let m: Metadata = ds
        .metadata
        .dataset_cache
        .after_transform(&probe, vec![], true);
    let new: Dataset = Dataset {
        dataframe: head,
        metadata: m,
    };
    assert_eq!(new.shape(), (4, 6));
    for column in ds.columns() {
        assert!(
            !same_stats(&ds, &new, &column),
            "{column} was not computed again"
        );
    }
    assert_eq!(new.try_numeric_stats("id").unwrap().max, 3.0);
}
