//! Splitting a [`Dataset`] into train and test rows (scikit-learn keeps these in
//! `sklearn.model_selection`).

use indexmap::IndexMap;
use polars::{
    error::PolarsError,
    prelude::{DataFrame, DataType, IdxCa, IdxSize, PlSmallStr},
};
use thiserror::Error;

use crate::{
    dataset::{Dataset, error::DatasetError},
    error::OrFatal,
    metadata::{
        Metadata,
        dataset::{DatasetCache, info::ColumnInfo},
        plot::PlotCache,
    },
};

/// What can fail while splitting a dataset.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SplitError {
    /// A column lookup failed, for example an unknown `stratify` column.
    #[error(transparent)]
    Dataset(#[from] DatasetError),

    /// A Polars query failed.
    #[error(transparent)]
    Polars(#[from] PolarsError),

    /// `test_size` is not a fraction between 0 and 1.
    #[error(
        "test_size = {test_size} is not valid; use a fraction between 0 and 1, for example 0.2"
    )]
    TestSize {
        /// The value given.
        test_size: f64,
    },

    /// One side of the split would have no rows.
    #[error(
        "test_size gives {train} train rows and {test} test rows from {rows}; each side needs at least one row"
    )]
    EmptySplit {
        /// Rows in the dataset.
        rows: usize,
        /// Train rows.
        train: usize,
        /// Test rows.
        test: usize,
    },

    /// A class of the `stratify` column has a single row, so it can't be on both sides.
    #[error(
        "class '{class}' of column '{column}' has only 1 row; stratify needs at least 2 rows in each class"
    )]
    ClassTooSmall {
        /// The `stratify` column.
        column: String,
        /// The class.
        class: String,
    },

    /// One side of the split has fewer rows than there are classes.
    #[error(
        "column '{column}' has {classes} classes, but the split has {train} train rows and {test} test rows; each side needs at least one row per class"
    )]
    TooManyClasses {
        /// The `stratify` column.
        column: String,
        /// Distinct classes.
        classes: usize,
        /// Train rows.
        train: usize,
        /// Test rows.
        test: usize,
    },
}

/// Splits `dataset` into `(train, test)` rows, like scikit-learn's `train_test_split`.
///
/// - `test_size` is the share of rows in `test`, between 0 and 1. `test` gets
///   `ceil(test_size × rows)` rows, as in scikit-learn.
/// - `random_state` seeds the shuffle: the same seed gives the same split on every run.
/// - `stratify` names a column whose classes keep their share in both sides, or is `""` for
///   none. A null is a class of its own.
///
/// Each side gets its own metadata, so a `fit` on `train` reads stats of the train rows only.
/// The stats are computed the first time something reads them.
///
/// ```
/// use rusmlai::prelude::*;
/// let ds = read_csv("tests/fixtures/data/students.csv");
/// let (train, test) = train_test_split(&ds, 0.25, 42, "");
/// assert_eq!(train.height() + test.height(), ds.height());
/// assert_eq!(test.height(), (ds.height() as f64 * 0.25).ceil() as usize);
/// ```
#[track_caller]
pub fn train_test_split(
    dataset: &Dataset,
    test_size: f64,
    random_state: u64,
    stratify: &str,
) -> (Dataset, Dataset) {
    try_train_test_split(dataset, test_size, random_state, stratify).or_fatal()
}

/// [`train_test_split`] that returns an error instead of panicking.
///
/// # Errors
///
/// - [`SplitError::TestSize`] when `test_size` is not between 0 and 1.
/// - [`SplitError::EmptySplit`] when one side would have no rows.
/// - [`SplitError::Dataset`] when the `stratify` column doesn't exist.
/// - [`SplitError::ClassTooSmall`] and [`SplitError::TooManyClasses`] when
///   the classes of `stratify` can't be on both sides.
///
/// ```
/// use rusmlai::{dataset::split::{SplitError, try_train_test_split}, prelude::*};
/// let ds = read_csv("tests/fixtures/data/students.csv");
/// let result = try_train_test_split(&ds, 1.5, 42, "");
/// assert!(matches!(result, Err(SplitError::TestSize { .. })));
/// ```
pub fn try_train_test_split(
    dataset: &Dataset,
    test_size: f64,
    random_state: u64,
    stratify: &str,
) -> Result<(Dataset, Dataset), SplitError> {
    if !(test_size > 0.0 && test_size < 1.0) {
        return Err(SplitError::TestSize { test_size });
    }
    let rows: usize = dataset.height();
    let test: usize = (test_size * rows as f64).ceil() as usize;
    let train: usize = rows.saturating_sub(test);
    if train == 0 || test == 0 {
        return Err(SplitError::EmptySplit { rows, train, test });
    }
    let mut rng: Rng = Rng::new(random_state);
    let (train_rows, test_rows): (Vec<IdxSize>, Vec<IdxSize>) = if stratify.is_empty() {
        let mut order: Vec<IdxSize> = (0..rows as IdxSize).collect();
        rng.shuffle(&mut order);
        let test_rows: Vec<IdxSize> = order.split_off(train);
        (order, test_rows)
    } else {
        stratified(dataset, stratify, test, &mut rng)?
    };
    Ok((take(dataset, train_rows)?, take(dataset, test_rows)?))
}

/// Train and test rows with each class of `column` split in the same proportion. The test
/// side gets `test` rows: each class its exact share rounded down, and the rows left over go to
/// the classes with the biggest remainders (scikit-learn's `_approximate_mode`, without its
/// random tie-break).
fn stratified(
    dataset: &Dataset,
    column: &str,
    test: usize,
    rng: &mut Rng,
) -> Result<(Vec<IdxSize>, Vec<IdxSize>), SplitError> {
    let rows: usize = dataset.height();
    let keys = dataset.column(column)?.cast(&DataType::String)?;
    let mut classes: IndexMap<Option<&str>, Vec<IdxSize>> = IndexMap::new();
    for (row, key) in keys.str()?.iter().enumerate() {
        classes.entry(key).or_default().push(row as IdxSize);
    }
    if let Some((class, _)) = classes.iter().find(|(_, members)| members.len() < 2) {
        return Err(SplitError::ClassTooSmall {
            column: column.into(),
            class: class.unwrap_or("null").into(),
        });
    }
    let train: usize = rows - test;
    if test < classes.len() || train < classes.len() {
        return Err(SplitError::TooManyClasses {
            column: column.into(),
            classes: classes.len(),
            train,
            test,
        });
    }
    let exact: Vec<f64> = classes
        .values()
        .map(|members| members.len() as f64 * test as f64 / rows as f64)
        .collect();
    let mut counts: Vec<usize> = exact.iter().map(|e| e.floor() as usize).collect();
    let mut by_remainder: Vec<usize> = (0..exact.len()).collect();
    by_remainder
        .sort_by(|&a, &b| (exact[b] - exact[b].floor()).total_cmp(&(exact[a] - exact[a].floor())));
    let missing: usize = test - counts.iter().sum::<usize>();
    for &class in by_remainder.iter().take(missing) {
        counts[class] += 1;
    }
    let (mut train_rows, mut test_rows): (Vec<IdxSize>, Vec<IdxSize>) = (Vec::new(), Vec::new());
    for (members, count) in classes.into_values().zip(counts) {
        let mut members: Vec<IdxSize> = members;
        rng.shuffle(&mut members);
        test_rows.extend_from_slice(&members[..count]);
        train_rows.extend_from_slice(&members[count..]);
    }
    rng.shuffle(&mut train_rows);
    rng.shuffle(&mut test_rows);
    Ok((train_rows, test_rows))
}

/// The rows of `dataset` at `rows`, in that order, with metadata of their own. The stats start
/// empty; the plot cache is new and writes no file.
fn take(dataset: &Dataset, rows: Vec<IdxSize>) -> Result<Dataset, SplitError> {
    let dataframe: DataFrame = dataset
        .dataframe
        .take(&IdxCa::from_vec(PlSmallStr::EMPTY, rows))?;
    let columns = dataframe
        .columns()
        .iter()
        .map(|column| {
            (
                column.name().to_string(),
                std::sync::Arc::new(ColumnInfo::new(column)),
            )
        })
        .collect();
    let metadata: Metadata = Metadata {
        filepath: dataset.metadata.filepath.clone(),
        height: dataframe.height(),
        width: dataframe.width(),
        dataset_cache: DatasetCache::new(columns),
        plot_cache: PlotCache::default(),
    };
    Ok(Dataset {
        dataframe,
        metadata,
    })
}

/// SplitMix64, a small seeded generator: a `random_state` gives the same rows on every run and
/// machine. `rand` is in the build only through Polars' cloud features, which rusmlai doesn't
/// use, so it is not a dependency here.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z: u64 = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Fisher-Yates. The modulo bias is below 2⁻³² for any slice that fits in memory.
    pub(crate) fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j: usize = (self.next_u64() % (i as u64 + 1)) as usize;
            items.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use polars::df;

    use super::*;

    fn ids(dataset: &Dataset) -> Vec<i32> {
        let mut ids: Vec<i32> = dataset
            .dataframe
            .column("id")
            .unwrap()
            .i32()
            .unwrap()
            .into_no_null_iter()
            .collect();
        ids.sort_unstable();
        ids
    }

    fn twelve() -> Dataset {
        let classes: Vec<&str> = [vec!["a"; 8], vec!["b"; 4]].concat();
        Dataset::new(
            df!("id" => (0..12).collect::<Vec<i32>>(), "y" => classes).unwrap(),
            "",
        )
    }

    #[test]
    fn the_sides_are_disjoint_complete_and_the_same_for_a_seed() {
        let ds: Dataset = twelve();
        let (train, test) = try_train_test_split(&ds, 0.25, 7, "").unwrap();
        assert_eq!((train.height(), test.height()), (9, 3));
        let mut all: Vec<i32> = [ids(&train), ids(&test)].concat();
        all.sort_unstable();
        assert_eq!(all, (0..12).collect::<Vec<i32>>());

        let (_, again) = try_train_test_split(&ds, 0.25, 7, "").unwrap();
        assert_eq!(ids(&again), ids(&test));
        let (_, other) = try_train_test_split(&ds, 0.25, 8, "").unwrap();
        assert_ne!(ids(&other), ids(&test));
    }

    #[test]
    fn stratify_keeps_each_class_share() {
        let ds: Dataset = twelve();
        let (train, test) = try_train_test_split(&ds, 0.25, 7, "y").unwrap();
        let count = |d: &Dataset, class: &str| {
            d.dataframe
                .column("y")
                .unwrap()
                .str()
                .unwrap()
                .iter()
                .filter(|v| *v == Some(class))
                .count()
        };
        assert_eq!((count(&test, "a"), count(&test, "b")), (2, 1));
        assert_eq!((count(&train, "a"), count(&train, "b")), (6, 3));
    }

    #[test]
    fn each_side_has_stats_of_its_own_rows_computed_on_first_use() {
        let ds: Dataset = twelve();
        let (train, _) = try_train_test_split(&ds, 0.25, 7, "").unwrap();
        assert!(
            train.metadata.dataset_cache.columns["id"]
                .stats
                .get()
                .is_none()
        );
        let ids: Vec<i32> = ids(&train);
        let mean: f64 = ids.iter().map(|&v| v as f64).sum::<f64>() / ids.len() as f64;
        assert_eq!(train.try_numeric_stats("id").unwrap().mean, mean);
    }

    #[test]
    fn errors_name_the_problem() {
        let ds: Dataset = twelve();
        for bad in [0.0, 1.0, -0.1, f64::NAN] {
            assert!(matches!(
                try_train_test_split(&ds, bad, 0, ""),
                Err(SplitError::TestSize { .. })
            ));
        }
        let one: Dataset = Dataset::new(df!("id" => [1]).unwrap(), "");
        assert!(matches!(
            try_train_test_split(&one, 0.5, 0, ""),
            Err(SplitError::EmptySplit { .. })
        ));
        let lonely: Dataset =
            Dataset::new(df!("id" => [1, 2, 3], "y" => ["a", "a", "b"]).unwrap(), "");
        let err: String = try_train_test_split(&lonely, 0.5, 0, "y")
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(
            err.starts_with("class 'b' of column 'y' has only 1 row"),
            "{err}"
        );
        assert!(matches!(
            try_train_test_split(&ds, 0.05, 0, "y"),
            Err(SplitError::TooManyClasses { .. })
        ));
        assert!(matches!(
            try_train_test_split(&ds, 0.25, 0, "nope"),
            Err(SplitError::Dataset(DatasetError::ColumnNotFound { .. }))
        ));
    }
}
