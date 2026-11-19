//! [`ColumnInfo`]: dtype, null count and the lazily computed stats of one column.

use polars::{
    datatypes::DataType,
    frame::{DataFrame, column::Column},
};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use crate::{
    metadata::{
        dataset::{
            dto::ColumnInfoDTO,
            role::{ColumnRole, detect},
            stats::ColumnStats,
        },
        error::MetadataError,
    },
    models::hist::Binned,
};

/// A column in buckets, as a tree model last made it (`models::hist::Bins`), with its
/// `max_bins`. The next tree model on the column takes it from here: a bag's copies, the
/// candidates of a search and a refit all share one. In memory only.
#[derive(Default)]
pub(crate) struct BinCache(Mutex<Option<(usize, Arc<Binned>)>>);

impl BinCache {
    /// The buckets, when they were made with `max_bins` for a column of `rows` rows.
    pub(crate) fn get(&self, max_bins: usize, rows: usize) -> Option<Arc<Binned>> {
        let kept = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        kept.as_ref()
            .filter(|(bins, binned)| *bins == max_bins && binned.codes.len() == rows)
            .map(|(_, binned)| Arc::clone(binned))
    }

    /// Keeps `binned`, in place of what was there, and gives it back shared.
    pub(crate) fn set(&self, max_bins: usize, binned: Binned) -> Arc<Binned> {
        let binned: Arc<Binned> = Arc::new(binned);
        *self.0.lock().unwrap_or_else(PoisonError::into_inner) = Some((max_bins, Arc::clone(&binned)));
        binned
    }
}

impl Clone for BinCache {
    fn clone(&self) -> Self {
        Self(Mutex::new(self.0.lock().unwrap_or_else(PoisonError::into_inner).clone()))
    }
}

impl std::fmt::Debug for BinCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("BinCache")
    }
}

/// What is known about one column: dtype, null count and stats.
///
/// ```
/// use rusmlai::prelude::*;
/// use rusmlai::polars::prelude::DataType;
/// let ds = read_csv("tests/fixtures/data/students.csv");
/// let info = ds.try_column_info("cgpa").unwrap();
/// assert_eq!(info.dtype, DataType::Float64);
/// assert_eq!(info.null_count, 1);
/// assert!(info.stats.get().is_some()); // read_csv computes every column
/// ```
///
/// [`ColumnInfo::new`] leaves the stats empty, and [`ColumnInfo::ensure_stats`] fills them once:
///
/// ```
/// use rusmlai::polars::prelude::Column;
/// use rusmlai::metadata::dataset::{info::ColumnInfo, stats::ColumnStats};
/// let column = Column::new("x".into(), [1.0, 2.0, 3.0]);
/// let info = ColumnInfo::new(&column);
/// assert!(info.stats.get().is_none());
/// let Ok(ColumnStats::Numeric(stats)) = info.ensure_stats(&column) else { panic!() };
/// assert_eq!(stats.median, 2.0);
/// ```
#[derive(Debug, Clone)]
pub struct ColumnInfo {
    /// The column's Polars dtype.
    pub dtype: DataType,
    /// Null values in the column.
    pub null_count: usize,
    /// The stats, empty until computed. Set at most once.
    pub stats: OnceLock<ColumnStats>,
    /// What the column is for, set with the stats, carried over to the two sides of a split,
    /// or chosen by [`Dataset::set_role`](crate::dataset::Dataset::set_role). Empty: a feature.
    pub(crate) role: OnceLock<ColumnRole>,
    /// The column in buckets, once a tree model made them.
    pub(crate) bins: BinCache,
}

impl ColumnInfo {
    /// Dtype and null count of `column`, with no stats yet.
    pub fn new(column: &Column) -> Self {
        ColumnInfo {
            dtype: column.dtype().clone(),
            null_count: column.null_count(),
            stats: OnceLock::new(),
            role: OnceLock::new(),
            bins: BinCache::default(),
        }
    }

    /// Dtype and null count of `column`, with `stats` already computed, and its role.
    pub fn with_stats(column: &Column, stats: ColumnStats) -> Self {
        let info: ColumnInfo = ColumnInfo::new(column);
        let _ = info.role.set(detect(column.name(), column.dtype(), info.null_count, column.len(), &stats));
        ColumnInfo {
            stats: OnceLock::from(stats),
            ..info
        }
    }

    /// Sets the stats computed for `column`, and its role unless it has one.
    pub(crate) fn fill(&self, column: &Column, stats: ColumnStats) {
        let _ = self.role.set(detect(column.name(), column.dtype(), self.null_count, column.len(), &stats));
        let _ = self.stats.set(stats); // already set: another thread was faster
    }

    /// What the column is for; a column whose stats were never computed is a feature.
    pub fn role(&self) -> ColumnRole {
        self.role.get().copied().unwrap_or(ColumnRole::Feature)
    }

    /// This info with `role`, whatever it had.
    pub(crate) fn with_role(&self, role: Option<ColumnRole>) -> ColumnInfo {
        ColumnInfo {
            role: role.map_or_else(OnceLock::new, OnceLock::from),
            ..self.clone()
        }
    }

    /// The column's stats, computed now if they aren't there yet.
    pub fn ensure_stats(&self, column: &Column) -> Result<&ColumnStats, MetadataError> {
        if let Some(stats) = self.stats.get() {
            return Ok(stats);
        }
        let dataframe: DataFrame = DataFrame::new(column.len(), vec![column.clone()])?;
        let stats: ColumnStats = ColumnStats::compute_stats(&dataframe)?.remove(0);
        // A role set by hand or carried from the whole table stays.
        self.fill(column, stats);
        Ok(self.stats.get().expect("just set"))
    }
}

impl From<&ColumnInfo> for ColumnInfoDTO {
    fn from(value: &ColumnInfo) -> Self {
        Self {
            dtype: value.dtype.clone(),
            null_count: value.null_count,
            stats: value.stats.get().cloned(),
            role: value.role.get().copied(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_stats_computes_missing_stats() {
        let column: Column = Column::new("x".into(), [1.0, 3.0, 2.0]);
        let info: ColumnInfo = ColumnInfo::new(&column);
        let stats: &ColumnStats = info.ensure_stats(&column).unwrap();
        assert!(matches!(stats, ColumnStats::Numeric(s) if s.median == 2.0));
    }
}
