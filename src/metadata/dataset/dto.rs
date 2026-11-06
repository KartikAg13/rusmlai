//! The serde shape of the stats cache file. NaN stats are written as `null`.

use std::sync::OnceLock;

use indexmap::IndexMap;
use polars::datatypes::DataType;
use serde::{Deserialize, Serialize};

use crate::metadata::dataset::{info::ColumnInfo, stats::ColumnStats};

/// The `{stem}.json` file.
#[derive(Serialize, Deserialize)]
pub struct DatasetCacheDTO {
    /// The CSV's `(size, mtime)` when the file was written; a mismatch makes the file stale.
    pub fingerprint: (u64, u64),
    /// One entry per column, in column order.
    pub columns: IndexMap<String, ColumnInfoDTO>,
}

/// [`ColumnInfo`] as stored on disk.
#[derive(Serialize, Deserialize)]
pub struct ColumnInfoDTO {
    /// The Polars dtype, stored through serde so every dtype round-trips.
    pub dtype: DataType,
    /// Null values in the column.
    pub null_count: usize,
    /// `None` when the stats were not computed.
    pub stats: Option<ColumnStats>,
}

impl From<ColumnInfoDTO> for ColumnInfo {
    fn from(dto: ColumnInfoDTO) -> Self {
        Self {
            dtype: dto.dtype,
            null_count: dto.null_count,
            stats: dto.stats.map_or_else(OnceLock::new, OnceLock::from),
        }
    }
}

#[cfg(test)]
mod tests {
    use polars::prelude::*;

    use super::*;

    #[test]
    fn column_info_round_trips_through_json() {
        let frame: DataFrame = df!(
            "i" => [1i64, 2],
            "f" => [1.5, f64::NAN],
            "s" => ["a", "b"],
            "b" => [true, false]
        )
        .unwrap();
        let mut columns: Vec<Column> = frame.columns().to_vec();
        columns.push(
            Column::new("d".into(), [0i32, 1])
                .cast(&DataType::Date)
                .unwrap(),
        );
        let frame: DataFrame = DataFrame::new(2, columns).unwrap();
        let stats: Vec<ColumnStats> = ColumnStats::compute_stats(&frame).unwrap();
        for (column, stats) in frame.columns().iter().zip(stats) {
            let info: ColumnInfo = ColumnInfo::with_stats(column, stats);
            let json: String = serde_json::to_string(&ColumnInfoDTO::from(&info)).unwrap();
            let back: ColumnInfo = serde_json::from_str::<ColumnInfoDTO>(&json).unwrap().into();
            assert_eq!(back.dtype, info.dtype, "{json}");
            assert_eq!(back.null_count, info.null_count);
            assert_eq!(
                format!("{:?}", back.stats.get()),
                format!("{:?}", info.stats.get())
            );
        }
    }

    #[test]
    fn nan_stats_are_written_as_null_and_read_back_as_nan() {
        let column: Column = Column::new("x".into(), [None::<f64>]);
        let stats: ColumnStats =
            ColumnStats::compute_stats(&DataFrame::new(1, vec![column.clone()]).unwrap())
                .unwrap()
                .remove(0);
        let json: String = serde_json::to_string(&ColumnInfoDTO::from(&ColumnInfo::with_stats(
            &column, stats,
        )))
        .unwrap();
        assert!(json.contains("\"mean\":null"), "{json}");
        let back: ColumnInfo = serde_json::from_str::<ColumnInfoDTO>(&json).unwrap().into();
        assert!(matches!(back.stats.get(), Some(ColumnStats::Numeric(s)) if s.mean.is_nan()));
    }
}
