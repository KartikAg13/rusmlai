//! Charts on a [`Dataset`], like seaborn. Each chart is a method that returns a [`Figure`], and
//! options chain on it: `ds.count("batch", "").title("Batches").palette("colorblind")`. In a
//! notebook the figure shows itself. In a program it saves a PNG to `metadata/plots/` next to
//! the CSV, unless you call `.save` or `.svg`. A column argument of `""` means "not used"
//! (`hue = ""`). Charts leave out rows with a null, NaN or infinite value in the columns they
//! use, like pandas. Every chart has a `try_` twin that returns `Result<Figure, PlotError>`;
//! `count` uses `try_bar(x, "", hue, "")`.
//!
//! Each chart file holds its `Dataset` methods (`foo` and `try_foo`), its
//! query and its renderer. The query helpers they share are here; the drawing helpers are in
//! `svg.rs`.

pub(crate) mod bar;
pub mod chart;
pub(crate) mod density;
pub(crate) mod dist;
pub mod error;
pub(crate) mod heat;
pub(crate) mod pair;
pub mod result;
pub mod spec;
pub mod style;
pub(crate) mod svg;
pub(crate) mod xy;

use std::{
    collections::{HashMap, hash_map::DefaultHasher},
    hash::{BuildHasher, BuildHasherDefault},
};

use polars::prelude::{
    DataFrame, DataType, Expr, IdxCa, IdxSize, IntoLazy, LazyFrame, PlSmallStr, SortOptions, col,
    lit,
};

use crate::{
    dataset::Dataset,
    display::Figure,
    plot::{
        chart::{Body, Chart, Texts},
        error::PlotError,
        style::Style,
    },
};

// ponytail: categories past 1,000 are cut before caching, even ones .order names (the error
// says so). `.order` comes after the query, so the fix is filtering the rows first (#36).
pub(crate) const MAX_GROUPS: usize = 1000; // categories kept in the cache, so a huge column can't fill the file
pub(crate) const SAMPLE: usize = 2500; // points that strip, scatter and regplot draw by default
pub(crate) const MAX_SAMPLE: usize = 50_000; // points kept from the data, the most .sample(n) draws
/// The largest magnitude a chart draws. Past it, `hi - lo` and the padding overflow to inf, so
/// such values are left out like infinities.
pub(crate) const BOUND: f64 = 1e300;
/// The aggregated column of a group-by.
pub(crate) const VALUE: &str = "__rusmlai_value";

impl Dataset {
    /// The chart as a `Figure` that auto-saves next to this dataset's CSV.
    pub(crate) fn figure(&self, name: &'static str, texts: Texts, body: Body) -> Figure {
        Figure::new(
            Chart { name, texts, body },
            self.metadata.filepath.as_deref(),
        )
    }
}

/// Category order: the metadata order (most frequent first, ties alphabetical) for text
/// columns, ascending for numbers. At most `MAX_GROUPS`.
pub(crate) fn labels(
    dataset: &Dataset,
    grouped: &DataFrame,
    column: &str,
) -> Result<Vec<String>, PlotError> {
    let mut labels: Vec<String> = match dataset.try_categorical_stats(column) {
        Ok(stats) => stats
            .categories
            .iter()
            .take(MAX_GROUPS)
            .map(|(name, _)| name.clone())
            .collect(),
        Err(_) => {
            let sorted: DataFrame = grouped
                .clone()
                .lazy()
                .select([col(column).unique().sort(SortOptions::default())])
                .collect()?;
            strings(&sorted, column)?
        }
    };
    labels.truncate(MAX_GROUPS);
    Ok(labels)
}

/// A column as text, the same way the metadata names categories.
pub(crate) fn strings(frame: &DataFrame, column: &str) -> Result<Vec<String>, PlotError> {
    Ok(frame
        .column(column)?
        .cast(&DataType::String)?
        .str()?
        .iter()
        .map(|v| v.unwrap_or_default().to_string())
        .collect())
}

/// For each row of `frame`, the place of its `column` value in `labels`: None for a value that
/// isn't drawn (past `MAX_GROUPS`). Without a column every row goes to place 0.
pub(crate) fn slots(
    frame: &DataFrame,
    column: Option<&str>,
    labels: &[String],
) -> Result<Vec<Option<usize>>, PlotError> {
    let Some(column) = column else {
        return Ok(vec![Some(0); frame.height()]);
    };
    let place: HashMap<&str, usize> = labels
        .iter()
        .enumerate()
        .map(|(i, label)| (label.as_str(), i))
        .collect();
    Ok(strings(frame, column)?
        .iter()
        .map(|value| place.get(value.as_str()).copied())
        .collect())
}

pub(crate) fn f64s(frame: &DataFrame, column: &str) -> Result<Vec<f64>, PlotError> {
    Ok(frame
        .column(column)?
        .cast(&DataType::Float64)?
        .f64()?
        .iter()
        .map(|v| v.unwrap_or(f64::NAN))
        .collect())
}

/// `""` (not used) as None.
pub(crate) fn optional(column: &str) -> Option<String> {
    (!column.is_empty()).then(|| column.to_string())
}

pub(crate) fn check_columns(
    dataset: &Dataset,
    x: Option<&str>,
    hue: Option<&str>,
) -> Result<(), PlotError> {
    if hue.is_some() && x.is_none() {
        return Err(PlotError::HueNeedsX);
    }
    for column in x.into_iter().chain(hue) {
        dataset.try_column_info(column)?;
    }
    Ok(())
}

/// The rows where each of `columns` holds a value: nulls dropped, and NaN, ±inf or a value past
/// `BOUND` in a float column too (pandas skips NaN; an infinite value has no place on an axis).
/// A column named twice (`hue` = `x`) is selected once.
pub(crate) fn rows(dataset: &Dataset, columns: &[&str]) -> LazyFrame {
    let names: Vec<&str> = unique(columns.iter().copied());
    let finite: Option<Expr> = names
        .iter()
        .filter(|c| {
            dataset
                .try_column_info(c)
                .is_ok_and(|info| info.dtype.is_float())
        })
        .map(|c| col(*c).gt_eq(lit(-BOUND)).and(col(*c).lt_eq(lit(BOUND)))) // NaN sorts above inf
        .reduce(|a, b| a.and(b));
    let rows: LazyFrame = dataset
        .dataframe
        .clone()
        .lazy()
        .select(names.iter().map(|c| col(*c)).collect::<Vec<Expr>>())
        .drop_nulls(None);
    match finite {
        Some(keep) => rows.filter(keep),
        None => rows,
    }
}

/// `names` without repeats, in order (`hue` may name the `x` column).
pub(crate) fn unique<'a>(names: impl IntoIterator<Item = &'a str>) -> Vec<&'a str> {
    let mut out: Vec<&str> = Vec::new();
    for name in names {
        if !out.contains(&name) {
            out.push(name);
        }
    }
    out
}

/// Up to `MAX_SAMPLE` rows with values in all `columns`: one random row from each run of k rows.
/// The seed is fixed, so the same data draws the same picture. Returns them and how many rows
/// had values.
pub(crate) fn sample(dataset: &Dataset, columns: &[&str]) -> Result<(DataFrame, usize), PlotError> {
    let rows: DataFrame = rows(dataset, columns).collect()?;
    let step: usize = rows.height().div_ceil(MAX_SAMPLE).max(1);
    if step == 1 {
        let n: usize = rows.height();
        return Ok((rows, n));
    }
    let seed: BuildHasherDefault<DefaultHasher> = BuildHasherDefault::default();
    let picks: Vec<IdxSize> = (0..rows.height() / step)
        .map(|run| (run * step + (seed.hash_one(run) % step as u64) as usize) as IdxSize)
        .collect();
    Ok((
        rows.take(&IdxCa::from_vec(PlSmallStr::EMPTY, picks))?,
        rows.height(),
    ))
}

/// `n` evenly spaced items, or all of them when there are fewer (`.sample(n)` on the stored
/// sample points).
pub(crate) fn thin<T: Copy>(items: &[T], n: usize) -> Vec<T> {
    if items.len() <= n {
        return items.to_vec();
    }
    (0..n).map(|i| items[i * items.len() / n]).collect()
}

/// "2500 of 630000 rows" when a sample is drawn, unless a subtitle is set.
pub(crate) fn sample_subtitle(style: &Style, subtitle: &str, shown: usize, rows: usize) -> String {
    match &style.subtitle {
        Some(text) => text.clone(),
        None if subtitle.is_empty() && shown < rows => format!("{shown} of {rows} rows"),
        None => subtitle.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use polars::df;

    use super::*;

    #[test]
    fn sample_takes_one_row_from_each_run() {
        let ys: Vec<f64> = (0..100_000).map(f64::from).collect();
        let ds: Dataset = Dataset::new(df!("y" => ys).unwrap(), "");
        let (sample, rows): (DataFrame, usize) = sample(&ds, &["y"]).unwrap();
        assert_eq!((sample.height(), rows), (MAX_SAMPLE, 100_000));
        let picked: Vec<f64> = f64s(&sample, "y").unwrap();
        assert!(
            picked
                .iter()
                .enumerate()
                .all(|(run, &y)| y as usize / 2 == run)
        );
        assert_eq!(thin(&picked, 8000).len(), 8000);
    }

    #[test]
    fn slots_place_rows_by_label() {
        let frame: DataFrame = df!("c" => ["b", "a", "z"]).unwrap();
        let labels: Vec<String> = vec!["a".to_string(), "b".to_string()];
        assert_eq!(
            slots(&frame, Some("c"), &labels).unwrap(),
            [Some(1), Some(0), None]
        );
        assert_eq!(slots(&frame, None, &[]).unwrap(), [Some(0); 3]);
    }
}
