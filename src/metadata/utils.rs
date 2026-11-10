//! File helpers for the cache: paths, fingerprints and JSONL.

use std::{
    borrow::Cow,
    fs::{self, Metadata, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    time::UNIX_EPOCH,
};

use polars::{datatypes::DataType, frame::DataFrame};
use serde::{Serialize, de::DeserializeOwned};

use crate::metadata::{constants::CACHE_DIRECTORY, error::MetadataError};

/// The first value of column `name` as `f64`; NaN if it is null.
pub fn unwrap_f64(frame: &DataFrame, name: &str) -> Result<f64, MetadataError> {
    Ok(frame
        .column(name)?
        .cast(&DataType::Float64)?
        .f64()?
        .get(0)
        .unwrap_or(f64::NAN))
}

/// `(size in bytes, mtime in nanoseconds)` of a file, the key that decides if a cache is stale.
/// Nanoseconds, so a rewrite of the same size within one second is still seen. A time before
/// 1970 counts as 0.
pub fn file_fingerprint(path: impl AsRef<Path>) -> io::Result<(u64, u64)> {
    let meta: Metadata = fs::metadata(path)?;
    let mtime: u64 = meta
        .modified()?
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_nanos() as u64);
    Ok((meta.len(), mtime))
}

/// Where the cache file with `suffix` for `csv` lives: `{csv dir}/metadata/{stem}{suffix}`.
///
/// ```
/// use std::path::Path;
/// use rusmlai::metadata::utils::cache_path;
/// let path = cache_path(Path::new("data/train.csv"), ".json");
/// assert_eq!(path, Path::new("data/metadata/train.json"));
/// ```
pub fn cache_path(csv: &Path, suffix: &str) -> PathBuf {
    let stem: Cow<str> = csv.file_stem().unwrap_or_default().to_string_lossy();
    csv.with_file_name(CACHE_DIRECTORY)
        .join(format!("{stem}{suffix}"))
}

/// The entries of a JSONL file whose first line is `fingerprint`. `None` if the file is missing
/// or its fingerprint differs; lines that don't parse are skipped.
pub fn read_jsonl<T: DeserializeOwned>(path: &Path, fingerprint: (u64, u64)) -> Option<Vec<T>> {
    let text: String = fs::read_to_string(path).ok()?;
    let mut lines = text.lines();
    if serde_json::from_str::<(u64, u64)>(lines.next()?).ok()? != fingerprint {
        return None;
    }
    Some(
        lines
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect(),
    )
}

/// Writes `fingerprint` and then one line per entry, replacing the file.
pub fn write_jsonl<T: Serialize>(
    path: &Path,
    fingerprint: (u64, u64),
    entries: &[T],
) -> io::Result<()> {
    fs::create_dir_all(path.parent().unwrap())?;
    let mut out: String = serde_json::to_string(&fingerprint)? + "\n";
    for entry in entries {
        out += &(serde_json::to_string(entry)? + "\n");
    }
    fs::write(path, out)?;
    print_saved(path);
    Ok(())
}

/// Appends one line. Fails with `NotFound` if the file doesn't exist.
pub fn append_jsonl<T: Serialize>(path: &Path, entry: &T) -> io::Result<()> {
    let line: String = serde_json::to_string(entry)? + "\n";
    OpenOptions::new()
        .append(true)
        .open(path)?
        .write_all(line.as_bytes())?;
    print_saved(path);
    Ok(())
}

/// Tells the user the full path of a file that was just created or updated.
pub(crate) fn print_saved(path: &Path) {
    let full: PathBuf = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    println!("[rusmlai] saved {}", full.display());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An empty folder of its own under `target/tmp/` (unit tests get no `CARGO_TARGET_TMPDIR`).
    fn scratch(name: &str) -> PathBuf {
        let dir: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target/tmp/utils")
            .join(name);
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn cache_path_keeps_the_stem_and_sits_in_metadata() {
        let p = |csv: &str, suffix: &str| cache_path(Path::new(csv), suffix);
        assert_eq!(
            p("data/data.v2.csv", ".json"),
            Path::new("data/metadata/data.v2.json")
        );
        assert_eq!(
            p("train.csv", ".plots.jsonl"),
            Path::new("metadata/train.plots.jsonl")
        );
    }

    #[test]
    fn read_jsonl_rejects_bad_files_and_skips_broken_lines() {
        let dir: PathBuf = scratch("read");
        let path: PathBuf = dir.join("x.jsonl");
        assert!(read_jsonl::<u32>(&path, (1, 2)).is_none()); // missing
        fs::write(&path, "").unwrap();
        assert!(read_jsonl::<u32>(&path, (1, 2)).is_none()); // empty
        fs::write(&path, "not json\n1\n").unwrap();
        assert!(read_jsonl::<u32>(&path, (1, 2)).is_none()); // bad header
        fs::write(&path, "[9,9]\n1\n").unwrap();
        assert!(read_jsonl::<u32>(&path, (1, 2)).is_none()); // stale fingerprint
        fs::write(&path, "[1,2]\n1\n2\n{\"cut").unwrap();
        assert_eq!(read_jsonl::<u32>(&path, (1, 2)), Some(vec![1, 2])); // broken last line
    }

    #[test]
    fn write_append_read_round_trip() {
        let path: PathBuf = scratch("round").join("deeper").join("x.jsonl");
        write_jsonl(&path, (3, 4), &["a".to_string()]).unwrap();
        append_jsonl(&path, &"b".to_string()).unwrap();
        assert_eq!(
            read_jsonl::<String>(&path, (3, 4)),
            Some(vec!["a".into(), "b".into()])
        );
    }

    #[test]
    fn append_to_a_missing_file_is_an_error() {
        let path: PathBuf = scratch("append").join("none.jsonl");
        let err: io::Error = append_jsonl(&path, &1).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn fingerprint_changes_when_the_file_grows() {
        let path: PathBuf = scratch("fingerprint").join("x.csv");
        fs::write(&path, "a\n1\n").unwrap();
        let before: (u64, u64) = file_fingerprint(&path).unwrap();
        fs::write(&path, "a\n1\n2\n").unwrap();
        assert_ne!(file_fingerprint(&path).unwrap(), before);
    }

    #[test]
    fn unwrap_f64_gives_nan_for_a_null() {
        let frame: DataFrame = polars::df!("x" => [None::<f64>]).unwrap();
        assert!(unwrap_f64(&frame, "x").unwrap().is_nan());
        assert!(unwrap_f64(&frame, "missing").is_err());
    }
}
