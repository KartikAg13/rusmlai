use std::{
    borrow::Cow,
    fs::{self, Metadata, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    sync::OnceLock,
    time::UNIX_EPOCH,
};

use polars::{datatypes::DataType, frame::DataFrame};
use serde::{Serialize, de::DeserializeOwned};

use crate::metadata::{constants::CACHE_DIRECTORY, error::MetadataError};

pub fn unwrap_f64(frame: &DataFrame, name: &str) -> Result<f64, MetadataError> {
    Ok(frame
        .column(name)?
        .cast(&DataType::Float64)?
        .f64()?
        .get(0)
        .unwrap_or(f64::NAN))
}

pub fn option_to_oncelock<T>(value: Option<T>) -> OnceLock<T> {
    let lock: OnceLock<T> = OnceLock::new();
    if let Some(v) = value {
        let _ = lock.set(v);
    }
    lock
}

pub fn file_fingerprint(path: impl AsRef<Path>) -> io::Result<(u64, u64)> {
    let meta: Metadata = fs::metadata(path)?;
    let mtime: u64 = meta
        .modified()?
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    Ok((meta.len(), mtime))
}

pub fn cache_path(csv: &Path, suffix: &str) -> PathBuf {
    let stem: Cow<str> = csv.file_stem().unwrap_or_default().to_string_lossy();
    csv.with_file_name(CACHE_DIRECTORY)
        .join(format!("{stem}{suffix}"))
}

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
