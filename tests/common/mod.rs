#![allow(dead_code)]

use std::{
    fs,
    path::{Path, PathBuf},
};

/// A copy of `tests/fixtures/data/<file>` in its own empty folder under `target/tmp/<name>/`, so cache
/// files never touch the tracked fixtures.
pub fn scratch_copy(name: &str, file: &str) -> PathBuf {
    let dir: PathBuf = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let csv: PathBuf = dir.join(file);
    fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/data")
            .join(file),
        &csv,
    )
    .unwrap();
    csv
}

/// A copy of `tests/fixtures/data/students.csv` (12 rows x 6 columns).
pub fn scratch_csv(name: &str) -> PathBuf {
    scratch_copy(name, "students.csv")
}

/// `metadata/<stem><suffix>` next to the CSV.
pub fn cache_file(csv: &Path, suffix: &str) -> PathBuf {
    rusmlai::metadata::utils::cache_path(csv, suffix)
}

pub fn lines(path: &Path) -> usize {
    fs::read_to_string(path).unwrap().lines().count()
}
