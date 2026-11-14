//! File names of the cache.

/// The folder next to the CSV that holds its cache files.
pub const CACHE_DIRECTORY: &str = "metadata";
/// Suffix of the column stats file: `{stem}.json`.
pub const DATASET_CACHE: &str = ".json";
/// Suffix of the chart cache file: `{stem}.plots.jsonl`.
pub const PLOTS_CACHE: &str = ".plots.jsonl";
/// Suffix of the model cache: `{stem}.models.jsonl`, one line per fit and per score.
pub const MODELS_CACHE: &str = ".models.jsonl";
/// Suffix of the training journal page: `{stem}.journal.html`.
pub const JOURNAL: &str = ".journal.html";
