# Contributing to rusmlai

Thanks for helping. This page says where things are, the rules the code follows, and what to run before you open a pull request.

## Set up

- Rust 1.95 or newer.
- On x86_64 Linux, `.cargo/config.toml` links with [mold](https://github.com/rui314/mold). Install mold, or delete that file locally.
- Python 3 with pandas, scikit-learn and seaborn, only for the scripts in `bench-python/` that compare rusmlai with Python.

```sh
cargo clippy --all-targets   # builds and lints everything; use it instead of `cargo check`
cargo test
cargo run --example students # a short tour on tests/fixtures/data/students.csv
```

The first build compiles Polars and takes several minutes. `cargo bench` builds Polars again in release mode and uses every core. Run it only when your change is about speed.

## Where things are

The prelude exports every public name, so users never see this layout. Find a type by its scikit-learn or pandas name: it is the same here.

| Folder or file | What it holds |
|---|---|
| `src/dataset/` | `Dataset` (a Polars table and its cached stats), the readers (`read.rs`), cleaning (`clean.rs`), pandas methods (`ops.rs`), `train_test_split` (`split.rs`) |
| `src/metadata/` | Every cache file next to a CSV: column stats (`dataset/`), chart results (`plot/`), fitted models (`models.rs`), the training journal page (`journal.rs`) |
| `src/plot/` | Charts. One file per chart family (`bar.rs`, `dist.rs`, `heat.rs`, `xy.rs`, `pair.rs`, `model.rs`), drawn by `svg.rs`, styled by `style.rs` |
| `src/preprocess/` | Transformers, one file per family, and `ColumnTransformer` |
| `src/models/mod.rs` | The `Model`, `Regressor` and `Classifier` traits, and `ModelError` |
| `src/models/hist.rs` | The histogram tree engine that every tree model shares |
| `src/models/design.rs` | `DesignMatrix`, the numbers that linear models read |
| `src/models/*.rs` | One file per model family: `linear.rs`, `logistic.rs`, `tree.rs`, `forest.rs`, `bagging.rs`, `boosting.rs`, `adaboost.rs`, `xgboost.rs`, `lightgbm.rs`, `catboost.rs`, `stacking.rs`, `hill.rs`. Model selection: `folds.rs`, `search.rs`. Files: `persist.rs` (save, load), `export.rs` (XGBoost, LightGBM and CatBoost files) |
| `src/pipeline.rs`, `src/metrics.rs` | `Pipeline`, and scikit-learn's metrics |
| `src/error.rs`, `src/display.rs`, `src/utils/` | `OrFatal` and warnings, table and figure display, shared helpers |
| `tests/` | Integration tests. `tests/fixtures/data/` holds the small tracked CSVs |
| `benches/` | Criterion benches, one file per area |
| `bench-python/` | The same work in Python, for speed and result comparisons |
| `examples/` | Runnable tours |
| `site/` | The docs website: plain HTML, CSS and JS, no build step |

## Rules the code follows

- **Python names.** Functions, types and settings take scikit-learn's, pandas' or seaborn's names and defaults. One exception: `random_state` defaults to 42, so every run repeats. What `fit` learns ends in `_` (`mean_`, `coef_`); settings don't.
- **One type per scikit-learn class.** `Ridge` and `Lasso` are separate types, even when they share a solver underneath.
- **Every setting that changes the result gets a setter**, with scikit-learn's name and default.
- **`try_*` holds the logic.** Each fallible operation is `try_foo`, returning `Result<_, ModuleError>`. The public `foo` is `#[track_caller]` and calls `try_foo(..).or_fatal()`, so a mistake panics at the user's line with a readable message.
- **One error enum per module** (`DatasetError`, `PlotError`, `ModelError`, ...), made with `thiserror` and `#[non_exhaustive]`.
- **An empty `&str` means "not used"**, like `hue=None` in Python.
- **Build column expressions with `crate::utils::col`**, never Polars' `col`: Polars reads `"*"` and `"^...$"` as patterns, not names.
- **Missing values stay missing.** Never turn a null into a silent `0.0`.
- **Use Polars as the engine.** Transformers return Polars expressions. Don't copy a column into a `Vec<f64>` and write it back.
- **Doc comments end with a Python twin.** Every public item with a Python counterpart ends its docs with `Python:` and one ```` ```python ```` block that does the same and asserts the same. `python3 bench-python/doc_python.py` runs them all.
- **Matrix maths goes through `faer`**, and parallel work runs on the Polars thread pool (`polars_core::runtime::THREAD_POOL.install`).

## Before you open a pull request

1. `cargo clippy --all-targets` shows no warnings.
2. `cargo test` passes.
3. A new public item has docs: what it does, an example, and its Python twin.
4. A new model or transformer matches scikit-learn on fixed data in a test. A tree model is tested on data without ties, because scikit-learn breaks ties at random.
5. If your change is about speed, give the bench before and after, on one core (`POLARS_MAX_THREADS=1`) and on all cores.

## What you won't find

`docs/`, `analysis/`, `notebooks/` and the Kaggle files in `tests/fixtures/` are local to the maintainer and not in git. Tests that need the Kaggle files are `#[ignore]`d, so `cargo test` runs without them.
