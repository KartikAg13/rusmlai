# rusmlai

A Rust library for tabular machine learning, with the names you know from pandas, scikit-learn and seaborn. Tables run on [Polars](https://docs.pola.rs/). The same code works in a `.rs` program and in a Jupyter notebook through the [evcxr](https://github.com/evcxr/evcxr) kernel.

rusmlai is young, but it works end to end today: load a CSV, look at it, draw 18 kinds of chart, preprocess, split, train linear and logistic models, score them, and compare every run in a training journal. On a 630,000-row Kaggle file the linear models train 9× to 26× faster than scikit-learn, and logistic regression 1.3× to 5.6× faster, with the same coefficients. Trees come next.

## Install

You need Rust 1.95 or newer ([rustup.rs](https://rustup.rs)). Then:

```sh
cargo new first-look
cd first-look
cargo add rusmlai
```

rusmlai runs on Polars, a large crate: the first build prints `Compiling` lines for a few minutes, once. Later builds take seconds.

## What works today

```rust
use rusmlai::prelude::*;

fn main() {
    let ds: Dataset = read_csv("train.csv"); // stats are cached in metadata/ next to the CSV

    ds.head(5);
    ds.info();               // columns, non-null counts, dtypes
    ds.describe();           // count, mean, std, min, quartiles, max
    ds.metadata(&[]);        // per-column stats, with the top categories of text columns

    let features: Dataset = ds.drop(&["id"]);
    let score = ds.try_numeric_stats("exam_score").unwrap();
    println!("{:?} median {}", features.shape(), score.median);
}
```

- `read_csv` parses the file with Polars and computes stats for every column in one pass. Numeric columns get mean, std, variance, min, max and quartiles. Text columns get category counts and the mode.
- Other sources give the same `Dataset`, with pandas' names:

  | Source | Call |
  |---|---|
  | CSV, or TSV by its `.tsv` extension | `read_csv("train.tsv")` |
  | Parquet | `read_parquet("train.parquet")` |
  | Excel (`.xlsx`, `.xls`, `.xlsb`) and `.ods` | `read_excel("train.xlsx", "")` (`""` is the first sheet) |
  | SQLite | `read_sql("SELECT * FROM students", "school.db")` |
  | PostgreSQL, MySQL | `read_sql(query, "postgresql://user:pw@host/db")` or `"mysql://..."` (no TLS) |

  Parquet caches its stats like a CSV. Excel and SQL compute them on every read: a workbook has several sheets, and a query's rows can change.
- The stats go into a `metadata/` folder next to the CSV, with the size and modification time of the file. The next `read_csv` of an unchanged file loads them and does not compute them again.
- The `metadata/` folder is trusted like code: its files set what scalers, imputers and models learn. For a data folder that came from someone else with its `metadata/` folder, call `set_cache(false)` first (or set `RUSMLAI_CACHE=off`). rusmlai then reads and writes nothing next to your data.
- `select` and `drop` share the stats of the original dataset. They do not copy or compute them again.
- Chart results get the same cache. The aggregated numbers of each chart are stored per CSV, so a chart comes back instantly, also after a restart.
- Tables print as text in a terminal and as HTML tables in a Jupyter notebook. `notebooks/s6e1.ipynb` and `s6e2.ipynb` are sample notebooks with every chart, one per cell.

Charts use seaborn's names and draw SVG with hover values:

```rust
ds.countplot("course", "");
ds.bar("gender", "exam_score", "course", "mean")
    .palette("colorblind")
    .title("Mean score by gender and course");
ds.hist("exam_score", "", 30);
ds.boxplot("sleep_quality", "exam_score", "gender");
ds.heatmap("course", "study_method", "exam_score", "mean").cmap("viridis");
ds.corrplot();
ds.regplot("study_hours", "exam_score", "");
ds.bar("course", "exam_score", "", "mean")
    .sort("value")
    .horizontal()
    .hline(62.5, "overall mean");
```

- The charts are `bar`, `countplot`, `describe_bar`, `pie`, `hist`, `boxplot`, `stripplot`, `violinplot`, `kdeplot`, `qqplot`, `heatmap`, `corrplot`, `scatter`, `plot`, `regplot`, `pairplot`, `missing` and `missing_matrix`.
- Options chain like keyword arguments in Python: `title`, `subtitle`, `xlabel`, `ylabel`, `palette`, `colors`, `color`, `color_map`, `theme`, `size`, `order`, `hue_order`, `sort`, `top`, `values`, `ylim`, `xlim`, `cmap`, `sample`, `horizontal`, `percent`, `stacked`, `hline`, `vline`, `legend`, `kde`. `set_theme` and `set_palette` set defaults for a whole notebook.
- Colors take hex codes or names: `"teal"`, `"tab:blue"`, or `"C0"` for the first color of the current palette. By default each category gets its own color from the `rusmlai` palette; `colorblind`, `tab10`, `deep` and five more palettes are options.
- Heatmaps draw a color scale. Long category names tilt instead of being cut.
- In a notebook, a chart shows itself. In a program, it saves itself to `metadata/plots/{title}.png` next to the CSV, unless you call `.save(path)` (`.png` or `.svg`).
- A PNG takes about 0.1 s to draw when dependencies are optimized, and about 0.5 s in a plain debug build. To make `cargo run` fast for charts and for Polars, add this to your `Cargo.toml` (the first build then takes longer):

  ```toml
  [profile.dev.package."*"]
  opt-level = 3
  ```

On a 630,000-row, 13-column Kaggle CSV (an i5-1135G7 laptop):

| Operation | Time |
|---|---:|
| first `read_csv` (parse and compute stats) | 120 ms |
| later `read_csv` (parse and load cached stats) | 62 ms |
| load the cached stats alone | 21 µs |
| stats for all 13 columns | 58 ms |

## Models

```rust
use rusmlai::{prelude::*, steps};

fn main() {
    let ds: Dataset = read_csv("train.csv").drop(&["id"]);
    let (train, test) = train_test_split(&ds, 0.2, 42, "");
    let prep = ColumnTransformer::new()
        .add("num", &["study_hours", "class_attendance"], steps![StandardScaler::new()])
        .add("cat", &["course", "sleep_quality"], steps![OneHotEncoder::new()]);
    let mut model = Pipeline::new(prep, LinearRegression::new());
    model.fit(&train, "exam_score");
    println!("R² on the test rows: {:.3}", model.score(&test, "exam_score"));
    ds.journal(); // every model trained on this CSV, with its settings and scores
}
```

- Transformers: `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, `SimpleImputer`, `OneHotEncoder`, `OrdinalEncoder` and `LabelEncoder`, with scikit-learn's settings. They learn from the cached stats, so a fit makes no pass over the data.
- Models: `LinearRegression` (Ridge, Lasso and ElasticNet are its `alpha` and `l1_ratio` settings), `LogisticRegression` (two or more classes) and `DummyRegressor`. They give scikit-learn's coefficients.
- A fit with the same settings on the same data loads the earlier result from the cache instead of training again. Every fit and score goes into `metadata/{stem}.journal.html`, an offline page with each run's settings, scores and loss curve. `model.load_run(&train, "r03")` brings a run back.
- Errors name the fix: `feature 'course' is str, not a number: encode it with OneHotEncoder or OrdinalEncoder in the ColumnTransformer`.

Fit times on the same 630,000-row file, against scikit-learn 1.9:

| Fit | rusmlai, 1 core | scikit-learn, 1 core | rusmlai, all cores | scikit-learn, all cores |
|---|---:|---:|---:|---:|
| `LinearRegression` | 48.9 ms | 544.5 ms | 20.6 ms | 528.5 ms |
| `LogisticRegression` (scikit-learn: `newton-cholesky`) | 175.0 ms | 398.8 ms | 84.5 ms | 476.8 ms |
| `Pipeline`: scale, one-hot, linear | 228.1 ms | 1083.7 ms | 103.9 ms | 1069.5 ms |

## Conventions

- Function and parameter names follow Python: `read_csv`, `head`, `describe`, `fit`, `predict`, `n_jobs`, `max_depth`. Defaults follow scikit-learn, so you can compare results directly.
- Each fallible method has a `try_*` form that returns a `Result`. The short form (`read_csv`, `select`) panics with a readable message that points at your line of code. In a notebook, this shows as a failed cell, like a Python exception, and the kernel keeps its variables.
- An empty `&str` means "not used", for example `hue = ""` in a chart.

## Roadmap

Done: data loading, stats, charts, the transformers, `ColumnTransformer`, `Pipeline`, `train_test_split`, linear and logistic regression, the model cache and the journal. Next:

1. Data operations that pandas users reach for: saving a dataset (`to_csv`), filtering and sorting rows, and cross validation. Until then, `ds.dataframe()` is the Polars table, and `Dataset::new` makes a dataset of a changed copy.
2. About 20 more models:
   - decision trees, random forests, bagging, AdaBoost and histogram gradient boosting (next);
   - kNN, Naive Bayes, linear SVM, an MLP;
   - k-means, DBSCAN and agglomerative clustering;
   - PCA, SVD, LDA and feature selection;
   - FP-growth and Apriori association rules.
3. A documentation website.

How the models will be fast:

- Trees put each feature into at most 255 bins (one byte per value) once, and use the bins for every tree, fold and refit. Gradient boosting follows the design of LightGBM.
- Linear models, PCA and cross validation share one pass that computes XᵀX. After that pass, they work on a small matrix.
- Values are stored in narrow types where it is safe (bytes, f32), and every sum and matrix solve uses f64.
- Work that a fit can reuse (bins, matrices, fitted models) can stay in the cache next to the CSV.

## Development

```sh
cargo build
cargo test
cargo run --example s6e1          # every chart, on the Kaggle Playground S6E1 file in tests/fixtures/s6e1/
cargo bench                       # criterion suites in benches/ (a cold build takes minutes and uses every core)
benches/profile.sh metadata '^stats/numeric/exam_score$' 5   # perf, flamegraph and heaptrack for one bench
```

`.cargo/config.toml` links with [mold](https://github.com/rui314/mold) on x86_64 Linux. Install mold, or remove that file, before you build.

## License

Apache-2.0.
