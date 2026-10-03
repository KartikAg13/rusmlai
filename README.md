# rusmlai

A Rust library for tabular machine learning, with the names you already know from pandas, scikit-learn and seaborn. Tables run on [Polars](https://docs.pola.rs/), and the same code works in a `.rs` program and in a Jupyter notebook through the [evcxr](https://github.com/evcxr/evcxr) kernel.

rusmlai is early. Loading data and column statistics work today. Preprocessing, charts and models are next.

## What works today

```rust
use rusmlai::dataset::{Dataset, read_csv};

fn main() {
    let ds: Dataset = read_csv("train.csv"); // stats are cached in metadata/ next to the CSV

    ds.head(5);
    ds.info();               // columns, non-null counts, dtypes
    ds.describe();           // count, mean, std, min, quartiles, max
    ds.metadata(&[]);        // per-column stats, including top categories for text columns

    let features: Dataset = ds.drop(&["id"]);
    let score = ds.try_numeric_stats("exam_score").unwrap();
    println!("{:?} median {}", features.shape(), score.median);
}
```

- `read_csv` parses the file with Polars and computes stats for every column in one pass. Numeric columns get mean, std, variance, min, max and quartiles; text columns get category counts and the mode.
- The stats are saved next to the CSV, in a `metadata/` folder, together with the file's size and modification time. The next `read_csv` of an unchanged file loads them instead of recomputing.
- `select` and `drop` share stats with the original dataset instead of copying or recomputing them.
- Plot results get the same treatment. Each chart's aggregated numbers are cached per CSV, so a chart drawn once comes back instantly, even after a restart.

On a 630,000-row, 13-column Kaggle CSV (an i5-1135G7 laptop):

| Operation | Time |
|---|---:|
| first `read_csv` (parse and compute stats) | 120 ms |
| later `read_csv` (parse and load cached stats) | 62 ms |
| loading the cached stats alone | 21 µs |
| stats for all 13 columns | 58 ms |

## Conventions

- Function and parameter names follow Python: `read_csv`, `head`, `describe`, `fit`, `predict`, `n_jobs`, `max_depth`. Defaults follow scikit-learn, so results can be compared directly.
- Every fallible method has a `try_*` form that returns a `Result`. The short form (`read_csv`, `select`) prints a readable error and panics, which in a notebook shows up as a failed cell, like a Python exception.
- An empty `&str` means "not used", for example `hue = ""` in a chart.

## Roadmap

1. Preprocessing: `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `SimpleImputer`, `OneHotEncoder` and friends, a `Pipeline`, `train_test_split` and cross validation. Scalers and imputers fit by reading the cached stats, so fitting on a freshly loaded CSV makes no pass over the data.
2. Charts in seaborn's style (`bar`, `hist`, `boxplot`, `scatter`, `heatmap`, `corrplot`), drawn as SVG, shown inline in notebooks and saved as files from programs.
3. About 25 models:
   - linear and logistic regression, Ridge, Lasso;
   - decision trees, random forests, bagging, AdaBoost and histogram gradient boosting;
   - kNN, Naive Bayes, linear SVM, an MLP;
   - k-means, DBSCAN and agglomerative clustering;
   - PCA, SVD, LDA and feature selection;
   - FP-growth and Apriori association rules.
4. A documentation website.

## Development

```sh
cargo build
cargo test
cargo bench                       # criterion suites in benches/
benches/profile.sh metadata '^stats/numeric/exam_score$' 5   # perf, flamegraph and heaptrack for one bench
```

`.cargo/config.toml` links with [mold](https://github.com/rui314/mold) on x86_64 Linux. Install it, or remove that file, before building.

## License

Apache-2.0.
