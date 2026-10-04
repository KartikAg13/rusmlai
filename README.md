# rusmlai

A Rust library for tabular machine learning, with the names you know from pandas, scikit-learn and seaborn. Tables run on [Polars](https://docs.pola.rs/). The same code works in a `.rs` program and in a Jupyter notebook through the [evcxr](https://github.com/evcxr/evcxr) kernel.

rusmlai is early. Data loading, column statistics and table display work today. Charts come next, then preprocessing and models.

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
- The stats go into a `metadata/` folder next to the CSV, with the size and modification time of the file. The next `read_csv` of an unchanged file loads them and does not compute them again.
- `select` and `drop` share the stats of the original dataset. They do not copy or compute them again.
- Chart results get the same cache. The aggregated numbers of each chart are stored per CSV, so a chart comes back instantly, also after a restart.
- Tables print as text in a terminal and as HTML tables in a Jupyter notebook. `notebooks/eda.ipynb` is a sample notebook.

On a 630,000-row, 13-column Kaggle CSV (an i5-1135G7 laptop):

| Operation | Time |
|---|---:|
| first `read_csv` (parse and compute stats) | 120 ms |
| later `read_csv` (parse and load cached stats) | 62 ms |
| load the cached stats alone | 21 µs |
| stats for all 13 columns | 58 ms |

## Conventions

- Function and parameter names follow Python: `read_csv`, `head`, `describe`, `fit`, `predict`, `n_jobs`, `max_depth`. Defaults follow scikit-learn, so you can compare results directly.
- Each fallible method has a `try_*` form that returns a `Result`. The short form (`read_csv`, `select`) panics with a readable message that points at your line of code. In a notebook, this shows as a failed cell, like a Python exception, and the kernel keeps its variables.
- An empty `&str` means "not used", for example `hue = ""` in a chart.

## Roadmap

1. Charts in the style of seaborn (`bar`, `count`, `hist`, `boxplot`, `scatter`, `heatmap`, `corrplot`). They are drawn as SVG with hover values, shown inline in notebooks, and saved as files from programs.
2. Preprocessing: `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `SimpleImputer`, `OneHotEncoder` and others, a `Pipeline`, `train_test_split` and cross validation. Scalers and imputers fit from the cached stats, so a fit on a newly loaded CSV makes no pass over the data.
3. About 25 models:
   - linear and logistic regression, Ridge, Lasso;
   - decision trees, random forests, bagging, AdaBoost and histogram gradient boosting;
   - kNN, Naive Bayes, linear SVM, an MLP;
   - k-means, DBSCAN and agglomerative clustering;
   - PCA, SVD, LDA and feature selection;
   - FP-growth and Apriori association rules.
4. A documentation website.

How the models will be fast:

- Trees put each feature into at most 255 bins (one byte per value) once, and use the bins for every tree, fold and refit. Gradient boosting follows the design of LightGBM.
- Linear models, PCA and cross validation share one pass that computes XᵀX. After that pass, they work on a small matrix.
- Values are stored in narrow types where it is safe (bytes, f32), and every sum and matrix solve uses f64.
- Work that a fit can reuse (bins, matrices, fitted models) can stay in the cache next to the CSV.

## Development

```sh
cargo build
cargo test
cargo bench                       # criterion suites in benches/ (a cold build takes minutes and uses every core)
benches/profile.sh metadata '^stats/numeric/exam_score$' 5   # perf, flamegraph and heaptrack for one bench
```

`.cargo/config.toml` links with [mold](https://github.com/rui314/mold) on x86_64 Linux. Install mold, or remove that file, before you build.

## License

Apache-2.0.
