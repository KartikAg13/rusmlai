# rusmlai

A Rust library for tabular machine learning, with the names you know from pandas, scikit-learn and seaborn. Tables run on [Polars](https://docs.pola.rs/). The same code works in a `.rs` program and in a Jupyter notebook through the [evcxr](https://github.com/evcxr/evcxr) kernel.

rusmlai is early. Data loading, column statistics, table display and charts work today. Preprocessing and models come next.

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

## Conventions

- Function and parameter names follow Python: `read_csv`, `head`, `describe`, `fit`, `predict`, `n_jobs`, `max_depth`. Defaults follow scikit-learn, so you can compare results directly.
- Each fallible method has a `try_*` form that returns a `Result`. The short form (`read_csv`, `select`) panics with a readable message that points at your line of code. In a notebook, this shows as a failed cell, like a Python exception, and the kernel keeps its variables.
- An empty `&str` means "not used", for example `hue = ""` in a chart.

## Roadmap

1. Preprocessing: `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `SimpleImputer`, `OneHotEncoder` and others, combined in a `ColumnTransformer` and joined to the model in a `Pipeline`, `train_test_split` and cross validation. Scalers and imputers fit from the cached stats, so a fit on a newly loaded CSV makes no pass over the data.
2. About 25 models:
   - linear regression (with Ridge, Lasso and ElasticNet penalties as settings) and logistic regression;
   - decision trees, random forests, bagging, AdaBoost and histogram gradient boosting;
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
