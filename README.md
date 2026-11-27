# rusmlai

A Rust library for tabular machine learning, with the names you know from pandas, scikit-learn and seaborn. Tables run on [Polars](https://docs.pola.rs/). The same code works in a `.rs` program and in a Jupyter notebook through the [evcxr](https://github.com/evcxr/evcxr) kernel.

rusmlai is young, but it works end to end today: load a CSV, look at it, clean it, draw 18 kinds of chart, preprocess, split, train linear models, trees, forests and boosters, score them, and compare every run in a training journal. On a 630,000-row Kaggle file the linear models train 9× to 26× faster than scikit-learn, and logistic regression 1.3× to 5.6× faster, with the same coefficients. Decision trees train 3.5× to 33× faster, forests and bags 2.7× to 5.4×, AdaBoost 24× to 41× and a grid search 17×. Histogram gradient boosting trains 1.17× to 1.86× faster. The scores are scikit-learn's.

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

- Transformers: `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, `SimpleImputer`, `OneHotEncoder`, `OrdinalEncoder`, `LabelEncoder`, `CountEncoder`, `TargetEncoder`, `KBinsDiscretizer` and `QuantileTransformer`, with scikit-learn's settings. They learn from the cached stats, so a fit makes no pass over the data.
- Models: `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `RidgeCV`, `LogisticRegression` (two or more classes), `LogisticRegressionCV`, `RidgeClassifier` and `DummyRegressor`. They give scikit-learn's coefficients. `DecisionTreeClassifier` and `DecisionTreeRegressor` grow scikit-learn's tree when each feature has at most 255 distinct values (more values go into 255 buckets), and take missing values as they are. `RandomForestClassifier` and `RandomForestRegressor` grow many of those trees on all cores, and `BaggingClassifier` and `BaggingRegressor` average copies of any model. `HistGradientBoostingClassifier` and `HistGradientBoostingRegressor` are scikit-learn's histogram gradient boosting, tree for tree: early stopping, its five regression losses (squared and absolute error, quantile, Poisson, gamma), class weights, and text columns as categorical features with no encoding; plus LightGBM's `subsample`, `l1_regularization` and `min_split_gain`. `AdaBoostClassifier` is SAMME on decision trees. `ExtraTreesClassifier` and `ExtraTreesRegressor` draw random splits. `XGBRegressor` / `XGBClassifier`, `LGBMRegressor` / `LGBMClassifier` and `CatBoostRegressor` / `CatBoostClassifier` keep their library's names, defaults and split rules. The boosters take an eval set (`fit_eval_set`), and XGB and LGBM take `early_stopping_rounds`, which stop where XGBoost and LightGBM stop. `StackingClassifier`, `StackingRegressor` and `HillClimbing` combine models. The tree and forest classifiers take `class_weight`, and every tree model takes `max_leaf_nodes`. Their scores agree with scikit-learn's as two of its own seeds agree.
- Metrics with scikit-learn's names: `r2_score`, `mean_squared_error`, `accuracy_score`, `balanced_accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `classification_report`, `log_loss`, `roc_auc_score`, `confusion_matrix` and more.
- Model selection: `GridSearchCV` (scikit-learn's folds, `cv_results_`, `best_params_`, nested names such as `estimator__max_depth`), `cross_val_score`, `cross_val_predict`, `KFold` and `StratifiedKFold`. The fits run in parallel, and the candidates of a fold share its bins.
- A fit with the same settings on the same data loads the earlier result from the cache instead of training again. Every fit and score goes into `metadata/{stem}.journal.html`, an offline page with each run's settings, scores and loss curve. `model.load_run(&train, "r03")` brings a run back. `model.save(path)` and `Model::load(path)` keep a model in one file, and the XGBoost, LightGBM and CatBoost types also write their library's own model file.
- Errors name the fix: `feature 'course' is str, not a number: encode it with OneHotEncoder or OrdinalEncoder in the ColumnTransformer`.

Fit times on the same 630,000-row file, against scikit-learn 1.9:

| Fit | rusmlai, 1 core | scikit-learn, 1 core | rusmlai, all cores | scikit-learn, all cores |
|---|---:|---:|---:|---:|
| `LinearRegression` | 48.9 ms | 544.5 ms | 20.6 ms | 528.5 ms |
| `LogisticRegression` (scikit-learn: `newton-cholesky`) | 175.0 ms | 398.8 ms | 84.5 ms | 476.8 ms |
| `Pipeline`: scale, one-hot, linear | 228.1 ms | 1083.7 ms | 103.9 ms | 1069.5 ms |
| `DecisionTreeClassifier`, full depth (180,000 nodes on both sides) | 411.1 ms | 2612.0 ms | 146.8 ms | 2609.7 ms |
| `DecisionTreeRegressor`, full depth (1.2 million nodes on both sides) | 897.3 ms | 3156.4 ms | 298.7 ms | 3159.2 ms |
| `DecisionTreeRegressor`, `max_depth=8` | 122.0 ms | 1287.9 ms | 47.8 ms | 1289.9 ms |
| `RandomForestClassifier`, 10 full trees (1.67 million nodes on both sides) | 2.00 s | 5.41 s | 0.75 s | 2.56 s |
| `RandomForestRegressor`, 10 full trees (7.7 million nodes on both sides) | 5.48 s | 20.85 s | 1.76 s | 7.71 s |
| `BaggingClassifier` of 10 full trees | 2.82 s | 14.98 s | 1.24 s | 6.70 s |
| `HistGradientBoostingRegressor`, 100 rounds of 31 leaves | 3.05 s | 4.47 s | 1.81 s | 2.13 s |
| the same with the 7 text columns as categories | 3.05 s | 5.69 s | 1.78 s | 2.84 s |
| `HistGradientBoostingClassifier`, 100 rounds of 31 leaves | 5.33 s | 6.99 s | 2.68 s | 3.15 s |
| `AdaBoostClassifier`, 50 stumps | 1.58 s | 38.46 s | 0.94 s | 38.45 s |
| `GridSearchCV`, 6 trees × 5 folds and the refit | 2.61 s | 44.75 s | 1.19 s | 20.14 s |

## Conventions

- Function and parameter names follow Python: `read_csv`, `head`, `describe`, `fit`, `predict`, `n_jobs`, `max_depth`. Defaults follow scikit-learn, so you can compare results directly.
- Each fallible method has a `try_*` form that returns a `Result`. The short form (`read_csv`, `select`) panics with a readable message that points at your line of code. In a notebook, this shows as a failed cell, like a Python exception, and the kernel keeps its variables.
- An empty `&str` means "not used", for example `hue = ""` in a chart.

## Roadmap

Done: everything above. Next, for 1.0: a learning-curve chart, more of scikit-learn's settings on each model, and monotone and interaction constraints on the boosters. kNN, naive Bayes and the unsupervised models (k-means, PCA and others) come in 1.2.

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) explains the setup, where each part of the code lives, the rules it follows, and what to run before a pull request.

## License

Apache-2.0.

rusmlai is not affiliated with or endorsed by scikit-learn, pandas, seaborn, XGBoost, LightGBM or CatBoost. Types such as `XGBRegressor`, `LGBMClassifier` and `CatBoostRegressor` reimplement those libraries' published algorithms under their names, so their users find the settings they know.
