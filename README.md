# rusmlai

Rust library for tables, sklearn-style models, and seaborn-style plots. Public names match Python: `fit`, `predict`, `transform`, `Pipeline`, `train_test_split`, `n_jobs`, `max_depth`, `hidden_layer_sizes`. Empty `&str` means unused. Public methods panic through `fatal!`; they do not return `Result`.

Crate version is `0.2.0`. Edition 2024. Tables go through [Polars](https://docs.pola.rs/) 0.55.2 with the `lazy` feature.

What exists in `src/` today: `dataset`, `metadata` (column stats plus an empty `PlotCache`), `plot` (`PlotSpec` / `PlotResult`), `preprocess` (stub), `pipeline.rs` (empty). Models, journal HTML, and Burn are planned. This file is the compute plan so we do not redo it from memory.

---

## Layout

| Path | Role |
| --- | --- |
| `src/dataset/` | `Dataset { dataframe, metadata }`, `read_csv`, `select`, `drop` |
| `src/metadata/` | Column profile `{stem}.json`, `PlotCache` for `{stem}.plots.json` |
| `src/plot/` | Chart drawing. Spec is the cache key, result is the numbers |
| `src/preprocess/` | `StandardScaler` and the rest. `fit` mutates, `transform` does not |
| `src/pipeline.rs` | sklearn `Pipeline`: transformers then one estimator |
| `src/linear_model/` | OLS, Ridge, Lasso, ElasticNet, logistic. [faer](https://github.com/sarah-ek/faer-rs) |
| `src/tree/` | One `DecisionTree`. Classifier and regressor only change impurity |
| `src/ensemble/` | `RandomForest*` = many of those trees |
| `src/neural_network/` | MLP via [Burn](https://burn.dev/books/burn/getting-started.html) |
| `src/metrics/` | `mean_squared_error`, `r2_score`, `accuracy_score`, … |
| `src/optim/` | `adam`, `sgd`, `rmsprop` names that map to Burn configs |
| `src/journal/` | Run log → `journal.json` and plotly `journal.html` |

`Dataset` is the table. It does not store fitted models or journal runs. `select` would clone them if it did.

Journal is not column metadata. Three files:

- `{csv_dir}/metadata/{stem}.json` : per-column stats
- `{csv_dir}/metadata/{stem}.plots.json` : bar / hist / heatmap / corr aggregates
- `journal.html` (path you pass) : training runs, any CSV

---

## Data path

Every `fit` copies columns once into a `DesignMatrix` (contiguous `f64`, feature names, target). Linear models wrap that as `faer::Mat`. Trees keep row-index slices into the same matrix. MLP copies into a Burn `Tensor`. Do not iterate Polars inside the inner loop.

```rust
let ds = read_csv("train.csv");
let (train, test) = train_test_split(&ds, 0.2, 0, "");
let mut pipe = Pipeline::new()
    .step("scale", StandardScaler::new())
    .step("model", Ridge::new().alpha(1.0));
pipe.fit(&train, "price");
let pred = pipe.predict(&test);
journal.log(&pipe, &r2_score(/* y */, &pred));
journal.to_html("journal.html");
```

`target` is the y column. X is the rest, skipping `id`. `.features(&["a", "b"])` comes later.

---

## Parallel vs sequential

Default for a single tree is sequential (`n_jobs = 1`) so a `random_state` run is easy to match. Forest `n_jobs = -1` uses all cores. Nested “trees × nodes × features” oversubscribes; pick one inner level unless a bench says otherwise.

### Decision tree (`src/tree/`)

Two kinds of parallelism, used at different depths.

**Feature-parallel (inside one node).** The root is one node. Left and right do not exist yet. Scan candidate features in parallel ([rayon](https://github.com/rayon-rs/rayon)) to pick the split. Fat nodes (lots of rows) stay here.

**Node-parallel (after a split).** The two children do not share rows, so they can grow at the same time, then the four grandchildren, and so on. Spawn a task per child. Stop spawning when `n_samples` is small (a few thousand). Task setup then costs more than the split.

Uneven 95/5 splits leave one worker with most of the work. Rayon work-stealing helps. It does not balance the row counts.

Do not use Burn or CUDA on the split finder unless a bench shows the CPU histogram is the limit.

### Random forest (`src/ensemble/`)

Same `DecisionTree` type. No second tree implementation.

Parallelize **across trees**: each worker gets a bootstrap sample and `max_features`. That usually fills the machine. Then grow each tree sequentially, or feature-parallel only at large nodes.

Predict: one pass over trees, then mean or vote. Parallelize over trees if predict shows up in a profile.

### Linear models (`src/linear_model/`)

| Model | Parallel? |
| --- | --- |
| `LinearRegression`, `Ridge` (closed form) | faer multithreaded QR / Cholesky on `XᵀX` / solve. No rayon of our own. |
| `Lasso`, `ElasticNet` (coordinate descent) | sequential CD first. Parallel coordinate blocks only if a bench wins. |
| `LogisticRegression` (`lbfgs`) | sequential solver unless the backend threads itself. |

Tiny OLS often loses to NumPy+MKL. Large `n`, no Python, is where this path wins. GPU launch for a 50-column `XᵀX` is usually slower than faer on CPU.

Do not use Polars, nalgebra, or a handwritten triple loop for GEMM. ndarray+OpenBLAS only if it beats faer on our sizes.

### Neural net (`src/neural_network/`)

Burn does the matmuls. `x.matmul(w)` on `Tensor<B, 2>` so autodiff stays intact. Do not call faer inside the epoch loop.

v1 backend: `ndarray` (CPU). Later `wgpu` or `cuda` by changing `B`, not `fit`. Optimizers (`Adam`, `Sgd` with momentum, `RmsProp`) are Burn’s. We only map sklearn/Keras names.

Batches can use Burn `num_workers` on the dataloader. That is I/O, not tree-style node parallel.

### Metrics and journal

`metrics` is sequential on `y_true` / `y_pred` vectors. Cheap. No Burn.

Journal write is sequential. Plotly HTML is not a training hotspot.

### Plots and metadata

Column stats: one lazy plan **per column** today (`compute_stats`). A single lazy plan for all numeric columns is a later `Metadata::new` change.

Plot cache is a `HashMap` lookup. The group-by on miss can use Polars lazy (already threaded). Drawing SVG is sequential.

---

## What we are not doing

- Writing GPU kernels in [cuda-oxide](https://github.com/NVlabs/cuda-oxide) or [cutile-rs](https://github.com/NVlabs/cutile-rs). Early, NVIDIA-only. Burn/CubeCL already has GEMM.
- Autograd by hand for MLP.
- A second decision tree for the forest.
- `fit` on `Dataset`.
- Scatter/line row samples in `{stem}.plots.json`. Grouped boxplot quantiles are also left out of the plot cache for now.
- `thiserror_context` on `DatasetError`. Keep it on `RusMLAIError` only.

---

## Links

- Polars: [https://docs.pola.rs/](https://docs.pola.rs/)
- faer: [https://github.com/sarah-ek/faer-rs](https://github.com/sarah-ek/faer-rs)
- Burn book: [https://burn.dev/books/burn/getting-started.html](https://burn.dev/books/burn/getting-started.html)
- rayon: [https://docs.rs/rayon](https://docs.rs/rayon)
- sklearn API (names we copy): [https://scikit-learn.org/stable/api/index.html](https://scikit-learn.org/stable/api/index.html)
