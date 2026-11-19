"""scikit-learn's side of the `tree` group of `benches/models.rs`: python3 bench-python/trees.py [repeats]

Like for like: the same rows, the same columns in the same order, the same settings and the same
call (a whole `fit`). On S6E1 (630,000 rows) the 7 text columns are OrdinalEncoder codes, followed
by the 4 numbers; on S6E2 (630,000 rows) the 13 features are as they are in the file, and the
target is its text labels. Each tree prints its size and its training score, so a faster fit of a
smaller or worse tree would show.

`DecisionTree*` is scikit-learn's model of the same name: an exact search over sorted values, on
one thread. `HistGradientBoostingRegressor` with one round is the same algorithm as rusmlai's tree
(255 buckets, histograms, OpenMP), so its row compares the code and not the algorithm. Only at
`max_depth=8`: a full tree of 600,000 leaves needs one histogram per leaf there, about 40 GB.

Each timing is the median of `repeats` runs (5 by default) after one warm-up run, on one thread
and on all cores.
"""

import statistics
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from models import NUMERIC, ROOT, TEXT


def median_ms(work, repeats: int) -> float:
    work()  # warm-up
    times: list[float] = []
    for _ in range(repeats):
        t: float = time.perf_counter()
        work()
        times.append(time.perf_counter() - t)
    return statistics.median(times) * 1000


def one_tree(**settings) -> HistGradientBoostingRegressor:
    """One histogram tree that predicts the mean of each leaf, like a decision tree."""
    return HistGradientBoostingRegressor(
        max_iter=1, learning_rate=1.0, max_leaf_nodes=None, min_samples_leaf=1, early_stopping=False, **settings
    )


def main() -> None:
    repeats: int = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    s6e1: pd.DataFrame = pd.read_csv(ROOT / "tests/fixtures/s6e1/train.csv")
    codes: np.ndarray = OrdinalEncoder().fit_transform(s6e1[TEXT])
    x: np.ndarray = np.ascontiguousarray(np.hstack([codes, s6e1[NUMERIC].to_numpy(dtype=np.float64)]))
    y: np.ndarray = s6e1["exam_score"].to_numpy(dtype=np.float64)
    s6e2: pd.DataFrame = pd.read_csv(ROOT / "tests/fixtures/s6e2/train.csv")
    xc: np.ndarray = np.ascontiguousarray(s6e2.drop(columns=["id", "Heart Disease"]).to_numpy(dtype=np.float64))
    yc: np.ndarray = s6e2["Heart Disease"].to_numpy()

    fits = {
        "fit/DecisionTreeRegressor": (lambda: DecisionTreeRegressor(random_state=42), x, y),
        "fit/DecisionTreeRegressor/max_depth_8": (lambda: DecisionTreeRegressor(max_depth=8, random_state=42), x, y),
        "fit/HistGradientBoostingRegressor 1 tree/max_depth_8": (lambda: one_tree(max_depth=8), x, y),
        "fit/DecisionTreeClassifier": (lambda: DecisionTreeClassifier(random_state=42), xc, yc),
        "fit/DecisionTreeClassifier/max_depth_8": (lambda: DecisionTreeClassifier(max_depth=8, random_state=42), xc, yc),
    }
    work = {}
    fitted = {}
    for name, (make, features, target) in fits.items():
        model = make().fit(features, target)
        fitted[name] = model
        if hasattr(model, "tree_"):
            t = model.tree_
            size: str = f"nodes {t.node_count} leaves {t.n_leaves} depth {t.max_depth}"
        else:
            p = model._predictors[0][0]
            size = f"nodes {len(p.nodes)} leaves {p.get_n_leaf_nodes()} depth {p.get_max_depth()}"
        kind: str = "accuracy" if hasattr(model, "classes_") else "r2"
        print(f"{name}: {size} train {kind} {model.score(features, target):.6f}", file=sys.stderr)
        work[name] = lambda make=make, features=features, target=target: make().fit(features, target)
    work["predict/DecisionTreeRegressor"] = lambda: fitted["fit/DecisionTreeRegressor"].predict(x)
    work["predict/DecisionTreeClassifier"] = lambda: fitted["fit/DecisionTreeClassifier"].predict(xc)
    work["predict_proba/DecisionTreeClassifier"] = lambda: fitted["fit/DecisionTreeClassifier"].predict_proba(xc)

    print(f"S6E1 {x.shape[0]} x {x.shape[1]}, S6E2 {xc.shape[0]} x {xc.shape[1]}, median of {repeats} runs", file=sys.stderr)
    print(f"{'bench':<56} {'1 thread':>10} {'all cores':>10}", file=sys.stderr)
    for name, run in work.items():
        with threadpool_limits(limits=1):
            one: float = median_ms(run, repeats)
        many: float = median_ms(run, repeats)
        print(f"{name:<56} {one:>8.1f}ms {many:>8.1f}ms", file=sys.stderr)


if __name__ == "__main__":
    main()
