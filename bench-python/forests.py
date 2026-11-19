"""scikit-learn's side of the `forest` group of `benches/models.rs`: python3 bench-python/forests.py [repeats]

Like for like: the same rows and columns as `trees.py`, the same settings (10 full trees, where
scikit-learn's default is 100, so that this script ends), and the same call (a whole `fit`). Each
forest prints its size and scores, so a faster fit of a smaller or worse forest would show.

`RandomForest*` and `Bagging*` grow scikit-learn's exact trees, one tree per thread with
`n_jobs=-1`. Each timing is the median of `repeats` runs (3 by default) after one warm-up run,
on one thread and on all cores.
"""

import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier, BaggingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from models import NUMERIC, ROOT, TEXT
from trees import median_ms


def main() -> None:
    repeats: int = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    s6e1: pd.DataFrame = pd.read_csv(ROOT / "tests/fixtures/s6e1/train.csv")
    codes: np.ndarray = OrdinalEncoder().fit_transform(s6e1[TEXT])
    x: np.ndarray = np.ascontiguousarray(np.hstack([codes, s6e1[NUMERIC].to_numpy(dtype=np.float64)]))
    y: np.ndarray = s6e1["exam_score"].to_numpy(dtype=np.float64)
    s6e2: pd.DataFrame = pd.read_csv(ROOT / "tests/fixtures/s6e2/train.csv")
    xc: np.ndarray = np.ascontiguousarray(s6e2.drop(columns=["id", "Heart Disease"]).to_numpy(dtype=np.float64))
    yc: np.ndarray = s6e2["Heart Disease"].to_numpy()

    # (make(n_jobs), features, target): `n_jobs` is 1 or -1.
    fits = {
        "fit/RandomForestRegressor": (
            lambda jobs, **kw: RandomForestRegressor(n_estimators=10, n_jobs=jobs, random_state=42, **kw), x, y),
        "fit/RandomForestClassifier": (
            lambda jobs, **kw: RandomForestClassifier(n_estimators=10, n_jobs=jobs, random_state=42, **kw), xc, yc),
        "fit/RandomForestClassifier/balanced": (
            lambda jobs, **kw: RandomForestClassifier(n_estimators=10, n_jobs=jobs, random_state=42, class_weight="balanced", **kw), xc, yc),
        "fit/BaggingClassifier": (
            lambda jobs, **kw: BaggingClassifier(DecisionTreeClassifier(), n_jobs=jobs, random_state=42, **kw), xc, yc),
        "fit/BaggingRegressor": (
            lambda jobs, **kw: BaggingRegressor(DecisionTreeRegressor(), n_jobs=jobs, random_state=42, **kw), x, y),
    }
    fitted = {}
    for name, (make, features, target) in fits.items():
        model = make(-1, oob_score=True).fit(features, target)
        fitted[name] = model
        nodes: int = sum(tree.tree_.node_count for tree in model.estimators_)
        kind: str = "accuracy" if hasattr(model, "classes_") else "r2"
        print(
            f"{name}: nodes {nodes} train {kind} {model.score(features, target):.6f} "
            f"oob {kind} {model.oob_score_:.6f}",
            file=sys.stderr,
        )

    print(f"S6E1 {x.shape[0]} x {x.shape[1]}, S6E2 {xc.shape[0]} x {xc.shape[1]}, median of {repeats} runs", file=sys.stderr)
    print(f"{'bench':<44} {'1 thread':>10} {'all cores':>10}", file=sys.stderr)
    for name, (make, features, target) in fits.items():
        with threadpool_limits(limits=1):
            one: float = median_ms(lambda: make(1).fit(features, target), repeats)
        many: float = median_ms(lambda: make(-1).fit(features, target), repeats)
        print(f"{name:<44} {one:>8.1f}ms {many:>8.1f}ms", file=sys.stderr)
    predicts = {
        "predict/RandomForestRegressor": (fitted["fit/RandomForestRegressor"], "predict", x),
        "predict/RandomForestClassifier": (fitted["fit/RandomForestClassifier"], "predict", xc),
        "predict_proba/RandomForestClassifier": (fitted["fit/RandomForestClassifier"], "predict_proba", xc),
    }
    for name, (model, method, features) in predicts.items():
        with threadpool_limits(limits=1):
            model.set_params(n_jobs=1)
            one = median_ms(lambda: getattr(model, method)(features), repeats)
        model.set_params(n_jobs=-1)
        many = median_ms(lambda: getattr(model, method)(features), repeats)
        print(f"{name:<44} {one:>8.1f}ms {many:>8.1f}ms", file=sys.stderr)


if __name__ == "__main__":
    main()
