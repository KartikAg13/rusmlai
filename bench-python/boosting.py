"""scikit-learn's side of the `boost` group of `benches/models.rs`: python3 bench-python/boosting.py [repeats]

Like for like: the rows and columns of `trees.py`, the same settings and the same call (a whole
`fit`). `HistGradientBoosting*` runs its 100 rounds without early stopping, so both sides grow the
same number of trees; one row times the default, with early stopping. Each model prints its size
and its training score, so a faster fit of a smaller or worse model would show.

`GridSearchCV` runs 6 trees x 5 folds and the refit: on one thread with `n_jobs=1`, on all
cores with `n_jobs=-1` (one process per core, scikit-learn's default backend).

Each timing is the median of `repeats` runs (3 by default) after one warm-up run.
"""

import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from models import NUMERIC, ROOT, TEXT
from trees import median_ms


def leaves(model) -> int:
    return sum(p.get_n_leaf_nodes() for it in model._predictors for p in it)


def main() -> None:
    repeats: int = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    s6e1: pd.DataFrame = pd.read_csv(ROOT / "tests/fixtures/s6e1/train.csv")
    codes: np.ndarray = OrdinalEncoder().fit_transform(s6e1[TEXT])
    x: np.ndarray = np.ascontiguousarray(np.hstack([codes, s6e1[NUMERIC].to_numpy(dtype=np.float64)]))
    y: np.ndarray = s6e1["exam_score"].to_numpy(dtype=np.float64)
    # The text columns as categories ("from_dtype", scikit-learn's default), not codes.
    xcat: pd.DataFrame = s6e1[TEXT + NUMERIC].astype({name: "category" for name in TEXT})
    s6e2: pd.DataFrame = pd.read_csv(ROOT / "tests/fixtures/s6e2/train.csv")
    xc: np.ndarray = np.ascontiguousarray(s6e2.drop(columns=["id", "Heart Disease"]).to_numpy(dtype=np.float64))
    yc: np.ndarray = s6e2["Heart Disease"].to_numpy()
    grid = {"max_depth": [4, 8, 12], "min_samples_leaf": [1, 50]}

    # name: (make(n_jobs), features, target)
    fits = {
        "fit/GradientBoostingRegressor": (lambda jobs: HistGradientBoostingRegressor(early_stopping=False), x, y),
        "fit/GradientBoostingRegressor/categorical": (lambda jobs: HistGradientBoostingRegressor(early_stopping=False), xcat, y),
        "fit/GradientBoostingClassifier": (lambda jobs: HistGradientBoostingClassifier(early_stopping=False), xc, yc),
        "fit/GradientBoostingClassifier/early_stopping": (lambda jobs: HistGradientBoostingClassifier(random_state=42), xc, yc),
        "fit/AdaBoostClassifier": (lambda jobs: AdaBoostClassifier(random_state=42), xc, yc),
        "fit/GridSearchCV": (lambda jobs: GridSearchCV(DecisionTreeClassifier(random_state=42), grid, cv=5, n_jobs=jobs), xc, yc),
    }
    fitted = {}
    for name, (make, features, target) in fits.items():
        model = make(-1).fit(features, target)
        fitted[name] = model
        kind: str = "accuracy" if hasattr(model, "classes_") else "r2"
        if hasattr(model, "_predictors"):
            size = f"rounds {model.n_iter_} leaves {leaves(model)}"
        elif hasattr(model, "best_params_"):
            size = f"best {model.best_params_} mean {np.round(model.cv_results_['mean_test_score'], 6).tolist()}"
        else:
            size = f"stumps {len(model.estimators_)}"
        print(f"{name}: {size} train {kind} {model.score(features, target):.6f}", file=sys.stderr)

    print(f"S6E1 {x.shape[0]} x {x.shape[1]}, S6E2 {xc.shape[0]} x {xc.shape[1]}, median of {repeats} runs", file=sys.stderr)
    print(f"{'bench':<52} {'1 thread':>10} {'all cores':>10}", file=sys.stderr)
    for name, (make, features, target) in fits.items():
        with threadpool_limits(limits=1):
            one: float = median_ms(lambda: make(1).fit(features, target), repeats)
        many: float = median_ms(lambda: make(-1).fit(features, target), repeats)
        print(f"{name:<52} {one:>8.1f}ms {many:>8.1f}ms", file=sys.stderr)
    predicts = {
        "predict/GradientBoostingRegressor": (fitted["fit/GradientBoostingRegressor"], "predict", x),
        "predict_proba/GradientBoostingClassifier": (fitted["fit/GradientBoostingClassifier"], "predict_proba", xc),
    }
    for name, (model, method, features) in predicts.items():
        with threadpool_limits(limits=1):
            one = median_ms(lambda: getattr(model, method)(features), repeats)
        many = median_ms(lambda: getattr(model, method)(features), repeats)
        print(f"{name:<52} {one:>8.1f}ms {many:>8.1f}ms", file=sys.stderr)


if __name__ == "__main__":
    main()
