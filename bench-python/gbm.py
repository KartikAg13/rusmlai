"""XGBoost's and LightGBM's side of the `gbm` group of `benches/models.rs`: python3 bench-python/gbm.py [repeats]

Like for like with rusmlai's `XGB*` / `LGBM*`: each library's defaults (100 rounds), the same
matrices as `trees.py` / `boosting.py` (S6E1 with its 7 text columns as `OrdinalEncoder` codes,
then the 4 numbers; S6E2 with its 13 numbers), and S6E1 with the text columns as pandas
categories (`enable_categorical=True` for XGBoost, LightGBM's default for that dtype). Each
model prints its leaves and its training score, so a faster fit of a smaller or worse model
would show.

One thread is `n_jobs=1`, all cores `n_jobs=os.cpu_count()` (rusmlai's pool has one thread per
logical core too). Each timing is the median of `repeats` runs (3 by default) after a warm-up.
"""

import os
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor, __version__ as lgbm_version
from sklearn.preprocessing import OrdinalEncoder
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier, XGBRegressor, __version__ as xgb_version

from models import NUMERIC, ROOT, TEXT
from trees import median_ms

ALL: int = os.cpu_count() or 1


def leaves(model) -> int:
    if hasattr(model, "booster_"):
        return sum(tree["num_leaves"] for tree in model.booster_.dump_model()["tree_info"])
    frame: pd.DataFrame = model.get_booster().trees_to_dataframe()
    return int((frame["Feature"] == "Leaf").sum())


def main() -> None:
    repeats: int = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    s6e1: pd.DataFrame = pd.read_csv(ROOT / "tests/fixtures/s6e1/train.csv")
    codes: np.ndarray = OrdinalEncoder().fit_transform(s6e1[TEXT])
    x: np.ndarray = np.ascontiguousarray(np.hstack([codes, s6e1[NUMERIC].to_numpy(dtype=np.float64)]))
    y: np.ndarray = s6e1["exam_score"].to_numpy(dtype=np.float64)
    # The CSV's column order, as rusmlai reads it; pandas sorts the categories, as rusmlai does.
    xcat: pd.DataFrame = s6e1.drop(columns=["id", "exam_score"]).astype({name: "category" for name in TEXT})
    s6e2: pd.DataFrame = pd.read_csv(ROOT / "tests/fixtures/s6e2/train.csv")
    xc: np.ndarray = np.ascontiguousarray(s6e2.drop(columns=["id", "Heart Disease"]).to_numpy(dtype=np.float64))
    # Absence 0, Presence 1: rusmlai's sorted classes_.
    yc: np.ndarray = (s6e2["Heart Disease"] == "Presence").to_numpy(dtype=np.int64)

    # name: (make(n_jobs), features, target)
    fits = {
        "fit/XGBRegressor": (lambda jobs: XGBRegressor(n_jobs=jobs), x, y),
        "fit/XGBRegressor/categorical": (lambda jobs: XGBRegressor(n_jobs=jobs, enable_categorical=True), xcat, y),
        "fit/XGBClassifier": (lambda jobs: XGBClassifier(n_jobs=jobs), xc, yc),
        "fit/LGBMRegressor": (lambda jobs: LGBMRegressor(n_jobs=jobs, verbose=-1), x, y),
        "fit/LGBMRegressor/categorical": (lambda jobs: LGBMRegressor(n_jobs=jobs, verbose=-1), xcat, y),
        "fit/LGBMClassifier": (lambda jobs: LGBMClassifier(n_jobs=jobs, verbose=-1), xc, yc),
    }
    fitted = {}
    for name, (make, features, target) in fits.items():
        model = make(ALL).fit(features, target)
        fitted[name] = model
        kind: str = "accuracy" if hasattr(model, "classes_") else "r2"
        print(f"{name}: leaves {leaves(model)} train {kind} {model.score(features, target):.6f}", file=sys.stderr)

    print(
        f"XGBoost {xgb_version}, LightGBM {lgbm_version}; S6E1 {x.shape[0]} x {x.shape[1]}, "
        f"S6E2 {xc.shape[0]} x {xc.shape[1]}, {ALL} threads, median of {repeats} runs",
        file=sys.stderr,
    )
    print(f"{'bench':<44} {'1 thread':>10} {'all cores':>10}", file=sys.stderr)
    for name, (make, features, target) in fits.items():
        with threadpool_limits(limits=1):
            one: float = median_ms(lambda: make(1).fit(features, target), repeats)
        many: float = median_ms(lambda: make(ALL).fit(features, target), repeats)
        print(f"{name:<44} {one:>8.1f}ms {many:>8.1f}ms", file=sys.stderr)
    predicts = {
        "predict/XGBRegressor": (fitted["fit/XGBRegressor"], "predict", x),
        "predict_proba/XGBClassifier": (fitted["fit/XGBClassifier"], "predict_proba", xc),
        "predict/LGBMRegressor": (fitted["fit/LGBMRegressor"], "predict", x),
        "predict_proba/LGBMClassifier": (fitted["fit/LGBMClassifier"], "predict_proba", xc),
    }
    for name, (model, method, features) in predicts.items():
        model.set_params(n_jobs=1)
        with threadpool_limits(limits=1):
            one = median_ms(lambda: getattr(model, method)(features), repeats)
        model.set_params(n_jobs=ALL)
        many = median_ms(lambda: getattr(model, method)(features), repeats)
        print(f"{name:<44} {one:>8.1f}ms {many:>8.1f}ms", file=sys.stderr)


if __name__ == "__main__":
    main()
