"""CatBoost's side of `benches/catboost.rs`: python3 bench-python/catboosting.py [repeats]

The same rows (S6E1, S6E2; `id` ending in 0 or 5 held out), the same settings (200 rounds,
depth 6, CatBoost's defaults, text columns as `cat_features`). Each timing is the median of
`repeats` runs (3 by default) after a warm-up, on one thread (`thread_count=1`) and on all
cores (`thread_count=-1`). Prints the held-out scores of 5 seeds: rusmlai draws its bootstrap
and noise from its own generator, so its score should sit in CatBoost's seed range.
"""

import sys
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score

from models import ROOT

ROUNDS = 200


def split(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(ROOT / f"tests/fixtures/{name}/train.csv")
    held = raw["id"] % 5 == 0
    return raw[~held].drop(columns=["id"]), raw[held].drop(columns=["id"])


def median_s(work, repeats: int) -> float:
    work()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        work()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def main() -> None:
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    train1, test1 = split("s6e1")
    train2, test2 = split("s6e2")
    target1, target2 = "exam_score", "Heart Disease"
    x1, y1 = train1.drop(columns=[target1]), train1[target1]
    x2, y2 = train2.drop(columns=[target2]), train2[target2]
    cats = [c for c in x1 if not pd.api.types.is_numeric_dtype(x1[c])]
    kw = dict(iterations=ROUNDS, verbose=0, allow_writing_files=False)

    def regressor(threads: int, seed: int = 0, **more):
        return CatBoostRegressor(**kw, **more, thread_count=threads, random_seed=seed).fit(x1, y1, cat_features=cats)

    def classifier(threads: int, seed: int = 0):
        return CatBoostClassifier(**kw, thread_count=threads, random_seed=seed).fit(x2, y2)

    for seed in range(5):
        r, c = regressor(-1, seed), classifier(-1, seed)
        p = c.predict_proba(test2.drop(columns=[target2]))[:, list(c.classes_).index("Presence")]
        print(
            f"seed {seed}: CatBoostRegressor held-out r2 {r2_score(test1[target1], r.predict(test1.drop(columns=[target1]))):.6f} "
            f"rate {r.learning_rate_:.6f}, without combinations "
            f"{r2_score(test1[target1], regressor(-1, seed, max_ctr_complexity=1).predict(test1.drop(columns=[target1]))):.6f}; CatBoostClassifier accuracy "
            f"{accuracy_score(test2[target2], c.predict(test2.drop(columns=[target2]))):.6f} "
            f"roc_auc {roc_auc_score(test2[target2] == 'Presence', p):.6f} rate {c.learning_rate_:.6f}",
            file=sys.stderr,
        )
    r, c = regressor(-1), classifier(-1)
    works = {
        "fit/CatBoostRegressor": lambda t: regressor(t),
        # rusmlai builds no combinations of categories yet: CatBoost without them too.
        "fit/CatBoostRegressor (max_ctr_complexity=1)": lambda t: regressor(t, max_ctr_complexity=1),
        "fit/CatBoostClassifier": lambda t: classifier(t),
        "predict/CatBoostRegressor": lambda t: r.predict(test1.drop(columns=[target1]), thread_count=t),
        "predict_proba/CatBoostClassifier": lambda t: c.predict_proba(test2.drop(columns=[target2]), thread_count=t),
    }
    print(f"median of {repeats} runs", file=sys.stderr)
    for name, work in works.items():
        one = median_s(lambda: work(1), repeats)
        many = median_s(lambda: work(-1), repeats)
        print(f"{name:<48} {one * 1000:>10.1f}ms {many * 1000:>10.1f}ms", file=sys.stderr)


if __name__ == "__main__":
    main()
