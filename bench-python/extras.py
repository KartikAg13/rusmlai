"""scikit-learn's and pandas' side of `benches/extras.rs`: python3 bench-python/extras.py [repeats]

Like for like: the same rows, columns, settings and calls as the Rust bench. Each timing is the
median of `repeats` runs (3 by default) after one warm-up run, on one thread
(`threadpool_limits(1)`, `n_jobs=1`) and on all cores (`n_jobs=-1`). Where scikit-learn has no
`n_jobs` (metrics, splitters, encoders), both columns are the same code; only BLAS threads differ.
"""

import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, StackingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import (
    KBinsDiscretizer,
    OneHotEncoder,
    OrdinalEncoder,
    QuantileTransformer,
    StandardScaler,
    TargetEncoder,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from models import NUMERIC, ROOT, TEXT
from trees import median_ms

HEART = "Heart Disease"


def main() -> None:
    repeats: int = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    s6e1: pd.DataFrame = pd.read_csv(ROOT / "tests/fixtures/s6e1/train.csv").drop(columns=["id"])
    s6e2: pd.DataFrame = pd.read_csv(ROOT / "tests/fixtures/s6e2/train.csv").drop(columns=["id"])
    x2: np.ndarray = StandardScaler().fit_transform(s6e2.drop(columns=[HEART]).to_numpy(dtype=np.float64))
    y2: np.ndarray = s6e2[HEART].to_numpy()
    design: np.ndarray = np.hstack([
        StandardScaler().fit_transform(s6e1[NUMERIC].to_numpy(dtype=np.float64)),
        OneHotEncoder(sparse_output=False).fit_transform(s6e1[TEXT]),
    ])
    y1: np.ndarray = s6e1["exam_score"].to_numpy(dtype=np.float64)
    assert design.shape == (630_000, 30)

    logistic = LogisticRegression(solver="newton-cholesky").fit(x2, y2)
    score: np.ndarray = logistic.predict_proba(x2)[:, 1]
    predicted: np.ndarray = logistic.predict(x2)
    print(f"roc_auc {roc_auc_score(y2, score):.6f}, balanced accuracy {balanced_accuracy_score(y2, predicted):.6f}", file=sys.stderr)

    kfold = KFold(5, shuffle=True, random_state=42)
    stratified = StratifiedKFold(5, shuffle=True, random_state=42)

    def stack(jobs: int) -> StackingClassifier:
        return StackingClassifier(
            [("logistic", LogisticRegression(solver="newton-cholesky")), ("tree", DecisionTreeClassifier(max_depth=8, random_state=42))],
            final_estimator=LogisticRegression(solver="newton-cholesky"),
            cv=5,
            n_jobs=jobs,
        )

    fitted = stack(-1).fit(x2, y2)
    print(f"stack train accuracy {fitted.score(x2, y2):.6f}", file=sys.stderr)

    codes: np.ndarray = np.hstack([OrdinalEncoder().fit_transform(s6e1[TEXT]), s6e1[NUMERIC].to_numpy(dtype=np.float64)])
    x2raw: np.ndarray = s6e2.drop(columns=[HEART]).to_numpy(dtype=np.float64)
    reg = ExtraTreesRegressor(n_estimators=10, n_jobs=-1, random_state=42).fit(codes, y1)
    clf = ExtraTreesClassifier(n_estimators=10, n_jobs=-1, random_state=42).fit(x2raw, y2)
    print(
        f"ExtraTreesRegressor: nodes {sum(e.tree_.node_count for e in reg.estimators_)} train r2 {reg.score(codes, y1):.6f}; "
        f"ExtraTreesClassifier: nodes {sum(e.tree_.node_count for e in clf.estimators_)} train accuracy {clf.score(x2raw, y2):.6f}",
        file=sys.stderr,
    )

    def pipe() -> Pipeline:
        prep = ColumnTransformer([("te", TargetEncoder(target_type="continuous", random_state=42), TEXT)], remainder="passthrough")
        return Pipeline([("prep", prep), ("model", Ridge())])

    x1 = s6e1.drop(columns=["exam_score"])
    print(f"Pipeline(TargetEncoder, Ridge): train r2 {pipe().fit(x1, y1).score(x1, y1):.6f}", file=sys.stderr)

    def count_encode() -> pd.DataFrame:
        return pd.DataFrame({c: s6e1[c].map(s6e1[c].value_counts()) for c in TEXT})

    # name -> work(jobs)
    works = {
        "metric/roc_auc_score": lambda jobs: roc_auc_score(y2, score),
        "metric/roc_curve": lambda jobs: roc_curve(y2, score, pos_label="Presence"),
        "metric/balanced_accuracy_score": lambda jobs: balanced_accuracy_score(y2, predicted),
        "metric/compute_sample_weight": lambda jobs: compute_sample_weight("balanced", y2),
        "split/KFold": lambda jobs: list(kfold.split(x2)),
        "split/StratifiedKFold": lambda jobs: list(stratified.split(x2, y2)),
        "oof/LogisticRegression": lambda jobs: cross_val_predict(
            LogisticRegression(solver="newton-cholesky"), x2, y2, cv=stratified, method="predict_proba", n_jobs=jobs),
        "fit/RidgeCV": lambda jobs: RidgeCV().fit(design, y1),
        "fit/StackingClassifier": lambda jobs: stack(jobs).fit(x2, y2),
        "fit_transform/TargetEncoder": lambda jobs: TargetEncoder(target_type="continuous", random_state=42).fit_transform(s6e1[TEXT], y1),
        "fit/Pipeline(TargetEncoder,Ridge)": lambda jobs: pipe().fit(x1, y1),
        "fit_transform/KBinsDiscretizer": lambda jobs: KBinsDiscretizer(random_state=42).fit_transform(s6e1[NUMERIC]),
        "fit_transform/QuantileTransformer": lambda jobs: QuantileTransformer(random_state=42).fit_transform(s6e1[NUMERIC]),
        "fit_transform/CountEncoder": lambda jobs: count_encode(),
        "fit/ExtraTreesRegressor": lambda jobs: ExtraTreesRegressor(n_estimators=10, n_jobs=jobs, random_state=42).fit(codes, y1),
        "fit/ExtraTreesClassifier": lambda jobs: ExtraTreesClassifier(n_estimators=10, n_jobs=jobs, random_state=42).fit(x2raw, y2),
    }
    only: str | None = sys.argv[2] if len(sys.argv) > 2 else None
    print(f"median of {repeats} runs", file=sys.stderr)
    print(f"{'bench':<44} {'1 thread':>10} {'all cores':>10}", file=sys.stderr)
    for name, work in works.items():
        if only and only not in name:
            continue
        with threadpool_limits(limits=1):
            one: float = median_ms(lambda: work(1), repeats)
        many: float = median_ms(lambda: work(-1), repeats)
        print(f"{name:<44} {one:>8.1f}ms {many:>8.1f}ms", file=sys.stderr)


if __name__ == "__main__":
    main()
