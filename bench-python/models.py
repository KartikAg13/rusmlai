"""Fit times of the scikit-learn models that rusmlai plans (docs/models.md), on the Kaggle files.

These are the numbers a rusmlai model must beat. Each model runs twice: on one thread
(BLAS, OpenMP and joblib all limited to 1) and on every core (`n_jobs=-1` where the model takes
it). Regressors use S6E1 (630,000 rows, 29 columns after the one-hot step), classifiers and the
unsupervised models use S6E2 (630,000 rows, 13 numeric columns).

    python3 bench-python/models.py                 # every model, both thread modes
    python3 bench-python/models.py --only Ridge    # the models whose name contains "Ridge"
    python3 bench-python/models.py --dump DIR      # also write the matrices as raw files

`--dump` writes little-endian, row-major files for prototypes in other languages:
`s6e1_x.f64` / `s6e1_y.f64`, `s6e2_x.f64` / `s6e2_y.f64` (0 or 1), and `s6e1_bins.u8` (the 11
raw features in scikit-learn's own 255 bins) with `shapes.txt`.
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from threadpoolctl import threadpool_limits

ROOT: Path = Path(__file__).resolve().parent.parent
NUMERIC: list[str] = ["age", "study_hours", "class_attendance", "sleep_hours"]
TEXT: list[str] = [
    "gender",
    "course",
    "internet_access",
    "sleep_quality",
    "study_method",
    "facility_rating",
    "exam_difficulty",
]
QUERY: int = 10_000  # rows that kNN predicts


def load() -> dict[str, np.ndarray]:
    """The design matrices, made the way `examples/s6e1.rs` and `s6e2.rs` make them."""
    s6e1: pd.DataFrame = pd.read_csv(ROOT / "tests/fixtures/s6e1/train.csv")
    prep = ColumnTransformer(
        [("num", StandardScaler(), NUMERIC), ("cat", OneHotEncoder(sparse_output=False), TEXT)]
    )
    codes: np.ndarray = OrdinalEncoder().fit_transform(s6e1[TEXT])
    raw: np.ndarray = np.hstack([s6e1[NUMERIC].to_numpy(dtype=np.float64), codes])
    s6e2: pd.DataFrame = pd.read_csv(ROOT / "tests/fixtures/s6e2/train.csv")
    features: pd.DataFrame = s6e2.drop(columns=["id", "Heart Disease"])
    return {
        "s6e1_x": np.ascontiguousarray(prep.fit_transform(s6e1)),
        "s6e1_y": s6e1["exam_score"].to_numpy(dtype=np.float64),
        "s6e1_raw": np.ascontiguousarray(raw),
        "s6e2_x": np.ascontiguousarray(StandardScaler().fit_transform(features)),
        "s6e2_y": (s6e2["Heart Disease"] == "Presence").to_numpy(dtype=np.float64),
    }


def models(jobs: int) -> list[tuple[str, str, object, str]]:
    """(name, data, estimator, note). `jobs` is 1 or -1 for the models that take `n_jobs`."""
    return [
        ("LinearRegression", "s6e1", LinearRegression(), ""),
        ("Ridge", "s6e1", Ridge(), ""),
        ("Lasso", "s6e1", Lasso(), ""),
        ("ElasticNet", "s6e1", ElasticNet(), ""),
        ("DecisionTreeRegressor", "s6e1", DecisionTreeRegressor(random_state=42), "full depth"),
        (
            "RandomForestRegressor",
            "s6e1",
            RandomForestRegressor(n_estimators=10, n_jobs=jobs, random_state=42),
            "10 trees, not 100",
        ),
        (
            "HistGradientBoostingRegressor",
            "s6e1",
            HistGradientBoostingRegressor(early_stopping=False),
            "100 iterations, no early stop",
        ),
        (
            "HistGradientBoostingRegressor raw",
            "s6e1_raw",
            HistGradientBoostingRegressor(early_stopping=False),
            "11 raw features, as the Rust prototype",
        ),
        (
            "GradientBoostingRegressor",
            "s6e1",
            GradientBoostingRegressor(n_estimators=10, random_state=42),
            "10 iterations, not 100",
        ),
        ("KNeighborsRegressor", "s6e1", KNeighborsRegressor(n_jobs=jobs), f"fit + predict {QUERY}"),
        ("MLPRegressor", "s6e1", MLPRegressor(max_iter=3, random_state=42), "3 epochs, not 200"),
        ("LogisticRegression lbfgs", "s6e2", LogisticRegression(), ""),
        (
            "LogisticRegression newton-cholesky",
            "s6e2",
            LogisticRegression(solver="newton-cholesky"),
            "",
        ),
        ("GaussianNB", "s6e2", GaussianNB(), ""),
        ("LinearDiscriminantAnalysis", "s6e2", LinearDiscriminantAnalysis(), ""),
        ("LinearSVC", "s6e2", LinearSVC(random_state=42), ""),
        ("DecisionTreeClassifier", "s6e2", DecisionTreeClassifier(random_state=42), "full depth"),
        (
            "RandomForestClassifier",
            "s6e2",
            RandomForestClassifier(n_estimators=10, n_jobs=jobs, random_state=42),
            "10 trees, not 100",
        ),
        (
            "HistGradientBoostingClassifier",
            "s6e2",
            HistGradientBoostingClassifier(early_stopping=False),
            "100 iterations, no early stop",
        ),
        (
            "AdaBoostClassifier",
            "s6e2",
            AdaBoostClassifier(n_estimators=10, random_state=42),
            "10 stumps, not 50",
        ),
        ("KNeighborsClassifier", "s6e2", KNeighborsClassifier(n_jobs=jobs), f"fit + predict {QUERY}"),
        ("MLPClassifier", "s6e2", MLPClassifier(max_iter=3, random_state=42), "3 epochs, not 200"),
        ("KMeans", "s6e2", KMeans(n_clusters=8, random_state=42), "k = 8"),
        ("PCA", "s6e2", PCA(n_components=5), ""),
        ("TruncatedSVD", "s6e2", TruncatedSVD(n_components=5, random_state=42), ""),
        ("DBSCAN", "s6e2[:20000]", DBSCAN(eps=1.5, n_jobs=jobs), "20,000 rows"),
        (
            "AgglomerativeClustering",
            "s6e2[:10000]",
            AgglomerativeClustering(n_clusters=8),
            "10,000 rows",
        ),
    ]


def run(name: str, data: str, model: object, arrays: dict[str, np.ndarray]) -> float:
    """Seconds for one fit (plus the prediction for kNN, whose fit only stores the rows)."""
    key, _, cut = data.partition("[:")
    rows: int | None = int(cut.rstrip("]")) if cut else None
    x: np.ndarray = arrays[key if key.endswith("_raw") else f"{key}_x"][:rows]
    y: np.ndarray = arrays[f"{key.removesuffix('_raw')}_y"][:rows]
    t: float = time.perf_counter()
    model.fit(x, y)
    if name.startswith("KNeighbors"):
        model.predict(x[:QUERY])
    return time.perf_counter() - t


def dump(arrays: dict[str, np.ndarray], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    shapes: list[str] = []
    for key in ["s6e1_x", "s6e1_y", "s6e2_x", "s6e2_y"]:
        arrays[key].astype("<f8").tofile(out / f"{key}.f64")
        shapes.append(f"{key} {' '.join(map(str, arrays[key].shape))}")
    t: float = time.perf_counter()
    bins: np.ndarray = _BinMapper(random_state=42).fit_transform(arrays["s6e1_raw"])
    shapes.append(f"binning_seconds {time.perf_counter() - t:.4f}")
    np.ascontiguousarray(bins).tofile(out / "s6e1_bins.u8")
    shapes.append(f"s6e1_bins {' '.join(map(str, bins.shape))}")
    (out / "shapes.txt").write_text("\n".join(shapes) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="run the models whose name contains this")
    parser.add_argument("--dump", type=Path, help="write the matrices as raw files here")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")  # the short MLP runs warn that they did not converge

    arrays: dict[str, np.ndarray] = load()
    if args.dump:
        dump(arrays, args.dump)
    print(f"scikit-learn {sklearn.__version__}, numpy {np.__version__}", file=sys.stderr)
    print(f"{'model':<36} {'1 thread':>10} {'all cores':>10}  note", file=sys.stderr)
    one: list[tuple[str, str, object, str]] = models(1)
    for (name, data, model, note), (_, _, parallel, _) in zip(one, models(-1)):
        if args.only not in name:
            continue
        with threadpool_limits(limits=1):
            single: float = run(name, data, model, arrays)
        many: float = run(name, data, parallel, arrays)
        print(f"{name:<36} {single:>9.3f}s {many:>9.3f}s  {note}", file=sys.stderr)


if __name__ == "__main__":
    main()
