"""scikit-learn's side of `benches/models.rs`: python3 bench-python/linear.py [repeats]

The same timings on the same matrices (`models.load()`): on S6E1, LinearRegression fit, Ridge
fit, LinearRegression predict, and a Pipeline fit from the loaded CSV; on S6E2, LogisticRegression
fit with the default L-BFGS and with Newton (rusmlai's solver), and predict_proba;
RidgeClassifier fit and predict, and LogisticRegressionCV fit (10 Cs, 5 folds; all cores
also run the folds as `n_jobs=-1` processes). Each one runs
on one thread and on all cores, and prints the median of `repeats` runs (7 by default) after one
warm-up run, so a single slow run does not decide the number.
"""

import statistics
import sys
import time

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV, Ridge, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from models import NUMERIC, ROOT, TEXT, load


def median_ms(work, repeats: int) -> float:
    work()  # warm-up
    times: list[float] = []
    for _ in range(repeats):
        t: float = time.perf_counter()
        work()
        times.append(time.perf_counter() - t)
    return statistics.median(times) * 1000


def main() -> None:
    repeats: int = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    arrays = load()
    x, y = arrays["s6e1_x"], arrays["s6e1_y"]
    raw: pd.DataFrame = pd.read_csv(ROOT / "tests/fixtures/s6e1/train.csv").drop(columns=["id"])
    raw_x, raw_y = raw.drop(columns=["exam_score"]), raw["exam_score"]
    fitted = LinearRegression().fit(x, y)
    xc, yc = arrays["s6e2_x"], arrays["s6e2_y"]
    newton = LogisticRegression(solver="newton-cholesky").fit(xc, yc)
    ridge = RidgeClassifier().fit(xc, yc)
    # The defaults, spelled out: these only silence scikit-learn 1.9's FutureWarnings.
    cv_kw = dict(scoring="accuracy", l1_ratios=(0.0,), use_legacy_attributes=False)
    cv = LogisticRegressionCV(**cv_kw).fit(xc, yc)
    print(
        f"RidgeClassifier: train accuracy {ridge.score(xc, yc):.6f}; LogisticRegressionCV: "
        f"C_ {cv.C_} train accuracy {cv.score(xc, yc):.6f}",
        file=sys.stderr,
    )

    def pipeline() -> Pipeline:
        prep = ColumnTransformer(
            [("num", StandardScaler(), NUMERIC), ("cat", OneHotEncoder(sparse_output=False), TEXT)]
        )
        return Pipeline([("prep", prep), ("model", LinearRegression())])

    work = {
        "fit/LinearRegression": lambda: LinearRegression().fit(x, y),
        "fit/Ridge": lambda: Ridge().fit(x, y),
        "predict/LinearRegression": lambda: fitted.predict(x),
        "pipeline_fit/LinearRegression": lambda: pipeline().fit(raw_x, raw_y),
        "fit/LogisticRegression lbfgs": lambda: LogisticRegression().fit(xc, yc),
        "fit/LogisticRegression newton": lambda: LogisticRegression(solver="newton-cholesky").fit(xc, yc),
        "predict_proba/LogisticRegression": lambda: newton.predict_proba(xc),
        "fit/RidgeClassifier": lambda: RidgeClassifier().fit(xc, yc),
        "predict/RidgeClassifier": lambda: ridge.predict(xc),
        "fit/LogisticRegressionCV lbfgs": lambda jobs=None: LogisticRegressionCV(**cv_kw, n_jobs=jobs).fit(xc, yc),
        "fit/LogisticRegressionCV newton": lambda jobs=None: LogisticRegressionCV(**cv_kw, solver="newton-cholesky", n_jobs=jobs).fit(xc, yc),
    }
    print(f"S6E1 {x.shape[0]} x {x.shape[1]}, S6E2 {xc.shape[0]} x {xc.shape[1]}, median of {repeats} runs", file=sys.stderr)
    print(f"{'bench':<40} {'1 thread':>10} {'all cores':>10}", file=sys.stderr)
    for name, run in work.items():
        with threadpool_limits(limits=1):
            one: float = median_ms(run, repeats)
        # A CV runs its folds in parallel only with n_jobs.
        many: float = median_ms((lambda: run(-1)) if "CV" in name else run, repeats)
        print(f"{name:<40} {one:>8.1f}ms {many:>8.1f}ms", file=sys.stderr)


if __name__ == "__main__":
    main()
