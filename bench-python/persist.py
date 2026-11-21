"""joblib's side of `benches/persist.rs`: python3 bench-python/persist.py [repeats]

The same models on the same rows (S6E1, text as ordinal codes): scikit-learn's random forest
of 50 full trees and XGBoost (300 rounds, xgboost's own
sklearn wrapper), written with `joblib.dump` and read with `joblib.load`, the usual way. The
median of `repeats` runs (5 by default) after a warm-up.
"""

import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor

from models import ROOT, TEXT
from trees import median_ms


def main() -> None:
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    raw = pd.read_csv(ROOT / "tests/fixtures/s6e1/train.csv").drop(columns=["id"])
    x = pd.concat([pd.DataFrame(OrdinalEncoder().fit_transform(raw[TEXT]), columns=TEXT), raw.drop(columns=TEXT + ["exam_score"])], axis=1)
    y = raw["exam_score"]
    forest = RandomForestRegressor(50, n_jobs=-1, random_state=42).fit(x, y)
    boosted = XGBRegressor(n_estimators=300).fit(x, y)
    folder = ROOT / "target/tmp"
    folder.mkdir(parents=True, exist_ok=True)
    files = {"RandomForestRegressor": folder / "forest.joblib", "XGBRegressor": folder / "boosted.joblib"}
    joblib.dump(forest, files["RandomForestRegressor"])
    joblib.dump(boosted, files["XGBRegressor"])
    print(
        f"forest: {sum(e.tree_.node_count for e in forest.estimators_)} nodes, "
        f"{os.path.getsize(files['RandomForestRegressor']) // 1_000_000} MB; boosting: {os.path.getsize(files['XGBRegressor']) // 1_000_000} MB",
        file=sys.stderr,
    )
    for name, model in [("RandomForestRegressor", forest), ("XGBRegressor", boosted)]:
        print(f"persist/save/{name:<24} {median_ms(lambda: joblib.dump(model, files[name]), repeats):>10.1f}ms", file=sys.stderr)
        print(f"persist/load/{name:<24} {median_ms(lambda: joblib.load(files[name]), repeats):>10.1f}ms", file=sys.stderr)


if __name__ == "__main__":
    main()
