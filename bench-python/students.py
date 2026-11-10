"""examples/students.rs in pandas and scikit-learn: python3 bench-python/students.py [file.csv]"""

import common  # first: starts the clock before the imports
from common import report, stage

import sys

import pandas as pd
from sklearn.metrics import mutual_info_score

BINS: int = 256  # information_gain puts a numeric column into 256 equal-width bins

path: str = sys.argv[1] if len(sys.argv) > 1 else str(common.ROOT / "tests/fixtures/data/students.csv")

with stage("load", "read_csv"):
    ds = pd.read_csv(path)

with stage("tables", "shape, head, info, describe"):
    print(f"shape: {ds.shape}")
    print(ds.head(5))
    ds.info()
    print(ds.describe())

columns: list[str] = list(ds.columns)
with stage("tables", "drop, quartiles"):
    rest = ds.drop(columns=[columns[0]])  # for example an id column
    print(f"without `{columns[0]}`: {rest.shape}")
    second = ds[columns[1]]
    if pd.api.types.is_numeric_dtype(second):
        q = second.quantile([0.25, 0.5, 0.75])
        print(f"{columns[1]}: median {q[0.5]:.2f}, q25 {q[0.25]:.2f}, q75 {q[0.75]:.2f}")
    else:
        print(f"column '{columns[1]}' is not numeric")


def key(column: pd.Series) -> pd.Series:
    """Text as it is, a number as its bin, like Dataset::information_gain."""
    if pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column):
        finite = column.where(column.abs() != float("inf"))
        return pd.cut(finite, BINS, labels=False)
    return column


with stage("information_gain", f"mutual information with `{columns[-1]}`"):
    target: str = columns[-1]
    y = key(rest[target])
    gains: list[tuple[str, float]] = []
    for column in rest.columns:
        if column == target:
            continue
        pair = pd.DataFrame({"x": key(rest[column]), "y": y}).dropna()
        gains.append((column, max(mutual_info_score(pair["y"], pair["x"]), 0.0)))
    gains.sort(key=lambda g: g[1], reverse=True)
    for column, gain in gains[:5]:
        print(f"information gain with `{target}`: {column} {gain:.3f}")

report("students")
