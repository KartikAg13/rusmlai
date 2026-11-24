"""pandas' side of `benches/clean.rs` (cleaning and the `dataset/ops.rs` methods):
python3 bench-python/clean.py [repeats]

The same rows (S6E1, 630,000 rows, in memory) and calls. pandas runs each on one thread; the
timing is the median of `repeats` runs (7 by default) after a warm-up.
"""

import sys

import pandas as pd

from models import ROOT
from trees import median_ms


def main() -> None:
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    ds = pd.read_csv(ROOT / "tests/fixtures/s6e1/train.csv")
    print(
        f"query rows {len(ds[ds['study_hours'] > 5.0])}, drop_duplicates rows "
        f"{len(ds.drop_duplicates(subset=['course', 'gender']))}, value_counts {ds['course'].value_counts().tolist()}",
        file=sys.stderr,
    )
    works = {
        "dropna": lambda: ds.dropna(),
        "fillna": lambda: ds.fillna({"study_hours": 0.0}),
        "astype": lambda: ds.astype({"age": "float64"}),
        "rename": lambda: ds.rename(columns={"age": "years"}),
        "query": lambda: ds[ds["study_hours"] > 5.0],
        "drop_duplicates": lambda: ds.drop_duplicates(subset=["course", "gender"]),
        "sample": lambda: ds.sample(100_000, random_state=42),
        "sort_values": lambda: ds.sort_values("exam_score", ascending=False),
        "value_counts": lambda: ds["course"].value_counts(),
    }
    means = ds.groupby("course").agg({"exam_score": "mean"}).reset_index()
    number = ds.select_dtypes("number")
    works |= {
        "merge": lambda: ds.merge(means, on="course", how="left"),
        "concat": lambda: pd.concat([ds, ds]),
        "groupby": lambda: ds.groupby(["course", "gender"])
        .agg({"exam_score": "mean", "study_hours": "max", "id": "count"})
        .reset_index(),
        "pivot_table": lambda: ds.pivot_table(values="exam_score", index="course", columns="gender", aggfunc="mean"),
        "melt": lambda: ds.melt(id_vars=["id"], value_vars=["study_hours", "sleep_hours"]),
        "get_dummies": lambda: pd.get_dummies(ds, columns=["course"]),
        "isna": lambda: ds.isna(),
        "ffill": lambda: ds.ffill(),
        "sum": lambda: ds.sum(numeric_only=True),
        "mean": lambda: ds.mean(numeric_only=True),
        "median": lambda: ds.median(numeric_only=True),
        "std": lambda: ds.std(numeric_only=True),
        "quantile": lambda: ds.quantile(0.25, numeric_only=True),
        "count": lambda: ds.count(),
        "nunique": lambda: ds.nunique(),
        "corr": lambda: ds.corr(numeric_only=True),
        "unique": lambda: ds["course"].unique(),
        "nlargest": lambda: ds.nlargest(100, "exam_score"),
        "select_dtypes": lambda: ds.select_dtypes("number"),
        "assign": lambda: ds.assign(older=ds["age"] + 1),
        "replace": lambda: ds.replace({"course": {"b.sc": "bsc"}}),
        "clip": lambda: number.clip(0, 50),
        "round": lambda: ds.round(1),
        "shift": lambda: ds.shift(1),
        "diff": lambda: number.diff(),
    }
    print(f"median of {repeats} runs", file=sys.stderr)
    for name, work in works.items():
        print(f"clean/{name:<24} {median_ms(work, repeats):>10.2f}ms", file=sys.stderr)


if __name__ == "__main__":
    main()
