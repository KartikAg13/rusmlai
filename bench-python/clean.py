"""pandas' side of `benches/clean.rs`: python3 bench-python/clean.py [repeats]

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
    print(f"median of {repeats} runs", file=sys.stderr)
    for name, work in works.items():
        print(f"clean/{name:<24} {median_ms(work, repeats):>10.2f}ms", file=sys.stderr)


if __name__ == "__main__":
    main()
