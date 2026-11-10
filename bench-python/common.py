"""Timing and chart helpers shared by the scripts that mirror examples/*.rs.

Each script does what its Rust example does, the way a Python user would write it with pandas,
scikit-learn, seaborn and matplotlib. Import this module first: its clock starts before pandas
and the other libraries load, so the report shows their import time too.
"""

import time

START: float = time.perf_counter()

import sys  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # files only, like a Rust program: no window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT: Path = Path(__file__).resolve().parent.parent
# Our charts are 720 x 469 px and save as PNG at 2x: 1440 x 938 px.
SIZE: tuple[float, float] = (7.2, 4.69)
DPI: int = 200

# (group, name, start, seconds) for every timed stage.
TIMES: list[tuple[str, str, float, float]] = []


def out_dir(name: str) -> Path:
    """target/bench-python/{name}/plots, so the runs never touch the fixtures."""
    path: Path = ROOT / "target" / "bench-python" / name / "plots"
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def stage(group: str, name: str):
    """Times the body of the `with` block."""
    t: float = time.perf_counter()
    yield
    TIMES.append((group, name, t, time.perf_counter() - t))


@contextmanager
def chart(kind: str, title: str, out: Path, path: Path | None = None):
    """Times one chart: a new figure, the drawing, the title and the PNG (or `path`).

    Yields the axes. A figure-level chart (pairplot, missingno's matrix) makes its own figure,
    which is then the current one, so that one is saved.
    """
    t: float = time.perf_counter()
    fig, ax = plt.subplots(figsize=SIZE)
    yield ax
    current = plt.gcf()
    if current is fig:
        ax.set_title(title)
    else:
        current.suptitle(title)
    current.savefig(path or out / f"{title.replace('/', '')}.png", dpi=DPI)  # "mg/dl" is no folder
    plt.close("all")
    TIMES.append(("charts", f"{kind}: {title}", t, time.perf_counter() - t))


def top_with_other(counts: pd.Series, top: int) -> pd.Series:
    """The `top` biggest counts and one "Other" for the rest, like our pie charts."""
    if len(counts) <= top:
        return counts
    return pd.concat([counts.iloc[:top], pd.Series({"Other": counts.iloc[top:].sum()})])


def report(name: str) -> None:
    """Prints every stage, then the totals per group, to stderr."""
    end: float = time.perf_counter()
    imports: float = TIMES[0][2] - START if TIMES else 0.0
    out = sys.stderr
    print(f"\n== {name}: Python {sys.version.split()[0]}, pandas {pd.__version__}", file=out)
    print(f"{'imports':<60} {imports * 1000:>10.1f} ms", file=out)
    for group, stage_name, _, seconds in TIMES:
        print(f"{group + ' / ' + stage_name:<60.60} {seconds * 1000:>10.1f} ms", file=out)
    print(file=out)
    groups: dict[str, float] = {}
    for group, _, _, seconds in TIMES:
        groups[group] = groups.get(group, 0.0) + seconds
    for group, seconds in groups.items():
        print(f"{'total ' + group:<60} {seconds * 1000:>10.1f} ms", file=out)
    print(f"{'total without imports':<60} {(end - TIMES[0][2]) * 1000:>10.1f} ms", file=out)
    print(f"{'total with imports':<60} {(end - START) * 1000:>10.1f} ms", file=out)
