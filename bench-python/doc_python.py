"""Runs every ```python block of the doc comments in src/: python3 bench-python/doc_python.py

Each block runs on its own, from the repo root, with matplotlib's file-only backend. Prints one
line per failing block (file:line and the error) and exits 1 if any failed.
"""

import os
import re
import sys
import traceback
import warnings
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent.parent
FENCE = re.compile(r"^\s*//[/!] ?```(\w*)")


def blocks(path: Path):
    """(first line number, code) of each ```python block in the doc comments of `path`."""
    lines: list[str] = path.read_text().splitlines()
    i: int = 0
    while i < len(lines):
        m = FENCE.match(lines[i])
        if m and m.group(1) == "python":
            start: int = i + 1
            body: list[str] = []
            i += 1
            while i < len(lines) and not FENCE.match(lines[i]):
                body.append(re.sub(r"^\s*//[/!] ?", "", lines[i]))
                i += 1
            yield start, "\n".join(body)
        i += 1


def main() -> int:
    os.chdir(ROOT)
    os.environ["MPLBACKEND"] = "Agg"
    # `plt.show()` is the honest equivalent in the examples; with Agg it only warns.
    warnings.filterwarnings("ignore", message=".*non-interactive.*")
    failed: int = 0
    total: int = 0
    for path in sorted((ROOT / "src").rglob("*.rs")):
        for line, code in blocks(path):
            total += 1
            try:
                exec(compile(code, f"{path.relative_to(ROOT)}:{line}", "exec"), {})
            except Exception:
                failed += 1
                print(f"{path.relative_to(ROOT)}:{line}", file=sys.stderr)
                traceback.print_exc(limit=-1)
            # Each block draws on a figure of its own.
            if "matplotlib.pyplot" in sys.modules:
                sys.modules["matplotlib.pyplot"].close("all")
    print(f"{total - failed} of {total} python blocks passed", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
