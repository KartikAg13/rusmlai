"""Python's side of `benches/hill.rs`: python3 bench-python/hill.py [repeats]

Chris Deotte's hill climbing (Kaggle S6E3, first place) on numpy instead of CuPy: every
candidate blend of a step at once, its AUC from ranks (`multiple_roc_auc_scores`, his
vectorized version), or its RMSE. 630,000 rows and 5 models from the same formulas as the Rust
side. numpy's argsort and sums use one thread whatever the setting, so one timing is given.
"""

import sys
import time

import numpy as np

ROWS = 630_000


def models():
    i = np.arange(ROWS, dtype=float)
    y = ((np.sin(i * 0.37) + np.cos(i * 0.011)) > 0.3).astype(np.int64)
    m = lambda a, b: y * a + (np.sin(i * b) * 0.5 + 0.5) * (1 - a)
    return np.column_stack([m(0.3, 1.7), m(0.25, 2.3), m(0.2, 0.9), m(0.15, 3.1), m(0.1, 0.7)]), y


def multiple_roc_auc_scores(y, preds):
    n_pos = y.sum()
    n_neg = y.size - n_pos
    ranks = np.argsort(np.argsort(preds, axis=0), axis=0) + 1
    return (ranks[y == 1, :].sum(axis=0) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def rmse_scores(y, preds):
    return -np.sqrt(((preds - y[:, None]) ** 2).mean(axis=0))


def climb(x, y, scores, tol=1e-5):
    ww = np.array([-0.5 + j * 0.01 for j in range(101)])
    singles = scores(y, x)
    first = int(np.argmax(singles))
    blend, best = x[:, first].copy(), float(singles[first])
    steps = 1
    while True:
        base = blend[:, None] * (1 - ww)[None, :]
        top = (-np.inf, None, None)
        for k in range(x.shape[1]):
            cand = base + x[:, k][:, None] * ww[None, :]
            s = scores(y, cand)
            j = int(np.argmax(s))
            if s[j] > top[0]:
                top = (float(s[j]), k, cand[:, j].copy())
        if top[0] - best < tol:
            return best, steps
        best, blend, steps = top[0], top[2], steps + 1


def main():
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    x, y = models()
    for name, scores in [("roc_auc", multiple_roc_auc_scores), ("neg_root_mean_squared_error", rmse_scores)]:
        print(f"{name}: score, steps {climb(x, y, scores)}", file=sys.stderr)
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            climb(x, y, scores)
            times.append(time.perf_counter() - start)
        print(f"fit/HillClimbing/{name:<32} {np.median(times) * 1000:>10.1f}ms", file=sys.stderr)


if __name__ == "__main__":
    main()
