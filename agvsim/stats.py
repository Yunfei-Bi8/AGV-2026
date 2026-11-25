"""Statistics for paired Monte-Carlo comparisons."""
import numpy as np
from scipy import stats


def boot_ci(x, fun=np.mean, n=10000, alpha=0.05, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)])
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = fun(x[rng.integers(0, len(x), (n, len(x)))], axis=1)
    return float(fun(x)), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def paired_test(a, b):
    """Two-sided Wilcoxon signed-rank test of paired samples; returns p-value and median of the ratio a/b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 5 or np.allclose(a, b):
        return 1.0, float(np.median(a / b)) if len(a) else np.nan
    p = stats.wilcoxon(a, b, zero_method='wilcox', alternative='two-sided').pvalue
    return float(p), float(np.median(a / np.maximum(b, 1e-15)))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj
