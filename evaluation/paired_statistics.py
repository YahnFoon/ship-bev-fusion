"""Condition-level paired inference; no frame or seed pseudo-replication."""
import numpy as np
from scipy.stats import rankdata


def exact_wilcoxon(differences):
    d = np.round(np.asarray(differences, dtype=float), 12)
    if d.ndim != 1 or not len(d) or not np.isfinite(d).all():
        raise ValueError('Expected nonempty, finite paired differences')
    nonzero = d[d != 0]
    n = len(nonzero)
    if not n:
        return dict(statistic=0., pvalue=1., nonzero_pairs=0, zero_pairs=len(d), ties=False)

    ranks2 = np.rint(2 * rankdata(np.abs(nonzero), method='average')).astype(int)
    counts = [0] * (int(ranks2.sum()) + 1)
    counts[0] = 1
    end = 0
    for weight in ranks2:
        weight = int(weight)
        for i in range(end, -1, -1):
            counts[i + weight] += counts[i]
        end += weight
    observed = int(ranks2[nonzero > 0].sum())
    p = min(1., 2 * min(sum(counts[:observed + 1]), sum(counts[observed:])) / (2 ** n))
    return dict(statistic=min(observed, end - observed) / 2., pvalue=p,
                nonzero_pairs=n, zero_pairs=len(d)-n,
                ties=len(set(np.abs(nonzero))) < n)


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid p values')
    order = np.argsort(p, kind='stable')
    adjusted = np.minimum(1., np.maximum.accumulate((len(p)-np.arange(len(p))) * p[order]))
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def paired_bootstrap(differences, seed, resamples=100000):
    d = np.asarray(differences, dtype=float)
    if d.ndim != 1 or len(d) < 2 or not np.isfinite(d).all():
        raise ValueError('At least two finite matched conditions required')
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for start in range(0, resamples, 10000):
        stop = min(start + 10000, resamples)
        means[start:stop] = d[rng.integers(0, len(d), size=(stop-start, len(d)))].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [.025, .975]))


def match_conditions(ours, rgb, seeds, metric):
    """Inputs map seed -> condition -> metric; require exact condition identities."""
    if set(ours) != set(seeds) or set(rgb) != set(seeds):
        raise ValueError('All predefined training runs are required')
    conditions = set(ours[seeds[0]])
    for method in (ours, rgb):
        for seed in seeds:
            if set(method[seed]) != conditions:
                raise ValueError('Condition identities do not match across methods/runs')
    included, excluded = [], []
    for condition in sorted(conditions, key=lambda s: (len(s), s)):
        a = np.array([float(ours[s][condition][metric]) for s in seeds])
        b = np.array([float(rgb[s][condition][metric]) for s in seeds])
        if np.isfinite(a).all() and np.isfinite(b).all():
            included.append(dict(condition=condition, ours_mean=float(a.mean()), rgb_mean=float(b.mean()),
                                 difference=float(a.mean()-b.mean()),
                                 **{'ours_'+str(s):float(v) for s,v in zip(seeds,a)},
                                 **{'rgb_'+str(s):float(v) for s,v in zip(seeds,b)}))
        else:
            excluded.append(dict(condition=condition, reason='Nonfinite metric in at least one of the six runs',
                                 ours_nonfinite_seeds=[s for s,v in zip(seeds,a) if not np.isfinite(v)],
                                 rgb_nonfinite_seeds=[s for s,v in zip(seeds,b) if not np.isfinite(v)]))
    return included, excluded
