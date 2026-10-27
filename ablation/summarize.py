"""Paired summary of ablate.py output (the statistics of Table 4 in the technical write-up).

Usage:
    python ablation/summarize.py results/*.jsonl

For each variant, compares the noise-corrected final-layer MSE with the `base` run network by network and prints the
geometric-mean MSE ratio with a 95% confidence interval (normal approximation on the per-network log ratios), the number
of networks whose MSE rose, and the mean relative change of the billed FLOPs.
"""
import sys, json, math, collections
import numpy as np

R = collections.defaultdict(dict)
for f in sys.argv[1:]:
    for line in open(f):
        r = json.loads(line)
        if 'mse_nc' in r:
            R[r['variant']][r['net']] = r
base = R.get('base')
if not base:
    sys.exit('no records of the base variant')
nb = sorted(base)
print('base: %d networks, mean noise-corrected MSE %.4e' % (len(nb), np.mean([base[n]['mse_nc'] for n in nb])))
print('%-20s %4s %10s %20s %7s %9s' % ('variant', 'n', 'MSE ratio', '95% CI', 'worse', 'dFLOPs'))
for v in sorted(R):
    nets = sorted(n for n in R[v] if n in base)
    if v == 'base' or len(nets) < 2:
        continue
    lr = np.array([math.log(R[v][n]['mse_nc'] / base[n]['mse_nc']) for n in nets])
    fr = np.array([R[v][n]['flops'] / base[n]['flops'] for n in nets])
    g, se = lr.mean(), lr.std(ddof=1) / math.sqrt(len(lr))
    print('%-20s %4d %10.3f   [%7.3f, %7.3f] %3d/%-3d %+8.2f%%' % (v, len(nets), math.exp(g), math.exp(g - 1.96 * se),
          math.exp(g + 1.96 * se), int((lr > 0).sum()), len(nets), 100 * (fr.mean() - 1)))
