"""Percentile-bootstrap 95% CI over per-concept detection accuracies"""
import argparse

import numpy as np


def bootstrap_ci(accs, n_boot=10000, seed=0, ci=95.0):
    """Return (mean, ci_low, ci_high) of a percentile bootstrap over `accs`."""
    accs = np.asarray(accs, dtype=float)
    rng = np.random.default_rng(seed)

    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(accs, size=len(accs), replace=True)
        boot_means.append(np.mean(sample))

    half = (100.0 - ci) / 2
    ci_low, ci_high = np.percentile(boot_means, [half, 100.0 - half])
    return float(np.mean(accs)), float(ci_low), float(ci_high)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--accs', type=float, nargs='+', required=True,
                        help='Per-concept accuracies in [0, 1], e.g. python bootstrap_ci.py --accs 0.01 0.14 0.0 0.0 0.04')
    parser.add_argument('--n_boot', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    mean_acc, ci_low, ci_high = bootstrap_ci(args.accs, n_boot=args.n_boot, seed=args.seed)
    print(f"Mean accuracy: {mean_acc:.4f}")
    print(f"95% bootstrap CI: [{ci_low:.4f}, {ci_high:.4f}]")


if __name__ == '__main__':
    main()
