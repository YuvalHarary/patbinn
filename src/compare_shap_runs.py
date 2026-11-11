#!/usr/bin/env python3
"""
Agreement between independently seeded Shapley attribution runs.

src/explain_router.py draws its explained reads, its background reads and its
permutations from --seed, so two runs with different seeds share nothing but
the router and the held-out split. This script reports how well their
attributions agree: if the ranking of k-mers were an artifact of Monte Carlo
noise or of the particular reads explained, it would not reproduce.

Reported, for every pair of runs:
  * Spearman correlation of the global attribution (mean |phi| over families)
  * Spearman correlation of each family's attribution, and its mean
  * overlap of each family's top-N k-mers, and its mean

No model and no torch are needed; the inputs are the matrix CSVs.

    python3 src/explain_router.py --seed 0 --matrix_csv results/router_shap_matrix.csv
    python3 src/explain_router.py --seed 1 --matrix_csv results/router_shap_matrix_seed1.csv \
        --out_csv /dev/null --mi_csv /dev/null --stats_csv /dev/null
    python3 src/compare_shap_runs.py results/router_shap_matrix.csv results/router_shap_matrix_seed1.csv
"""

import argparse
import csv
import itertools


def read_matrix(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    families = [c for c in rows[0] if c != "kmer"]
    return families, {r["kmer"]: [float(r[f]) for f in families] for r in rows}


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):                      # average rank over ties
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for t in range(i, j + 1):
                r[order[t]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("matrices", nargs="+", help="two or more router_shap_matrix CSVs")
    p.add_argument("--top_n", type=int, default=20)
    p.add_argument("--out_csv", default=None)
    args = p.parse_args()
    if len(args.matrices) < 2:
        raise SystemExit("need at least two runs to compare")

    runs = [read_matrix(m) for m in args.matrices]
    families = runs[0][0]
    kmers = sorted(runs[0][1])
    out = []
    for (ia, (_, a)), (ib, (_, b)) in itertools.combinations(enumerate(runs), 2):
        ga = [sum(a[km]) / len(families) for km in kmers]
        gb = [sum(b[km]) / len(families) for km in kmers]
        rho_global = spearman(ga, gb)
        print(f"\n{args.matrices[ia]}  vs  {args.matrices[ib]}")
        print(f"  global attribution, Spearman over {len(kmers)} k-mers: {rho_global:.3f}")
        print(f"  {'family':<20}{'Spearman':>10}{'top-' + str(args.top_n) + ' shared':>16}")
        rhos, overlaps = [], []
        for fi, fam in enumerate(families):
            xa = [a[km][fi] for km in kmers]
            xb = [b[km][fi] for km in kmers]
            rho = spearman(xa, xb)
            top_a = set(sorted(kmers, key=lambda km: -a[km][fi])[: args.top_n])
            top_b = set(sorted(kmers, key=lambda km: -b[km][fi])[: args.top_n])
            shared = len(top_a & top_b)
            rhos.append(rho); overlaps.append(shared)
            print(f"  {fam:<20}{rho:10.3f}{shared:16d}")
            out.append({"run_a": args.matrices[ia], "run_b": args.matrices[ib], "family": fam,
                        "spearman": f"{rho:.4f}", f"top{args.top_n}_shared": shared})
        print(f"  {'mean':<20}{sum(rhos) / len(rhos):10.3f}{sum(overlaps) / len(overlaps):16.1f}")
        out.append({"run_a": args.matrices[ia], "run_b": args.matrices[ib], "family": "global",
                    "spearman": f"{rho_global:.4f}", f"top{args.top_n}_shared": ""})

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            w.writeheader(); w.writerows(out)
        print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
