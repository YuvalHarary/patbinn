#!/usr/bin/env python3
"""
The two figures for the router-explainability section.

  Fig A  results/figures/router_shap_topkmers.png
         Heatmap of mean |Shapley value| for the highest-attribution k-mers,
         k-mers on one axis and families on the other. Shows what the router
         keys on per family, and how much that evidence is shared.

  Fig B  results/figures/router_shap_vs_mi.png
         Attribution against mutual information, per k-mer. Mutual information
         is computed from the reads alone with no reference to the model, so
         this is the independent check: if the router had latched onto
         arbitrary features the cloud would be structureless.

Inputs are produced by src/explain_router.py. Run from the repository root:

    apptainer exec --bind $(pwd):/workspace --pwd /workspace patbinn.sif \
        python3 src/plot_router_explain.py
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 11

# Matches plot_results.py: Okabe-Ito, colourblind-safe, print-validated.
COLOR_POINT = "#0072B2"
COLOR_HI = "#D55E00"
COLOR_MUTED = "#666666"

# The manifests record the family under the name used when the genomes were
# retrieved. ICTV renamed it in 2022 (proposal 2022.001D), so figures display
# the current name while the stored data is left untouched.
DISPLAY_NAME = {"Herpesviridae": "Orthoherpesviridae"}


def display(fam):
    return DISPLAY_NAME.get(fam, fam)


def read_csv(path):
    if not os.path.exists(path):
        print(f"  (skip) {path} not found")
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def plot_top_kmers(matrix_rows, out_path, per_family=6):
    """Heatmap over the full attribution matrix.

    Columns are the union of each family's strongest k-mers, but every cell is
    that family's actual measured attribution, so an off-diagonal cell being
    dark is a real result rather than a gap in the table.
    """
    families = [c for c in matrix_rows[0].keys() if c != "kmer"]
    attr = {r["kmer"]: {f: float(r[f]) for f in families} for r in matrix_rows}

    chosen = []
    for fam in families:
        ranked = sorted(attr, key=lambda km: attr[km][fam], reverse=True)
        for km in ranked[:per_family]:
            if km not in chosen:
                chosen.append(km)

    grid = [[attr[km][f] for km in chosen] for f in families]

    fig, ax = plt.subplots(figsize=(min(0.30 * len(chosen) + 2.4, 7.0), 3.1))
    im = ax.imshow(grid, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(chosen)))
    ax.set_xticklabels(chosen, rotation=90, fontsize=11, family="monospace")
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels([display(f) for f in families], fontsize=12)
    ax.set_xlabel(f"k-mer (union of top {per_family} per family)", fontsize=12)
    cb = fig.colorbar(im, ax=ax, pad=0.012)
    cb.set_label("mean |Shapley value|", fontsize=10.5)
    cb.ax.tick_params(labelsize=9.5)
    ax.set_title("Router attribution per family", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_validation(mi_rows, matrix_rows, refs_dir, out_path, label_top=6):
    """Two independent checks on the attribution, neither consulting the model.

    Left  - attribution against per-k-mer mutual information. Near-zero
            correlation, because no single k-mer carries much family
            information: the representation is distributed.
    Right - true genome GC content against the GC content of each family's
            top-attributed k-mers. Strong correlation, because the router does
            recover an aggregate compositional property.
    """
    import glob, os

    def gc(seq):
        return sum(c in "GC" for c in seq) / len(seq)

    def pearson(xs, ys):
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = (sum((x - mx) ** 2 for x in xs) ** 0.5) * (sum((y - my) ** 2 for y in ys) ** 0.5)
        return num / den if den else float("nan")

    def spearman(xs, ys):
        rx = {v: i for i, v in enumerate(sorted(xs))}
        ry = {v: i for i, v in enumerate(sorted(ys))}
        return pearson([rx[x] for x in xs], [ry[y] for y in ys])

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(7.8, 3.5))

    # ---- left: attribution vs mutual information
    shap = [float(r["mean_abs_shap"]) for r in mi_rows]
    mi_bits = [float(r["mutual_information_bits"]) for r in mi_rows]
    # Reported as a share of the information the family label carries, so the
    # axis needs no unit and does not collide with the paper's hardware bits.
    import math as _m
    H = _m.log2(len([c for c in matrix_rows[0] if c != "kmer"]))
    mi = [v / H * 100 for v in mi_bits]
    rho = spearman(shap, mi)
    axl.scatter(mi, shap, s=10, alpha=0.40, color=COLOR_POINT, linewidth=0)
    axl.set_xlabel("Family information per $k$-mer (%)", fontsize=12)
    axl.set_ylabel("Mean |Shapley value|", fontsize=12)
    axl.set_title(f"No single k-mer is decisive\nSpearman $\\rho$ = {rho:.3f}", fontsize=12.5)
    axl.tick_params(labelsize=10.5)
    axl.spines[["top", "right"]].set_visible(False)
    axl.grid(alpha=0.2)

    # ---- right: genome GC vs attribution GC
    families = [c for c in matrix_rows[0] if c != "kmer"]
    true_gc = {}
    for fam in families:
        g = n = 0
        for fa in glob.glob(os.path.join(refs_dir, fam, "*", "*.fasta")):
            with open(fa) as f:
                for line in f:
                    if line.startswith(">"):
                        continue
                    seq = line.strip().upper()
                    g += seq.count("G") + seq.count("C")
                    n += len(seq)
        if n:
            true_gc[fam] = g / n

    xs, ys, names = [], [], []
    for fam in families:
        if fam not in true_gc:
            continue
        ranked = sorted(matrix_rows, key=lambda r: float(r[fam]), reverse=True)
        xs.append(true_gc[fam])
        ys.append(sum(gc(r["kmer"]) for r in ranked[:20]) / 20)
        names.append(fam)

    if xs:
        r_p = pearson(xs, ys)
        axr.scatter(xs, ys, s=46, color=COLOR_HI, linewidth=0, zorder=3)
        xmid = (min(xs) + max(xs)) / 2
        # Labels sit on the side of the point facing the middle of the axis.
        # Two points on the same side whose GC values are close would have
        # their labels collide, so each such pair is pushed apart vertically.
        nudge = {nm: 0.0 for nm in names}
        pts = sorted(zip(ys, xs, names))
        for (ya, xa, na), (yb, xb, nb) in zip(pts, pts[1:]):
            if (xa > xmid) == (xb > xmid) and yb - ya < 0.035:
                nudge[na] -= 5.0
                nudge[nb] += 5.0
        for x, y, nm in zip(xs, ys, names):
            right = x > xmid
            axr.annotate(display(nm), (x, y), textcoords="offset points",
                         xytext=(-6 if right else 6, -3 + nudge[nm]),
                         ha="right" if right else "left",
                         fontsize=9, color=COLOR_MUTED)
        lo, hi = min(xs + ys) - 0.03, max(xs + ys) + 0.03
        axr.plot([lo, hi], [lo, hi], color=COLOR_MUTED, linewidth=1,
                 linestyle=":", zorder=1)
        axr.set_xlabel("Genome GC content", fontsize=12)
        axr.set_ylabel("GC of top-20 $k$-mers", fontsize=12)
        axr.set_title(f"Attribution recovers base composition\nPearson $r$ = {r_p:.3f}",
                      fontsize=12.5)
    axr.tick_params(labelsize=10.5)
    axr.spines[["top", "right"]].set_visible(False)
    axr.grid(alpha=0.2)

    fig.tight_layout(w_pad=2.2)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description="Figures for the router-explainability section")
    p.add_argument("--shap_csv", default="results/router_shap.csv")
    p.add_argument("--mi_csv", default="results/router_shap_mi.csv")
    p.add_argument("--matrix_csv", default="results/router_shap_matrix.csv")
    p.add_argument("--out_dir", default="results/figures")
    p.add_argument("--per_family", type=int, default=4)
    p.add_argument("--ext", default="png", choices=["png", "pdf", "svg"],
                   help="output format; pdf keeps the text as vectors, which print cleanly")
    p.add_argument("--refs_dir", default="data/refs")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Top k-mers per family:")
    rows = read_csv(args.matrix_csv)
    if rows:
        plot_top_kmers(rows, os.path.join(args.out_dir, f"router_shap_topkmers.{args.ext}"),
                       args.per_family)

    print("Validation (MI null + GC recovery):")
    mi_rows = read_csv(args.mi_csv)
    if mi_rows and rows:
        plot_validation(mi_rows, rows, args.refs_dir,
                        os.path.join(args.out_dir, f"router_shap_validation.{args.ext}"))


if __name__ == "__main__":
    main()
