#!/usr/bin/env python3
"""
The five results figures of the paper, regenerated from the result CSVs.

  fig09-cv-compare.pdf        hierarchical / float32 / flat under 5-fold CV (bars, mean +/- sd)
  fig10-encoding.pdf          (a) accuracy per encoding  (b) binary weights per encoding
  fig11-reads-per-sample.pdf  per-sample accuracy against reads voted, one curve per fold
  fig12-error-rate.pdf        accuracy against injected substitution error (one fold)
  fig13-kraken-compare.pdf    PatBiNN against Kraken2 under injected error, per read and per sample

Every number is read from a CSV named on the command line; nothing is typed in.
The hierarchical per-read bar in fig09 is PIPELINE accuracy (each read routed by
the router's own prediction), the same 58-way task the flat baseline solves.

Run from the repository root, for example:

    apptainer exec --bind $(pwd):/workspace --pwd /workspace patbinn.sif \
        python3 src/plot_paper_figures.py --out_dir patbinn_paper_EIAA/figs
"""

import argparse
import csv
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito, colourblind-safe; the same palette as plot_results.py
BLUE, GREEN, ORANGE, RED, PURPLE = "#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7"
W = 3.46  # single-column width in inches

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
    "axes.labelsize": 8.5, "axes.titlesize": 8.5, "legend.fontsize": 7, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.5,
    "grid.linewidth": 0.5, "lines.linewidth": 1.2, "lines.markersize": 4, "legend.framealpha": 1,
    "legend.edgecolor": "0.3", "axes.linewidth": 0.6,
})


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def col(rs, key):
    return [float(r[key]) for r in rs]


def save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("  wrote", path)


# --------------------------------------------------------------------------- fig09
def fig09(a, out_dir):
    hb, hf, fl = rows(a.cv_binary), rows(a.cv_float32), rows(a.cv_flat)
    # pipeline per-read for the hierarchical models; the flat baseline is 58-way already
    key = "per_read_leaf_acc" if "per_read_leaf_acc" in hb[0] else "per_read_expert_acc_oracle"
    if key != "per_read_leaf_acc":
        print("  WARNING fig09: no per_read_leaf_acc column, falling back to oracle per-read")
    groups = [("Hierarchical\n(binary)", col(hb, key), col(hb, "sample_leaf_acc_pipeline")),
              ("Hierarchical\n(float32)", col(hf, key), col(hf, "sample_leaf_acc_pipeline")),
              ("Flat 58-way\n(binary)", col(fl, "per_read_leaf_acc"), col(fl, "sample_leaf_acc"))]
    fig, ax = plt.subplots(figsize=(W, 2.12))
    x = range(len(groups)); bw = 0.38
    for i, (name, pr, ps) in enumerate(groups):
        for off, vals, colr, lab in ((-bw / 2, pr, BLUE, "Per-read accuracy"), (bw / 2, ps, GREEN, "Per-sample accuracy")):
            m, s = st.mean(vals), st.stdev(vals)
            ax.bar(i + off, m, bw, yerr=s, color=colr, edgecolor="black", linewidth=0.5,
                   capsize=2.5, error_kw={"linewidth": 0.8}, label=lab if i == 0 else None)
            ax.text(i + off, m - 6 if m > 20 else m + 2, f"{m:.1f}", ha="center", va="top" if m > 20 else "bottom",
                    color="white" if m > 20 else "black", fontsize=7)
    ax.set_xticks(list(x)); ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100); ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right")
    save(fig, out_dir, "fig09-cv-compare.pdf")


# --------------------------------------------------------------------------- fig10
def fig10(a, out_dir):
    enc = rows(a.encoding)   # columns: encoding, width, router, expert_oracle, sample
    labels = [r["encoding"] for r in enc]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(W, 4.56), gridspec_kw={"height_ratios": [1, 1.15]})
    x = list(range(len(enc)))
    for key, colr, mk, ls, lab in (("router", BLUE, "o", "-", "Family accuracy (per read)"),
                                   ("expert_oracle", GREEN, "s", "--", "Variant accuracy, oracle-routed (per read)"),
                                   ("sample", ORANGE, "^", ":", "Accuracy (per sample)")):
        ax1.plot(x, col(enc, key), marker=mk, linestyle=ls, color=colr, markerfacecolor="white", label=lab)
    ax1.set_xticks(x); ax1.set_xticklabels(labels); ax1.set_ylabel("Accuracy (%)"); ax1.legend(loc="lower left")
    ax1.text(-0.16, 1.02, "(a)", transform=ax1.transAxes, fontweight="bold")
    # weights follow from the input width: router = in*1024 + 1024*128 + 128*8, experts = 8 of the same shape
    router = [(int(r["width"]) * 1024 + 1024 * 128 + 128 * 8) / 1e6 for r in enc]
    experts = [8 * (int(r["width"]) * 1024 + 1024 * 128 + 128 * 8) / 1e6 for r in enc]
    ax2.bar(x, router, 0.6, color=BLUE, edgecolor="black", linewidth=0.5, label="Router")
    ax2.bar(x, experts, 0.6, bottom=router, color=GREEN, edgecolor="black", linewidth=0.5, hatch="//", label="Experts (x8)")
    for i in x:
        ax2.text(i, router[i] / 2, f"{router[i]:.1f}", ha="center", va="center", color="white", fontsize=6.5)
        ax2.text(i, router[i] + experts[i] - 0.5, f"{experts[i]:.1f}", ha="center", va="top", color="white", fontsize=6.5)
    ax2.set_xticks(x); ax2.set_xticklabels(labels); ax2.set_xlabel("Encoding"); ax2.set_ylabel("Binary weights (millions)")
    ax2.grid(axis="x", visible=False); ax2.legend(loc="upper right")
    ax2.text(-0.16, 1.02, "(b)", transform=ax2.transAxes, fontweight="bold")
    save(fig, out_dir, "fig10-encoding.pdf")


# --------------------------------------------------------------------------- fig11
def fig11(a, out_dir):
    rs = rows(a.read_count)
    fig, ax = plt.subplots(figsize=(W, 2.12))
    styles = [(BLUE, "o", "-"), (GREEN, "s", "--"), (ORANGE, "^", ":"), (RED, "D", "-."), (PURPLE, "v", "-.")]
    for fold in sorted({int(r["fold"]) for r in rs}):
        fr = sorted((r for r in rs if int(r["fold"]) == fold), key=lambda r: int(r["max_votes_per_sample"]))
        c, mk, ls = styles[fold % len(styles)]
        ax.plot([int(r["max_votes_per_sample"]) for r in fr], col(fr, "sample_leaf_acc_pipeline"),
                marker=mk, linestyle=ls, color=c, markerfacecolor="white", label=f"Fold {fold}")
    ax.set_xscale("log"); counts = sorted({int(r["max_votes_per_sample"]) for r in rs})
    # 20 and 25 sit too close on a log axis to both carry a label
    ax.set_xticks(counts); ax.set_xticklabels(["" if c == 25 else str(c) for c in counts]); ax.minorticks_off()
    ax.set_xlabel("Reads per sample (majority vote)"); ax.set_ylabel("Per-sample accuracy (%)")
    ax.set_ylim(85, 100.5); ax.legend(loc="lower right", ncol=2)
    save(fig, out_dir, "fig11-reads-per-sample.pdf")


# --------------------------------------------------------------------------- fig12
def fig12(a, out_dir):
    rs = sorted(rows(a.error_sweep), key=lambda r: float(r["error_rate"]))
    x = [100 * float(r["error_rate"]) for r in rs]
    fig, ax = plt.subplots(figsize=(W, 2.12))
    for key, colr, mk, ls, lab in (("per_read_family_acc", BLUE, "o", "-", "Family (per read)"),
                                   ("per_read_variant_acc_oracle", GREEN, "s", "--", "Variant, oracle-routed (per read)"),
                                   ("sample_leaf_acc_pipeline", ORANGE, "^", ":", "Full-class (per sample)")):
        ax.plot(x, col(rs, key), marker=mk, linestyle=ls, color=colr, markerfacecolor="white", label=lab)
    ax.set_xlabel("Injected per-base error rate (%)"); ax.set_ylabel("Accuracy (%)")
    lo = min(min(col(rs, k)) for k in ("per_read_family_acc", "per_read_variant_acc_oracle", "sample_leaf_acc_pipeline"))
    ax.set_ylim(5 * int((lo - 3) // 5), 101); ax.set_xticks(range(0, 11, 2)); ax.legend(loc="lower left")
    save(fig, out_dir, "fig12-error-rate.pdf")


# --------------------------------------------------------------------------- fig13
def fig13(a, out_dir):
    k = sorted(rows(a.kraken), key=lambda r: float(r["error_rate"]))
    kn = sorted(rows(a.kraken_normalized), key=lambda r: float(r["error_rate"]))
    p = sorted(rows(a.error_sweep), key=lambda r: float(r["error_rate"]))
    x = [100 * float(r["error_rate"]) for r in k]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(W, 3.56), sharex=True)
    ax1.plot(x, col(k, "per_read_variant_acc"), marker="o", color=BLUE, markerfacecolor="white", label="Kraken2")
    ax1.plot(x, col(k, "pct_unclassified"), linestyle="--", color=PURPLE, label="Kraken2 unclassified")
    ax1.plot(x, col(p, "per_read_leaf_acc"), marker="^", linestyle=":", color=ORANGE, markerfacecolor="white", label="PatBiNN")
    ax1.set_ylabel("Per-read accuracy (%)"); ax1.set_ylim(0, 100); ax1.legend(loc="center left")
    ax1.text(-0.16, 1.02, "(a)", transform=ax1.transAxes, fontweight="bold")
    ax2.plot(x, col(kn, "sample_variant_acc_lenient"), marker="o", color=BLUE, markerfacecolor="white", label="Kraken2 (lenient)")
    ax2.plot(x, col(kn, "sample_variant_acc_strict"), marker="s", linestyle="--", color=GREEN, markerfacecolor="white", label="Kraken2 (fair)")
    ax2.plot(x, col(p, "sample_leaf_acc_pipeline"), marker="^", linestyle=":", color=ORANGE, markerfacecolor="white", label="PatBiNN")
    ax2.set_ylabel("Per-sample accuracy (%)"); ax2.set_xlabel("Injected per-base error rate (%)")
    lo = min(min(col(kn, "sample_variant_acc_strict")), min(col(p, "sample_leaf_acc_pipeline")))
    ax2.set_ylim(5 * int((lo - 3) // 5), 101); ax2.set_xticks(range(0, 11, 2)); ax2.legend(loc="lower left")
    ax2.text(-0.16, 1.02, "(b)", transform=ax2.transAxes, fontweight="bold")
    save(fig, out_dir, "fig13-kraken-compare.pdf")


def main():
    ap = argparse.ArgumentParser(description="Regenerate the paper's five results figures from CSVs")
    ap.add_argument("--cv_binary", default="results/cv_results.csv")
    ap.add_argument("--cv_float32", default="results/cv_results_float32.csv")
    ap.add_argument("--cv_flat", default="results/cv_results_flat.csv")
    ap.add_argument("--encoding", default="results/encoding_table.csv",
                    help="columns: encoding,width,router,expert_oracle,sample")
    ap.add_argument("--read_count", default="results/read_count_sweep.csv")
    ap.add_argument("--error_sweep", default="results/patbinn_fold1_error_sweep.csv")
    ap.add_argument("--kraken", default="results/kraken2_error_sweep.csv")
    ap.add_argument("--kraken_normalized", default="results/kraken2_sample_acc_normalized.csv")
    ap.add_argument("--out_dir", default="patbinn_paper_EIAA/figs")
    ap.add_argument("--only", nargs="*", default=None, help="subset, e.g. fig09 fig12")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for name, fn in (("fig09", fig09), ("fig10", fig10), ("fig11", fig11), ("fig12", fig12), ("fig13", fig13)):
        if a.only and name not in a.only:
            continue
        try:
            fn(a, a.out_dir)
        except FileNotFoundError as e:
            print(f"  skip {name}: {e}")


if __name__ == "__main__":
    main()
