#!/usr/bin/env python3
"""
Margin-based abstention at the router, over every read of a split.

The router's decision margin is the difference between its first- and
second-ranked family scores. Each score is a sum of +/-1 products over the
last hidden layer, so the margin is an even integer and a margin of 2*d means
the two leading output rows differ by d in POPCOUNT. This script measures how
well a threshold on that margin separates misrouted reads from correctly routed
ones: reads whose margin is at or below the threshold are withheld, and the
accuracy on the reads kept and the share of routing errors removed are
reported at every threshold.

Thresholds, not quantiles, define the operating points: the margin is an
integer with many ties, so "the least confident 5% of reads" is not a
well-defined set, whereas "margin at most 8" is.

Every read of every sample in the split is scored, so the accuracy with
nothing withheld is the router's per-read accuracy on that split as reported
by src/evaluate_moe.py.

Outputs:
  results/router_abstention.csv            one row per margin threshold
  results/router_abstention_fractions.csv  the same at fixed fractions withheld
  results/router_margin.csv       margin and read statistics, correct vs misrouted

Run from the repository root:

    apptainer exec --nv --bind $(pwd):/workspace --pwd /workspace patbinn.sif \
        python3 src/abstain_router.py
"""

import argparse
import csv
import os
from multiprocessing import Pool

import numpy as np
import torch

from evaluate_moe import load_checkpoint
from kmer_encoding import parse_fastq, encode_sequence_kmers
from moe_dataset import remap_variant_idx

_ENC = {}


def _init(k, kmer_to_idx, vocab_size, canonical):
    _ENC.update(k=k, kmer_to_idx=kmer_to_idx, vocab_size=vocab_size, canonical=canonical)
    torch.set_num_threads(1)


def _encode_sample(row):
    """All reads of one sample as a bit-packed (n_reads, vocab_size / 8) array."""
    seqs = parse_fastq(row["r1_path"]) + parse_fastq(row["r2_path"])
    if not seqs:
        return row["family_idx"], None
    vecs = torch.stack([encode_sequence_kmers(s, _ENC["k"], _ENC["kmer_to_idx"],
                                              _ENC["vocab_size"], canonical=_ENC["canonical"])
                        for s in seqs])
    return row["family_idx"], np.packbits(vecs.numpy().astype(bool), axis=1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/cv_hierarchical_binary/fold1.pth")
    p.add_argument("--sample_manifest",
                   default="data/manifests_hierarchical_binary/cv_fold1_sample_manifest.csv")
    p.add_argument("--split", default="test")
    p.add_argument("--out_csv", default="results/router_abstention.csv")
    p.add_argument("--margin_csv", default="results/router_margin.csv")
    p.add_argument("--fractions_csv", default="results/router_abstention_fractions.csv")
    p.add_argument("--fractions", type=float, nargs="+", default=[0, 1, 2, 5, 10, 20],
                   help="percentages of reads withheld to report")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else args.device if args.device != "auto" else "cpu")
    ckpt, router, _ = load_checkpoint(args.checkpoint, device)
    router.eval()
    vocab_size = ckpt["vocab_size"]

    with open(args.sample_manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["family_idx"] = int(r["family_idx"])
        r["variant_idx"] = int(r["variant_idx"])
    remap_variant_idx(rows)
    rows = [r for r in rows if r["split"] == args.split]
    if not rows:
        raise SystemExit(f"no rows for split={args.split!r} in {args.sample_manifest}")
    print(f"router: {args.checkpoint}   device={device}   {len(rows)} samples")

    margins, correct, richness = [], [], []
    init = (ckpt["k"], ckpt["kmer_vocab"], vocab_size, ckpt.get("canonical", False))
    with Pool(args.workers, initializer=_init, initargs=init) as pool:
        for n, (fam, packed) in enumerate(pool.imap_unordered(_encode_sample, rows, chunksize=1), 1):
            if packed is None:
                continue
            bits = torch.from_numpy(np.unpackbits(packed, axis=1)[:, :vocab_size])
            with torch.no_grad():
                for i in range(0, len(bits), 8192):
                    x = bits[i:i + 8192].to(device).float()
                    scores = router(x).float()
                    top2 = scores.topk(2, dim=1).values
                    margins.append((top2[:, 0] - top2[:, 1]).cpu())
                    # argmax, as in evaluate_moe.py, so that tied reads (margin 0)
                    # are resolved exactly as they are in the reported accuracy
                    correct.append((scores.argmax(dim=1) == fam).cpu())
                    richness.append(x.sum(1).cpu())
            if n % 100 == 0:
                print(f"  {n}/{len(rows)} samples", flush=True)

    margins = torch.cat(margins); correct = torch.cat(correct); richness = torch.cat(richness)
    n = len(margins); n_err = int((~correct).sum())
    base = float(correct.float().mean()) * 100
    print(f"\n  {n} reads, routing accuracy {base:.2f}%, {n_err} misrouted")

    stat_rows = [
        ("n_reads", int(correct.sum()), n_err),
        ("mean_decision_margin", float(margins[correct].mean()), float(margins[~correct].mean())),
        ("median_decision_margin", float(margins[correct].median()), float(margins[~correct].median())),
        ("mean_distinct_vocab_kmers", float(richness[correct].mean()), float(richness[~correct].mean())),
    ]
    print(f"\n  {'':<28}{'correct':>12}{'misrouted':>12}")
    for name, a, b in stat_rows:
        print(f"  {name:<28}{a:12.2f}{b:12.2f}")

    # Sweep the threshold over every margin value that occurs.
    out = []
    print(f"\n  {'margin <=':>10}{'POPCOUNT diff <=':>18}{'withheld %':>12}{'kept acc %':>12}{'errors removed %':>18}")
    for thr in [None] + sorted(set(margins.tolist())):
        held = torch.zeros(n, dtype=torch.bool) if thr is None else margins <= thr
        kept = ~held
        if not kept.any():
            break
        row = {"margin_threshold": "" if thr is None else int(thr),
               "popcount_difference": "" if thr is None else thr / 2,
               "reads_withheld_pct": float(held.float().mean()) * 100,
               "accuracy_kept_pct": float(correct[kept].float().mean()) * 100,
               "errors_removed_pct": (float((held & ~correct).sum()) / n_err * 100) if n_err else 0.0}
        out.append(row)
        if row["reads_withheld_pct"] <= 30:
            print(f"  {str(row['margin_threshold']):>10}{str(row['popcount_difference']):>18}"
                  f"{row['reads_withheld_pct']:12.2f}{row['accuracy_kept_pct']:12.2f}"
                  f"{row['errors_removed_pct']:18.1f}")

    # The same sweep at fixed fractions of reads withheld. The margin is an
    # integer with many ties, so the reads sharing the margin at the cut are
    # withheld in proportion: the expected result of breaking ties at random.
    print(f"\n  {'withheld %':>10}{'kept acc %':>12}{'errors removed %':>18}")
    fractions = []
    for frac in args.fractions:
        hi = next(i for i, r in enumerate(out) if r["reads_withheld_pct"] >= frac)
        if hi == 0:
            removed = 0.0
        else:
            a, b = out[hi - 1], out[hi]
            t = (frac - a["reads_withheld_pct"]) / (b["reads_withheld_pct"] - a["reads_withheld_pct"])
            removed = a["errors_removed_pct"] + t * (b["errors_removed_pct"] - a["errors_removed_pct"])
        kept_acc = 100 - (100 - base) * (1 - removed / 100) / (1 - frac / 100)
        fractions.append({"reads_withheld_pct": frac, "accuracy_kept_pct": kept_acc,
                          "errors_removed_pct": removed})
        print(f"  {frac:10.0f}{kept_acc:12.2f}{removed:18.1f}")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.fractions_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fractions[0].keys()))
        w.writeheader()
        for row in fractions:
            w.writerow({kk: f"{v:.4f}" for kk, v in row.items()})
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        for row in out:
            w.writerow({kk: (f"{v:.4f}" if isinstance(v, float) else v) for kk, v in row.items()})
    with open(args.margin_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "correct", "misrouted"])
        for name, a, b in stat_rows:
            w.writerow([name, f"{a:.4f}" if isinstance(a, float) else a,
                        f"{b:.4f}" if isinstance(b, float) else b])
    print(f"\nwrote {args.out_csv}\nwrote {args.fractions_csv}\nwrote {args.margin_csv}")


if __name__ == "__main__":
    main()
