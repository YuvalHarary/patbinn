#!/usr/bin/env python3
"""
What the router gets wrong, and whether its errors follow the mechanism the
Shapley attribution implies.

src/explain_router.py shows the router keys on distributed base composition
rather than on marker k-mers. That makes a falsifiable prediction: families
whose genomes have *similar* GC content should be the ones confused with each
other. This script tests that prediction, and characterises misrouted reads.

Reported:
  * per-family read-level routing accuracy
  * the 8x8 read-level confusion matrix
  * correlation between family-pair confusion rate and GC similarity
  * how misrouted reads differ from correctly routed ones (decision margin,
    GC content, number of distinct vocabulary k-mers hit)

No Shapley values here - this is forward passes only, so it is fast.

Run from the repository root:

    apptainer exec --nv --bind $(pwd):/workspace --pwd /workspace patbinn.sif \
        python3 src/analyse_router_errors.py
"""

import argparse
import csv
import glob
import os
import random

import torch

from evaluate_moe import load_checkpoint
from kmer_encoding import parse_fastq, encode_sequence_kmers
from moe_dataset import remap_variant_idx


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) ** 0.5) * (sum((y - my) ** 2 for y in ys) ** 0.5)
    return num / den if den else float("nan")


def genome_gc(refs_dir, families):
    out = {}
    for fam in families:
        g = n = 0
        for fa in glob.glob(os.path.join(refs_dir, fam, "*", "*.fasta")):
            with open(fa) as f:
                for line in f:
                    if line.startswith(">"):
                        continue
                    s = line.strip().upper()
                    g += s.count("G") + s.count("C")
                    n += len(s)
        if n:
            out[fam] = g / n
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/cv_hierarchical_binary/fold1.pth")
    p.add_argument("--sample_manifest",
                   default="data/manifests_hierarchical_binary/cv_fold1_sample_manifest.csv")
    p.add_argument("--split", default="test")
    p.add_argument("--reads_per_family", type=int, default=4000)
    p.add_argument("--refs_dir", default="data/refs")
    p.add_argument("--out_csv", default="results/router_errors.csv")
    p.add_argument("--confusion_csv", default="results/router_confusion.csv")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    rng = random.Random(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else args.device if args.device != "auto" else "cpu")

    ckpt, router, _ = load_checkpoint(args.checkpoint, device)
    router.eval()
    kmer_to_idx, vocab_size, k = ckpt["kmer_vocab"], ckpt["vocab_size"], ckpt["k"]
    canonical = ckpt.get("canonical", False)
    families = ckpt["family_names"]
    F = len(families)

    with open(args.sample_manifest, newline="") as f:
        all_rows = list(csv.DictReader(f))
    for r in all_rows:
        r["family_idx"] = int(r["family_idx"])
        r["variant_idx"] = int(r["variant_idx"])
    remap_variant_idx(all_rows)
    rows = [r for r in all_rows if r["split"] == args.split]
    rng.shuffle(rows)

    # Collect raw sequences per family so read-level properties (GC, k-mer
    # richness) can be measured alongside the routing decision.
    seqs_by_fam = {f: [] for f in range(F)}
    for row in rows:
        fam = row["family_idx"]
        if len(seqs_by_fam[fam]) >= args.reads_per_family:
            continue
        s = parse_fastq(row["r1_path"]) + parse_fastq(row["r2_path"])
        rng.shuffle(s)
        seqs_by_fam[fam].extend(s[: args.reads_per_family - len(seqs_by_fam[fam])])
        if all(len(v) >= args.reads_per_family for v in seqs_by_fam.values()):
            break

    confusion = [[0] * F for _ in range(F)]
    stats = {"correct": {"margin": [], "gc": [], "richness": []},
             "wrong": {"margin": [], "gc": [], "richness": []}}

    print(f"router: {args.checkpoint}   device={device}")
    for fam in range(F):
        seqs = seqs_by_fam[fam]
        if not seqs:
            continue
        vecs = torch.stack([encode_sequence_kmers(s, k, kmer_to_idx, vocab_size,
                                                  canonical=canonical) for s in seqs])
        preds, margins = [], []
        with torch.no_grad():
            for i in range(0, len(vecs), 2048):
                logits = router(vecs[i:i + 2048].to(device)).float()
                top2 = logits.topk(2, dim=1)
                preds.extend(top2.indices[:, 0].cpu().tolist())
                # margin between the chosen family and the runner-up
                margins.extend((top2.values[:, 0] - top2.values[:, 1]).cpu().tolist())

        for s, pred, marg, vec in zip(seqs, preds, margins, vecs):
            confusion[fam][pred] += 1
            bucket = "correct" if pred == fam else "wrong"
            su = s.upper()
            stats[bucket]["margin"].append(marg)
            stats[bucket]["gc"].append((su.count("G") + su.count("C")) / max(len(su), 1))
            stats[bucket]["richness"].append(int(vec.sum().item()))

        n = sum(confusion[fam])
        print(f"  {families[fam]:<18} {confusion[fam][fam]/n*100:6.2f}% of {n} reads routed correctly")

    total = sum(sum(r) for r in confusion)
    correct = sum(confusion[i][i] for i in range(F))
    print(f"\noverall read-level routing accuracy: {correct/total*100:.2f}%  ({total} reads)")

    # ---- does confusion follow GC similarity?
    gc = genome_gc(args.refs_dir, families)
    dgc, rate = [], []
    for i in range(F):
        n = sum(confusion[i])
        for j in range(F):
            if i == j or families[i] not in gc or families[j] not in gc:
                continue
            dgc.append(abs(gc[families[i]] - gc[families[j]]))
            rate.append(confusion[i][j] / n)
    r_conf = pearson(dgc, rate)
    print(f"Pearson(|GC_i - GC_j|, confusion rate i->j) = {r_conf:+.3f}  over {len(dgc)} ordered pairs")
    print("  negative means closer GC -> more confusion, i.e. errors follow composition")

    # ---- what distinguishes a misrouted read?
    print("\nmisrouted vs correctly routed reads:")
    print(f"  {'property':<28}{'correct':>10}{'wrong':>10}")
    for key, name in (("margin", "decision margin"),
                      ("gc", "read GC content"),
                      ("richness", "distinct vocab k-mers")):
        c = stats["correct"][key]
        w = stats["wrong"][key]
        if c and w:
            print(f"  {name:<28}{sum(c)/len(c):10.3f}{sum(w)/len(w):10.3f}")
    print(f"  {'n reads':<28}{len(stats['correct']['margin']):10d}{len(stats['wrong']['margin']):10d}")

    os.makedirs(os.path.dirname(args.confusion_csv) or ".", exist_ok=True)
    with open(args.confusion_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true_family"] + list(families))
        for i in range(F):
            w.writerow([families[i]] + confusion[i])
    print(f"\nwrote {args.confusion_csv}")

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "correct", "wrong"])
        for key, name in (("margin", "mean_decision_margin"),
                          ("gc", "mean_read_gc"),
                          ("richness", "mean_distinct_vocab_kmers")):
            c, wr = stats["correct"][key], stats["wrong"][key]
            w.writerow([name, f"{sum(c)/len(c):.6f}" if c else "",
                        f"{sum(wr)/len(wr):.6f}" if wr else ""])
        w.writerow(["n_reads", len(stats["correct"]["margin"]), len(stats["wrong"]["margin"])])
        w.writerow(["read_routing_accuracy_pct", f"{correct/total*100:.4f}", ""])
        w.writerow(["pearson_gcdist_vs_confusion", f"{r_conf:.6f}", ""])
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
