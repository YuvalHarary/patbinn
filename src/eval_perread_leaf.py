#!/usr/bin/env python3
"""
Pipeline-mode per-read evaluation across every cross-validation fold.

`evaluate_moe.evaluate_checkpoint` routes each sample's reads by the
majority-voted family, which is right for the sample decision but not for a
per-read figure. This script instead routes every read by its own router
prediction, so the reported per-read accuracy is the one directly comparable
to an external classifier's exact assignment. The oracle-routed figure stored
in the fold JSONs is handed the true family and measures an easier task.

One pass produces, for all five folds:

  results/cv_perread_leaf.csv       per fold: family, oracle and leaf accuracy
  results/cv_class_metrics.csv      per leaf class: precision, recall, F1, support
  results/cv_family_confusion.csv   family confusion matrix, summed over folds
  results/cv_family_readcounts.csv  held-out reads contributed per family

Run from the repository root:

    apptainer exec --nv --bind $(pwd):/workspace --pwd /workspace patbinn.sif \
        python3 src/eval_perread_leaf.py
"""

import argparse
import collections
import csv
import os
import time

import torch

from evaluate_moe import load_checkpoint, majority_vote, run_model_batched
from kmer_encoding import parse_fastq, encode_sequence_kmers
from moe_dataset import encode_sequences_parallel
from moe_dataset import remap_variant_idx


def evaluate_fold(rows, router, experts, kmer_to_idx, vocab_size, k, device,
                   canonical, families, max_reads, bs=8192, workers=32):
    """Per-read and per-sample metrics for one fold, routing each read itself."""
    leaf_true, leaf_pred = [], []          # dense leaf-class ids, for P/R/F1
    fam_true, fam_pred = [], []            # family ids, for the confusion matrix
    reads_per_family = collections.Counter()
    oracle_correct = oracle_total = 0
    n_samples = fam_correct = leaf_correct = 0

    for row in rows:
        fi, vi = row["family_idx"], row["variant_idx"]
        seqs = parse_fastq(row["r1_path"]) + parse_fastq(row["r2_path"])
        if max_reads is not None:
            seqs = seqs[:max_reads]
        if not seqs:
            continue

        # Encoding, not inference, dominates at this scale; spread it over cores.
        if len(seqs) >= 2000 and workers > 1:
            vecs = encode_sequences_parallel(seqs, k, kmer_to_idx, vocab_size,
                                             num_workers=workers, canonical=canonical)
        else:
            vecs = torch.stack([encode_sequence_kmers(s, k, kmer_to_idx, vocab_size,
                                                       canonical=canonical) for s in seqs])
        fam_preds = run_model_batched(router, vecs, device, batch_size=bs)

        # Per-read leaf: group by predicted family so each expert runs once over
        # its own slice rather than every expert over every read.
        by_pred = collections.defaultdict(list)
        for i, f in enumerate(fam_preds):
            by_pred[f].append(i)
        pred_variant = [None] * len(seqs)
        for f, idxs in by_pred.items():
            sub = run_model_batched(experts[f], vecs[idxs], device, batch_size=bs)
            for j, i in enumerate(idxs):
                pred_variant[i] = sub[j]

        for i, f in enumerate(fam_preds):
            leaf_true.append((fi, vi))
            leaf_pred.append((f, pred_variant[i]))
        fam_true.extend([fi] * len(seqs))
        fam_pred.extend(fam_preds)
        reads_per_family[fi] += len(seqs)

        oracle = run_model_batched(experts[fi], vecs, device, batch_size=bs)
        oracle_correct += sum(p == vi for p in oracle)
        oracle_total += len(oracle)

        maj = majority_vote(fam_preds)
        pipe = majority_vote(run_model_batched(experts[maj], vecs, device, batch_size=bs))
        n_samples += 1
        fam_correct += int(maj == fi)
        leaf_correct += int(maj == fi and pipe == vi)

    n = len(leaf_true)
    return {
        "n_samples": n_samples,
        "n_reads": n,
        "per_read_family_acc": 100.0 * sum(a == b for a, b in zip(fam_true, fam_pred)) / max(n, 1),
        "per_read_variant_acc_oracle": 100.0 * oracle_correct / max(oracle_total, 1),
        "per_read_leaf_acc": 100.0 * sum(a == b for a, b in zip(leaf_true, leaf_pred)) / max(n, 1),
        "sample_family_acc": 100.0 * fam_correct / max(n_samples, 1),
        "sample_leaf_acc_pipeline": 100.0 * leaf_correct / max(n_samples, 1),
        "_leaf_true": leaf_true, "_leaf_pred": leaf_pred,
        "_fam_true": fam_true, "_fam_pred": fam_pred,
        "_reads_per_family": reads_per_family,
    }


def prf(true, pred, labels):
    """Per-class precision, recall and F1 from raw prediction lists."""
    tp = collections.Counter(); fp = collections.Counter(); fn = collections.Counter()
    for t, p in zip(true, pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    out = {}
    for c in labels:
        P = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        R = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        out[c] = (P, R, F, tp[c] + fn[c])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints_dir", default="checkpoints/cv_hierarchical_binary")
    p.add_argument("--manifests_dir", default="data/manifests_hierarchical_binary")
    p.add_argument("--k_folds", type=int, default=5)
    p.add_argument("--folds", default=None,
                   help="comma-separated subset of folds to evaluate, e.g. 1 or 0,2; default all. "
                        "Folds are independent, so separate processes can run them in parallel")
    p.add_argument("--split", default="test")
    p.add_argument("--max_reads_per_sample", type=int, default=None,
                   help="cap reads per sample; default is every held-out read")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--workers", type=int, default=1,
                   help="parallel encode workers; the spawn pool costs more than it saves at one call per sample")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else args.device if args.device != "auto" else "cpu")
    per_fold, all_leaf_t, all_leaf_p, all_fam_t, all_fam_p = [], [], [], [], []
    reads_per_family = collections.Counter()
    families = None
    variant_names = None

    folds = [int(f) for f in args.folds.split(",")] if args.folds else range(args.k_folds)
    for fold in folds:
        ck = os.path.join(args.checkpoints_dir, f"fold{fold}.pth")
        mf = os.path.join(args.manifests_dir, f"cv_fold{fold}_sample_manifest.csv")
        if not (os.path.exists(ck) and os.path.exists(mf)):
            print(f"fold {fold}: missing checkpoint or manifest, skipped")
            continue

        t0 = time.time()
        ckpt, router, experts = load_checkpoint(ck, device)
        router.eval()
        for e in experts:
            e.eval()
        families = ckpt["family_names"]
        variant_names = ckpt["variant_names_by_family"]

        with open(mf, newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["family_idx"] = int(r["family_idx"]); r["variant_idx"] = int(r["variant_idx"])
        remap_variant_idx(rows)
        rows = [r for r in rows if r["split"] == args.split]

        res = evaluate_fold(rows, router, experts, ckpt["kmer_vocab"], ckpt["vocab_size"],
                             ckpt["k"], device, ckpt.get("canonical", False), families,
                             args.max_reads_per_sample, args.batch_size, args.workers)
        res["fold"] = fold
        all_leaf_t += res.pop("_leaf_true"); all_leaf_p += res.pop("_leaf_pred")
        all_fam_t += res.pop("_fam_true");   all_fam_p += res.pop("_fam_pred")
        reads_per_family.update(res.pop("_reads_per_family"))
        per_fold.append(res)
        print(f"fold {fold}: {res['n_samples']} samples, {res['n_reads']} reads, "
              f"leaf {res['per_read_leaf_acc']:.2f}%  family {res['per_read_family_acc']:.2f}%  "
              f"oracle {res['per_read_variant_acc_oracle']:.2f}%  [{time.time()-t0:.0f}s]")

    if not per_fold:
        raise SystemExit("no folds evaluated")

    os.makedirs(args.out_dir, exist_ok=True)
    cols = ["fold", "n_samples", "n_reads", "per_read_family_acc",
            "per_read_variant_acc_oracle", "per_read_leaf_acc",
            "sample_family_acc", "sample_leaf_acc_pipeline"]
    with open(os.path.join(args.out_dir, "cv_perread_leaf.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in per_fold:
            w.writerow({c: r[c] for c in cols})

    # Macro-averaged per-class metrics over the 58 leaf classes, pooled over folds.
    labels = sorted(set(all_leaf_t))
    m = prf(all_leaf_t, all_leaf_p, labels)
    with open(os.path.join(args.out_dir, "cv_class_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "variant", "precision", "recall", "f1", "support_reads"])
        for (fi, vi) in labels:
            P, R, F, S = m[(fi, vi)]
            fam = families[fi]
            var = variant_names[fam][vi] if vi < len(variant_names[fam]) else str(vi)
            w.writerow([fam, var, f"{P:.6f}", f"{R:.6f}", f"{F:.6f}", S])
    macro = [sum(x) / len(labels) for x in zip(*[(m[c][0], m[c][1], m[c][2]) for c in labels])]
    print(f"\nmacro over {len(labels)} leaf classes: "
          f"precision {macro[0]*100:.2f}%  recall {macro[1]*100:.2f}%  F1 {macro[2]*100:.2f}%")

    F = len(families)
    conf = [[0]*F for _ in range(F)]
    for t, p_ in zip(all_fam_t, all_fam_p):
        conf[t][p_] += 1
    with open(os.path.join(args.out_dir, "cv_family_confusion.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["true_family"] + list(families))
        for i in range(F):
            w.writerow([families[i]] + conf[i])

    total = sum(reads_per_family.values())
    with open(os.path.join(args.out_dir, "cv_family_readcounts.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["family", "held_out_reads", "share_pct"])
        for i in range(F):
            n = reads_per_family[i]
            w.writerow([families[i], n, f"{100.0*n/max(total,1):.4f}"])
    print(f"held-out reads: {total}")
    print(f"wrote cv_perread_leaf.csv, cv_class_metrics.csv, "
          f"cv_family_confusion.csv, cv_family_readcounts.csv in {args.out_dir}/")


if __name__ == "__main__":
    main()
