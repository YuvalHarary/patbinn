#!/usr/bin/env python3
"""
Shapley attribution for the family router's decisions.

Answers "which k-mers does the router rely on to recognise each family?", and
then checks that answer against an independent statistic computed from the data
rather than from the model.

Two things make this cheaper and more defensible here than in a typical
explainability study:

  * The features are 1024 named DNA k-mers, so an attribution lands directly on
    an interpretable unit with no decoding step.
  * The features are binary. A feature whose value already matches the
    background contributes exactly zero to the Shapley sum, so each permutation
    sweep only has to touch the symmetric difference between the explained read
    and its background read - typically ~200 of 1024 features, not all of them.

Estimator: permutation-sampling (Castro et al. 2009) of the interventional
Shapley value, i.e. the same estimand as KernelSHAP, evaluated against a
background drawn from real reads. The model is treated as a black box and run
exactly as deployed - binarized weights, sign activations - so the attribution
describes the network the hardware implements, not a differentiable surrogate
of it. Gradient-based variants (DeepSHAP, GradientSHAP) would instead
backpropagate through the straight-through estimator, which is not the function
being executed.

Outputs:
  results/router_shap.csv       mean |phi| per (family, k-mer), plus signed mean
  results/router_shap_mi.csv    per-k-mer attribution vs mutual information

Run from the repository root:

    apptainer exec --nv --bind $(pwd):/workspace --pwd /workspace patbinn.sif \
        python3 src/explain_router.py
"""

import argparse
import csv
import math
import os
import random

import torch

from evaluate_moe import load_checkpoint
from kmer_encoding import parse_fastq, encode_sequence_kmers
from moe_dataset import remap_variant_idx


# ----------------------------------------------------------------- data

def load_reads_by_family(manifest_path, split, k, kmer_to_idx, vocab_size,
                         canonical, reads_per_family, rng):
    """Encoded reads for `split`, grouped by true family index."""
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["family_idx"] = int(r["family_idx"])
        r["variant_idx"] = int(r["variant_idx"])
    # Remap from ALL rows before filtering, for the same reason evaluate_moe does:
    # a split-filtered subset can produce a different dense index mapping.
    remap_variant_idx(rows)
    rows = [r for r in rows if r["split"] == split]
    if not rows:
        raise SystemExit(f"no rows for split={split!r} in {manifest_path}")

    by_family = {}
    rng.shuffle(rows)
    for row in rows:
        fam = row["family_idx"]
        have = by_family.setdefault(fam, [])
        if len(have) >= reads_per_family:
            continue
        seqs = parse_fastq(row["r1_path"]) + parse_fastq(row["r2_path"])
        if not seqs:
            continue
        rng.shuffle(seqs)
        for s in seqs[: reads_per_family - len(have)]:
            have.append(encode_sequence_kmers(s, k, kmer_to_idx, vocab_size,
                                              canonical=canonical))
        if all(len(v) >= reads_per_family for v in by_family.values()) \
           and len(by_family) == len({r["family_idx"] for r in rows}):
            if min(len(v) for v in by_family.values()) >= reads_per_family:
                break
    return {f: torch.stack(v) for f, v in by_family.items() if v}


# ------------------------------------------------------- Shapley estimator

@torch.no_grad()
def shapley_for_read(model, x, backgrounds, target_class, n_perm, device,
                     max_batch=8192):
    """Permutation-sampling Shapley values of each feature for one read.

    x           (D,)    the explained read, binary
    backgrounds (B, D)  pool of real reads to draw the reference from
    returns     (D,)    phi, contributions summing to f(x) - E[f(background)]
    """
    D = x.numel()
    phi = torch.zeros(D, device=device)

    for p in range(n_perm):
        b = backgrounds[torch.randint(len(backgrounds), (1,), device=device)].squeeze(0)

        # Features already equal to the background can never change the output,
        # whatever order they are added in: their marginal contribution is
        # identically zero. Only the symmetric difference matters.
        diff = (x != b).nonzero(as_tuple=True)[0]
        m = diff.numel()
        if m == 0:
            continue
        order = diff[torch.randperm(m, device=device)]

        # Coalition t has the first t features of `order` taken from x, the rest
        # from b. Build all m+1 coalitions at once as a prefix mask.
        coalitions = b.unsqueeze(0).repeat(m + 1, 1)
        # reveal[t, j] is True when feature order[j] is present at step t:
        # row 0 is the empty coalition, row m the full one.
        reveal = torch.zeros(m + 1, m, dtype=torch.bool, device=device)
        reveal[1:] = torch.tril(torch.ones(m, m, dtype=torch.bool, device=device))
        coalitions[:, order] = torch.where(reveal, x[order].expand(m + 1, m),
                                           b[order].expand(m + 1, m))

        outs = []
        for i in range(0, m + 1, max_batch):
            outs.append(model(coalitions[i:i + max_batch])[:, target_class])
        f = torch.cat(outs).float()

        phi.index_add_(0, order, f[1:] - f[:-1])

    return phi / max(n_perm, 1)


# ------------------------------------------------------------ validation

def mutual_information(features, labels, n_classes):
    """MI in bits between each binary feature and the class label.

    Computed from the data alone, with no reference to the model, so it is an
    independent check on whether the router's attributions land on k-mers that
    really are family-discriminative.
    """
    N, D = features.shape
    mi = torch.zeros(D)
    py = torch.bincount(labels, minlength=n_classes).float() / N
    px1 = features.float().mean(0)                      # P(x=1)
    for c in range(n_classes):
        sel = labels == c
        if not sel.any():
            continue
        # P(x=1, y=c) and P(x=0, y=c)
        p11 = features[sel].float().sum(0) / N
        p01 = py[c] - p11
        for p_joint, p_x in ((p11, px1), (p01, 1.0 - px1)):
            denom = p_x * py[c]
            ok = (p_joint > 0) & (denom > 0)
            mi[ok] += p_joint[ok] * torch.log2(p_joint[ok] / denom[ok])
    return mi.clamp(min=0.0)


def spearman(a, b):
    def rank(v):
        order = torch.argsort(v)
        r = torch.empty_like(v)
        r[order] = torch.arange(len(v), dtype=v.dtype)
        return r
    ra, rb = rank(a), rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return float((ra @ rb) / (ra.norm() * rb.norm() + 1e-12))


# ------------------------------------------------------------------ main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/cv_hierarchical_binary/fold1.pth")
    p.add_argument("--sample_manifest",
                   default="data/manifests_hierarchical_binary/cv_fold1_sample_manifest.csv")
    p.add_argument("--split", default="test")
    p.add_argument("--reads_per_family", type=int, default=40,
                   help="reads explained per family")
    p.add_argument("--permutations", type=int, default=24,
                   help="permutation samples per read; variance falls as 1/sqrt(n)")
    p.add_argument("--background", type=int, default=64,
                   help="reads forming the Shapley reference distribution")
    p.add_argument("--top_k", type=int, default=15, help="k-mers listed per family")
    p.add_argument("--out_csv", default="results/router_shap.csv")
    p.add_argument("--mi_csv", default="results/router_shap_mi.csv")
    p.add_argument("--matrix_csv", default="results/router_shap_matrix.csv")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else args.device if args.device != "auto" else "cpu")

    ckpt, router, _ = load_checkpoint(args.checkpoint, device)
    router.eval()
    kmer_to_idx = ckpt["kmer_vocab"]
    vocab_size, k = ckpt["vocab_size"], ckpt["k"]
    canonical = ckpt.get("canonical", False)
    family_names = ckpt["family_names"]
    idx_to_kmer = {v: kk for kk, v in kmer_to_idx.items()}

    print(f"router: {args.checkpoint}")
    print(f"  {len(family_names)} families, vocab {vocab_size}, k={k}, canonical={canonical}")
    print(f"  device={device}")

    by_family = load_reads_by_family(args.sample_manifest, args.split, k, kmer_to_idx,
                                     vocab_size, canonical, args.reads_per_family, rng)
    print(f"  explained reads: {sum(len(v) for v in by_family.values())} "
          f"across {len(by_family)} families")

    all_reads = torch.cat([v for v in by_family.values()]).to(device)
    all_labels = torch.cat([torch.full((len(v),), f, dtype=torch.long)
                            for f, v in by_family.items()])
    bg_idx = torch.randperm(len(all_reads))[: args.background]
    backgrounds = all_reads[bg_idx]

    # -------- attribution
    D = vocab_size
    abs_sum = torch.zeros(len(family_names), D, device=device)
    signed_sum = torch.zeros(len(family_names), D, device=device)
    counts = torch.zeros(len(family_names), device=device)

    for fam in sorted(by_family):
        reads = by_family[fam].to(device)
        for i in range(len(reads)):
            phi = shapley_for_read(router, reads[i], backgrounds, fam,
                                   args.permutations, device)
            abs_sum[fam] += phi.abs()
            signed_sum[fam] += phi
            counts[fam] += 1
        print(f"  {family_names[fam]:<18} {int(counts[fam]):4d} reads explained")

    mean_abs = (abs_sum / counts.clamp(min=1).unsqueeze(1)).cpu()
    mean_signed = (signed_sum / counts.clamp(min=1).unsqueeze(1)).cpu()

    # -------- independent check: mutual information, model never consulted
    mi = mutual_information(all_reads.cpu(), all_labels, len(family_names))
    global_attr = mean_abs.mean(0)
    rho = spearman(global_attr, mi)
    print(f"\nSpearman(mean |phi|, mutual information) = {rho:.3f}  over {D} k-mers")

    # -------- write
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "rank", "kmer", "mean_abs_shap", "mean_signed_shap",
                    "mutual_information_bits"])
        for fam, name in enumerate(family_names):
            order = torch.argsort(mean_abs[fam], descending=True)[: args.top_k]
            for r, j in enumerate(order.tolist(), 1):
                w.writerow([name, r, idx_to_kmer[j],
                            f"{mean_abs[fam][j]:.6f}", f"{mean_signed[fam][j]:.6f}",
                            f"{mi[j]:.6f}"])
    print(f"wrote {args.out_csv}")

    with open(args.mi_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kmer", "mean_abs_shap", "mutual_information_bits"])
        for j in range(D):
            w.writerow([idx_to_kmer[j], f"{global_attr[j]:.6f}", f"{mi[j]:.6f}"])
    print(f"wrote {args.mi_csv}  (spearman_rho={rho:.4f})")

    # Full attribution matrix. The top-k CSV above is a convenience view; a
    # heatmap built from it alone would show zeros wherever a k-mer fell outside
    # a family's top-k, which reads as "the router ignores this k-mer" when it
    # only means "not listed for that family".
    with open(args.matrix_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kmer"] + list(family_names))
        for j in range(D):
            w.writerow([idx_to_kmer[j]] + [f"{mean_abs[fam][j]:.6f}"
                                            for fam in range(len(family_names))])
    print(f"wrote {args.matrix_csv}  (full {len(family_names)} x {D} matrix)")

    # The single most and least informative k-mers, and what the top one is
    # actually worth: these are the concrete figures quoted in the paper, so
    # they are reported here rather than derived by hand.
    top_j = int(torch.argmax(mi)); low_j = int(torch.argmin(mi))
    present = all_reads[:, top_j] > 0
    print(f"\nMost informative k-mer: {idx_to_kmer[top_j]} at {mi[top_j]:.4f} bits "
          f"({mi[top_j]/math.log2(len(family_names))*100:.1f}% of the "
          f"{math.log2(len(family_names)):.0f}-bit family question)")
    print(f"Least informative k-mer: {idx_to_kmer[low_j]} at {mi[low_j]:.4f} bits")
    if present.any():
        print(f"  {'family':<18}{'P(family)':>11}{'P(family | present)':>22}")
        for f, name in enumerate(family_names):
            prior = float((all_labels == f).float().mean())
            post = float((all_labels[present.cpu()] == f).float().mean())
            print(f"  {name:<18}{prior*100:10.1f}%{post*100:21.1f}%")

    print("\nTop k-mers per family:")
    for fam, name in enumerate(family_names):
        order = torch.argsort(mean_abs[fam], descending=True)[:6]
        tops = ", ".join(f"{idx_to_kmer[j]}({mean_abs[fam][j]:.2f})" for j in order.tolist())
        print(f"  {name:<18} {tops}")


if __name__ == "__main__":
    main()
