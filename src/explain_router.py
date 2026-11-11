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

The explained reads are drawn evenly from every held-out sample of a family,
so an attribution reflects the family rather than one isolate. Mutual
information and the conditional-probability example are computed over a
separate, larger balanced pool (--mi_reads_per_family), because the plug-in
mutual-information estimate is biased upward on a few hundred reads.

Mean |phi| is biased upward by Monte Carlo noise (the absolute value of a noisy
estimate of zero is positive), so the permutation count matters: --checkpoints
writes the attribution matrix at intermediate permutation counts so that
convergence can be inspected, and src/compare_shap_runs.py measures the
agreement between independently seeded runs.

Outputs:
  results/router_shap.csv         top k-mers per family: mean |phi|, signed mean
  results/router_shap_mi.csv      per-k-mer attribution vs mutual information
  results/router_shap_matrix.csv  full family x k-mer matrix of mean |phi|
  results/router_shap_stats.csv   summary statistics at each permutation count

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
    """Encoded reads for `split`, grouped by true family index.

    Reads are drawn evenly from every sample of a family (the same number from
    each, in shuffled sample order, until `reads_per_family` is reached), so a
    family is represented by all of its held-out genomes and not by the first
    one encountered.
    """
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

    rows_by_family = {}
    for row in rows:
        rows_by_family.setdefault(row["family_idx"], []).append(row)

    by_family = {}
    for fam in sorted(rows_by_family):
        fam_rows = rows_by_family[fam]
        rng.shuffle(fam_rows)
        per_sample = math.ceil(reads_per_family / len(fam_rows))
        have = []
        for row in fam_rows:
            if len(have) >= reads_per_family:
                break
            seqs = parse_fastq(row["r1_path"]) + parse_fastq(row["r2_path"])
            if not seqs:
                continue
            take = min(per_sample, reads_per_family - len(have), len(seqs))
            for s in rng.sample(seqs, take):
                have.append(encode_sequence_kmers(s, k, kmer_to_idx, vocab_size,
                                                  canonical=canonical))
        if have:
            by_family[fam] = torch.stack(have)
    return by_family


# ------------------------------------------------------- Shapley estimator

@torch.no_grad()
def shapley_for_read(model, x, backgrounds, target_class, checkpoints, device,
                     gen, chunk=32, max_rows=16384):
    """Permutation-sampling Shapley values of each feature for one read.

    x            (D,)    the explained read, binary
    backgrounds  (B, D)  pool of real reads to draw the reference from
    checkpoints          increasing permutation counts, e.g. [32, 128, 512]
    returns      list of (D,) tensors, phi after each checkpoint; each sums to
                 f(x) - E[f(background)] over the permutations drawn so far
    """
    D = x.numel()
    phi_sum = torch.zeros(D, device=device)
    out, done = [], 0
    for target in checkpoints:
        while done < target:
            c = min(chunk, target - done)
            b = backgrounds[torch.randint(len(backgrounds), (c,), device=device, generator=gen)]
            # Features already equal to the background can never change the
            # output, whatever order they are added in: their marginal
            # contribution is identically zero. Only the symmetric difference
            # is ordered; everything else is pushed to the end of the order.
            diff = b != x.unsqueeze(0)                                   # (c, D)
            keys = torch.rand(c, D, device=device, generator=gen)
            keys[~diff] = 2.0
            order = torch.argsort(keys, dim=1)                           # (c, D)
            rank = torch.empty_like(order)
            rank.scatter_(1, order, torch.arange(D, device=device).expand(c, D))
            M = int(diff.sum(1).max())
            if M > 0:
                # Coalition t of permutation p takes from x the features whose
                # rank is below t, and the rest from that permutation's
                # background: t = 0 is the background, t = m_p the read itself.
                t = torch.arange(M + 1, device=device).view(1, M + 1, 1)
                coalitions = torch.where(rank.unsqueeze(1) < t, x.view(1, 1, D), b.unsqueeze(1))
                flat = coalitions.reshape(-1, D)
                f = torch.cat([model(flat[i:i + max_rows])[:, target_class].float()
                               for i in range(0, flat.shape[0], max_rows)]).view(c, M + 1)
                # Marginal contribution of the feature at position t-1; zero by
                # construction beyond m_p, so the padding positions add nothing.
                phi_sum.index_add_(0, order[:, :M].reshape(-1), (f[:, 1:] - f[:, :-1]).reshape(-1))
            done += c
        out.append(phi_sum / max(done, 1))
    return out


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
    p.add_argument("--reads_per_family", type=int, default=60,
                   help="reads explained per family, drawn evenly from its samples")
    p.add_argument("--permutations", type=int, default=2048,
                   help="permutation samples per read; Monte Carlo error falls as 1/sqrt(n)")
    p.add_argument("--checkpoints", type=int, nargs="*", default=[32, 128, 512],
                   help="intermediate permutation counts at which statistics are also reported")
    p.add_argument("--background", type=int, default=128,
                   help="size of the background pool the reference reads are drawn from")
    p.add_argument("--mi_reads_per_family", type=int, default=5000,
                   help="balanced pool for mutual information and the background")
    p.add_argument("--top_k", type=int, default=15, help="k-mers listed per family")
    p.add_argument("--out_csv", default="results/router_shap.csv")
    p.add_argument("--mi_csv", default="results/router_shap_mi.csv")
    p.add_argument("--matrix_csv", default="results/router_shap_matrix.csv")
    p.add_argument("--stats_csv", default="results/router_shap_stats.csv")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else args.device if args.device != "auto" else "cpu")
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    ckpt, router, _ = load_checkpoint(args.checkpoint, device)
    router.eval()
    kmer_to_idx = ckpt["kmer_vocab"]
    vocab_size, k = ckpt["vocab_size"], ckpt["k"]
    canonical = ckpt.get("canonical", False)
    family_names = ckpt["family_names"]
    idx_to_kmer = {v: kk for kk, v in kmer_to_idx.items()}
    n_fam = len(family_names)

    print(f"router: {args.checkpoint}")
    print(f"  {n_fam} families, vocab {vocab_size}, k={k}, canonical={canonical}")
    print(f"  device={device}  seed={args.seed}")

    # Two independent draws from the same split: the reads to explain, and a
    # larger balanced pool for the background and the data-only statistics.
    by_family = load_reads_by_family(args.sample_manifest, args.split, k, kmer_to_idx,
                                     vocab_size, canonical, args.reads_per_family,
                                     random.Random(args.seed))
    print(f"  explained reads: {sum(len(v) for v in by_family.values())} "
          f"across {len(by_family)} families")
    pool = load_reads_by_family(args.sample_manifest, args.split, k, kmer_to_idx,
                                vocab_size, canonical, args.mi_reads_per_family,
                                random.Random(args.seed + 10007))
    pool_reads = torch.cat([v for v in pool.values()])
    pool_labels = torch.cat([torch.full((len(v),), f, dtype=torch.long)
                             for f, v in pool.items()])
    print(f"  balanced pool: {len(pool_reads)} reads, "
          f"{min(len(v) for v in pool.values())} to {max(len(v) for v in pool.values())} per family")
    bg_idx = torch.randperm(len(pool_reads))[: args.background]
    backgrounds = pool_reads[bg_idx].to(device)

    # -------- attribution
    D = vocab_size
    checkpoints = sorted(set(c for c in args.checkpoints if 0 < c < args.permutations)) \
        + [args.permutations]
    abs_sum = torch.zeros(len(checkpoints), n_fam, D, device=device)
    signed_sum = torch.zeros(n_fam, D, device=device)
    counts = torch.zeros(n_fam, device=device)

    for fam in sorted(by_family):
        reads = by_family[fam].to(device)
        for i in range(len(reads)):
            phis = shapley_for_read(router, reads[i], backgrounds, fam, checkpoints, device, gen)
            for c, phi in enumerate(phis):
                abs_sum[c, fam] += phi.abs()
            signed_sum[fam] += phis[-1]
            counts[fam] += 1
        print(f"  {family_names[fam]:<18} {int(counts[fam]):4d} reads explained", flush=True)

    denom = counts.clamp(min=1).view(1, n_fam, 1)
    mean_abs_all = (abs_sum / denom).cpu()                # (checkpoints, families, D)
    mean_abs = mean_abs_all[-1]
    mean_signed = (signed_sum / counts.clamp(min=1).unsqueeze(1)).cpu()

    # -------- independent check: mutual information, model never consulted
    mi = mutual_information(pool_reads, pool_labels, n_fam)
    global_attr = mean_abs.mean(0)
    rho = spearman(global_attr, mi)

    # -------- convergence: mean |phi| is inflated by Monte Carlo noise, so the
    # statistics are reported at every permutation count, not only the last
    print(f"\n  {'permutations':>12}{'median |phi|':>14}{'max |phi|':>11}"
          f"{'max/median':>12}{'Spearman vs MI':>16}")
    stats = []
    for c, n_perm in enumerate(checkpoints):
        g = mean_abs_all[c].mean(0)
        row = {"permutations": n_perm, "median_abs_phi": float(g.median()),
               "max_abs_phi": float(g.max()),
               "max_over_median": float(g.max() / g.median()),
               "spearman_vs_mutual_information": spearman(g, mi)}
        stats.append(row)
        print(f"  {n_perm:12d}{row['median_abs_phi']:14.3f}{row['max_abs_phi']:11.3f}"
              f"{row['max_over_median']:12.2f}{row['spearman_vs_mutual_information']:16.3f}")

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
    print(f"\nwrote {args.out_csv}")

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
                                            for fam in range(n_fam)])
    print(f"wrote {args.matrix_csv}  (full {n_fam} x {D} matrix)")

    with open(args.stats_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
        w.writeheader()
        for row in stats:
            w.writerow({kk: (f"{v:.6f}" if isinstance(v, float) else v) for kk, v in row.items()})
    print(f"wrote {args.stats_csv}")

    # The single most and least informative k-mers, and what the top one is
    # actually worth: these are the concrete figures quoted in the paper, so
    # they are reported here rather than derived by hand. Population: the
    # balanced pool, so every family has the same prior.
    top_j = int(torch.argmax(mi)); low_j = int(torch.argmin(mi))
    present = pool_reads[:, top_j] > 0
    bits = math.log2(n_fam)
    print(f"\nMost informative k-mer: {idx_to_kmer[top_j]} at {mi[top_j]:.4f} bits "
          f"({mi[top_j]/bits*100:.2f}% of the {bits:.0f}-bit family question), "
          f"present in {float(present.float().mean())*100:.1f}% of {len(pool_reads)} reads")
    print(f"Least informative k-mer: {idx_to_kmer[low_j]} at {mi[low_j]:.5f} bits "
          f"({mi[low_j]/bits*100:.3f}%)")
    if present.any():
        print(f"  {'family':<18}{'P(family)':>11}{'P(family | present)':>22}")
        for f, name in enumerate(family_names):
            prior = float((pool_labels == f).float().mean())
            post = float((pool_labels[present] == f).float().mean())
            print(f"  {name:<18}{prior*100:10.1f}%{post*100:21.1f}%")

    print("\nTop k-mers per family:")
    for fam, name in enumerate(family_names):
        order = torch.argsort(mean_abs[fam], descending=True)[:6]
        tops = ", ".join(f"{idx_to_kmer[j]}({mean_abs[fam][j]:.2f})" for j in order.tolist())
        print(f"  {name:<18} {tops}")


if __name__ == "__main__":
    main()
