# Results

Every number here is read directly from the CSV/JSON files in
[`../results/`](../results/). Nothing is rounded from memory; the file backing
each table is named above it.

Unless stated otherwise:

- **Per-read** accuracy is over individual 150 bp reads.
- **Specimen-level** accuracy is the per-genome majority vote over that
  specimen's reads: the operational number, since you classify a sample rather
  than a read.
- All cross-validation is **genome-level**: whole isolates are withheld, never
  merely some of their reads.

**Correction, 2026-11-08.** One genome, `NC_024781.1` (Ravn virus), had been
retrieved under both `Filoviridae/Marburg` and `Filoviridae/Ravn`, byte for byte
identical. The Marburg copy was removed: the dataset is 542 genomes and 3,252
samples. The duplicate sat in fold 1's test set, so every fold-1 figure below was
re-evaluated on 654 specimens with the existing checkpoints; fold 1's training
set never contained it. The test sets of folds 0, 2, 3 and 4 are unchanged.
Their models were trained with the duplicate present, labelled Marburg, which
cannot inflate their test numbers. Before the correction the model called all
six fold-1 replicates of that genome Ravn, which they are, and was scored wrong
for each; those six were 40% of fold 1's specimen errors.

---

## 1. Main cross-validation result

*Source: `results/cv_results.csv`, `results/cv_results_float32.csv`,
`results/cv_results_flat.csv`*

5-fold mean ± standard deviation:

| Condition | Per-read, family (router) | Per-read, leaf (pipeline) | Per-read, variant (oracle-routed) | Specimen-level |
|---|---|---|---|---|
| **Hierarchical MoE, binary** | **97.76% ± 0.69** | **90.09% ± 1.74** | **91.91% ± 1.60** | **99.72% ± 0.62** |
| Hierarchical MoE, float32 | 99.57% ± 0.22 | 98.60% ± 0.48 | 98.87% ± 0.35 | 99.91% ± 0.14 |
| Flat 58-way, binary | n/a | 43.55% ± 3.19 | n/a | 88.94% ± 4.17 |

**The pipeline column is the per-read comparison to make.** Each read is routed
by the router's own prediction and both family and variant must be right, which
is the 58-way task the flat classifier solves. Oracle routing hands the expert
the true family and measures an easier task; it is reported to separate expert
quality from router quality, not to flatter the hierarchy.

Per-fold specimen-level accuracy, hierarchical binary:

| Fold | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Test specimens | 678 | 654 | 654 | 636 | 630 |
| Specimen accuracy | 100.00% | 98.62% | 100.00% | 100.00% | 100.00% |
| Family accuracy | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

Family routing is perfect at the specimen level in every fold: all residual
error is within-family variant confusion, concentrated in fold 1.

### Three findings

1. **The hierarchy is doing real work.** 99.72% vs 88.94% specimen-level, and
   the per-read gap is far wider. A flat 58-way softmax must separate a
   Flaviviridae variant from a Coronaviridae variant using the same weights
   that separate two Coronaviridae variants from each other; splitting those
   two very different similarity scales is what buys the improvement.
2. **Binarization is close to free.** 0.18 points at the specimen level
   (99.72% vs 99.91%), inside the binary network's fold-to-fold noise, even though every weight
   and activation is constrained to ±1. Per read the binary penalty is real and
   visible (90.09% vs 98.60% in pipeline mode); majority voting absorbs almost all
   of it.
3. **Majority voting is the mechanism.** The jump from ~90% per read to ~99.7%
   per specimen is the entire operational case for this design. Section 5
   quantifies how many reads that actually requires.

---

## 2. Router size and depth

*Source: `results/router_sweep.csv` (fold 1). Experts are restored from an
existing checkpoint, so expert quality is held fixed and only the router
varies.*

| Router shape | Layers | Parameters | Per-read family | Specimen-level |
|---|---|---|---|---|
| 64 | 1 | 66,305 | 70.16% | 81.04% |
| 128 | 1 | 132,609 | 64.32% | 93.43% |
| 256 | 1 | 265,217 | 67.36% | 96.79% |
| 512 | 1 | 530,433 | 84.79% | 93.43% |
| 128×64 | 2 | 140,546 | 75.59% | 98.62% |
| 256×128 | 2 | 297,474 | 82.20% | 98.62% |
| **1024×128** | 2 | 1,185,282 | **97.11%** | **98.62%** |
| 1024×512×128 | 3 | 1,646,083 | 97.99% | 98.62% |

**The second hidden layer is a requirement, not a refinement.** No
single-hidden-layer shape reaches any two-layer shape at the specimen level,
and the best one-layer router (512, at 530k parameters) is beaten by the
smallest two-layer router (128×64, at 141k), a quarter of the parameters.

**Beyond two layers buys nothing.** A third hidden layer adds 0.87 points per
read and exactly zero at the specimen level, for 460k additional parameters.

**Specimen-level accuracy saturates long before per-read accuracy does.**
Every two-layer router lands on the identical 98.62%, while per-read accuracy
still climbs from 75.59% to 97.99% across that range. Majority voting absorbs
the entire 22-point per-read deficit, so a router chosen on per-read accuracy
alone would be sized far larger than the task actually needs.

---

## 3. Expert size and depth

*Source: `results/expert_sweep.csv` (fold 1). The router is held fixed, so only
expert capacity varies.*

| Expert shape | Layers | Parameters (all 8) | Per-read variant (oracle) | Specimen-level |
|---|---|---|---|---|
| 128 | 1 | 1,060,104 | 63.16% | 96.64% |
| 256 | 1 | 2,120,200 | 73.35% | 97.86% |
| 512 | 1 | 4,240,392 | 69.96% | 96.33% |
| 1024 | 1 | 8,480,776 | 75.89% | 97.71% |
| 256×128 | 2 | 2,379,024 | 82.96% | 98.01% |
| 512×128 | 2 | 4,746,512 | 85.78% | 99.54% |
| **1024×128** | 2 | 9,481,488 | 89.87% | **99.85%** |
| 1024×256 | 2 | 10,541,584 | 91.93% | 99.08% |

Two layers beat one on the expert side as well, at every width. `1024×128` is
the best specimen-level shape; `1024×256` buys 2 points per read but *loses*
0.76 points per specimen for a further million parameters, so the extra
capacity is not converting into better specimen calls.

`512×128` is the interesting economy point: half the parameters of `1024×128`
for 0.31 points less at the specimen level.

---

## 4. Input encoding

*Source: `results/encoding_variants.csv` (fold 1). Each row is a full
retrain (router and experts) at that encoding.*

| Encoding | Vocabulary | Total parameters | Per-read family | Per-read variant (oracle) | Specimen-level |
|---|---|---|---|---|---|
| k=5, 1024 (baseline) | 1024 | 10,666,770 | 97.37% | 94.48% | 98.62% |
| k=6, 1024 | 1024 | 10,666,770 | 96.92% | 76.50% | 99.39% |
| **k=5 canonical, 512** | **512** | **5,948,178** | **97.96%** | 92.38% | **99.24%** |
| k=4, 256 | 256 | 3,588,882 | 75.42% | 74.73% | 95.41% |

**Canonical k=5 is the result worth acting on.** Collapsing every k-mer with
its reverse complement halves the vector to 512 dimensions, which removes
**44%** of the classifier's parameters (10.67M → 5.95M) while *improving*
specimen-level accuracy (98.62% → 99.24%) and per-read family accuracy
(97.37% → 97.96%).

Why it works: reads come off both strands (measured 211 forward / 177 reverse
in 400 SARS-CoV-2 reads), and the plain encoding is verifiably **not**
strand-invariant: the same locus read from opposite strands produces two
disjoint k-mer sets, and the network has to learn that equivalence from data
instead of getting it for free. Odd `k` admits no self-complementary k-mer, so
1024 5-mers map onto exactly 512 with no collisions.

**k=6 is not an improvement**: the same parameter count, worse per-read variant
accuracy (76.50% vs 94.48%), because a 1024-word vocabulary covers a far
smaller fraction of the 4096 possible 6-mers. **k=4 is too coarse**: 256
dimensions cannot separate 58 classes, and everything degrades together.

---

## 5. How many reads does a confident call need?

*Source: `results/read_count_sweep.csv` (fold 1). Pure inference; the reads
contributing to each specimen's majority vote are capped at N.*

| Reads voted | 1 | 5 | 10 | 25 | 50 | 100 | 200 | 500 | 1000 |
|---|---|---|---|---|---|---|---|---|---|
| Specimen accuracy | 90.37% | 95.57% | 97.55% | 98.01% | 98.78% | 98.47% | 98.47% | 98.62% | 98.47% |

**The specimen decision saturates at roughly 50 reads**, at ~98.5-98.8%. A
specimen at 30× coverage supplies thousands of reads, so well over 99% of the
available reads are surplus to the decision, which is the entire argument for
deploying this on a small, power-constrained device.

Accuracy is not perfectly monotonic past saturation (98.78% at best, 98.47% at
worst between 50 and 1000 reads). That is a 2-specimen difference out of 654, i.e.
±0.2 points of vote-order noise, not a real decline.

---

## 6. Robustness to sequencing error

*Source: `results/error_rate_sweep.csv` (fold 4, 630 specimens). Substitution
errors are injected into held-out reads at inference time; no retraining.*

| Injected error | 0% | 2% | 4% | 6% | 8% | 10% |
|---|---|---|---|---|---|---|
| Per-read leaf | 94.62% | 86.90% | 74.03% | 59.37% | 46.40% | 35.33% |
| Specimen-level | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 97.46% |

Per-read accuracy collapses roughly linearly with substitution rate, which is expected:
a single substitution corrupts up to `k` k-mers at once, and the
bag-of-k-mers vector degrades globally rather than losing one feature.

**Majority voting is remarkably effective at hiding this.** Specimen-level
accuracy is perfect through 8% injected error, where barely half of individual
reads are still classified correctly, and is still 97.46% at 10%, a
substitution rate far above any real Illumina run.

---

## 7. Comparison with Kraken2

*Source: `results/kraken2_error_sweep.csv`,
`results/kraken2_sample_acc_normalized.csv`,
`results/patbinn_fold1_error_sweep.csv`,
`results/canonical_fold1_error_sweep.csv`; all fold 1, 654 specimens.*

Kraken2's database is built from exactly the genomes PatBiNN was allowed to
train on (every genome outside the held-out test fold), so neither method has
seen a test isolate. Both are scored on byte-identical reads, corrupted by the
same function with the same seed.

| Error | Kraken2 per-read | PatBiNN per-read | Kraken2 unclassified | Kraken2 lenient | Kraken2 fair | PatBiNN specimen | Canonical specimen |
|---|---|---|---|---|---|---|---|
| 0% | 87.89% | **90.57%** | 8.20% | 97.55% | 91.13% | **98.47%** | 99.08% |
| 2% | **84.84%** | 81.53% | 10.51% | 98.17% | 91.59% | **98.47%** | 98.78% |
| 4% | **79.47%** | 68.35% | 15.15% | 98.17% | 88.53% | 98.17% | 98.47% |
| 6% | **68.77%** | 54.32% | 25.42% | 97.25% | 81.50% | 96.64% | 98.01% |
| 8% | **53.83%** | 42.30% | 40.83% | 95.87% | 73.09% | **96.18%** | 94.95% |
| 10% | **37.92%** | 32.47% | 57.91% | 95.11% | 33.64% | 94.04% | 76.76% |

Two scoring conventions are reported for Kraken2 because it can decline to
classify a read:

- **Lenient**: Kraken2's own convention: the specimen vote runs only over the
  reads it chose to classify. Unclassified reads are ignored.
- **Fair**: unclassified reads are counted as errors, the same standard
  PatBiNN is held to, since PatBiNN has no abstention mechanism and must answer
  for every read.

### This is a parity claim, and should not be upgraded to a win

- On **clean reads** the two are close, with PatBiNN slightly ahead per read
  (90.57% vs 87.89%) and at the specimen level (98.47% vs 97.55% lenient).
- **Under noise Kraken2 is clearly more robust per read**: 74.69% vs 61.22% at
  5% error, and the gap widens from 2% onward. Kraken2 needs a single surviving
  exact k-mer match; a bag-of-k-mers vector degrades globally. This is a real
  and material advantage for Kraken2.
- Kraken2's failure mode is **abstention** (8% → 58% unclassified), not
  misclassification. Under its own lenient convention it therefore looks nearly
  flat across the whole sweep; under the fair convention it falls off a cliff
  at 10%. Neither view alone is honest on its own, which is why both are
  reported.

The supportable claim is **comparable accuracy from a far smaller, wholly
binary computation** (a network of ~6-10M binary weights against a multi-GB
k-mer database), not superior accuracy.

Note the canonical 512-bit encoding tracks or beats the 1024-bit model
everywhere up to 8% error, then **collapses at 10%** (76.76%): halving the
activation vector leaves too little redundancy to absorb that much corruption.

---

## 8. Limitations

Stated plainly, because they bound what these numbers mean.

**Reads are simulated, not sequenced.** Every read comes from ART's Illumina
error model applied to a real NCBI genome. Real runs carry adapter
contamination, PCR duplicates, coverage bias and host background that ART does
not reproduce. The error-rate sweep in section 6 is a stress test for
substitutions only; it is not a substitute for real sequencing data.

**Class balance is uneven by nature.** Families range from 3 to 8 variants and
from 21 to 80 genomes (Retroviridae is the smallest at 3 variants / 21
genomes). Specimen-level accuracy is dominated by the larger families.

**Closed-world assumption.** The classifier always returns one of 58 classes.
There is no "none of the above" output, so a read from an unrepresented
organism is silently forced into the nearest class. Kraken2, by contrast, can
abstain, which is exactly why section 7 reports both scoring conventions. An
abstention mechanism is the most obvious next addition.

**Single-fold sweeps.** Sections 2, 3, 4, 5 and 7 are measured on fold 1
only, not across all five folds, because each sweep point is a full retrain.
They are reliable for ranking shapes and encodings against each other; their
absolute values carry single-fold variance and should not be quoted as
five-fold results.

**Specimen-level accuracy is a majority vote over many reads.** It is the right
operational metric, but it is not comparable to a per-read number from another
tool. Wherever an external classifier is compared, per-read and specimen-level
figures are reported side by side and the oracle-routed figure is excluded.
