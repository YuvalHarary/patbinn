# Results

Every experiment's output. These files are small, checked in, and are the
actual scientific record: the tables in
[docs/RESULTS.md](../docs/RESULTS.md) are read directly from them.

Figures are **not** checked in; regenerate them with
`python3 src/plot_results.py`.

## Cross-validation (5 folds, genome-level)

| Path | Contents |
|---|---|
| `cv_hierarchical_binary/fold{0..4}_result.json` | Per-fold metrics, hierarchical MoE, binary. **The main result** |
| `cv_hierarchical_float32/fold{0..4}_result.json` | Per-fold metrics, float32 precision control |
| `cv_flat_binary/fold{0..4}_result.json` | Per-fold metrics, flat 58-way baseline |
| `cv_results.csv` | The five hierarchical-binary folds aggregated |
| `cv_results_float32.csv` | The five float32 folds aggregated |
| `cv_results_flat.csv` | The five flat-baseline folds aggregated |

Aggregated by `src/aggregate_cv_results.py`. Metric columns:

- `per_read_router_acc`: family accuracy on individual reads
- `per_read_expert_acc_oracle`: variant accuracy **given the true family**.
  An easier task than end-to-end classification; not comparable to another
  classifier's per-read accuracy.
- `per_read_leaf_acc` *(flat only)*: family and variant both correct, per read
- `sample_family_acc`: family correct, per specimen majority vote
- `sample_leaf_acc_pipeline`: family and variant both correct, per specimen,
  using the **real router**. This is the headline metric.
- `sample_variant_acc_oracle`: per specimen, with oracle routing

## Sweeps (fold 1 unless noted)

| File | Contents |
|---|---|
| `router_sweep.csv` | 8 router shapes, 1-3 hidden layers. Experts held fixed, so only router capacity varies. |
| `expert_sweep.csv` | 8 expert shapes, 1-2 hidden layers. Router held fixed. |
| `cv_perread_leaf.csv` | Pipeline-mode per-read accuracy per fold: each read routed by the router's own prediction, family and variant both required. The number comparable to the flat baseline. |
| `cv_class_metrics.csv` | Precision, recall and F1 per leaf class, pooled over the five folds. |
| `cv_family_confusion.csv`, `cv_family_readcounts.csv` | Family-level read confusion and held-out reads per family, pooled over folds. |
| `f32_perread/` | The same four files for the float32 counterpart. |
| `encoding_table.csv` | The four encodings in the layout `plot_paper_figures.py` reads. |
| `encoding_variants.csv` | Input encodings: k=4/5/6, canonical on/off. Each row is a full retrain. Where a label appears twice, the **last** row is current. |
| `read_count_sweep.csv` | Specimen accuracy vs number of reads voted (1-1000). Covers **all five folds**. |
| `error_rate_sweep.csv` | Injected substitution error 0-10%, **fold 4**. |
| `router_shap.csv`, `router_shap_mi.csv`, `router_shap_matrix.csv` | Shapley attribution of the fold-1 router (`src/explain_router.py`, seed 0): 480 reads drawn from all held-out samples, 2,048 permutations per read. Top k-mers per family, attribution against mutual information over a balanced pool of 40,000 reads, and the full family by k-mer matrix. |
| `router_shap_stats.csv` | Median and maximum attribution at 32, 128, 512 and 2,048 permutations. Mean absolute attribution is inflated by sampling noise at low counts. |
| `router_shap_matrix_seed1.csv`, `router_shap_agreement.csv` | An independent run with seed 1, and its agreement with seed 0 (`src/compare_shap_runs.py`): Spearman 0.907 over the 1,024 k-mers. |
| `router_abstention.csv`, `router_abstention_fractions.csv`, `router_margin.csv` | Router decision margin over all 4,183,826 held-out reads of fold 1 (`src/abstain_router.py`): accuracy on the reads kept and errors removed, by margin threshold and at fixed fractions withheld, and margin statistics for correct and misrouted reads. |

## Kraken2 comparison (fold 1, 654 specimens)

| File | Contents |
|---|---|
| `kraken2_error_sweep.csv` | Kraken2 on the error sweep: per-read accuracy, and `pct_unclassified`, the share of reads it declines to classify. |
| `kraken2_sample_acc_normalized.csv` | Kraken2 specimen accuracy under both conventions: `sample_variant_acc_lenient` (votes only over reads it classified) and `sample_variant_acc_strict` (unclassified counted as errors; the **fair** comparison, since PatBiNN cannot abstain). |
| `patbinn_fold1_error_sweep.csv` | PatBiNN on byte-identical reads, 1024-bit encoding. `per_read_leaf_acc` is the metric comparable to Kraken2's per-read accuracy. |
| `canonical_fold1_error_sweep.csv` | The same sweep for the canonical 512-bit encoding. |

**Read the Kraken2 numbers as a parity result, not a win.** Kraken2 is clearly
more robust per read under noise; PatBiNN is slightly ahead on clean reads and
at the specimen level. See
[docs/RESULTS.md](../docs/RESULTS.md#7-comparison-with-kraken2).

## Sweeps whose output is not shipped

`src/sweep_architecture.py` (joint router × expert grid, writing `size_sweep.csv`)
and `src/sweep_cross_combos.py` (interaction check) still run, but their outputs are
**not** checked in. The joint grid was the original approach and was superseded
by the separate `src/sweep_router.py` and `src/sweep_experts.py`, which vary one
component at a time against a fixed counterpart and are therefore easier to
interpret. The cross-combo run exists only to confirm that the two components do
not interact, which justifies sweeping them independently; it is a check, not a
result.

Both regenerate on demand if you want them.

## Reproducing any of these

Each file is regenerated by the script of the same name: `router_sweep.csv` by
`src/sweep_router.py`, `expert_sweep.csv` by `src/sweep_experts.py`, and so on. The
Kraken2 files come from `src/compare_kraken2.py --build` followed by
`--classify`. See [docs/EXPERIMENTS.md](../docs/EXPERIMENTS.md#status) for the
full mapping.

## A note on superseded files

Two retrieval bugs in `data/download_genomes.py` were fixed partway through the
project, and every result was re-run on the corrected dataset. Results produced
before that fix have been removed rather than kept alongside the current ones,
because the two are not comparable and mixing them is how wrong numbers end up
in a paper. Anything here is from the corrected dataset: 542 genomes after the
removal of one duplicated, mislabelled genome (see `docs/RESULTS.md`).

The current test-set sizes are 678 / 654 / 654 / 636 / 630 specimens for folds
0-4; a file reporting any other count is from a superseded dataset.
