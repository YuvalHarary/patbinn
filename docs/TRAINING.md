# Training procedure

Everything needed to reproduce the trained checkpoints: optimizer, schedule,
stopping rule, how class imbalance is handled, what differs between the binary
and full precision runs, and where determinism is and is not guaranteed.

Entry points are `src/run_cv.py` (drives the five folds) and `src/train_moe.py`
(trains one fold). See [PIPELINE.md](PIPELINE.md) for how the reads and
manifests are produced, and [ARCHITECTURE.md](ARCHITECTURE.md) for the network
shapes.

## Stage-wise training

The hierarchical classifier is not trained end to end. Each stage is trained
separately, against its own true labels:

```
fold manifest (train / val / test)
        |
        v
1. build k-mer vocabulary        <- train split only
        |
        v
2. train the family router       <- labels: family, 8-way
        |
        v
3. train 8 per-family experts    <- labels: variant, one expert per family,
   (independently, in a loop)       on that family's reads by TRUE family label
        |
        v
combined checkpoint: checkpoints/<run>/fold<N>.pth
```

Because an expert only ever sees reads selected by their true family label, an
expert is independent of the router and stays valid if the router is retrained.
That is what makes the router and expert sweeps separable, and it is why the
router sweep can restore experts from the cross-validation checkpoint of the
same fold instead of retraining them.

## Hyperparameters

These are the values stored in the `args` key of every checkpoint, so any
checkpoint can be interrogated for the settings it was actually trained with.

| Setting | Value | Source |
|---|---|---|
| Optimizer | SGD, momentum 0.9, weight decay 1e-4 | `train_moe.py:74` |
| Learning rate | 0.1 | `--lr` |
| LR schedule | `ReduceLROnPlateau`, on validation accuracy, patience 5, factor 0.1 | `train_moe.py:75` |
| Loss | cross entropy | `train_moe.py:76` |
| Batch size | 512 | `--batch_size` |
| Max epochs | 30 per stage | `--router_epochs`, `--expert_epochs` |
| Early-stopping patience | 10 epochs | `--patience` |
| Reads per sample, training | 2000 | `--max_reads_per_sample_train` |
| Reads per sample, vocabulary | 500 | `--max_reads_per_sample_for_vocab` |
| k | 5 | `--k` |
| Vector length | 1024 bits (512 when `--canonical`) | `--max_kmers` |
| Router shape | 1024 x 128 | `--router_hidden_sizes` |
| Expert shape | 1024 x 128 | `--expert_hidden_sizes` |
| Folds | 5 | `--k_folds` |

The learning rate and patience are shared by the router and the expert stages.
Only the epoch budget is set separately, and both default to 30.

## Early stopping and model selection

Validation accuracy is evaluated after every epoch. The rule in
`train_moe.py:88-99` is:

* if validation accuracy improves on the best seen so far, the full state dict
  is cloned as the new best and the patience counter resets to 0;
* otherwise the counter increments, and training stops once it reaches 10;
* when the loop ends, by budget or by patience, the best state dict is loaded
  back into the model.

So the saved weights are the best-validation epoch, never the last epoch. The
accuracy recorded in the checkpoint (`router_val_acc`, `expert_val_accs`) is
that best validation accuracy, not a test figure.

The LR scheduler and the stopping rule both watch validation accuracy but with
different patience, 5 and 10. The learning rate therefore drops by 10x once
before early stopping can trigger.

## Class imbalance

Classes are unbalanced at two levels: isolates per class range from 3 to 10,
and because coverage is fixed, reads per sample scale with genome length, so a
125 to 236 kb herpesvirus genome contributes far more reads than a 3 kb
hepadnavirus one.

Two mechanisms address this during training:

* **Per-sample read cap.** At most 2000 reads are drawn from any one sample,
  which bounds the contribution of the longest genomes.
* **Balanced sampler.** Training loaders use a `WeightedRandomSampler` with
  weight `1/count` per class, drawing `len(dataset)` samples with replacement
  (`moe_dataset.py:269-279`). The expected class frequency within a batch is
  therefore equal.

Validation loaders are deliberately **not** balanced and not shuffled, so
validation accuracy is measured on the natural class mix.

Neither mechanism applies at evaluation time. The 2000-read cap is a training
setting only, and reported per-read accuracy is a micro-average over every
held-out read at the natural family mix. See [RESULTS.md](RESULTS.md).

## Binary versus full precision runs

`--precision binary` and `--precision float32` share the schedule, shapes and
sampler. They differ in one respect:

* **Latent weight clipping.** In binary runs, all non-BatchNorm parameters are
  clipped to `[-1.5, 1.5]` after every optimizer step
  (`training_utils.py:47-53`). These latent full precision weights exist only
  during training and are discarded at inference in favour of their binarized
  values. The clip keeps them inside the range over which the straight-through
  estimator passes gradient.
* The clip is **disabled** for the full precision baseline, so that the
  precision comparison is not confounded by it (`train_moe.py:215`).

## Splits and vocabulary

Folds are assigned per isolate at retrieval time, in round-robin order, and
every read simulated from an isolate inherits its fold. For a given test fold:

```
test  = this fold
val   = (this fold + 1) % k_folds        run_cv.py:38
train = everything else
```

So no read of a held-out genome appears in training, and the validation
genomes are also disjoint from both.

The k-mer vocabulary is built from the **train split only**
(`train_moe.py:195-208`), capped at 500 reads per sample, and cached per fold
at `data/vocab_<run>/kmer_vocab_fold<N>.json`. Validation and test genomes
never contribute to vocabulary selection.

## Determinism and seeds

**No random seed is set anywhere in the training path.** `run_cv.py`,
`train_moe.py`, `moe_dataset.py` and `training_utils.py` do not call
`torch.manual_seed`, `random.seed` or `np.random.seed`. Consequently the
following are not reproducible run to run:

* weight initialization;
* the draw order of the `WeightedRandomSampler`;
* which 2000 reads are taken from a sample when it holds more.

What *is* deterministic is the data partition. Fold assignment is round-robin
over isolate index, not random, so the train/val/test split of genomes is
identical on every run and is recorded in
`data/manifests_<run>/cv_fold<N>_sample_manifest.csv`. Reported fold-to-fold
standard deviations therefore reflect which genomes were withheld, which is
the intended comparison, and not a reshuffled split.

Exact numerical reproduction of a published checkpoint is not possible from
source for this reason. The checkpoints themselves are the reproducible
artifact, and every accuracy figure in [RESULTS.md](RESULTS.md) can be
recomputed from them. Seeding the training path is a known gap rather than a
deliberate choice.

Two scripts that *do* seed, because their sampling is the measurement rather
than an incidental detail, are `src/explain_router.py` (`--seed`, default 0)
and `src/sweep_maj_and.py`.

## Software environment

| Component | Version |
|---|---|
| PyTorch | 2.5.1+cu124 |
| CUDA | 12.4 |
| Python | 3.11.10 |

The container recipe is `patbinn.def`; `requirements.txt` pins the Python
dependencies. All runs reported in [RESULTS.md](RESULTS.md) were produced
inside that container on a single NVIDIA H100.

## Reproducing

Five-fold cross validation, hierarchical binary classifier:

```bash
apptainer exec --nv --bind $(pwd):/workspace --pwd /workspace patbinn.sif \
    python3 src/run_cv.py \
        --classifier hierarchical \
        --precision binary \
        --k_folds 5
```

Swap `--precision float32` for the full precision counterpart, or
`--classifier flat` for the 58-way baseline. Add `--canonical` for the
strand-invariant 512-bit encoding.

Per-fold checkpoints land in `checkpoints/<run>/fold<N>.pth` and per-fold
result JSONs in `results/<run>/fold<N>_result.json`. Aggregate them with:

```bash
python3 src/aggregate_cv_results.py --results_dir results/cv_hierarchical_binary
```

To read back the settings a checkpoint was trained with:

```python
import torch
ck = torch.load("checkpoints/cv_hierarchical_binary/fold0.pth",
                map_location="cpu", weights_only=False)
print(vars(ck["args"]) if hasattr(ck["args"], "__dict__") else ck["args"])
```
