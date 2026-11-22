# Part 2: training sweep

Part 1 ([README.md](README.md)) found that a corpus fine-tune gave potion-base-8M only +0.004 to
+0.006 NDCG@10. This sweep asks why, and how much better the fine-tune can be. It changes, one
at a time:

1. **Training length.** Train to convergence and keep the best checkpoint.
2. **Learning rate and start-model scale.** Learning rates from 1e-3 to 3e-1, and potion-base-8M
   with its vectors divided by 10.5.
3. The same grid on all three datasets.
4. **Target.** bge's CLS embedding (its search embedding) instead of its mean token vector.
5. **More training text.** Train-split queries and sentences of the corpus documents.
6. **Retrain on 100% of the corpus** for the number of epochs found with the 10% hold-out.

Step 7, added during the run: **C4 replay**. The new recipe forgot a distant domain (FiQA), so a
variant mixes potion's general training text (C4) into the fine-tune.

## Results

### Test NDCG@10 (main result)

Test queries only, as in Part 1. "new" is the setting chosen on dev queries for each dataset
(below), 5 seeds (mean ± sample std).

| Model | SciFact | NFCorpus | FiQA |
|---|---|---|---|
| 1: bge-base-en-v1.5 (CLS pooling) | 0.7451 | 0.3667 | 0.3916 |
| 1m: bge-base-en-v1.5 (mean pooling) | 0.7601 | 0.3733 | 0.3984 |
| 2: distilled (256d) | 0.4391 | 0.1954 | 0.1153 |
| 3: potion-base-8M | 0.4879 | 0.2308 | 0.1662 |
| 4: Part 1, potion + fine-tune | 0.4937 ± 0.0014 | 0.2348 ± 0.0005 | 0.1705 ± 0.0010 |
| 4b: Part 1, distilled + fine-tune | 0.4967 ± 0.0041 | 0.2365 ± 0.0033 | 0.1809 ± 0.0012 |
| **new** | **0.6534 ± 0.0055** | **0.3253 ± 0.0026** | **0.2279 ± 0.0051** |
| new, retrained on 100% of the corpus | 0.6611 ± 0.0037 | 0.3251 ± 0.0043 | 0.2308 ± 0.0036 |
| new + C4 replay | 0.6514 ± 0.0074 | 0.3092 ± 0.0057 | 0.2236 ± 0.0017 |

At @10, the new models are also better on recall, MRR and precision:

| Model | SciFact R / MRR / P | NFCorpus R / MRR / P | FiQA R / MRR / P |
|---|---|---|---|
| 1: bge (CLS) | 0.8686 / 0.7092 / 0.0980 | 0.1775 / 0.5737 / 0.2752 | 0.4590 / 0.4742 / 0.1106 |
| 4b: Part 1 distilled | 0.6477 / 0.4552 / 0.0719 | 0.1111 / 0.4086 / 0.1800 | 0.2445 / 0.2166 / 0.0553 |
| new | 0.7971 / 0.6146 / 0.0887 | 0.1592 / 0.5038 / 0.2487 | 0.3013 / 0.2714 / 0.0695 |
| new, 100% corpus | 0.8054 / 0.6221 / 0.0895 | 0.1591 / 0.4999 / 0.2478 | 0.3061 / 0.2750 / 0.0705 |

`results/final/tables.md` has all four metrics at k = 1..10 for every model, and every paired
comparison at k = 1, 5 and 10.

Paired bootstrap 95% CIs over test queries (NDCG@10; a multi-seed model's per-query score is its
mean over seeds):

| A − B | SciFact | NFCorpus | FiQA |
|---|---|---|---|
| new − 4 (Part 1, potion) | +0.160 [+0.124, +0.196] | +0.091 [+0.073, +0.109] | +0.057 [+0.041, +0.074] |
| new − 4b (Part 1, distilled) | +0.157 [+0.124, +0.190] | +0.089 [+0.073, +0.106] | +0.047 [+0.034, +0.061] |
| 100% corpus − new | +0.008 [+0.002, +0.014] | −0.000 [−0.004, +0.004] | +0.003 [−0.000, +0.006] |
| C4 replay − new | −0.002 [−0.012, +0.008] | −0.016 [−0.027, −0.007] | −0.004 [−0.010, +0.001] |
| 1 (bge) − new | +0.092 [+0.064, +0.120] | +0.041 [+0.025, +0.059] | +0.164 [+0.141, +0.186] |

The gap to bge-base-en-v1.5 is now 0.09 (SciFact), 0.04 (NFCorpus) and 0.16 (FiQA) NDCG@10. In
Part 1 it was 0.25, 0.13 and 0.21.

### The final settings

| Dataset | Start model | Target | Learning rate | Extra text | Dev NDCG@10 (5 seeds) | Best epoch (mean) | 100% retrain epochs |
|---|---|---|---|---|---|---|---|
| SciFact | distilled | CLS | 1e-1 | train queries + sentences | 0.6674 ± 0.0036 | 20.2 | 21 |
| NFCorpus | potion ÷ 10.5 | mean | 3e-2 | none | 0.3208 ± 0.0044 | 23.0 | 24 |
| FiQA | distilled | mean | 3e-1 | train queries | 0.2183 ± 0.0036 | 16.6 | 18 |

Several settings tie within seed noise on each dataset (see step 2). The table shows the one with
the highest mean dev score.

### Forgetting: the final models on the other datasets

Part 1's fine-tune lost nothing on the other datasets. The new recipe moves the vectors much more.
It helps related domains (science, medicine) and hurts the distant one (finance). Test NDCG@10,
change from the start model (95% CI):

| Trained on → evaluated on | new | new + C4 replay | Part 1 fine-tune |
|---|---|---|---|
| SciFact → NFCorpus | +0.036 [+0.017, +0.055] | +0.053 [+0.035, +0.070] | +0.016 |
| SciFact → FiQA | **−0.044** [−0.060, −0.028] | +0.002 [−0.014, +0.018] | +0.000 |
| NFCorpus → SciFact | +0.023 [−0.001, +0.047] | +0.030 [+0.006, +0.054] | −0.000 |
| NFCorpus → FiQA | **−0.047** [−0.061, −0.034] | −0.013 [−0.023, −0.004] | −0.000 |
| FiQA → SciFact | −0.009 [−0.046, +0.029] | +0.034 [−0.001, +0.070] | +0.023 |
| FiQA → NFCorpus | +0.006 [−0.010, +0.021] | +0.029 [+0.013, +0.045] | +0.009 |

C4 replay removes most of the forgetting. It is above the start model on 5 of 6 pairs (the CI
excludes 0 on 3 of them); on NFCorpus → FiQA it is still 0.013 below. In domain, it costs 0.016 on
NFCorpus and nothing measurable on SciFact and FiQA.

### What each step did (dev NDCG@10)

**Step 1, train to convergence.** Same start model and learning rate (1e-3) as Part 1; only early
stopping, the scheduler threshold and the kept checkpoint change:

| Dataset | potion: Part 1 → converged | distilled: Part 1 → converged |
|---|---|---|
| SciFact | 0.4865 → 0.5106 (236 epochs) | 0.5108 → 0.5294 |
| NFCorpus | 0.2362 → 0.2466 | 0.2358 → 0.2511 |
| FiQA | 0.1605 → 0.1839 | 0.1768 → 0.1830 |

**Steps 2-4, learning rate, start-model scale and target.** Mean dev NDCG@10 (2 seeds; FiQA 1
seed; 5 seeds for the settings that were confirmed):

SciFact:

| Start, target | 1e-3 | 1e-2 | 3e-2 | 1e-1 | 3e-1 |
|---|---|---|---|---|---|
| potion, mean | 0.5106 | 0.5467 | 0.5671 | 0.5887 | 0.6180 |
| potion ÷ 10.5, mean | 0.5537 | 0.6000 | 0.6190 | 0.6312 | 0.6121 |
| distilled, mean | 0.5294 | 0.5790 | 0.6177 | 0.6267 | 0.6144 |
| potion ÷ 10.5, CLS | 0.5539 | 0.5988 | 0.6220 | 0.6267 | – |
| distilled, CLS | 0.5311 | 0.5789 | 0.6131 | 0.6321 | – |

NFCorpus:

| Start, target | 1e-3 | 1e-2 | 3e-2 | 1e-1 | 3e-1 |
|---|---|---|---|---|---|
| potion, mean | 0.2466 | 0.2676 | 0.2758 | 0.3001 | 0.3096 |
| potion ÷ 10.5, mean | 0.2740 | 0.3026 | 0.3208 | 0.3159 | 0.3111 |
| distilled, mean | 0.2511 | 0.2873 | 0.3173 | 0.3200 | 0.3107 |
| potion ÷ 10.5, CLS | 0.2723 | 0.3031 | 0.3199 | 0.3154 | – |
| distilled, CLS | 0.2480 | 0.2840 | 0.3134 | 0.3126 | – |

FiQA:

| Start, target | 1e-3 | 1e-2 | 3e-2 | 1e-1 | 3e-1 | 1 |
|---|---|---|---|---|---|---|
| potion, mean | 0.1839 | 0.1909 | 0.1915 | 0.1916 | 0.1915 | – |
| potion ÷ 10.5, mean | 0.1918 | 0.1902 | 0.1962 | 0.2001 | 0.2053 | 0.2042 |
| distilled, mean | 0.1830 | 0.1893 | 0.2009 | 0.1969 | 0.2054 | 0.2015 |
| potion ÷ 10.5, CLS | 0.1852 | 0.1837 | 0.1797 | 0.1920 | – | – |
| distilled, CLS | 0.1788 | 0.1899 | 0.1909 | 0.1949 | – | – |

- The learning rate is the largest effect. From 1e-3 to the best value, dev NDCG@10 goes up by
  +0.07 to +0.11 on SciFact, +0.05 to +0.07 on NFCorpus and +0.01 to +0.02 on FiQA, for every
  start model and target.
- **Scale and learning rate are the same knob.** Unscaled potion at a learning rate 10× higher
  scores like potion ÷ 10.5 (SciFact: 0.6180 at 3e-1 vs 0.6190 at 3e-2; 0.5887 at 1e-1 vs 0.6000
  at 1e-2). This supports the explanation in the method section: what matters is the step size
  relative to the vector length.
- With a good learning rate, potion ÷ 10.5 and the plain distilled model end within noise of each
  other. So potion's C4 training gives no advantage after the corpus fine-tune.
- **The CLS target does not help.** It ties the mean target on SciFact and NFCorpus, and is worse
  on FiQA (best 0.1949 vs 0.2054). On test, bge itself with mean pooling is as good as or better
  than with CLS pooling (SciFact +0.015 [+0.003, +0.027]; NFCorpus +0.007 [−0.000, +0.014]; FiQA
  +0.007 [−0.001, +0.014]). So the mean target was not the limit.

**Step 5, extra text.** Best setting of each dataset, seeds 42 and 43 (mean):

| Extra text | SciFact | NFCorpus | FiQA |
|---|---|---|---|
| none | 0.6354 | 0.3206 | 0.2049 |
| train queries | 0.6473 | 0.3206 | **0.2160** |
| sentences | 0.6455 | 0.3109 | 0.1844 |
| both | **0.6705** | 0.3215 | 0.1870 |

A variant was kept only if it beat its base by more than the base's 5-seed std: SciFact (both,
+0.035) and FiQA (train queries, +0.011). The kept variants were then trained with seeds 44-46 and
became the final settings.

**Step 6, 100% of the corpus.** See the test table: +0.008 on SciFact (CI excludes 0), no
measurable change on NFCorpus and FiQA.

### Validation loss does not rank settings

The lowest validation loss is often not the best search setting. NFCorpus: potion at 1e-2 has a
validation loss of 0.0772 and dev NDCG@10 0.2676; potion ÷ 10.5 at 3e-2 has 0.0811 and 0.3208.
Validation loss measures the 256→768 head together with the vectors, and the head is discarded.


## Method

### What changed in training

`sweep_finetune.py` keeps Part 1's model and loss (`StaticModelForSimilarity`, `n_layers=0`,
cosine loss, batch size 256, Adam) and changes these settings:

| Setting | Part 1 (model2vec defaults) | Sweep |
|---|---|---|
| Early stopping | `min_delta=0.001`, patience 5 | `min_delta=0`, patience 10, at most 300 epochs |
| LR scheduler (`ReduceLROnPlateau`) | threshold 3% (relative) | threshold 1e-4 (PyTorch default) |
| Saved checkpoint | last epoch | lowest validation loss |
| Learning rate | 1e-3 | 1e-3, 1e-2, 3e-2, 1e-1, 3e-1 |
| Start model | potion-base-8M or distilled | also potion-base-8M ÷ 10.5 |
| Validation split | model2vec's internal split | `train_test_split(test_size=0.1, random_state=seed)` over corpus documents |

Why these changes:

- In Part 1, every run stopped while its validation loss was still going down: the last epoch's
  loss was below the loss at the last improvement that early stopping counted.
- model2vec's scheduler halves the learning rate when the validation loss improves by less than
  3% in 3 epochs. So the learning rate soon gets near its 1e-6 floor, and more epochs do little.
- potion-base-8M's token vectors are about 10× longer than the distilled model's (median L2 norm
  97.3 vs 9.3). Adam moves each weight by about the learning rate per step, whatever the weight's
  size, so the same learning rate turns potion's vectors about 10× less. Dividing every vector by
  10.5 does not change potion's search results (each text vector is normalized), but it makes
  the same step turn the vectors more.

`train_info_sweep/<model>.json` records each run's settings, its validation loss and learning
rate at every epoch, and the epoch it kept.

### Choosing settings: dev queries, not validation loss or test queries

The validation loss decides when one run stops. It is not used to compare settings, for two
reasons. First, it measures how well documents match the teacher, not search quality. Second,
the CLS and mean targets give losses on different scales.

Settings are compared on **dev queries** with testoperator, as in Part 1:

| Dataset | Dev queries | Source |
|---|---|---|
| SciFact | 436 | the half of the `train` split whose query-id SHA-256 starts with an even byte |
| NFCorpus | 324 | `dev` split |
| FiQA | 500 | `dev` split |

`fetch_dev_qrels.py` pins these files and records their checksums in `SHA256SUMS.sweep`. No dev
or test query is ever used as training text. `query_sets.json` records the overlaps, which are
all 0. **Test queries are used only for the final tables.**

### Steps

```
 grid (steps 1-4): start model × learning rate × target, 2 seeds (FiQA: 1), dev NDCG@10
        │
        ▼
 confirm: top 3 settings per dataset, seeds 42-46, dev NDCG@10 → best setting per dataset
        │
        ▼
 extras (step 5): best setting + {train queries, sentences, both}, seeds 42-43, dev
        │
        ▼
 final: best setting, seeds 42-46, on test; and its 100% retrain (step 6), seeds 42-46, on test
```

- Grid: 3 start models × 5 learning rates for the mean target. The CLS grid leaves out unscaled
  potion, because on SciFact it scored like potion ÷ 10.5 at a 10× lower learning rate. It uses
  learning rates 1e-3 to 1e-1.
- FiQA grid runs use one seed. Their runs are about 10× longer, and Part 1's FiQA seed spread
  was 0.001 NDCG@10.
- 100% retrain: train on every corpus document for the mean best epoch of the 5 held-out runs,
  plus one, with no early stopping.

### Extra training texts (step 5)

- `trainq`: the text of train-split queries that are not dev or test queries. SciFact: 373,
  NFCorpus: 2,590, FiQA: 5,500.
- `sents`: up to 2 sentences per corpus document (at least 4 words), chosen at random with a
  fixed seed.

Both kinds of text are featurized with the same teacher and target as the corpus. They are added
to the training split only. The validation split stays corpus documents.

### CLS targets

`featurize_texts.py --target cls` embeds each text, cut to 512 tokens, with bge's own
sentence-transformers pipeline (CLS pooling, then normalization). FiQA's 57,600 texts are
embedded in chunks of 4,096, each in a new process. Running them all in one process made MPS
memory grow past its limit. The chunked output is the same as the unchunked output: cosine 1.0
on 300 SciFact documents.

## Caveats

- **Settings were chosen on dev queries; test queries were used only for the final tables.** The
  final settings' seeds 42 and 43 were also used to choose them, on dev queries; this does not
  touch the test numbers.
- SciFact's dev queries and its train-query training text are the two halves of the same `train`
  split. They do not overlap, but they come from the same source. Test queries are separate.
- FiQA grid points use one seed. The confirmed settings use five.
- 3 of 357 search runs needed a second attempt. One was a port collision between two plans that
  this harness caused: spiced exited at startup, and testoperator then waited for 40 minutes
  instead of failing. Every reported number comes from a passed run.
- The first run of mean-pooled bge on FiQA shared the GPU with featurization. It was stopped after
  2.8 hours to free the GPU for steps 5-6, and run again alone after them (1.9 hours, passed).
- C4 replay (step 7) was not in the plan. It was added after the forgetting check.
- The bge "mean pooling" row and the mean training target are not the same average. spiced's mean
  pooling averages all token outputs, CLS and SEP included, the same as sentence-transformers
  (cosine ≥ 0.99999 on 5 texts). tokenlearn's mean target leaves CLS and SEP out, because a static
  model has no such tokens.
- Part 1's caveats also apply: no query instruction for bge, `text` column only, ties broken in
  hash-map order.

## Reproduce

```bash
cd experiments/static-finetune/scripts
export PYTHONPATH=$(git rev-parse --show-toplevel)
PY=~/Github/tokenlearn/.venv/bin/python

$PY fetch_dev_qrels.py
$PY fetch_c4.py
for ds in scifact nfcorpus fiqa; do
  $PY featurize_texts.py --dataset $ds --source corpus --target cls
  for src in trainq sents; do for tgt in mean cls; do
    $PY featurize_texts.py --dataset $ds --source $src --target $tgt
  done; done
done
for tgt in mean cls; do $PY featurize_texts.py --dataset c4 --source c4 --target $tgt; done

# The plans in ../plans/, in the order they ran. Plans after the grid read earlier dev scores;
# the committed plan files are the ones that ran. Use distinct --port-base values when plans run
# at the same time (each plan uses base..base+workers and base+100..).
for plan in grid-mean grid-mean-hi grid-cls grid-mean-fiqa-top baselines baselines-fill \
            confirm-scifact-nfcorpus confirm-fiqa extras-scifact-nfcorpus confirm-scifact \
            final-scifact-nfcorpus cross-scifact-nfcorpus replay-scifact-nfcorpus \
            extras-fiqa confirm-fiqa-trainq final-fiqa cross-fiqa replay-fiqa; do
  $PY sweep.py ../plans/$plan.json
done
$PY sweep_report.py    # dev tables: results/sweep/dev_report.md
$PY sweep_final.py     # test tables: results/final/tables.md
```
