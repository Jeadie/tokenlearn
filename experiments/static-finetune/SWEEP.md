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

RESULTS_PLACEHOLDER

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

## Reproduce

```bash
cd experiments/static-finetune/scripts
export PYTHONPATH=$(git rev-parse --show-toplevel)
PY=~/Github/tokenlearn/.venv/bin/python

$PY fetch_dev_qrels.py
for ds in scifact nfcorpus fiqa; do
  $PY featurize_texts.py --dataset $ds --source corpus --target cls
  for src in trainq sents; do for tgt in mean cls; do
    $PY featurize_texts.py --dataset $ds --source $src --target $tgt
  done; done
done

$PY plans.py grid-mean grid-mean-hi grid-cls baselines
for plan in grid-mean grid-mean-hi grid-cls baselines; do $PY sweep.py ../plans/$plan.json; done
$PY plans.py confirm --top 3                     # needs the grid's dev scores
$PY sweep.py ../plans/confirm-scifact-nfcorpus-fiqa.json
$PY plans.py extras && $PY sweep.py ../plans/extras-scifact-nfcorpus-fiqa.json
$PY plans.py final && $PY sweep.py ../plans/final-scifact-nfcorpus-fiqa.json
$PY sweep_report.py                              # dev tables
$PY sweep_final.py                               # test tables
```

The plan files in `plans/` are the ones that ran. Some plans ran per dataset; see their names.
