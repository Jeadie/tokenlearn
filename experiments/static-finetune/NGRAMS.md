# Part 3: additive token n-grams

Part 2 ([SWEEP.md](SWEEP.md)) found a fine-tune setting for each dataset. This part adds n-gram
rows to those settings and measures the change in search quality, through spiced.

In short:

- N-gram rows improve test NDCG@10 on all three datasets: +0.049 (SciFact), +0.017 (NFCorpus) and
  +0.053 (FiQA). All the gain is in the n-gram rows.
- 2-grams are best. The best K and n-gram learning rate are different for each dataset.
- FiQA's n-gram model loses 0.067 on SciFact and 0.040 on NFCorpus. C4 replay does not remove
  this loss. The SciFact and NFCorpus n-gram models lose nothing on the other datasets.
- Cost: the model file is 3× to 17× larger, and encoding is 13% to 19% slower (P99 and P99.9).
- spiced reads the n-gram rows only with a change to model2vec-rs (branch `ngrams` of
  `Jeadie/model2vec-rs`).

## Results

### Test NDCG@10 (main result)

Test queries only. "with n-grams" is each dataset's final Part 2 setting with the n-gram setting
chosen on dev queries (below). "no n-grams" is Part 2's final models, searched again with the
n-gram build of spiced. 5 seeds each (mean ± sample std).

| Model | SciFact | NFCorpus | FiQA |
|---|---|---|---|
| bge-base-en-v1.5 (CLS pooling), from Part 2 | 0.7451 | 0.3667 | 0.3916 |
| bge-base-en-v1.5 (mean pooling), from Part 2 | 0.7601 | 0.3733 | 0.3984 |
| no n-grams (Part 2 final) | 0.6534 ± 0.0055 | 0.3254 ± 0.0027 | 0.2279 ± 0.0051 |
| **with n-grams** | **0.7023 ± 0.0087** | **0.3420 ± 0.0009** | **0.2813 ± 0.0078** |
| with n-grams + C4 replay (FiQA only, see Forgetting) | | | 0.2816 ± 0.0065 |

Paired bootstrap 95% CIs over test queries, with n-grams − no n-grams:

| Metric | SciFact (300 queries) | NFCorpus (323) | FiQA (648) |
|---|---|---|---|
| NDCG@1 | +0.0580 [+0.0267, +0.0907] | +0.0229 [+0.0050, +0.0418] | +0.0630 [+0.0392, +0.0864] |
| NDCG@10 | **+0.0489** [+0.0317, +0.0669] | **+0.0166** [+0.0106, +0.0228] | **+0.0534** [+0.0372, +0.0691] |
| Recall@10 | +0.0425 [+0.0215, +0.0646] | +0.0076 [+0.0039, +0.0115] | +0.0506 [+0.0310, +0.0704] |
| MRR@10 | +0.0502 [+0.0299, +0.0716] | +0.0180 [+0.0056, +0.0305] | +0.0650 [+0.0442, +0.0852] |

All 12 CIs exclude 0. The gap to bge-base-en-v1.5 (CLS pooling) goes from 0.092 to 0.043
(SciFact), from 0.041 to 0.025 (NFCorpus) and from 0.164 to 0.110 (FiQA) NDCG@10. (These gaps
are differences of means, with no CI.)

### The chosen settings

| Dataset | N-grams | K (n-grams kept) | N-gram lr | Dev NDCG@10, 5 seeds | Dev, no n-grams | Listed n-grams per corpus text | Model file |
|---|---|---|---|---|---|---|---|
| SciFact | 2-grams | 64k (64,000) | × 0.3 | 0.7003 ± 0.0033 | 0.6674 ± 0.0036 | 230 (303 tokens) | 96.5 MB |
| NFCorpus | 2-grams | 256k (93,636) | × 0.3 | 0.3354 ± 0.0057 | 0.3208 ± 0.0042 | 279 (326 tokens) | 127.2 MB |
| FiQA | 2-grams | 1M (466,211) | × 1 | 0.2855 ± 0.0091 | 0.2183 ± 0.0036 | 150 (164 tokens) | 513.2 MB |

The model file without n-grams is 30.2 MB (float32). Corpus counts are means over the first 2,000
corpus texts, seed-42 model.

### What each setting did (dev NDCG@10, seeds 42 and 43)

Change from no n-grams. `*`: the 95% CI excludes 0. Blank: not run.

| Setting | SciFact | NFCorpus | FiQA |
|---|---|---|---|
| 2+3-grams, K=16k | −0.0074 | +0.0036 | +0.0199 * |
| 2+3-grams, K=64k | −0.0013 | +0.0028 | +0.0294 * |
| 2-grams, K=16k | −0.0036 | +0.0090 * | +0.0233 * |
| 2-grams, K=64k | +0.0087 | +0.0094 * | +0.0407 * |
| 2-grams, K=256k | +0.0081 | +0.0112 * | +0.0538 * |
| 2-grams, K=1M | | | **+0.0615** * |
| 2-grams, K=64k, n-gram lr × 0.1 | +0.0233 * | +0.0058 * | +0.0135 * |
| 2-grams, K=64k, n-gram lr × 0.3 | **+0.0276** * | +0.0123 * | +0.0374 * |
| 2-grams, K=256k, n-gram lr × 0.3 | +0.0261 * | **+0.0133** * | +0.0498 * |
| 2+3-grams, K=64k, n-gram lr × 0.3 | +0.0247 * | +0.0102 * | |

- **2-grams are better than 2- and 3-grams** with the same K, on all three datasets.
- **A larger K helps FiQA most**: +0.023, +0.041, +0.054 and +0.062 for K = 16k, 64k, 256k and
  1M. On SciFact and NFCorpus, a K above 64k adds little. (NFCorpus has only 93,636 2-grams with
  count ≥ 2.)
- **SciFact needs a lower n-gram learning rate.** With the token rows' rate, its change is not
  significant (+0.009); with × 0.3 it is +0.028. FiQA is the opposite: × 0.3 and × 0.1 are worse
  than × 1.
- **Validation loss does not rank these settings** (as in Part 2). FiQA's n-gram models have a
  worse best validation loss than the model without n-grams (0.102 to 0.108 against 0.097, seeds
  42 and 43), but better search.
- **Short queries use few n-grams.** NFCorpus dev queries have 5.1 tokens on average, and 72% to
  76% of them contain a listed n-gram (K = 64k or 256k). On SciFact and FiQA, 95% to 100% do.
  NFCorpus gains least.

### Forgetting: each dataset's models on the other datasets

Test NDCG@10, 5 seeds; Δ = with n-grams − no n-grams (95% CI).

| Trained on → evaluated on | No n-grams | With n-grams | Δ |
|---|---|---|---|
| SciFact → NFCorpus | 0.2310 | 0.2349 | +0.0038 [−0.0052, +0.0128] |
| SciFact → FiQA | 0.0716 | 0.0752 | +0.0035 [−0.0036, +0.0108] |
| NFCorpus → SciFact | 0.5106 | 0.5124 | +0.0018 [−0.0095, +0.0132] |
| NFCorpus → FiQA | 0.1189 | 0.1242 | +0.0053 [+0.0009, +0.0099] |
| FiQA → SciFact | 0.4299 | 0.3633 | **−0.0666** [−0.0942, −0.0396] |
| FiQA → NFCorpus | 0.2012 | 0.1609 | **−0.0403** [−0.0533, −0.0280] |

The SciFact and NFCorpus n-gram models lose nothing on the other datasets. The FiQA n-gram model
loses 0.067 on SciFact and 0.040 on NFCorpus. It is then below its start model (the distilled
model: 0.4391 on SciFact and 0.1954 on NFCorpus, from Part 2).

#### Where the change comes from

The token rows of an n-gram model train together with its n-gram rows. To split the change, each
chosen n-gram model (5 seeds) was also searched with its n-gram tensors removed from the file
(`make_ngram_drop.py`), so that only its token rows encode. Test NDCG@10 (95% CI):

| Trained on → evaluated on | No n-grams | With n-grams | Rows removed | Token rows: removed − no n-grams | N-gram rows: with − removed |
|---|---|---|---|---|---|
| SciFact → SciFact | 0.6534 | 0.7023 | 0.6417 | −0.0117 [−0.0277, +0.0039] | **+0.0606** [+0.0433, +0.0786] |
| NFCorpus → NFCorpus | 0.3254 | 0.3420 | 0.3187 | −0.0067 [−0.0112, −0.0023] | **+0.0233** [+0.0162, +0.0307] |
| FiQA → FiQA | 0.2279 | 0.2813 | 0.1953 | −0.0326 [−0.0428, −0.0228] | **+0.0860** [+0.0702, +0.1021] |
| FiQA → SciFact | 0.4299 | 0.3633 | 0.4114 | −0.0185 [−0.0424, +0.0051] | **−0.0481** [−0.0670, −0.0299] |
| FiQA → NFCorpus | 0.2012 | 0.1609 | 0.1643 | **−0.0369** [−0.0497, −0.0247] | −0.0034 [−0.0110, +0.0042] |

- In domain, all of the gain is in the n-gram rows. The token rows alone are a little worse than
  Part 2's models (the CI excludes 0 on NFCorpus and FiQA).
- The two losses of the FiQA model have different sources: on SciFact, mostly the n-gram rows;
  on NFCorpus, the token rows.
- Removing the n-gram rows is therefore not a fix: the model then loses the in-domain gain and
  keeps the NFCorpus loss.

#### Two FiQA variants against the forgetting

These two variants were added after the forgetting result above, on the test queries. Both are
reported. Neither changes the main table.

| FiQA variant (5 seeds) | SciFact | NFCorpus | FiQA (in domain) |
|---|---|---|---|
| no n-grams (Part 2 final) | 0.4299 | 0.2012 | 0.2279 |
| with n-grams (chosen: 2-grams, K=1M) | 0.3633 | 0.1609 | 0.2813 |
| 2-grams, K=256k, n-gram lr × 0.3 | 0.4288 | 0.1882 | 0.2663 |
| no n-grams + C4 replay (Part 2 step 7) | 0.4730 | 0.2245 | 0.2236 |
| with n-grams (chosen) + C4 replay | 0.4146 | 0.2027 | 0.2816 |

Paired differences (95% CI):

| A − B | SciFact | NFCorpus | FiQA |
|---|---|---|---|
| K=256k, lr × 0.3 − no n-grams | −0.0011 [−0.0165, +0.0143] | −0.0130 [−0.0193, −0.0069] | +0.0385 [+0.0288, +0.0482] |
| chosen + C4 − no n-grams | −0.0154 [−0.0432, +0.0121] | +0.0015 [−0.0110, +0.0139] | +0.0537 [+0.0384, +0.0688] |
| chosen + C4 − no n-grams + C4 | **−0.0584** [−0.0883, −0.0290] | **−0.0217** [−0.0337, −0.0103] | +0.0580 [+0.0420, +0.0739] |

- C4 replay does **not** remove the loss that the n-grams cause. It raises both models: by 0.043
  (SciFact) and 0.023 (NFCorpus) without n-grams, and by 0.051 and 0.042 with n-grams. With C4 on
  both sides, the n-gram model still loses 0.058 on SciFact and 0.022 on NFCorpus.
- The n-gram model with C4 replay is at the level of Part 2's final model on SciFact and NFCorpus
  (the CIs include 0) and 0.054 above it on FiQA. If Part 2's final model is the reference, this
  model is better on FiQA and no worse elsewhere. If Part 2's C4 model is the reference, it trades
  0.02 to 0.06 out of domain for 0.058 in domain.
- A lower n-gram learning rate (K=256k, × 0.3) keeps SciFact, loses 0.013 on NFCorpus, and keeps
  only 0.039 of FiQA's gain.

### Size and speed

- The n-gram rows make the model file larger: 3.2× (SciFact), 4.2× (NFCorpus) and 17× (FiQA:
  513 MB against 30 MB). Smaller rows (float16, quantization) and pruning were not tried.

- Encoding is slower. model2vec-rs (branch `ngrams`, release build, one text per call, one
  thread), Apple M1 Max, idle. FiQA's model without n-grams against FiQA's chosen n-gram model
  (466,211 n-grams). Each model also ran twice in one process, and the run was repeated with the
  models in reverse order. µs per text (`results/ngrams/latency.json`):

  | Texts | Samples per model and order | No n-grams P99 / P99.9 | Same model again (A/A) P99 / P99.9 | With n-grams P99 / P99.9 |
  |---|---|---|---|---|
  | FiQA test queries (648) | 25,920 | 39.4–39.8 / 45.0–45.3 | 39.5–39.7 / 45.0–45.5 | **44.7–45.1** / **51.5–52.6** |
  | FiQA corpus, first 1,000 texts | 20,000 | 762–763 / 797 | 761–763 / 794–795 | **902–904** / **929–942** |

  Ranges are over the two orders. The A/A difference is at most 0.6 µs (queries) and 3 µs
  (corpus); the n-gram model is about 13% to 16% slower on queries and 16% to 19% slower on corpus
  texts, at both P99 and P99.9. This is the encode step only, not a search request through spiced.
  A first run, started while the machine was still busy after the searches, had an A/A P99.9 of
  798 against 1,287 µs on the corpus texts; it is not used.

## Caveats

- **Settings were chosen on dev queries.** Each dataset had 9 n-gram settings; the chosen
  setting's dev score is biased up. The test queries were used once for the chosen settings. The
  FiQA forgetting variants and the rows-removed models (above) were added after the test results,
  as analyses. They did not change a choice.
- **NFCorpus results move by up to about 0.0008 NDCG@10 between two runs of the same model and
  build** (see Gates). SciFact and FiQA results do not move.
- **The n-gram list comes from each run's training texts**, as the token rows do. These include
  the corpus documents that the search runs over (as in Part 2), and SciFact's and FiQA's train
  queries. They include no dev or test queries.
- **The spiced build is local**: `spiceai/spiceai` trunk `96ada40216` and a Cargo `[patch]` to
  `Jeadie/model2vec-rs` branch `ngrams` (`47359dc`). No spiced release can read the n-gram rows.
  With a release (or the Python model2vec library), an n-gram model encodes as its token rows
  only, with no error.
- FiQA's validation loss and dev score disagree on n-grams (above). The n-gram settings were not
  chosen by validation loss.
- Part 1's and Part 2's caveats also apply.

## The model

A static model gives each token one vector. A text's vector is the mean of its token vectors, so
the model cannot see that two tokens are next to each other. This part adds one vector for each
of the K most frequent token n-grams, like Sent2Vec and fastText:

```
text:     "low interest rate"
tokens:   low      interest      rate                (no special tokens, UNK dropped, at most 512)
rows:     U[low]   U[interest]   U[rate]             one row per token, as before
        + G[low interest]  G[interest rate]          one row per listed 2-gram in the text
        + G[low interest rate]                       one row per listed 3-gram in the text
vector  = normalize( sum of the rows / number of rows )
```

- The n-grams are token n-grams (not word n-grams). An n-gram can be inside a word
  (`un ##happy`) or go across punctuation.
- Only the K n-grams in the list get a row. An n-gram that is not in the list adds nothing, and
  does not count in the mean. (Hashing every n-gram into K buckets, as fastText does, would give
  unseen n-grams a shared row that other n-grams trained.)
- The list is the K most frequent n-grams of the training texts (count ≥ 2), ties broken by token
  ids. It is mined again for each training run, from that run's training texts only.
- An n-gram that occurs twice in a text adds its row twice, as a token does.

### The model file

The n-grams are two more tensors in `model.safetensors`:

| Tensor | Shape | Content |
|---|---|---|
| `embeddings` | [V, d] | the token rows, as before |
| `ngram_ids` | [K, 3], int32 | the token ids of each n-gram; `-1` after a 2-gram's two ids |
| `ngram_embeddings` | [K, d] | the n-gram rows |

The Python model2vec library and older builds of model2vec-rs ignore the two tensors. A model that
has them then encodes as if it had no n-grams, with no error. So every search run here records the
spiced build it used (`run.json`), and `sweep.py --expect-version` refuses to search with another
build.

### spiced

spiced pools model2vec models in [model2vec-rs](https://github.com/spiceai/model2vec-rs). The
change is in that crate only (branch `ngrams` of `Jeadie/model2vec-rs`, on top of the `spiceai`
branch that spiced uses). It reads the two tensors, rejects malformed ones, and adds the n-gram
rows in `pool_ids`. spiced itself does not change: the spiced build for this part is
`spiceai/spiceai` trunk with a Cargo `[patch]` that points model2vec-rs at that branch.

### Training

Training is Part 2's recipe with a larger table. The K n-gram rows go below the V token rows of
one table, and each text's input ids are its token ids followed by `V + row` for each listed n-gram
in it. model2vec's own mean pooling then computes the vector above, and the token rows and n-gram
rows train together.

- The n-gram rows start at zero. A zero row changes only the length of a text vector, and the
  vectors are normalized, so before training the model gives the same results as the model without
  n-grams. Training starts from the same point as Part 2's.
- Same seed, same split, same head start and same batch order as the Part 2 model it is compared
  with.
- The n-gram rows can have a learning rate of their own (`--ngram-lr-scale c`): the model uses
  `c × stored row`. Adam moves each stored value by about the learning rate per step, so the rows
  then move `c` times as fast as the token rows. This is Part 2's potion ÷ 10.5 trick, for the
  n-gram rows only. With `c = 1` the n-gram rows have the token rows' learning rate.
- Early stopping and the best checkpoint by validation loss, as in Part 2.

## Gates

| Gate | What it checks | How |
|---|---|---|
| G1 | The new spiced build gives the same results as the build of Parts 1 and 2 on models without n-grams. | The same text vectors (`check_ngram_spiced.py`); the same per-query metrics on dev and test for each dataset's final model, 5 seeds (`ngram_report.py`). |
| G2 | spiced adds the n-gram rows as the model above says. | A model with random n-gram rows: spiced's vectors against `NgramEncoder`, the Python reference (`check_ngram_spiced.py`). The texts include texts over 512 tokens and texts with unknown tokens. |
| G3 | Zero n-gram rows change nothing. | Each dataset's final model (seed 42) plus 64,000 zero n-gram rows, searched on dev, against the same model without them. |

To show that G2 can fail, it was also run with the old build in the place of the new one
(`results/ngrams/spiced_check_old_binary.json`): the old build ignores the n-gram rows, and the
minimum cosine to the reference is 0.22.

Results (all in [results/ngrams/report.md](results/ngrams/report.md)):

- **G2 passes.** spiced against the reference, 405 texts: minimum cosine 0.99999999998, largest
  coordinate difference 1.8e-6 (float32 against float64). The same spiced vectors against the model
  without its n-gram rows: minimum cosine 0.22, so the rows are used.
- **G1 passes on text vectors**: the two builds give identical vectors (largest difference 0).
- **G1 and G3 pass on search for SciFact and FiQA**: every metric@k of every query is identical
  (dev and test, 5 seeds; 376,800 values for G1).
- **On NFCorpus, G1 and G3 show differences**: 419 of 64,800 dev values and 212 of 64,600 test
  values (G1), and 57 of 12,960 (G3). The build is not the cause. The old build, searched twice
  with the same models, differs as much: 226 of 400 run-level values, up to 0.0008 NDCG@10 per run
  (up to 0.009 at other metric@k). SciFact and FiQA do not change between runs. NFCorpus has 40
  pairs of documents with the same text, so with the same vector. A likely cause (not verified) is
  that their tied scores come out in a different order from run to run. This adds up to about
  0.0008 NDCG@10 of noise to each NFCorpus run.

## Method

- Datasets, queries and splits: as in Part 2. Dev queries choose the setting; test queries are
  used once, for the final tables.
- Every n-gram model starts from its dataset's final Part 2 setting (start model, target, learning
  rate, extra text). The reference is the same setting without n-grams, with the same seeds,
  searched with the new build.
- The dev grid has three rounds, seeds 42 and 43. Each round was planned from the results of the
  one before:

  | Round | Settings | Why |
  |---|---|---|
  | 1 | {2-grams, 2- and 3-grams} × K ∈ {16k, 64k} | the plan |
  | 2 | the best round-1 setting (2-grams, K=64k on all three) with n-gram lr × 0.1 or × 0.3; and with K=256k | FiQA's validation loss was worse with n-grams; every dataset was best at the largest K |
  | 3 | FiQA: K=1M; K=256k with lr × 0.3. SciFact, NFCorpus: K=256k with lr × 0.3; 2- and 3-grams, K=64k, lr × 0.3 | combine what helped: a lower n-gram lr (SciFact, NFCorpus) and a larger K (FiQA) |

  K is an upper limit: only n-grams with count ≥ 2 are kept, so a large K can keep fewer.
- The chosen setting of each dataset is the one with the best mean dev NDCG@10 over seeds 42 and
  43, across all rounds. With 9 settings per dataset, the chosen setting's dev score is biased up;
  the test queries, used once, give the unbiased number.
- Confirmation (dev): the chosen setting with seeds 44-46 too.
- Test: the 5 seeds of the best setting against the 5 seeds without n-grams, on the test queries
  of the training dataset (in domain) and of the other two datasets (forgetting).
- Comparisons are paired over queries: a model with several seeds scores each query with its mean
  over seeds; the 95% CI is a bootstrap over queries (10,000 resamples).

## Reproduce

Part 2's data, features and models must exist first ([SWEEP.md](SWEEP.md), Reproduce).

The spiced build (the command that built the binary for these results):

```bash
git clone -b ngrams https://github.com/Jeadie/model2vec-rs.git /tmp/model2vec-rs-worktrees/ngrams
cd <spiceai/spiceai checkout at trunk 96ada40216>
cp ~/.spice/bin/spiced ~/.spice/bin/spiced_60eaf209cd   # Part 2's build, for gate G1
cat >> Cargo.toml <<'EOF'
[patch."https://github.com/spiceai/model2vec-rs.git"]
model2vec-rs = { path = "/tmp/model2vec-rs-worktrees/ngrams" }
EOF
make install SPICED_NON_DEFAULT_FEATURES='metal,models,elasticsearch,spicebench,odbc,snapshots,rate-control'
~/.spice/bin/spiced --version   # v2.5.0-unstable+models.metal (840011c44d) here
```

The experiment:

```bash
cd experiments/static-finetune/scripts
export PYTHONPATH=$(git rev-parse --show-toplevel)
PY=~/Github/tokenlearn/.venv/bin/python
NEW="--expect-version 840011c44d --scores ../results/ngrams/scores.jsonl"

# Gates. G1 searches Part 2's final models again with Part 2's build (run tag "old").
$PY make_ngram_zero.py
$PY check_ngram_spiced.py          # G1 and G2 on text vectors: results/ngrams/spiced_check.json
$PY sweep.py ../plans/ngram-baselines-old.json --spiced ~/.spice/bin/spiced_60eaf209cd \
  --expect-version 60eaf209cd --scores ../results/ngrams/scores.jsonl

# The plans in ../plans/, in the order they ran. Each plan after the grid reads the earlier dev
# scores; the committed plan files are the ones that ran.
for plan in ngram-gates ngram-baselines ngram-grid ngram-grid2 ngram-grid3 \
            ngram-confirm-scifact-nfcorpus-fiqa ngram-final-scifact-nfcorpus-fiqa \
            ngram-forgetting ngram-forgetting-reference; do
  $PY sweep.py ../plans/$plan.json $NEW
done
$PY make_ngram_drop.py
$PY sweep.py ../plans/ngram-drop.json $NEW

$PY ngram_report.py                # results/ngrams/report.md, paired_comparisons.csv
$PY ngram_latency.py               # results/ngrams/latency.json; run on an idle machine
```

`plans.py <name>` writes a plan file again from the scores so far (`ngram-grid2`, `ngram-grid3`,
`ngram-confirm`, `ngram-final`, `ngram-forgetting`, `ngram-forgetting-reference`, `ngram-drop`).
