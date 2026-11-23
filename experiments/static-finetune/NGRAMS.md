# Part 3: additive token n-grams

Part 2 ([SWEEP.md](SWEEP.md)) found a fine-tune setting for each dataset. This part adds n-gram
rows to those settings and measures the change in search quality, through spiced.

RESULTS_PLACEHOLDER

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
- The chosen setting of each dataset is the one with the best mean dev NDCG@10 over seeds 42 and 43,
  across all rounds. With 9 settings per dataset, the chosen setting's dev score is biased up; the test queries, used once, give the unbiased number.
- Confirmation (dev): the chosen setting with seeds 44-46 too.
- Test: the 5 seeds of the best setting against the 5 seeds without n-grams, on the test queries
  of the training dataset (in domain) and of the other two datasets (forgetting).
- Comparisons are paired over queries: a model with several seeds scores each query with its mean
  over seeds; the 95% CI is a bootstrap over queries (10,000 resamples).

## Reproduce

REPRODUCE_PLACEHOLDER
