# Static embedding models: search quality before and after corpus fine-tuning

This experiment measures the search quality of static (Model2Vec) embedding models in Spice. It
compares them with the transformer teacher that they come from, and it measures how much a short
tokenlearn fine-tune on the search corpus improves them.

All runs use `testoperator run search`, which does an embedding-only vector search through `spiced`.

> **Part 2:** [SWEEP.md](SWEEP.md) trains the fine-tune to convergence with a higher learning
> rate, and gets +0.05 to +0.16 test NDCG@10 over this part's fine-tuned models. It also shows that
> the stronger fine-tune loses quality on a distant domain, and that replaying general text fixes
> most of that loss.
>
> **Part 3:** [NGRAMS.md](NGRAMS.md) adds a trained vector for each frequent token 2-gram to Part
> 2's models (+0.017 to +0.053 test NDCG@10). spiced reads them with a model2vec-rs change.

## Models

| # | Model | How it is made | Dims |
|---|---|---|---|
| 1 | `BAAI/bge-base-en-v1.5` | The transformer itself (`huggingface:` embeddings, CLS pooling, `truncate: END`). | 768 |
| 2 | `bge-base-en-v1.5-distill-256` | `model2vec.distill(BAAI/bge-base-en-v1.5, pca_dims=256)`. Teacher tokenizer vocabulary. | 256 |
| 3 | `minishlab/potion-base-8M` | Published tokenlearn model: distilled from bge-base-en-v1.5, then trained on `minishlab/tokenlearn-c4-en-bge-base-v1.5` (model card). | 256 |
| 4 | potion-base-8M + corpus fine-tune | Model 3, then tokenlearn training on the eval corpus (see below). 5 seeds. | 256 |
| 4b | distilled + corpus fine-tune | Model 2, then the same training as model 4. 5 seeds. Control: it shows how much of model 3's C4 training a corpus fine-tune can replace. | 256 |

Models 4 and 4b are trained on each of the three corpora. Every trained model is evaluated on all
three datasets (the cross matrix below), so each fine-tune is also checked for loss of quality on
other domains.

## Method

```
 mteb/<ds> corpus.parquet ──► featurize_corpus.py ──► (text, bge mean token vector) per doc
                                  (bge-base-en-v1.5,            │
                                   max_length 512)              ▼
 model 2 or model 3 ───────────────────────────────► finetune.py (seed 42..46) ──► model 4 / 4b
                                                                                    │
 make_spicepods.py ──► spicepods/<eval ds>/<model>-arrow.yaml ◄────────────────────┘
        │
        ▼
 run_benchmarks.py ──► testoperator run search (own HTTP + Flight ports per run)
        │                         │
        │                         └─ OTLP gRPC ──► otel collector ──► metrics.jsonl
        ▼                                                                  │
 results/runs/<ds>/<name>/run.json                 collect.py ◄────────────┘
                                                       │
                                                       ▼
                                   run_metrics.csv + per_query.parquet ──► analyze.py ──► results/tables.md
```

### Data

The datasets are MTEB SciFact, NFCorpus and FiQA, from Hugging Face's parquet API:
`https://huggingface.co/api/datasets/mteb/<ds>/parquet/<config>/<split>/0.parquet`, with
`corpus/corpus`, `queries/queries` and `default/test` (qrels). Local copies of these files are
pinned in `~/Github/tokenlearn/data/static-finetune/mteb/` and every run reads them. The SHA-256 of
each file is listed below. `provenance.json` records the dataset revisions.

```
2cf6fa2ccaf67044354edc7d6717d36df9e45c923bc824a2843eafaf10dc1c72  scifact/corpus.parquet
958ffe3e92c8d9624663d632573f5adfad92ea68ac2ca697bef93df29597fc95  scifact/qrels_test.parquet
79f5dd332d88069091a77116bae1702174572fa372d41b2ff04ea4d04285c294  scifact/queries.parquet
249d0ee2135afd70345ac2eb48b007bef8c7c4ca2340900fb9f0a23b3e849d1f  nfcorpus/corpus.parquet
9898f0df06319009953e3366cc97df1498ded220e72096b7ba3f065c9a9e8430  nfcorpus/qrels_test.parquet
b4a2ce497642af342ead987008c8d95bdf547a816b4ea34351ed33adbd141f07  nfcorpus/queries.parquet
b49a742488f918c745fe69f7ea7f439f8d65f7b061f1303c9042bdee61bf9fb5  fiqa/corpus.parquet
70a6e45a057e8c71d1876efe6915c75069559d02497cf8aba553106acec7c97b  fiqa/qrels_test.parquet
5fac4bc056fb9a7985a377516d4c49ae0def078c0de04bc2cc04a37b39ccfdb6  fiqa/queries.parquet
```

| Dataset | Corpus docs | Judged test queries |
|---|---|---|
| SciFact | 5,183 | 300 |
| NFCorpus | 3,633 | 323 |
| FiQA | 57,638 | 648 |

### Fine-tuning (models 4 and 4b)

1. `featurize_corpus.py` embeds the `text` column of the corpus with bge-base-en-v1.5
   (`max_length` 512), with `tokenlearn.featurize`. The target for each document is the mean of
   its token vectors, without CLS and SEP. The stored training text is cut to the same 512
   tokens. The script sorts texts by length to reduce padding; this changes only the batch order.
2. `finetune.py` loads the start model, and calls
   `StaticModelForSimilarity.from_static_model(n_layers=0, freeze_weights=True)` and `fit()`:
   cosine loss, batch size 256, learning rate 1e-3, 10% validation split, early stopping on
   validation loss. The token embeddings are trained; the per-token weights are frozen
   (`freeze_weights`). The 256→768 projection head is used only in training: the saved model
   (`to_static_model()`) is the updated 256-dim token embedding table. Model 4b loads the model 2
   directory with `StaticModel.from_pretrained`, so its vocabulary is the teacher vocabulary.
3. Seeds 42–46 for each (start model, corpus) pair. 30 fine-tuned models in total.

`train_info/*.json` records each run: rows, epochs, best and final validation loss, wall time.

| Fine-tune | Train rows | Epochs (min–max over seeds) | Final val loss (min–max) |
|---|---|---|---|
| potion-8M + scifact | 4,664 | 37–42 | 0.1022–0.1037 |
| potion-8M + nfcorpus | 3,269 | 45–48 | 0.0885–0.0891 |
| potion-8M + fiqa | 51,840 | 18–19 | 0.0956–0.0957 |
| distilled + scifact | 4,664 | 42–44 | 0.1008–0.1012 |
| distilled + nfcorpus | 3,269 | 48–54 | 0.0875–0.0900 |
| distilled + fiqa | 51,840 | 19–22 | 0.0894–0.0896 |

### Search runs

- `make_spicepods.py` writes 99 spicepods: 33 models × 3 eval datasets. They differ only in the
  `embeddings:` entry. Each spicepod has the three pinned files as `file:` datasets with the Arrow
  accelerator, an embedding column on `corpus.text`, and a `test_queries` view that keeps only
  the queries that have qrels. Static models set `embed_max_token_length: 512`.
- `run_benchmarks.py` runs `testoperator run search` for each spicepod. Many runs go at the
  same time, so each run gets its own spiced HTTP and Flight port. This needs the testoperator
  from [spiceai/spiceai#16623](https://github.com/spiceai/spiceai/pull/16623), which adds
  `--spiced-http-port`, `--spiced-flight-port` and `--per-query-all-k`, and which makes
  `run search` export to `--otlp-endpoint`.
- testoperator gets the top 10 results per query, so every metric is reported at k = 1..10.
- Each run exports its run-level metrics and its per-query metrics (every query, every k) over
  OTLP to a local collector (`otel/collector.yaml`). `collect.py` keeps the newest export of each
  run.

### Analysis

`analyze.py` first checks that every run-level metric@k is equal to the mean of the per-query
values for that run (3,960 pairs, largest difference 8.9e-16). Then it writes:

- metric@k, k = 1..10, for ndcg, recall, mrr and precision. Models 4 and 4b show the mean ± sample
  standard deviation over 5 seeds, for the model trained on the eval dataset.
- Paired comparisons A − B on the same queries. For each query, a fine-tuned model's score is
  the mean over its 5 seeds. The 95% confidence interval is a paired bootstrap over queries
  (10,000 resamples). The table also shows the mean delta of each seed.
- The cross matrix: each fine-tuned model minus its start model, for all
  (train dataset, eval dataset) pairs.

## Results

Full tables (all four metrics at k = 1..10, all comparisons at k = 1, 5, 10) are in
[`results/tables.md`](results/tables.md) and `results/tables/*.csv`.

### NDCG@10

| Model | SciFact | NFCorpus | FiQA |
|---|---|---|---|
| 1: bge-base-en-v1.5 | 0.7451 | 0.3667 | 0.3916 |
| 2: distilled (256d) | 0.4391 | 0.1954 | 0.1153 |
| 3: potion-base-8M | 0.4879 | 0.2308 | 0.1662 |
| 4: potion-base-8M + corpus fine-tune | 0.4937 ± 0.0014 | 0.2348 ± 0.0005 | 0.1705 ± 0.0010 |
| 4b: distilled + corpus fine-tune | 0.4967 ± 0.0041 | 0.2365 ± 0.0033 | 0.1809 ± 0.0012 |

### Recall@10, MRR@10, Precision@10

| Model | SciFact R | NFCorpus R | FiQA R | SciFact MRR | NFCorpus MRR | FiQA MRR | SciFact P | NFCorpus P | FiQA P |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.8686 | 0.1775 | 0.4590 | 0.7092 | 0.5737 | 0.4742 | 0.0980 | 0.2752 | 0.1106 |
| 2 | 0.5562 | 0.0909 | 0.1472 | 0.4093 | 0.3674 | 0.1519 | 0.0617 | 0.1440 | 0.0327 |
| 3 | 0.6268 | 0.1012 | 0.2188 | 0.4516 | 0.4154 | 0.2074 | 0.0693 | 0.1746 | 0.0481 |
| 4 | 0.6328 | 0.1030 | 0.2212 | 0.4571 | 0.4181 | 0.2129 | 0.0699 | 0.1788 | 0.0485 |
| 4b | 0.6477 | 0.1111 | 0.2445 | 0.4552 | 0.4086 | 0.2166 | 0.0719 | 0.1800 | 0.0553 |

Models 4 and 4b are means over 5 seeds. `results/tables.md` gives the standard deviations.

### Paired comparisons, NDCG@10 (95% bootstrap CI)

| A − B | SciFact | NFCorpus | FiQA |
|---|---|---|---|
| 4 − 3 (fine-tune gain on potion) | +0.0058 [+0.0009, +0.0115] | +0.0040 [+0.0021, +0.0060] | +0.0043 [+0.0008, +0.0079] |
| 4b − 2 (fine-tune gain on distilled) | +0.0576 [+0.0424, +0.0731] | +0.0411 [+0.0328, +0.0496] | +0.0657 [+0.0519, +0.0800] |
| 4 − 4b | −0.0030 [−0.0240, +0.0176] | −0.0017 [−0.0122, +0.0087] | −0.0104 [−0.0214, +0.0005] |
| 3 − 2 (C4 tokenlearn vs plain distill) | +0.0488 [+0.0259, +0.0719] | +0.0354 [+0.0247, +0.0465] | +0.0509 [+0.0378, +0.0644] |
| 1 − 4 (gap to the teacher after fine-tune) | +0.2514 [+0.2089, +0.2951] | +0.1318 [+0.1104, +0.1538] | +0.2211 [+0.1963, +0.2469] |

### Cross matrix: fine-tuned − start model, NDCG@10 (mean over 5 seeds, 95% CI; * = CI excludes 0)

Model 4 (potion-base-8M + fine-tune) − model 3:

| Trained on ↓ / evaluated on → | SciFact | NFCorpus | FiQA |
|---|---|---|---|
| SciFact | +0.0058 [+0.0009, +0.0115] * | +0.0008 [−0.0005, +0.0023] | +0.0001 [−0.0014, +0.0018] |
| NFCorpus | −0.0004 [−0.0023, +0.0010] | +0.0040 [+0.0021, +0.0060] * | −0.0004 [−0.0017, +0.0008] |
| FiQA | −0.0000 [−0.0052, +0.0045] | −0.0011 [−0.0031, +0.0008] | +0.0043 [+0.0008, +0.0079] * |

Model 4b (distilled + fine-tune) − model 2:

| Trained on ↓ / evaluated on → | SciFact | NFCorpus | FiQA |
|---|---|---|---|
| SciFact | +0.0576 [+0.0424, +0.0731] * | +0.0157 [+0.0104, +0.0210] * | +0.0001 [−0.0048, +0.0046] |
| NFCorpus | +0.0075 [−0.0026, +0.0179] | +0.0411 [+0.0328, +0.0496] * | +0.0038 [−0.0005, +0.0081] |
| FiQA | +0.0229 [+0.0022, +0.0445] * | +0.0091 [+0.0011, +0.0173] * | +0.0657 [+0.0519, +0.0800] * |

### Findings

1. The transformer (model 1) is much better than every static model: +0.13 to +0.25 NDCG@10
   over the best static model on each dataset.
2. A corpus fine-tune of potion-base-8M (4 − 3) gives a small gain on all three datasets:
   +0.004 to +0.006 NDCG@10. The CI excludes 0 each time, and all 5 seeds are positive.
3. A corpus fine-tune of the plain distilled model (4b − 2) gives a gain 10× to 15× larger:
   +0.04 to +0.07 NDCG@10. After the fine-tune, 4b's NDCG@10 is higher than 4's on all three
   datasets (+0.002 to +0.010), and each of the 5 seeds of 4 is below 4b's mean. But no 95% CI
   excludes 0, so the query sample does not establish the difference. At small k it reverses:
   4 is higher at NDCG@1 on all three datasets (+0.003 to +0.017, CIs include 0). 4b has a
   higher recall@10 on all three (FiQA: +0.023, CI excludes 0). So a short fine-tune on the
   search corpus gets about the same result as potion-base-8M's training on C4, or slightly
   better at k = 10, but it does not add to that training.
4. No loss of quality on other domains: the off-diagonal cells for model 4 are all within
   ±0.0011 and every CI includes 0. Model 4b, trained on one corpus, is also better on some
   other corpora (for example, trained on FiQA: +0.023 on SciFact).
5. Seed variance is small compared with the effects: the standard deviation of NDCG@10 over
   seeds is 0.0005 to 0.0041.

## Caveats

- No query instruction is used for bge-base-en-v1.5 (the model card recommends one for
  retrieval queries), and the static models get none either. Model 1's scores can be a bit
  lower than published MTEB values for this reason. **Unverified, not compared in this
  experiment.**
- Only the corpus `text` column is embedded and used for training. The `title` column is not
  used.
- FiQA has 38 corpus documents with empty text. Their featurized target is NaN (no tokens after
  CLS and SEP are removed), so tokenlearn drops them from training (57,638 → 57,600 rows).
  They are still in the search index.
- model2vec reloads the last epoch checkpoint after early stopping, not the checkpoint with the
  best validation loss. The difference in validation loss is small (see `train_info/`).
- testoperator sorts results with the same score in hash-map order, which is not deterministic.
  Of the 9 runs that were done twice (next item), 8 gave the same metric tables. One NFCorpus run
  differed at k = 4–5 only, by at most 0.0007 (precision@4); its k = 10 values were the same.
- 9 runs first failed the testoperator health check (latency over 125 ms) because the host was
  under very high load from the parallel runs. Their metric tables were complete. They were run
  again with 2 workers, and all passed; `run.attempt1.json` keeps the first attempts. The results
  use the passed runs only.
- The timing and memory metrics in `results/tables.md` come from runs that were done at the same
  time on one host. They are not a performance comparison.
- spiced logs a deprecation warning for the `arrow` accelerator engine. The engine was still used.

## Provenance

- spiced: `~/.spice/bin/spiced` v2.5.0-unstable+models.metal (60eaf209cd). Not rebuilt.
- testoperator: built from the spiceai/spiceai#16623 branch with
  `cargo build -p testoperator --release --features models`, from the working tree that was then
  committed as b91e686ad5. All 99 runs used the same binary (SHA-256 `120f804fb7b1eb1c…`, in each
  `run.json`).
- Models: `BAAI/bge-base-en-v1.5` at a5beb1e3e68b9ab74eb54cfd186867f64f240e1a,
  `minishlab/potion-base-8M` at bf8b056651a2c21b8d2565580b8569da283cab23.
- Libraries: model2vec 0.8.2, torch 2.7.1, lightning 2.6.1, sentence-transformers 5.4.1,
  transformers 5.3.0, tokenizers 0.22.2, datasets 4.8.4.
- OTLP collector: `otel/opentelemetry-collector-contrib:0.120.0`.
- Featurize ran on MPS. Fine-tunes ran on CPU, with 2 torch threads each.

## Reproduce

Paths below are relative to this directory. Models and featurized data go to
`~/Github/tokenlearn/{models,data}/static-finetune/`; they are not in git.

```bash
PY=~/Github/tokenlearn/.venv/bin/python

# 1. Data and models.
$PY scripts/provenance.py
$PY scripts/distill.py                      # model 2
$PY scripts/run_all.py --workers 4          # featurize each corpus, then 30 fine-tunes

# 2. Spicepods and the OTLP collector.
$PY scripts/make_spicepods.py
docker run -d --name static-finetune-otel -p 127.0.0.1:4317:4317 \
  -v "$PWD/otel/collector.yaml:/etc/otelcol-contrib/config.yaml:ro" \
  -v ~/Github/tokenlearn/data/static-finetune/otel:/out \
  otel/opentelemetry-collector-contrib:0.120.0

# 3. Search runs (skips runs that already passed).
$PY scripts/run_benchmarks.py --workers 6

# 4. Tables.
$PY scripts/collect.py
$PY scripts/analyze.py
```
