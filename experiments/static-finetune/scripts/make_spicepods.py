"""Generate one embedding-only search spicepod per (model, eval dataset) and a run manifest.

Every spicepod differs only in its `embeddings:` entry. The corpus, queries and qrels are pinned
local copies of the MTEB parquet files, so every run reads identical bytes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
SPICEPOD_DIR = EXPERIMENT_DIR / "spicepods"
MANIFEST_PATH = EXPERIMENT_DIR / "manifest.json"

DATA_DIR = Path.home() / "Github/tokenlearn/data/static-finetune/mteb"
MODEL_DIR = Path.home() / "Github/tokenlearn/models/static-finetune"

DATASETS = ("scifact", "nfcorpus", "fiqa")
SEEDS = (42, 43, 44, 45, 46)
DISTILL = "bge-base-en-v1.5-distill-256"
POTION = "potion-base-8M"


@dataclass(frozen=True)
class Model:
    """An embedding model under test."""

    # Spicepod name prefix, per test/spicepods/search/README.md naming.
    label: str
    # `embeddings[].from` value.
    source: str
    # Model family: 1 = bge, 2 = distilled, 3 = potion, 4 = potion + ft, 4b = distilled + ft.
    family: str
    params: tuple[tuple[str, str | int], ...]
    train_dataset: str | None = None
    seed: int | None = None
    # Local model directory that must exist before the run can start.
    local_path: str | None = None
    # Hugging Face model id the run downloads; runs sharing one are not started together cold.
    hf_model: str | None = None


@dataclass(frozen=True)
class Run:
    """One `testoperator run search` invocation."""

    name: str
    eval_dataset: str
    spicepod_path: str
    model: Model


def static_model(family: str, label: str, path: Path, train_dataset: str | None = None, seed: int | None = None) -> Model:
    return Model(
        label=label,
        source=f"model2vec:{path}",
        family=family,
        params=(("embed_max_token_length", 512),),
        train_dataset=train_dataset,
        seed=seed,
        local_path=str(path),
    )


def models() -> list[Model]:
    result = [
        Model(
            label="hf[bge-base-en-v1.5]",
            source="huggingface:huggingface.co/BAAI/bge-base-en-v1.5",
            family="1",
            params=(("truncate", "END"),),
            hf_model="BAAI/bge-base-en-v1.5",
        ),
        static_model("2", f"model2vec[{DISTILL}]", MODEL_DIR / DISTILL),
        Model(
            label=f"model2vec[{POTION}]",
            source=f"model2vec:minishlab/{POTION}",
            family="3",
            params=(("embed_max_token_length", 512),),
            hf_model=f"minishlab/{POTION}",
        ),
    ]
    for train_dataset in DATASETS:
        for seed in SEEDS:
            variant = f"ft-{train_dataset}-s{seed}"
            result.append(
                static_model(
                    "4", f"model2vec[{POTION}[{variant}]]", MODEL_DIR / f"{POTION}-{variant}", train_dataset, seed
                )
            )
            result.append(
                static_model(
                    "4b", f"model2vec[{DISTILL}[{variant}]]", MODEL_DIR / f"{DISTILL}-{variant}", train_dataset, seed
                )
            )
    return result


def render(name: str, eval_dataset: str, model: Model) -> str:
    data = DATA_DIR / eval_dataset
    params = "".join(f"\n      {key}: {value}" for key, value in model.params)
    return f"""version: v1
kind: Spicepod
name: {name}

metadata:
  benchmark_dataset: {eval_dataset}_retrieval

datasets:

  # MTEB {eval_dataset} retrieval (https://huggingface.co/datasets/mteb/{eval_dataset}), pinned local copy.

  - from: file:{data / "corpus.parquet"}
    name: corpus
    acceleration:
      enabled: true
      engine: arrow
    columns:
      - name: text
        embeddings:
          - from: embed
            row_id:
              - _id

  - from: file:{data / "queries.parquet"}
    name: queries
    acceleration:
      enabled: true
      engine: arrow

  - from: file:{data / "qrels_test.parquet"}
    name: relevance_data
    acceleration:
      enabled: true
      engine: arrow

views:
  # Only judged queries: scoring ignores the rest, so sending them only costs time.
  - name: test_queries
    sql: SELECT _id, text FROM queries WHERE _id IN (SELECT DISTINCT "query-id" FROM relevance_data)

embeddings:
  - from: {model.source}
    name: embed
    params:{params}
"""


def main() -> None:
    runs: list[Run] = []
    for eval_dataset in DATASETS:
        (SPICEPOD_DIR / eval_dataset).mkdir(parents=True, exist_ok=True)
        for model in models():
            name = f"{model.label}-arrow"
            path = SPICEPOD_DIR / eval_dataset / f"{name}.yaml"
            path.write_text(render(name, eval_dataset, model))
            runs.append(Run(name=name, eval_dataset=eval_dataset, spicepod_path=str(path), model=model))
    MANIFEST_PATH.write_text(json.dumps([asdict(run) for run in runs], indent=2) + "\n")
    print(f"Wrote {len(runs)} spicepods and {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
