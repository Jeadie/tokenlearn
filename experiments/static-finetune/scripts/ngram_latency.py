"""Encode latency with and without n-grams, through model2vec-rs (bench/ngram_latency).

Runs the benchmark on FiQA's judged test queries and on 1,000 FiQA corpus texts with three models:
FiQA's final model without n-grams (seed 42), the same model again (the A/A noise floor), and
FiQA's chosen n-gram model (seed 42). One text per call, single thread, release build. Writes
results/ngrams/latency.json. Run it on an otherwise idle machine.
"""

import json
import logging
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

import pandas as pd
from common import EXPERIMENT_DIR
from make_spicepods import DATA_DIR
from plans import FINAL_SEEDS, best_ngram_setting, best_setting
from sweep_finetune import SWEEP_MODELS_DIR

logger = logging.getLogger(__name__)

BENCH_DIR = EXPERIMENT_DIR / "bench" / "ngram_latency"
OUT_PATH = EXPERIMENT_DIR / "results" / "ngrams" / "latency.json"
# Passes over each text set: enough samples for P99.9 to have 10 or more samples above it.
PASSES = {"queries": 40, "corpus": 20}
CORPUS_TEXTS = 1000


def texts() -> dict[str, list[str]]:
    """FiQA's judged test queries, and the first corpus texts."""
    queries = pd.read_parquet(DATA_DIR / "fiqa" / "queries.parquet")
    judged = set(pd.read_parquet(DATA_DIR / "fiqa" / "qrels_test.parquet")["query-id"].astype(str))
    corpus = pd.read_parquet(DATA_DIR / "fiqa" / "corpus.parquet")
    return {
        "queries": queries[queries["_id"].astype(str).isin(judged)]["text"].tolist(),
        "corpus": corpus["text"].head(CORPUS_TEXTS).tolist(),
    }


def main() -> None:
    """Build the benchmark, run it on both text sets, and write the percentiles."""
    subprocess.run(["cargo", "build", "--release", "--quiet"], cwd=BENCH_DIR, check=True)
    binary = BENCH_DIR / "target" / "release" / "ngram_latency"
    base = SWEEP_MODELS_DIR / replace(best_setting("fiqa", min_seeds=len(FINAL_SEEDS)), seed=42).name
    ngram = SWEEP_MODELS_DIR / replace(best_ngram_setting("fiqa"), seed=42).name
    models = [f"no-ngrams={base}", f"no-ngrams-again={base}", f"ngrams:{ngram.name}={ngram}"]
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, values in texts().items():
            path = Path(tmp) / f"{name}.json"
            path.write_text(json.dumps(values))
            output = subprocess.run(
                [str(binary), str(path), str(PASSES[name]), *models], capture_output=True, text=True, check=True
            ).stdout
            for line in output.splitlines():
                results.append({"texts": name, **json.loads(line)})
                logger.info(json.dumps(results[-1]))
    OUT_PATH.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
