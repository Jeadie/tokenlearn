"""Collect the n-gram results into one JSON file for the report page: results/ngrams/page_data.json.

Reads only committed records: results/ngrams/{scores.jsonl, paired_comparisons.csv, gates.csv,
spiced_check.json, latency.json}, Part 2's results/final/<dataset>_ndcg.csv (the bge rows), and
the model files (sizes). Run ngram_report.py first. The file holds one table per page section and
this script's own text, so readers can check each number.
"""

import json
import logging
import re
import statistics
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd
from common import DATASETS, EXPERIMENT_DIR
from plans import (
    FINAL_SEEDS,
    NGRAM_SCORES,
    NGRAM_TAG,
    best_ngram_setting,
    best_setting,
    ngram_forgetting_variants,
    without_ngrams,
)
from sweep import TrainSpec
from sweep_finetune import SWEEP_MODELS_DIR

logger = logging.getLogger(__name__)

OUT_DIR = EXPERIMENT_DIR / "results" / "ngrams"
OUT_PATH = OUT_DIR / "page_data.json"
BGE_ROWS = {
    "1: bge-base-en-v1.5 (CLS)": "bge-base-en-v1.5, CLS pooling",
    "1m: bge-base-en-v1.5 (mean pooling)": "bge-base-en-v1.5, mean pooling",
}
NAMES = {"scifact": "SciFact", "nfcorpus": "NFCorpus", "fiqa": "FiQA"}


@dataclass
class Score:
    """Mean test NDCG@10 of one model over its seeds."""

    eval: str
    model: str
    ndcg10: float
    sd: float | None
    seeds: int


@dataclass
class Delta:
    """One paired comparison (A − B) of NDCG@10 or another metric, with its bootstrap 95% CI."""

    group: str
    trained_on: str
    eval: str
    label: str
    metric: str
    queries: int
    delta: float
    lo: float
    hi: float
    excludes0: bool


@dataclass
class Latency:
    """Encode latency percentiles of one model, in one model order."""

    texts: str
    order: str
    model: str
    samples: int
    p50_us: float
    p99_us: float
    p99_9_us: float


@dataclass
class Size:
    """model.safetensors size without and with the chosen n-gram rows (seed 42)."""

    eval: str
    ngrams_kept: int
    without_mb: float
    with_mb: float


@dataclass
class Gate:
    """One gate check."""

    gate: str
    what: str
    compared: int
    different: int
    worst: float


def ndcg10(model: str, dataset: str, split: str = "test") -> list[float]:
    """Run-level NDCG@10 of a model's passed n-gram-build runs on a dataset."""
    out = []
    for record in map(json.loads, NGRAM_SCORES.read_text().splitlines()):
        if (record["model"], record["eval_dataset"], record["split"]) != (model, dataset, split):
            continue
        if record.get("tag", "") == NGRAM_TAG and record["exit_code"] == 0 and record["metrics"]:
            out.append(record["metrics"]["ndcg"]["10"])
    return out


def score(dataset: str, label: str, spec: TrainSpec) -> Score:
    """Mean and sample std over FINAL_SEEDS."""
    values = [v for seed in FINAL_SEEDS for v in ndcg10(replace(spec, seed=seed).name, dataset)]
    if len(values) != len(FINAL_SEEDS):
        raise ValueError(f"{spec.name} on {dataset}: {len(values)} runs, expected {len(FINAL_SEEDS)}")
    # Run-level scores are recorded to 4 decimals, so a mean can differ from the per-query mean in
    # report.md by up to 0.0001.
    return Score(NAMES[dataset], label, statistics.mean(values), statistics.stdev(values), len(values))


def test_scores() -> list[Score]:
    """Bge (Part 2), no n-grams, with n-grams, and FiQA's n-gram model with C4 replay; in domain, test."""
    rows = []
    for dataset in DATASETS:
        table = pd.read_csv(EXPERIMENT_DIR / "results" / "final" / f"{dataset}_ndcg.csv", index_col=0)
        for key, label in BGE_ROWS.items():
            rows.append(Score(NAMES[dataset], label, float(table.loc[key, "@10"]), None, 1))
        rows.append(score(dataset, "No n-grams (Part 2 final)", best_setting(dataset, min_seeds=len(FINAL_SEEDS))))
        rows.append(score(dataset, "With n-grams", best_ngram_setting(dataset)))
    c4 = next(v for v in ngram_forgetting_variants() if "c4" in v.extras)
    rows.append(score("fiqa", "With n-grams + C4 replay", c4))
    return rows


def cross_scores() -> list[Score]:
    """Every model the forgetting sections compare, on every dataset's test queries."""
    fiqa = [("No n-grams + C4 replay", without_ngrams(v)) for v in ngram_forgetting_variants() if "c4" in v.extras]
    fiqa += [("K=256k, n-gram lr ×0.3", v) for v in ngram_forgetting_variants() if "c4" not in v.extras]
    fiqa += [("With n-grams + C4 replay", v) for v in ngram_forgetting_variants() if "c4" in v.extras]
    rows = []
    for trained in DATASETS:
        models = [
            ("No n-grams", best_setting(trained, min_seeds=len(FINAL_SEEDS))),
            ("With n-grams", best_ngram_setting(trained)),
        ]
        models += fiqa if trained == "fiqa" else []
        for label, spec in models:
            for dataset in DATASETS:
                row = score(dataset, label, spec)
                rows.append(replace(row, model=f"{NAMES[trained]}: {label}"))
    return rows


def number(value: str | float) -> float:
    """'+0.0489', '−0.01' or a parsed number as a float."""
    return float(str(value).replace("−", "-"))


def deltas() -> list[Delta]:
    """The comparisons the page shows, from paired_comparisons.csv."""
    table = pd.read_csv(OUT_DIR / "paired_comparisons.csv")
    table.columns = ["eval", "context", "a_b", "metric", "queries", "delta", "ci", "excludes0"]
    rows = []
    for r in table.itertuples(index=False):
        if r.context == "test":
            group, trained, label = "test", r.eval, "with n-grams − no n-grams"
        elif r.metric != "ndcg@10":
            continue
        elif r.context == "dev grid":
            group, trained, label = "dev grid", r.eval, r.a_b.removesuffix(" − none")
        elif m := re.fullmatch(r"trained on (\w+)", r.context):
            group, trained, label = "forgetting", m[1], "with n-grams − no n-grams"
        elif m := re.fullmatch(r"trained on (\w+): (n-gram|token) rows", r.context):
            group, trained, label = "parts", m[1], f"{m[2]} rows"
        elif r.context.startswith("FiQA candidate: "):
            group, trained = "variants", "fiqa"
            label = r.context.removeprefix("FiQA candidate: ") + (
                " − no n-grams" if r.a_b == "candidate − none" else ""
            )
        else:
            continue
        low, high = (number(x) for x in r.ci.strip("[]").split(","))
        rows.append(
            Delta(
                group=group,
                trained_on=NAMES[trained],
                eval=NAMES[r.eval],
                label=label,
                metric=r.metric,
                queries=int(r.queries),
                delta=number(r.delta),
                lo=low,
                hi=high,
                excludes0=r.excludes0 == "yes",
            )
        )
    return rows


def latency() -> list[Latency]:
    """The encode latency benchmark's percentiles (ngram_latency.py)."""
    rows = json.loads((OUT_DIR / "latency.json").read_text())
    return [
        Latency(
            texts="FiQA test queries" if r["texts"] == "queries" else "FiQA corpus texts",
            order=r["order"],
            model="With n-grams"
            if r["model"].startswith("ngrams")
            else {"no-ngrams": "No n-grams", "no-ngrams-again": "No n-grams, again (A/A)"}[r["model"]],
            samples=r["samples"],
            p50_us=r["p50_us"],
            p99_us=r["p99_us"],
            p99_9_us=r["p99_9_us"],
        )
        for r in rows
    ]


def sizes() -> list[Size]:
    """Model file sizes, as ngram_report.py reports them."""
    rows = []
    for dataset in DATASETS:
        spec = best_ngram_setting(dataset)
        base = SWEEP_MODELS_DIR / replace(best_setting(dataset, min_seeds=len(FINAL_SEEDS)), seed=42).name
        ngram = SWEEP_MODELS_DIR / spec.name
        kept = json.loads((ngram / "config.json").read_text())["ngram_count"]
        mb = [round((path / "model.safetensors").stat().st_size / 1e6, 1) for path in (base, ngram)]
        rows.append(Size(NAMES[dataset], kept, mb[0], mb[1]))
    return rows


def gates() -> list[Gate]:
    """G1 and G2 on text vectors (check_ngram_spiced.py), G1 and G3 on search (ngram_report.py)."""
    rows = []
    for check in json.loads((OUT_DIR / "spiced_check.json").read_text()):
        what = {
            "G2": "spiced vectors vs the Python reference (min cosine)",
            "G2 rows matter": "spiced vectors vs the same model without n-gram rows (min cosine)",
            "G1": "new build vs Part 2 build, text vectors (min cosine)",
        }[check["name"]]
        rows.append(Gate(check["name"], what, check["texts"], 0, check["min_cosine"]))
    for r in pd.read_csv(OUT_DIR / "gates.csv").itertuples(index=False):
        what = f"{NAMES[r.eval]} {r.split}: {r.runs}, every metric@k of every query"
        rows.append(Gate(r.gate, what, int(r.values), int(r.different), float(r.max_abs_diff)))
    return rows


def main() -> None:
    """Write page_data.json."""
    tables: dict[str, list[Any]] = {
        "test_scores": test_scores(),
        "cross_scores": cross_scores(),
        "deltas": deltas(),
        "latency": latency(),
        "sizes": sizes(),
        "gates": gates(),
    }
    meta = [
        {"label": Path(__file__).name, "value": Path(__file__).read_text(), "type": "code", "lang": "python"},
        {
            "label": "Input",
            "value": "results/ngrams/{scores.jsonl, paired_comparisons.csv, gates.csv, spiced_check.json, latency.json}, "
            "results/final/<dataset>_ndcg.csv, model files",
        },
    ]
    out = {"meta": meta, "tables": {name: [asdict(row) for row in rows] for name, rows in tables.items()}}
    OUT_PATH.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {OUT_PATH}: " + ", ".join(f"{name} {len(rows)}" for name, rows in tables.items()))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
