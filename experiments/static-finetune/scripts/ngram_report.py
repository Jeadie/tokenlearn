"""Results of the n-gram experiment (NGRAMS.md): gates, dev grid, confirmation, test and forgetting.

Reads search scores from results/ngrams/scores.jsonl (and the first sweep's scores for the old
build's runs), per-query metrics from the OTLP collector's output, and training records from
train_info_sweep/. Writes results/ngrams/report.md and paired_comparisons.csv. Sections whose
runs do not exist yet are left out.

All comparisons are paired over queries: a model with several seeds scores each query with its
mean over seeds, and the 95% CI is a bootstrap over queries (10,000 resamples).
"""

import json
import logging
from dataclasses import replace

import numpy as np
import pandas as pd
from analyze import to_markdown
from common import DATASETS, EXPERIMENT_DIR
from plans import (
    FINAL_SEEDS,
    NGRAM_OLD_TAG,
    NGRAM_SCORES,
    NGRAM_SEEDS,
    NGRAM_TAG,
    best_ngram_setting,
    best_setting,
    ngram_grid,
    zero_name,
)
from sweep import SCORES_PATH, TrainSpec
from make_spicepods import DATA_DIR
from ngrams import NgramEncoder
from sweep_final import Entry, check, compare, metric_table, read_runs
from sweep_finetune import SWEEP_INFO_DIR, SWEEP_MODELS_DIR

logger = logging.getLogger(__name__)

OUT_DIR = EXPERIMENT_DIR / "results" / "ngrams"
SPICED_CHECK = OUT_DIR / "spiced_check.json"


def passed_runs() -> set[tuple[str, str]]:
    """(run name, eval dataset) of every passed search run, from both score files."""
    runs = set()
    for path in (SCORES_PATH, NGRAM_SCORES):
        if path.exists():
            for record in map(json.loads, path.read_text().splitlines()):
                if record["exit_code"] == 0 and record["metrics"]:
                    runs.add((record["run_name"], record["eval_dataset"]))
    return runs


def run(spec: TrainSpec, split: str, tag: str = NGRAM_TAG) -> str:
    """The run name of one model on one split."""
    return f"{spec.name}-{split}" + (f"-{tag}" if tag else "")


def entry(key: str, label: str, specs: list[TrainSpec], split: str, tag: str = NGRAM_TAG) -> Entry:
    """A table entry over the given seeds."""
    return Entry(key=key, label=label, runs=tuple(run(s, split, tag) for s in specs))


def seeds(spec: TrainSpec, values: tuple[int, ...]) -> list[TrainSpec]:
    """`spec` with each seed."""
    return [replace(spec, seed=seed) for seed in values]


def label(spec: TrainSpec) -> str:
    """`2-grams, K=16k` for an n-gram setting; `no n-grams` without."""
    if not spec.ngram_k:
        return "no n-grams"
    orders = "+".join(map(str, spec.ngram_orders))
    return f"{orders}-grams, K={spec.ngram_k // 1000}k"


def info(spec: TrainSpec) -> dict:
    """The training record of one model."""
    return json.loads((SWEEP_INFO_DIR / f"{spec.name}.json").read_text())


def available(e: Entry, dataset: str, have: set[tuple[str, str]]) -> bool:
    """True when every run of an entry passed."""
    return all((r, dataset) in have for r in e.runs)


def ndcg10(per_query: pd.DataFrame, e: Entry, dataset: str) -> pd.Series:
    """Per-query NDCG@10 of each run of an entry, one column per run."""
    rows = per_query[(per_query["dataset"] == dataset) & per_query["run"].isin(e.runs)]
    rows = rows[(rows["metric"] == "ndcg") & (rows["k"] == 10)]
    return rows.pivot_table(index="query_id", columns="run", values="value")


def same_runs(per_query: pd.DataFrame, dataset: str, a: tuple[str, ...], b: tuple[str, ...]) -> dict:
    """How far apart two lists of runs (pairwise) are, query by query, at every metric@k."""
    diffs = []
    for run_a, run_b in zip(a, b, strict=True):
        left = per_query[(per_query["dataset"] == dataset) & (per_query["run"] == run_a)]
        right = per_query[(per_query["dataset"] == dataset) & (per_query["run"] == run_b)]
        joined = left.merge(right, on=["query_id", "metric", "k"], suffixes=("_a", "_b"))
        diffs.append((joined["value_a"] - joined["value_b"]).abs().to_numpy())
    flat = np.concatenate(diffs)
    return {"values": len(flat), "different": int((flat > 1e-9).sum()), "max_abs_diff": float(flat.max())}


def gate_lines(per_query: pd.DataFrame, have: set[tuple[str, str]]) -> list[str]:
    """G1 (vectors and search), G2 (vectors) and G3 (search)."""
    lines = ["## Gates", ""]
    if SPICED_CHECK.exists():
        lines += ["Single text vectors through `/v1/embeddings` (`check_ngram_spiced.py`):", ""]
        checks = pd.DataFrame(json.loads(SPICED_CHECK.read_text()))
        lines += [to_markdown(checks, index=False), ""]
    rows = []
    for dataset in DATASETS:
        best = best_setting(dataset, min_seeds=len(FINAL_SEEDS))
        for split in ("dev", "test"):
            old = entry("old", "", seeds(best, FINAL_SEEDS), split, tag=NGRAM_OLD_TAG)
            new = entry("new", "", seeds(best, FINAL_SEEDS), split)
            if available(old, dataset, have) and available(new, dataset, have):
                rows.append(
                    {"gate": "G1", "eval": dataset, "split": split, "runs": "old build vs new build, 5 seeds"}
                    | same_runs(per_query, dataset, old.runs, new.runs)
                )
        zero = (f"{zero_name(dataset)}-dev-{NGRAM_TAG}",)
        base = (run(replace(best, seed=42), "dev"),)
        if all((r, dataset) in have for r in zero + base):
            rows.append(
                {"gate": "G3", "eval": dataset, "split": "dev", "runs": "zero n-gram rows vs none, seed 42"}
                | same_runs(per_query, dataset, zero, base)
            )
    if rows:
        lines += ["Search results, every metric@k of every query (`different`: |Δ| > 1e-9):", ""]
        lines += [to_markdown(pd.DataFrame(rows), index=False), ""]
    return lines


def grid_lines(per_query: pd.DataFrame, have: set[tuple[str, str]], comparisons: list[dict]) -> list[str]:
    """Dev NDCG@10 of each n-gram setting next to no n-grams, seeds 42 and 43."""
    lines = ["## Dev grid (seeds 42 and 43)", ""]
    for dataset in DATASETS:
        best = best_setting(dataset, min_seeds=len(FINAL_SEEDS))
        base = entry("none", "no n-grams", seeds(best, NGRAM_SEEDS), "dev")
        if not available(base, dataset, have):
            continue
        base_scores = ndcg10(per_query, base, dataset).mean(axis=1)
        rows = [
            {
                "setting": "no n-grams",
                "NDCG@10": f"{base_scores.mean():.4f}",
                "Δ [95% CI]": "",
                "n-grams": 0,
                "n-grams per text": "",
                "best epoch": np.mean([info(s)["best_epoch"] for s in seeds(best, NGRAM_SEEDS)]),
                "train min": round(np.mean([info(s)["wall_time_s"] for s in seeds(best, NGRAM_SEEDS)]) / 60, 1),
            }
        ]
        for spec in [t for t in ngram_grid().train if t.dataset == dataset and t.seed == NGRAM_SEEDS[0]]:
            e = entry(label(spec), label(spec), seeds(spec, NGRAM_SEEDS), "dev")
            if not available(e, dataset, have):
                continue
            result = compare(per_query, dataset, e, base, "dev grid")
            comparisons += result
            delta = next(r for r in result if r["metric"] == "ndcg@10")
            infos = [info(s) for s in seeds(spec, NGRAM_SEEDS)]
            rows.append(
                {
                    "setting": label(spec),
                    "NDCG@10": f"{ndcg10(per_query, e, dataset).mean(axis=1).mean():.4f}",
                    "Δ [95% CI]": f"{delta['mean Δ']} {delta['95% CI']}",
                    "n-grams": infos[0]["n_ngrams"],
                    "n-grams per text": np.mean([i["ngrams_per_train_text"] for i in infos]),
                    "best epoch": np.mean([i["best_epoch"] for i in infos]),
                    "train min": round(np.mean([i["wall_time_s"] for i in infos]) / 60, 1),
                }
            )
        lines += [f"### {dataset}", "", to_markdown(pd.DataFrame(rows), index=False), ""]
    return lines


def coverage_lines() -> list[str]:
    """How many listed n-grams dev queries and corpus texts contain (seed-42 model of each grid setting)."""
    rows = []
    for spec in [t for t in ngram_grid().train if t.seed == NGRAM_SEEDS[0]]:
        path = SWEEP_MODELS_DIR / spec.name
        if not path.exists():
            continue
        encoder = NgramEncoder(path)
        queries = pd.read_parquet(DATA_DIR / spec.dataset / "queries.parquet")
        dev_ids = set(pd.read_parquet(DATA_DIR / spec.dataset / "qrels_dev.parquet")["query-id"].astype(str))
        texts = {
            "dev queries": queries[queries["_id"].astype(str).isin(dev_ids)]["text"].tolist(),
            "corpus": pd.read_parquet(DATA_DIR / spec.dataset / "corpus.parquet")["text"].head(2000).tolist(),
        }
        row = {"eval": spec.dataset, "setting": label(spec), "n-grams": len(encoder.table.ngrams)}
        for name, values in texts.items():
            tokens = [encoder.token_ids(text) for text in values]
            found = [len(encoder.table.find(ids)) for ids in tokens]
            row[f"{name}: tokens"] = round(float(np.mean([len(t) for t in tokens])), 1)
            row[f"{name}: n-grams"] = round(float(np.mean(found)), 1)
            row[f"{name}: with any"] = f"{np.mean([f > 0 for f in found]):.0%}"
        rows.append(row)
    if not rows:
        return []
    return [
        "## Coverage: listed n-grams per text (seed-42 models; corpus: first 2,000 texts)",
        "",
        to_markdown(pd.DataFrame(rows), index=False),
        "",
    ]


def chosen(dataset: str) -> TrainSpec | None:
    """The best n-gram setting of a dataset, or None before the grid has scores."""
    try:
        return best_ngram_setting(dataset)
    except (FileNotFoundError, ValueError, StopIteration):
        return None


def final_lines(per_query: pd.DataFrame, have: set[tuple[str, str]], comparisons: list[dict]) -> list[str]:
    """The best n-gram setting of each dataset (5 seeds) against no n-grams: dev, test and forgetting."""
    lines = []
    for split, title in (("dev", "Confirmation on dev (5 seeds)"), ("test", "Test (5 seeds)")):
        section = [f"## {title}", ""]
        for dataset in DATASETS:
            spec = chosen(dataset)
            best = best_setting(dataset, min_seeds=len(FINAL_SEEDS))
            if spec is None:
                continue
            ng = entry("ngrams", f"with n-grams ({label(spec)})", seeds(spec, FINAL_SEEDS), split)
            base = entry("none", "no n-grams", seeds(best, FINAL_SEEDS), split)
            if not (available(ng, dataset, have) and available(base, dataset, have)):
                continue
            result = compare(per_query, dataset, ng, base, split)
            comparisons += result
            section += [f"### {dataset}", "", to_markdown(metric_table(per_query, dataset, "ndcg", [base, ng])), ""]
            shown = [r for r in result if r["metric"] in ("ndcg@1", "ndcg@10", "recall@10", "mrr@10")]
            section += [to_markdown(pd.DataFrame(shown), index=False), ""]
        lines += section if len(section) > 2 else []
    section = ["## Forgetting: each dataset's models on the other datasets' test queries", ""]
    rows = []
    for dataset in DATASETS:
        spec = chosen(dataset)
        best = best_setting(dataset, min_seeds=len(FINAL_SEEDS))
        if spec is None:
            continue
        for other in (d for d in DATASETS if d != dataset):
            ng = entry("ngrams", "", seeds(spec, FINAL_SEEDS), "test")
            base = entry("none", "", seeds(best, FINAL_SEEDS), "test")
            if available(ng, other, have) and available(base, other, have):
                result = compare(per_query, other, ng, base, f"trained on {dataset}")
                comparisons += result
                rows += [r for r in result if r["metric"] == "ndcg@10"]
    if rows:
        lines += section + [to_markdown(pd.DataFrame(rows), index=False), ""]
    return lines


def main() -> None:
    """Write the report."""
    have = passed_runs()
    wanted = set()
    for dataset in DATASETS:
        best = best_setting(dataset, min_seeds=len(FINAL_SEEDS))
        specs = seeds(best, FINAL_SEEDS) + [t for t in ngram_grid().train if t.dataset == dataset]
        if (spec := chosen(dataset)) is not None:
            specs += seeds(spec, FINAL_SEEDS)
        names = {s.name for s in specs} | {zero_name(dataset)}
        wanted |= {(r, d) for r, d in have if any(r.startswith(n + "-") for n in names)}
    per_query, run_level = read_runs(wanted)
    check(per_query, run_level)
    comparisons: list[dict] = []
    lines = ["# N-gram experiment: results", "", "Generated by `scripts/ngram_report.py`.", ""]
    lines += gate_lines(per_query, have)
    lines += grid_lines(per_query, have, comparisons)
    lines += coverage_lines()
    lines += final_lines(per_query, have, comparisons)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.md").write_text("\n".join(lines))
    pd.DataFrame(comparisons).to_csv(OUT_DIR / "paired_comparisons.csv", index=False)
    logger.info(f"Wrote {OUT_DIR / 'report.md'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
