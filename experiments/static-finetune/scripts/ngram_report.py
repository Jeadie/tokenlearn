"""Results of the n-gram experiment (NGRAMS.md): gates, dev grid, confirmation, test and forgetting.

Reads search scores from results/ngrams/scores.jsonl, per-query metrics from the OTLP collector's output, and training records from
train_info_sweep/. Writes results/ngrams/report.md and paired_comparisons.csv. Sections whose
runs do not exist yet are left out.

All comparisons are paired over queries: a model with several seeds scores each query with its
mean over seeds, and the 95% CI is a bootstrap over queries (10,000 resamples).
"""

import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from analyze import to_markdown
from common import DATASETS, EXPERIMENT_DIR
from make_spicepods import DATA_DIR
from ngrams import NgramEncoder
from plans import (
    FINAL_SEEDS,
    NGRAM_OLD_TAG,
    NGRAM_SCORES,
    NGRAM_SEEDS,
    NGRAM_TAG,
    best_ngram_setting,
    best_setting,
    ngram_forgetting_variants,
    ngram_settings,
    zero_name,
)
from sweep import SCORES_PATH, TrainSpec
from sweep_final import Entry, check, compare, metric_table, read_runs
from sweep_finetune import SWEEP_INFO_DIR, SWEEP_MODELS_DIR

logger = logging.getLogger(__name__)

OUT_DIR = EXPERIMENT_DIR / "results" / "ngrams"
SPICED_CHECK = OUT_DIR / "spiced_check.json"


def passed_runs() -> set[tuple[str, str]]:
    """(run name, eval dataset) of every passed search run of this part (both builds)."""
    records = map(json.loads, NGRAM_SCORES.read_text().splitlines())
    return {(r["run_name"], r["eval_dataset"]) for r in records if r["exit_code"] == 0 and r["metrics"]}


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
    scale = f", n-gram lr ×{spec.ngram_lr_scale:g}" if spec.ngram_lr_scale != 1 else ""
    return f"{orders}-grams, K={spec.ngram_k // 1000}k{scale}"


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


def run_level(path: Path, tag: str) -> dict[tuple[str, str, str], dict]:
    """metric@k of each passed run with this tag: (model, eval dataset, split) -> metrics."""
    out = {}
    for record in map(json.loads, path.read_text().splitlines()):
        if record.get("tag", "") == tag and record["exit_code"] == 0 and record["metrics"]:
            out[(record["model"], record["eval_dataset"], record["split"])] = record["metrics"]
    return out


def rerun_lines() -> list[str]:
    """Run-level metric@k of the same models searched twice: with the same build, and with each build.

    The first searches (Part 2) lost their per-query metrics, so this compares run-level values only.
    """
    first = run_level(SCORES_PATH, "")
    old = run_level(NGRAM_SCORES, NGRAM_OLD_TAG)
    new = run_level(NGRAM_SCORES, NGRAM_TAG)
    rows = []
    for name, a, b in (("old build, Part 2 vs again", first, old), ("old build vs new build", old, new)):
        for dataset in DATASETS:
            keys = [k for k in b if k in a and k[1] == dataset]
            diffs = [abs(a[k][m][i] - b[k][m][i]) for k in keys for m in a[k] for i in a[k][m]]
            ndcg = [abs(a[k]["ndcg"]["10"] - b[k]["ndcg"]["10"]) for k in keys]
            rows.append(
                {
                    "runs compared": name,
                    "eval": dataset,
                    "runs": len(keys),
                    "metric@k values": len(diffs),
                    "different": sum(d > 0 for d in diffs),
                    "max |Δ|": round(max(diffs), 4),
                    "max |Δ| NDCG@10": round(max(ndcg), 4),
                }
            )
    return [
        "The same models searched twice, run-level metric@1..10 (4 metrics × 10 k per run):",
        "",
        to_markdown(pd.DataFrame(rows), index=False),
        "",
    ]


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
        for spec, _ in sorted(ngram_settings(dataset).values(), key=lambda item: label(item[0])):
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
                    "n-grams per text": round(float(np.mean([i["ngrams_per_train_text"] for i in infos])), 1),
                    "best epoch": float(np.mean([i["best_epoch"] for i in infos])),
                    "train min": round(np.mean([i["wall_time_s"] for i in infos]) / 60, 1),
                }
            )
        lines += [f"### {dataset}", "", to_markdown(pd.DataFrame(rows), index=False), ""]
    return lines


def coverage_lines() -> list[str]:
    """How many listed n-grams dev queries and corpus texts contain (seed-42 model of each grid setting)."""
    rows = []
    specs = [spec for dataset in DATASETS for spec, _ in ngram_settings(dataset).values() if spec.ngram_lr_scale == 1]
    for spec in sorted(specs, key=lambda s: (s.dataset, label(s))):
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


def candidate_lines(per_query: pd.DataFrame, have: set[tuple[str, str]], comparisons: list[dict]) -> list[str]:
    """FiQA's forgetting candidates (5 seeds) on each dataset's test queries, next to the chosen n-gram model."""
    chosen_spec = chosen("fiqa")
    if chosen_spec is None:
        return []
    base = entry("none", "no n-grams", seeds(best_setting("fiqa", min_seeds=len(FINAL_SEEDS)), FINAL_SEEDS), "test")
    picked = entry("chosen", "chosen", seeds(chosen_spec, FINAL_SEEDS), "test")
    rows = []
    for variant in ngram_forgetting_variants():
        name = label(variant) + (" + C4 replay" if "c4" in variant.extras else "")
        candidate = entry("candidate", name, seeds(variant, FINAL_SEEDS), "test")
        for dataset in DATASETS:
            if not all(available(e, dataset, have) for e in (base, picked, candidate)):
                continue
            result = compare(per_query, dataset, candidate, base, f"FiQA candidate: {name}")
            comparisons += result
            delta = next(r for r in result if r["metric"] == "ndcg@10")
            rows.append(
                {
                    "variant": name,
                    "eval": dataset,
                    "no n-grams": f"{ndcg10(per_query, base, dataset).mean(axis=1).mean():.4f}",
                    f"chosen ({label(chosen_spec)})": f"{ndcg10(per_query, picked, dataset).mean(axis=1).mean():.4f}",
                    "candidate": f"{ndcg10(per_query, candidate, dataset).mean(axis=1).mean():.4f}",
                    "candidate − no n-grams [95% CI]": f"{delta['mean Δ']} {delta['95% CI']}",
                }
            )
    if not rows:
        return []
    return [
        "## Forgetting: FiQA candidates (test NDCG@10, 5 seeds)",
        "",
        to_markdown(pd.DataFrame(rows), index=False),
        "",
    ]


def size_lines() -> list[str]:
    """Size of each dataset's model file without n-grams and with its chosen n-grams (seed 42)."""
    rows = []
    for dataset in DATASETS:
        spec = chosen(dataset)
        if spec is None:
            continue
        base = SWEEP_MODELS_DIR / replace(best_setting(dataset, min_seeds=len(FINAL_SEEDS)), seed=42).name
        ngram = SWEEP_MODELS_DIR / replace(spec, seed=42).name
        rows.append(
            {
                "eval": dataset,
                "setting": label(spec),
                "n-grams kept": info(replace(spec, seed=42))["n_ngrams"],
                "model.safetensors without (MB)": round((base / "model.safetensors").stat().st_size / 1e6, 1),
                "with (MB)": round((ngram / "model.safetensors").stat().st_size / 1e6, 1),
            }
        )
    if not rows:
        return []
    return ["## Model size (float32, seed 42)", "", to_markdown(pd.DataFrame(rows), index=False), ""]


def main() -> None:
    """Write the report."""
    have = passed_runs()
    wanted = set()
    for dataset in DATASETS:
        best = best_setting(dataset, min_seeds=len(FINAL_SEEDS))
        specs = seeds(best, FINAL_SEEDS)
        specs += [s for spec, _ in ngram_settings(dataset).values() for s in seeds(spec, NGRAM_SEEDS)]
        if (spec := chosen(dataset)) is not None:
            specs += seeds(spec, FINAL_SEEDS)
        if dataset == "fiqa" and spec is not None:
            specs += [s for v in ngram_forgetting_variants() for s in seeds(v, FINAL_SEEDS)]
        names = {s.name for s in specs} | {zero_name(dataset)}
        wanted |= {(r, d) for r, d in have if any(r.startswith(n + "-") for n in names)}
    per_query, run_level = read_runs(wanted)
    check(per_query, run_level)
    comparisons: list[dict] = []
    lines = ["# N-gram experiment: results", "", "Generated by `scripts/ngram_report.py`.", ""]
    lines += gate_lines(per_query, have)
    lines += rerun_lines()
    lines += grid_lines(per_query, have, comparisons)
    lines += coverage_lines()
    lines += final_lines(per_query, have, comparisons)
    lines += candidate_lines(per_query, have, comparisons)
    lines += size_lines()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.md").write_text("\n".join(lines))
    pd.DataFrame(comparisons).to_csv(OUT_DIR / "paired_comparisons.csv", index=False)
    logger.info(f"Wrote {OUT_DIR / 'report.md'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
