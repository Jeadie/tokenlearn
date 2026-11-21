"""Test-query results of the sweep's chosen settings, next to every earlier model.

Reads per-query metrics from the OTLP collector's output (the newest export of each run), and
writes results/final/tables.md and CSVs:
- metric@k (k = 1..10) for ndcg, recall, mrr and precision, per eval dataset. Multi-seed models
  show mean ± sample std over seeds.
- paired bootstrap 95% CIs over queries (10,000 resamples) for the main comparisons at k = 1, 5
  and 10. A multi-seed model's per-query score is its mean over seeds.

Before the tables, it checks that each run's metric@k equals the mean of its per-query values.
"""

import json
import logging
from dataclasses import dataclass, replace

import pandas as pd
from analyze import BOOTSTRAP_RESAMPLES, bootstrap_ci, fmt, to_markdown
from collect import AT_K_METRICS, OTLP_PATH, PER_QUERY_METRICS, attrs, point_value
from common import EXPERIMENT_DIR
from plans import FINAL_SEEDS, best_setting, full_epochs

logger = logging.getLogger(__name__)

FINAL_DIR = EXPERIMENT_DIR / "results" / "final"
DATASETS = ("scifact", "nfcorpus", "fiqa")
METRICS = ("ndcg", "recall", "mrr", "precision")
KS = tuple(range(1, 11))


@dataclass(frozen=True)
class Entry:
    """One model in the tables: a label and the run names of its seeds on one dataset."""

    label: str
    runs: tuple[str, ...]


def entries(dataset: str) -> list[Entry]:
    """Every model to report on one dataset, in table order."""
    best = best_setting(dataset, min_seeds=len(FINAL_SEEDS))
    full = replace(best, full_epochs=full_epochs(best))
    setting = best.name.removesuffix("-s42")
    full_setting = full.name.removesuffix("-s42")
    return [
        Entry("1: bge-base-en-v1.5 (CLS)", ("hf[bge-base-en-v1.5]-arrow",)),
        Entry("1m: bge-base-en-v1.5 (mean pooling)", ("hf-mean[bge-base-en-v1.5]-test",)),
        Entry("2: distilled", ("model2vec[bge-base-en-v1.5-distill-256]-arrow",)),
        Entry("3: potion-base-8M", ("model2vec[potion-base-8M]-arrow",)),
        Entry(
            "4: old recipe, potion", tuple(f"model2vec[potion-base-8M[ft-{dataset}-s{s}]]-arrow" for s in FINAL_SEEDS)
        ),
        Entry(
            "4b: old recipe, distilled",
            tuple(f"model2vec[bge-base-en-v1.5-distill-256[ft-{dataset}-s{s}]]-arrow" for s in FINAL_SEEDS),
        ),
        Entry(f"new: {setting}", tuple(f"{setting}-s{s}-test" for s in FINAL_SEEDS)),
        Entry(f"new, 100% corpus: {full_setting}", tuple(f"{full_setting}-s{s}-test" for s in FINAL_SEEDS)),
    ]


def parse_points(resource_metrics: dict, key: tuple[str, str]) -> tuple[int, list[tuple], list[tuple]]:
    """The export time, per-query rows and run-level rows of one exported run."""
    time_ns, query_rows, run_rows = 0, [], []
    for scope in resource_metrics.get("scopeMetrics", []):
        for metric in scope.get("metrics", []):
            for point in metric.get("gauge", {}).get("dataPoints", []):
                time_ns = max(time_ns, int(point.get("timeUnixNano", 0)))
                p = attrs(point.get("attributes"))
                if metric["name"] in PER_QUERY_METRICS:
                    name = PER_QUERY_METRICS[metric["name"]]
                    query_rows.append((*key, p["query_id"], name, int(p["k"]), point_value(point)))
                elif metric["name"] in AT_K_METRICS and "k" in p:
                    run_rows.append((*key, AT_K_METRICS[metric["name"]], int(p["k"]), point_value(point)))
    return time_ns, query_rows, run_rows


def read_runs(wanted: set[tuple[str, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-query and run-level metrics of the wanted (run name, dataset) pairs, newest export each."""
    newest: dict[tuple[str, str], tuple[int, list, list]] = {}
    with OTLP_PATH.open() as lines:
        for line in filter(str.strip, lines):
            for resource_metrics in json.loads(line).get("resourceMetrics", []):
                resource = attrs(resource_metrics.get("resource", {}).get("attributes"))
                key = (resource.get("name", ""), resource.get("benchmark_dataset", "").removesuffix("_retrieval"))
                if resource.get("type") != "search" or key not in wanted:
                    continue
                parsed = parse_points(resource_metrics, key)
                if key not in newest or parsed[0] >= newest[key][0]:
                    newest[key] = parsed
    missing = wanted - set(newest)
    if missing:
        raise SystemExit(f"no export for {sorted(missing)}")
    per_query = pd.DataFrame(
        [r for _, q, _ in newest.values() for r in q], columns=["run", "dataset", "query_id", "metric", "k", "value"]
    )
    run_level = pd.DataFrame(
        [r for _, _, run in newest.values() for r in run], columns=["run", "dataset", "metric", "k", "value"]
    )
    return per_query, run_level


def check(per_query: pd.DataFrame, run_level: pd.DataFrame) -> None:
    """Each run-level metric@k must equal the mean of the run's per-query values."""
    means = per_query.groupby(["run", "dataset", "metric", "k"], as_index=False)["value"].mean()
    joined = run_level.merge(means, on=["run", "dataset", "metric", "k"], suffixes=("_run", "_mean"))
    worst = float((joined["value_run"] - joined["value_mean"]).abs().max())
    logger.info(f"run-level vs per-query mean: {len(joined)} pairs of {len(run_level)}, max |diff| = {worst:.2e}")
    if len(joined) != len(run_level) or worst > 1e-9:
        raise SystemExit("per-query data does not reproduce the run-level metrics")


def scores(per_query: pd.DataFrame, entry: Entry, dataset: str) -> pd.DataFrame:
    """Per-query values of one entry, one column per seed; index (query_id, metric, k)."""
    rows = per_query[(per_query["dataset"] == dataset) & per_query["run"].isin(entry.runs)]
    return rows.pivot_table(index=["query_id", "metric", "k"], columns="run", values="value")


def metric_table(per_query: pd.DataFrame, dataset: str, metric: str) -> pd.DataFrame:
    """metric@1..10 of every entry on one dataset; mean ± std over seeds."""
    table = {}
    for entry in entries(dataset):
        per_seed = scores(per_query, entry, dataset).xs(metric, level="metric").groupby(level="k").mean()
        multi = per_seed.shape[1] > 1
        label = entry.label + (f" (n={per_seed.shape[1]} seeds)" if multi else "")
        table[label] = {
            f"@{k}": fmt(float(row.mean()), float(row.std(ddof=1)) if multi else None) for k, row in per_seed.iterrows()
        }
    return pd.DataFrame.from_dict(table, orient="index").reindex(columns=[f"@{k}" for k in KS])


def comparison_rows(per_query: pd.DataFrame, dataset: str) -> list[dict]:
    """Paired bootstrap comparisons of the new model with the others."""
    by_label = {e.label.split(":")[0]: e for e in entries(dataset)}
    new, new_full = by_label["new"], by_label["new, 100% corpus"]
    pairs = [
        (new, by_label["4"]),
        (new, by_label["4b"]),
        (new, by_label["3"]),
        (new, by_label["2"]),
        (new_full, new),
        (by_label["1"], new),
        (by_label["1m"], by_label["1"]),
    ]
    rows = []
    for a, b in pairs:
        a_scores = scores(per_query, a, dataset).mean(axis=1)
        b_scores = scores(per_query, b, dataset).mean(axis=1)
        for metric in METRICS:
            for k in (1, 5, 10):
                paired = pd.concat(
                    [a_scores.xs((metric, k), level=("metric", "k")), b_scores.xs((metric, k), level=("metric", "k"))],
                    axis=1,
                    join="inner",
                ).dropna()
                deltas = (paired.iloc[:, 0] - paired.iloc[:, 1]).to_numpy()
                low, high = bootstrap_ci(deltas)
                rows.append(
                    {
                        "eval": dataset,
                        "A − B": f"{a.label.split(':')[0]} − {b.label.split(':')[0]}",
                        "metric": f"{metric}@{k}",
                        "queries": len(deltas),
                        "mean Δ": f"{deltas.mean():+.4f}",
                        "95% CI": f"[{low:+.4f}, {high:+.4f}]",
                        "excludes 0": "yes" if low > 0 or high < 0 else "no",
                    }
                )
    return rows


def main() -> None:
    """Write results/final/tables.md and its CSVs."""
    wanted = {(run, ds) for ds in DATASETS for e in entries(ds) for run in e.runs}
    per_query, run_level = read_runs(wanted)
    check(per_query, run_level)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    sections = []
    for dataset in DATASETS:
        labels = "\n".join(f"- {e.label}: {len(e.runs)} run(s)" for e in entries(dataset))
        sections.append(f"## {dataset}\n\n{labels}\n")
        for metric in METRICS:
            table = metric_table(per_query, dataset, metric)
            table.to_csv(FINAL_DIR / f"{dataset}_{metric}.csv")
            sections.append(f"### {dataset}: {metric}@k (test)\n\n{to_markdown(table)}\n")
    comparisons = pd.DataFrame([row for ds in DATASETS for row in comparison_rows(per_query, ds)])
    comparisons.to_csv(FINAL_DIR / "paired_comparisons.csv", index=False)
    sections.append(
        f"## Paired comparisons (test, bootstrap, {BOOTSTRAP_RESAMPLES} resamples)\n\n"
        f"{to_markdown(comparisons, index=False)}\n"
    )
    (FINAL_DIR / "tables.md").write_text("\n".join(sections))
    logger.info(f"Wrote {FINAL_DIR / 'tables.md'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
