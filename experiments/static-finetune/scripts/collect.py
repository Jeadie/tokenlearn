"""Turn the OTLP collector's JSON-lines output into tidy CSVs, joined with the run manifest.

Writes:
- results/run_metrics.csv: one row per (run, metric, k) — the run-level metrics testoperator emits.
- data/static-finetune/results/per_query.parquet (outside git; ~1.7M rows): one row per
  (run, query, metric, k).

A run is keyed by its spicepod name and eval dataset. When a run was exported more than once
(a rerun), only its newest export is kept.
"""

from __future__ import annotations

import csv
import json

import pyarrow as pa
import pyarrow.parquet as pq
from collections import defaultdict
from dataclasses import astuple, dataclass, fields
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
OTLP_PATH = Path.home() / "Github/tokenlearn/data/static-finetune/otel/metrics.jsonl"
MANIFEST_PATH = EXPERIMENT_DIR / "manifest.json"
RESULTS_DIR = EXPERIMENT_DIR / "results"
PER_QUERY_PATH = Path.home() / "Github/tokenlearn/data/static-finetune/results/per_query.parquet"

PER_QUERY_METRICS = {
    "search_query_ndcg": "ndcg",
    "search_query_recall": "recall",
    "search_query_reciprocal_rank": "mrr",
    "search_query_precision": "precision",
}
AT_K_METRICS = {"score": "ndcg", "search_recall": "recall", "search_mrr": "mrr", "search_precision": "precision"}


@dataclass(frozen=True)
class RunKey:
    name: str
    eval_dataset: str


@dataclass(frozen=True)
class RunInfo:
    family: str
    train_dataset: str
    seed: str


@dataclass(frozen=True)
class RunMetricRow:
    name: str
    eval_dataset: str
    family: str
    train_dataset: str
    seed: str
    metric: str
    k: str
    value: float


@dataclass(frozen=True)
class PerQueryRow:
    name: str
    eval_dataset: str
    family: str
    train_dataset: str
    seed: str
    query_id: str
    metric: str
    k: int
    value: float


@dataclass
class Export:
    """All points of one testoperator export (one run)."""

    time_unix_nano: int
    run_points: list[tuple[str, str, float]]
    query_points: list[tuple[str, str, int, float]]


def attr_value(value: dict) -> str:
    for key in ("stringValue", "intValue", "doubleValue", "boolValue"):
        if key in value:
            return str(value[key])
    return json.dumps(value)


def attrs(items: list[dict] | None) -> dict[str, str]:
    return {item["key"]: attr_value(item["value"]) for item in items or []}


def point_value(point: dict) -> float:
    return float(point["asDouble"]) if "asDouble" in point else float(point.get("asInt", "nan"))


def manifest_info() -> dict[RunKey, RunInfo]:
    info = {}
    for run in json.loads(MANIFEST_PATH.read_text()):
        model = run["model"]
        info[RunKey(run["name"], run["eval_dataset"])] = RunInfo(
            family=model["family"],
            train_dataset=model["train_dataset"] or "",
            seed=str(model["seed"] or ""),
        )
    return info


def parse_resource(resource_metrics: dict) -> tuple[RunKey, Export] | None:
    """The run key and points of one exported resource, or None for anything but a search run."""
    resource = attrs(resource_metrics.get("resource", {}).get("attributes"))
    if resource.get("type") != "search" or resource.get("name", "").startswith("smoketest-"):
        return None
    key = RunKey(resource["name"], resource["benchmark_dataset"].removesuffix("_retrieval"))
    export = Export(time_unix_nano=0, run_points=[], query_points=[])
    for scope in resource_metrics.get("scopeMetrics", []):
        for metric in scope.get("metrics", []):
            for point in metric.get("gauge", {}).get("dataPoints", []):
                export.time_unix_nano = max(export.time_unix_nano, int(point.get("timeUnixNano", 0)))
                point_attrs = attrs(point.get("attributes"))
                value = point_value(point)
                if metric["name"] in PER_QUERY_METRICS:
                    metric_name = PER_QUERY_METRICS[metric["name"]]
                    export.query_points.append((point_attrs["query_id"], metric_name, int(point_attrs["k"]), value))
                else:
                    metric_name = AT_K_METRICS.get(metric["name"], metric["name"])
                    export.run_points.append((metric_name, point_attrs.get("k", ""), value))
    return key, export


def read_exports() -> dict[RunKey, Export]:
    newest: dict[RunKey, Export] = {}
    # One line per OTLP export request; stream it, since the file holds every per-query point.
    with OTLP_PATH.open() as lines:
        for line in lines:
            if not line.strip():
                continue
            for resource_metrics in json.loads(line).get("resourceMetrics", []):
                parsed = parse_resource(resource_metrics)
                if parsed is None:
                    continue
                key, export = parsed
                previous = newest.get(key)
                if previous is None or export.time_unix_nano >= previous.time_unix_nano:
                    newest[key] = export
    return newest


def write_csv(path: Path, row_type: type, rows: list) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([f.name for f in fields(row_type)])
        writer.writerows(astuple(row) for row in rows)


def main() -> None:
    info = manifest_info()
    exports = read_exports()
    run_rows: list[RunMetricRow] = []
    query_rows: list[PerQueryRow] = []
    unknown = [key for key in exports if key not in info]
    for key, export in sorted(exports.items(), key=lambda item: (item[0].eval_dataset, item[0].name)):
        if key not in info:
            continue
        run = info[key]
        common = (key.name, key.eval_dataset, run.family, run.train_dataset, run.seed)
        run_rows.extend(RunMetricRow(*common, metric, k, value) for metric, k, value in export.run_points)
        query_rows.extend(PerQueryRow(*common, *point) for point in export.query_points)
    RESULTS_DIR.mkdir(exist_ok=True)
    write_csv(RESULTS_DIR / "run_metrics.csv", RunMetricRow, run_rows)
    PER_QUERY_PATH.parent.mkdir(parents=True, exist_ok=True)
    columns = {f.name: [getattr(row, f.name) for row in query_rows] for f in fields(PerQueryRow)}
    pq.write_table(pa.table(columns), PER_QUERY_PATH, compression="zstd")

    queries_per_run: dict[RunKey, set[int]] = defaultdict(set)
    for row in query_rows:
        queries_per_run[RunKey(row.name, row.eval_dataset)].add(row.k)
    print(f"{len(exports)} runs exported ({len(unknown)} not in manifest: {unknown})")
    print(f"{len(run_rows)} run-level rows, {len(query_rows)} per-query rows")
    missing = [key for key in info if key not in exports]
    print(f"{len(missing)} manifest runs without exported metrics")


if __name__ == "__main__":
    main()
