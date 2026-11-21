"""Summarize the sweep's dev scores, to choose settings. Test scores are reported elsewhere.

Writes results/sweep/dev_report.md: for each eval dataset, one row per setting (seeds averaged),
sorted by dev NDCG@10, next to the first experiment's models on the same dev queries.
"""

import json
import logging
from dataclasses import dataclass

import numpy as np
from common import EXPERIMENT_DIR
from sweep import SCORES_PATH
from sweep_finetune import SWEEP_INFO_DIR

logger = logging.getLogger(__name__)

REPORT_PATH = EXPERIMENT_DIR / "results" / "sweep" / "dev_report.md"
DATASETS = ("scifact", "nfcorpus", "fiqa")


@dataclass
class Row:
    setting: str
    seeds: int
    ndcg10: float
    ndcg10_std: float
    recall10: float
    mrr10: float
    epochs: float | None
    best_epoch: float | None
    best_val_loss: float | None


def setting_of(model: str) -> str:
    """A sweep model name without its seed: the setting that seeds are averaged over."""
    return model.rsplit("-s", 1)[0] if model.startswith("sw-") else model


def baseline_setting(model: str, dataset: str) -> str:
    """The first experiment's models, with seeds folded together."""
    for seed in (42, 43, 44, 45, 46):
        model = model.replace(f"[ft-{dataset}-s{seed}]", f"[ft-{dataset}]")
    return model


def load_scores(split: str) -> list[dict]:
    """The newest passed score of each (run, dataset) on one split."""
    newest: dict[tuple[str, str], dict] = {}
    if SCORES_PATH.exists():
        for line in SCORES_PATH.read_text().splitlines():
            record = json.loads(line)
            if record["split"] == split and record["metrics"]:
                newest[(record["run_name"], record["eval_dataset"])] = record
    return list(newest.values())


def train_info(model: str) -> dict | None:
    """The training record of a sweep model."""
    path = SWEEP_INFO_DIR / f"{model}.json"
    return json.loads(path.read_text()) if path.exists() else None


def rows_for(dataset: str, scores: list[dict]) -> list[Row]:
    """One row per setting on one eval dataset."""
    groups: dict[str, list[dict]] = {}
    for record in scores:
        if record["eval_dataset"] != dataset:
            continue
        key = (
            setting_of(record["model"])
            if record["model"].startswith("sw-")
            else baseline_setting(record["model"], dataset)
        )
        groups.setdefault(key, []).append(record)
    rows = []
    for setting, records in groups.items():
        ndcg = [r["metrics"]["ndcg"]["10"] for r in records]
        infos = [i for i in (train_info(r["model"]) for r in records) if i]

        def mean_of(key: str, infos: list[dict] = infos) -> float | None:
            values = [i[key] for i in infos if i.get(key) is not None]
            return float(np.mean(values)) if values else None

        rows.append(
            Row(
                setting=setting,
                seeds=len(records),
                ndcg10=float(np.mean(ndcg)),
                ndcg10_std=float(np.std(ndcg, ddof=1)) if len(ndcg) > 1 else float("nan"),
                recall10=float(np.mean([r["metrics"]["recall"]["10"] for r in records])),
                mrr10=float(np.mean([r["metrics"]["mrr"]["10"] for r in records])),
                epochs=mean_of("epochs_run"),
                best_epoch=mean_of("best_epoch"),
                best_val_loss=mean_of("best_val_loss"),
            )
        )
    return sorted(rows, key=lambda r: -r.ndcg10)


def fmt(value: float | None, digits: int = 4) -> str:
    """A number for the table, or a dash."""
    return "–" if value is None or np.isnan(value) else f"{value:.{digits}f}"


def main() -> None:
    """Write the dev report."""
    scores = load_scores("dev")
    sections = []
    for dataset in DATASETS:
        rows = rows_for(dataset, scores)
        if not rows:
            continue
        lines = [
            f"### {dataset} (dev)",
            "",
            "| setting | seeds | ndcg@10 | std | recall@10 | mrr@10 | epochs | best epoch | best val loss |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        lines += [
            f"| {r.setting} | {r.seeds} | {fmt(r.ndcg10)} | {fmt(r.ndcg10_std)} | {fmt(r.recall10)} | {fmt(r.mrr10)} "
            f"| {fmt(r.epochs, 1)} | {fmt(r.best_epoch, 1)} | {fmt(r.best_val_loss)} |"
            for r in rows
        ]
        sections.append("\n".join(lines) + "\n")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(sections))
    logger.info(REPORT_PATH.read_text())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
