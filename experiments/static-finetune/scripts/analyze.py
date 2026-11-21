"""Build the result tables from collect.py's output.

Writes results/tables.md and CSVs under results/tables/:
- metric@k (k = 1..10) for every model on every eval dataset. Fine-tuned models (families 4, 4b)
  are shown for train dataset == eval dataset, as mean ± sample std over seeds.
- the cross-dataset matrix: fine-tuned model minus its starting model, for every
  (train dataset, eval dataset) pair.
- paired bootstrap 95% confidence intervals on per-query deltas. Fine-tuned models use the per-query
  score averaged over seeds, so the interval reflects query sampling; seed spread is reported apart.
- testoperator's run-level timing and memory metrics, as raw output (not a performance comparison).

Before any table, it checks that each run-level metric@k equals the mean of its per-query values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
PER_QUERY_PATH = Path.home() / "Github/tokenlearn/data/static-finetune/results/per_query.parquet"

DATASETS = ("scifact", "nfcorpus", "fiqa")
METRICS = ("ndcg", "recall", "mrr", "precision")
KS = tuple(range(1, 11))
FAMILY_LABELS = {
    "1": "1: bge-base-en-v1.5",
    "2": "2: distilled (bge → static, 256d)",
    "3": "3: potion-base-8M",
    "4": "4: potion-base-8M + corpus fine-tune",
    "4b": "4b: distilled + corpus fine-tune",
}
# Fine-tuned family → the model it starts from.
BASELINE = {"4": "3", "4b": "2"}
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0


@dataclass(frozen=True)
class Comparison:
    """A paired comparison A − B on one eval dataset, at one metric@k."""

    eval_dataset: str
    a: str
    b: str
    metric: str
    k: int
    n_queries: int
    mean_delta: float
    ci_low: float
    ci_high: float
    # Mean delta of each seed of A (when A is fine-tuned), to show training variance.
    seed_deltas: str


def check_run_means(run_metrics: pd.DataFrame, per_query: pd.DataFrame) -> None:
    """Each run-level metric@k must equal the mean over that run's judged queries."""
    at_k = run_metrics[run_metrics["metric"].isin(METRICS)].copy()
    at_k["k"] = at_k["k"].astype(int)
    means = per_query.groupby(["name", "eval_dataset", "metric", "k"], as_index=False)["value"].mean()
    joined = at_k.merge(means, on=["name", "eval_dataset", "metric", "k"], suffixes=("_run", "_mean"))
    worst = (joined["value_run"] - joined["value_mean"]).abs().max()
    missing = len(at_k) - len(joined)
    print(f"run-level vs per-query mean: {len(joined)} pairs, max |diff| = {worst:.2e}, {missing} without per-query data")
    if missing or worst > 1e-9:
        raise SystemExit("per-query data does not reproduce the run-level metrics")


def to_markdown(table: pd.DataFrame, index: bool = True) -> str:
    """A GitHub Markdown table (pandas' own needs `tabulate`, which the venv lacks)."""
    frame = table.reset_index() if index else table
    header = [str(c) if str(c) != "index" else "" for c in frame.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


def fmt(mean: float, std: float | None) -> str:
    return f"{mean:.4f}" if std is None or np.isnan(std) else f"{mean:.4f} ± {std:.4f}"


def model_rows(run_metrics: pd.DataFrame, eval_dataset: str, metric: str) -> pd.DataFrame:
    """One row per model family with metric@1..10 on `eval_dataset`."""
    rows = run_metrics[(run_metrics["eval_dataset"] == eval_dataset) & (run_metrics["metric"] == metric)].copy()
    rows["k"] = rows["k"].astype(int)
    # Fine-tuned models: only those trained on the eval dataset (the off-diagonal is the cross matrix).
    in_domain = ~rows["family"].isin(BASELINE) | (rows["train_dataset"] == eval_dataset)
    grouped = rows[in_domain].groupby(["family", "k"])["value"].agg(["mean", "std", "count"]).reset_index()
    table = {}
    for family, group in grouped.groupby("family"):
        seeds = int(group["count"].max())
        label = FAMILY_LABELS[family] + (f" (n={seeds} seeds)" if family in BASELINE else "")
        table[label] = {f"@{int(r.k)}": fmt(r.mean, r.std if family in BASELINE else None) for r in group.itertuples()}
    return pd.DataFrame.from_dict(table, orient="index").reindex(columns=[f"@{k}" for k in KS])


def seed_averaged(per_query: pd.DataFrame, eval_dataset: str, family: str, train_dataset: str | None) -> pd.Series:
    """Per-query scores of one family on one eval dataset, averaged over seeds; index (query_id, metric, k)."""
    rows = per_query[(per_query["eval_dataset"] == eval_dataset) & (per_query["family"] == family)]
    if train_dataset is not None:
        rows = rows[rows["train_dataset"] == train_dataset]
    return rows.groupby(["query_id", "metric", "k"])["value"].mean()


def bootstrap_ci(deltas: np.ndarray) -> tuple[float, float]:
    # A fresh generator per comparison: the same comparison always gets the same interval.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.integers(0, len(deltas), size=(BOOTSTRAP_RESAMPLES, len(deltas)))
    means = deltas[samples].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(
    per_query: pd.DataFrame,
    eval_dataset: str,
    a: tuple[str, str | None],
    b: tuple[str, str | None],
    metric: str,
    k: int,
) -> Comparison:
    a_scores = seed_averaged(per_query, eval_dataset, *a).xs((metric, k), level=("metric", "k"))
    b_scores = seed_averaged(per_query, eval_dataset, *b).xs((metric, k), level=("metric", "k"))
    paired = pd.concat([a_scores, b_scores], axis=1, join="inner").dropna()
    deltas = (paired.iloc[:, 0] - paired.iloc[:, 1]).to_numpy()
    low, high = bootstrap_ci(deltas)

    seed_deltas = ""
    family_a, train_a = a
    if family_a in BASELINE:
        rows = per_query[
            (per_query["eval_dataset"] == eval_dataset)
            & (per_query["family"] == family_a)
            & (per_query["train_dataset"] == train_a)
            & (per_query["metric"] == metric)
            & (per_query["k"] == k)
        ]
        per_seed = rows.groupby("seed")["value"].mean() - b_scores.mean()
        seed_deltas = ", ".join(f"s{seed}: {delta:+.4f}" for seed, delta in per_seed.items())

    def label(model: tuple[str, str | None]) -> str:
        family, train = model
        return family + (f"[ft {train}]" if train else "")

    return Comparison(
        eval_dataset=eval_dataset,
        a=label(a),
        b=label(b),
        metric=metric,
        k=k,
        n_queries=len(deltas),
        mean_delta=float(deltas.mean()),
        ci_low=low,
        ci_high=high,
        seed_deltas=seed_deltas,
    )


def comparison_table(comparisons: list[Comparison]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "eval": c.eval_dataset,
                "A − B": f"{c.a} − {c.b}",
                "metric": f"{c.metric}@{c.k}",
                "queries": c.n_queries,
                "mean Δ": f"{c.mean_delta:+.4f}",
                "95% CI": f"[{c.ci_low:+.4f}, {c.ci_high:+.4f}]",
                "excludes 0": "yes" if c.ci_low > 0 or c.ci_high < 0 else "no",
                "per-seed Δ": c.seed_deltas,
            }
            for c in comparisons
        ]
    )


def main() -> None:
    run_metrics = pd.read_csv(RESULTS_DIR / "run_metrics.csv", dtype={"seed": str, "k": str, "train_dataset": str})
    run_metrics = run_metrics.fillna({"seed": "", "train_dataset": "", "k": ""})
    per_query = pd.read_parquet(PER_QUERY_PATH)
    per_query["seed"] = per_query["seed"].astype(str)
    check_run_means(run_metrics, per_query)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    sections: list[str] = []

    # 1. metric@k for every model on every eval dataset.
    for eval_dataset in DATASETS:
        for metric in METRICS:
            table = model_rows(run_metrics, eval_dataset, metric)
            if table.empty:
                continue
            table.to_csv(TABLES_DIR / f"{eval_dataset}_{metric}.csv")
            sections.append(f"### {eval_dataset}: {metric}@k\n\n{to_markdown(table)}\n")

    # 2. Paired comparisons on each eval dataset (fine-tuned models trained on that dataset).
    comparisons: list[Comparison] = []
    for eval_dataset in DATASETS:
        pairs = [
            (("4", eval_dataset), ("3", None)),
            (("4b", eval_dataset), ("2", None)),
            (("4", eval_dataset), ("4b", eval_dataset)),
            (("3", None), ("2", None)),
            (("1", None), ("2", None)),
            (("1", None), ("3", None)),
            (("1", None), ("4", eval_dataset)),
        ]
        for a, b in pairs:
            for metric in METRICS:
                for k in (1, 5, 10):
                    try:
                        comparisons.append(compare(per_query, eval_dataset, a, b, metric, k))
                    except KeyError:
                        continue
    table = comparison_table(comparisons)
    table.to_csv(TABLES_DIR / "paired_comparisons.csv", index=False)
    sections.append(f"### Paired comparisons (bootstrap, {BOOTSTRAP_RESAMPLES} resamples)\n\n{to_markdown(table, index=False)}\n")

    # 3. Cross-dataset matrix: fine-tuned minus its starting model, NDCG@10.
    cross: list[Comparison] = []
    for family in BASELINE:
        for train_dataset in DATASETS:
            for eval_dataset in DATASETS:
                try:
                    cross.append(
                        compare(per_query, eval_dataset, (family, train_dataset), (BASELINE[family], None), "ndcg", 10)
                    )
                except KeyError:
                    continue
    table = comparison_table(cross)
    table.to_csv(TABLES_DIR / "cross_matrix_ndcg10.csv", index=False)
    sections.append(f"### Cross-dataset matrix: fine-tuned − starting model, ndcg@10\n\n{to_markdown(table, index=False)}\n")

    # 4. Run-level timing and memory, as testoperator reports them.
    ops = run_metrics[~run_metrics["metric"].isin(METRICS)]
    ops = ops.pivot_table(index=["eval_dataset", "name"], columns="metric", values="value", aggfunc="first")
    ops.to_csv(TABLES_DIR / "testoperator_ops_metrics.csv")
    sections.append(
        "### testoperator timing and memory (raw output; one run each, concurrent runs, P95 only — "
        f"not a performance comparison)\n\n{to_markdown(ops)}\n"
    )

    (RESULTS_DIR / "tables.md").write_text("\n".join(sections))
    print(f"Wrote {RESULTS_DIR / 'tables.md'}")


if __name__ == "__main__":
    main()
