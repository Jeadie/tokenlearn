"""Write the sweep plans (JSON) that sweep.py runs.

grid: starting model × learning rate × seed for one target, scored on dev queries of the train
dataset. FiQA gets one seed: its runs are ~10× longer, and its seed spread in the first
experiment was 0.001 NDCG@10.

baselines: the first experiment's models 2, 3, 4 and 4b (seeds 42 and 43) on dev queries, so
the sweep has a dev reference; and bge with mean pooling on test queries, which shows how good
the mean-pooled target can be.

ngram-*: the n-gram experiment (NGRAMS.md). Each dataset's final sweep setting with additive
token n-grams. Its search runs use a spiced build with n-gram support, so they carry the tag
`NGRAM_TAG`, and their scores go to results/ngrams/scores.jsonl.
"""

import json
import logging
from argparse import ArgumentParser
from collections.abc import Callable
from dataclasses import asdict, replace

from common import DATASETS, EXPERIMENT_DIR
from sweep import Plan, SearchSpec, TrainSpec
from sweep_finetune import SWEEP_INFO_DIR
from sweep_report import load_scores, rows_for, setting_of

logger = logging.getLogger(__name__)

PLANS_DIR = EXPERIMENT_DIR / "plans"
STARTS = (("potion", 1.0), ("potion", 10.5), ("distill", 1.0))
LRS = (1e-3, 1e-2, 3e-2)
# The SciFact grid was best at its top learning rate (3e-2), so the grid is extended upward.
HIGH_LRS = (1e-1, 3e-1)
# The CLS grid: unscaled potion is left out, because on SciFact it scored like potion ÷10.5 at a
# 10× lower learning rate; 1e-1 is added, the best SciFact learning rate of the mean grid.
CLS_STARTS = (("potion", 10.5), ("distill", 1.0))
# FiQA's best learning rate was the top of the extended grid (3e-1), so it gets one more point.
FIQA_TOP_LRS = (1.0,)
CLS_LRS = (1e-3, 1e-2, 3e-2, 1e-1)
# Short SciFact and NFCorpus runs first; FiQA waits for its CLS targets.
CLS_ORDER = ("scifact", "nfcorpus", "fiqa")
FINAL_SEEDS = (42, 43, 44, 45, 46)
# Extra-text variants are compared with the same seeds of the setting they extend.
EXTRAS_SEEDS = (42, 43)
GRID_SEEDS = {"scifact": (42, 43), "fiqa": (42,), "nfcorpus": (42, 43)}
# Long FiQA runs before the short NFCorpus runs, so the short ones fill the end.
GRID_ORDER = ("scifact", "fiqa", "nfcorpus")
# The n-gram grid: (orders, K) on each dataset's final setting.
NGRAM_SETTINGS = (((2,), 16_000), ((2,), 64_000), ((2, 3), 16_000), ((2, 3), 64_000))
NGRAM_SEEDS = (42, 43)
# Round 2, on each dataset's best round-1 setting: the n-gram rows' learning rate as a fraction of the
# token rows', and a larger K (round 1 was better at its largest K on every dataset).
NGRAM_LR_SCALES = (0.1, 0.3)
NGRAM_BIG_K = 256_000
NGRAM_ORDER = ("fiqa", "scifact", "nfcorpus")
NGRAM_TAG = "ng"
# The same baselines searched again with the build of Parts 1 and 2, for gate G1 (the collector lost
# the per-query metrics of the first searches).
NGRAM_OLD_TAG = "old"
NGRAM_SCORES = EXPERIMENT_DIR / "results" / "ngrams" / "scores.jsonl"


def grid(
    target: str,
    lrs: tuple[float, ...] = LRS,
    name: str = "",
    starts: tuple[tuple[str, float], ...] = STARTS,
    order: tuple[str, ...] = GRID_ORDER,
) -> Plan:
    """Every starting model and learning rate on every dataset, scored on dev."""
    train = [
        TrainSpec(base=base, scale=scale, dataset=ds, lr=lr, seed=seed, target=target)
        for ds in order
        for seed in GRID_SEEDS[ds]
        for base, scale in starts
        for lr in lrs
    ]
    search = [SearchSpec(model=t.name, eval_dataset=t.dataset, split="dev") for t in train]
    return Plan(name=name or f"grid-{target}", train=train, search=search)


def baselines() -> Plan:
    """Dev scores of the first experiment's models; test scores of mean-pooled bge."""
    # bge runs use the GPU; FiQA's takes ~2 h, so it goes last.
    search = [SearchSpec(model="hf-mean[bge-base-en-v1.5]", eval_dataset=ds, split="test") for ds in DATASETS[:2]]
    for ds in DATASETS:
        labels = ["model2vec[bge-base-en-v1.5-distill-256]", "model2vec[potion-base-8M]"]
        for seed in (42, 43):
            labels.append(f"model2vec[potion-base-8M[ft-{ds}-s{seed}]]")
            labels.append(f"model2vec[bge-base-en-v1.5-distill-256[ft-{ds}-s{seed}]]")
        search += [SearchSpec(model=label, eval_dataset=ds, split="dev") for label in labels]
    search.append(SearchSpec(model="hf-mean[bge-base-en-v1.5]", eval_dataset="fiqa", split="test"))
    return Plan(name="baselines", train=[], search=search)


def confirm(datasets: tuple[str, ...], top: int) -> Plan:
    """The `top` settings of each dataset by dev NDCG@10, trained with seeds 42-46 and scored on dev."""
    scores = load_scores("dev")
    train = []
    for dataset in datasets:
        rows = [r for r in rows_for(dataset, scores) if r.setting.startswith("sw-")][:top]
        for row in rows:
            info = json.loads((SWEEP_INFO_DIR / f"{row.setting}-s42.json").read_text())
            train += [
                TrainSpec(
                    base=info["base"],
                    scale=info["scale"],
                    dataset=dataset,
                    lr=info["lr"],
                    seed=seed,
                    target=info["target"],
                    extras=tuple(info["extras"]),
                )
                for seed in FINAL_SEEDS
            ]
    search = [SearchSpec(model=t.name, eval_dataset=t.dataset, split="dev") for t in train]
    return Plan(name=f"confirm-{'-'.join(datasets)}", train=train, search=search)


def best_setting(dataset: str, min_seeds: int) -> TrainSpec:
    """The sweep setting with the best mean dev NDCG@10 among those with at least `min_seeds` seeds (seed 42)."""
    # 100% retrains are results of a setting, not settings to choose from.
    rows = [
        r
        for r in rows_for(dataset, load_scores("dev"))
        if r.setting.startswith("sw-") and "-full" not in r.setting and r.seeds >= min_seeds
    ]
    info = json.loads((SWEEP_INFO_DIR / f"{rows[0].setting}-s42.json").read_text())
    return TrainSpec(
        base=info["base"],
        scale=info["scale"],
        dataset=dataset,
        lr=info["lr"],
        seed=42,
        target=info["target"],
        extras=tuple(info["extras"]),
        full_epochs=info["full_epochs"],
    )


def extras(datasets: tuple[str, ...]) -> Plan:
    """The confirmed best setting of each dataset, with extra training texts, scored on dev."""
    train = []
    for dataset in datasets:
        best = best_setting(dataset, min_seeds=len(FINAL_SEEDS))
        for extra in (("trainq",), ("sents",), ("sents", "trainq")):
            train += [replace(best, extras=extra, seed=seed) for seed in EXTRAS_SEEDS]
    search = [SearchSpec(model=t.name, eval_dataset=t.dataset, split="dev") for t in train]
    return Plan(name=f"extras-{'-'.join(datasets)}", train=train, search=search)


def full_epochs(spec: TrainSpec) -> int:
    """The number of epochs for a 100% retrain: the mean best epoch over seeds, plus one (epochs count from 0)."""
    epochs = [
        json.loads((SWEEP_INFO_DIR / f"{replace(spec, seed=seed).name}.json").read_text())["best_epoch"]
        for seed in FINAL_SEEDS
    ]
    return round(sum(epochs) / len(epochs)) + 1


def final(datasets: tuple[str, ...]) -> Plan:
    """The chosen setting of each dataset on test (5 seeds), and its 100% retrain on dev and test."""
    train = []
    search = []
    for dataset in datasets:
        best = best_setting(dataset, min_seeds=len(FINAL_SEEDS))
        held_out = [replace(best, seed=seed) for seed in FINAL_SEEDS]
        full = [replace(best, seed=seed, full_epochs=full_epochs(best)) for seed in FINAL_SEEDS]
        train += held_out + full
        search += [SearchSpec(model=t.name, eval_dataset=dataset, split="test") for t in held_out + full]
        search += [SearchSpec(model=t.name, eval_dataset=dataset, split="dev") for t in full]
    return Plan(name=f"final-{'-'.join(datasets)}", train=train, search=search)


def cross(datasets: tuple[str, ...]) -> Plan:
    """Each dataset's final models (held-out and 100%), scored on the other datasets' test queries."""
    search = []
    for dataset in datasets:
        best = best_setting(dataset, min_seeds=len(FINAL_SEEDS))
        models = [replace(best, seed=seed) for seed in FINAL_SEEDS]
        models += [replace(best, seed=seed, full_epochs=full_epochs(best)) for seed in FINAL_SEEDS]
        search += [
            SearchSpec(model=m.name, eval_dataset=other, split="test")
            for other in DATASETS
            if other != dataset
            for m in models
        ]
    return Plan(name=f"cross-{'-'.join(datasets)}", train=[], search=search)


def replay(datasets: tuple[str, ...]) -> Plan:
    """Each dataset's final setting plus C4 replay (seeds 42-46): dev and test in-domain, test elsewhere."""
    train = []
    search = []
    for dataset in datasets:
        best = best_setting(dataset, min_seeds=len(FINAL_SEEDS))
        runs = [replace(best, seed=seed, extras=tuple(sorted((*best.extras, "c4")))) for seed in FINAL_SEEDS]
        train += runs
        search += [SearchSpec(model=t.name, eval_dataset=dataset, split="dev") for t in runs]
        search += [SearchSpec(model=t.name, eval_dataset=other, split="test") for other in DATASETS for t in runs]
    return Plan(name=f"replay-{'-'.join(datasets)}", train=train, search=search)


def ngram_baselines(tag: str = NGRAM_TAG) -> Plan:
    """Each dataset's final setting without n-grams (seeds 42-46), on dev and test, with the build `tag` names."""
    search = []
    for dataset in DATASETS:
        best = best_setting(dataset, min_seeds=len(FINAL_SEEDS))
        search += [
            SearchSpec(model=replace(best, seed=seed).name, eval_dataset=dataset, split=split, tag=tag)
            for split in ("dev", "test")
            for seed in FINAL_SEEDS
        ]
    return Plan(name="ngram-baselines" + ("" if tag == NGRAM_TAG else f"-{tag}"), train=[], search=search)


def ngram_grid() -> Plan:
    """Each dataset's final setting with each n-gram setting, seeds 42 and 43, scored on dev."""
    train = [
        replace(best_setting(dataset, min_seeds=len(FINAL_SEEDS)), seed=seed, ngram_orders=orders, ngram_k=k)
        for dataset in NGRAM_ORDER
        for seed in NGRAM_SEEDS
        for orders, k in NGRAM_SETTINGS
    ]
    search = [SearchSpec(model=t.name, eval_dataset=t.dataset, split="dev", tag=NGRAM_TAG) for t in train]
    return Plan(name="ngram-grid", train=train, search=search)


def ngram_gates() -> Plan:
    """The zero-row n-gram models (make_ngram_zero.py) on dev: they must score as their base models."""
    search = [
        SearchSpec(model=zero_name(dataset), eval_dataset=dataset, split="dev", tag=NGRAM_TAG) for dataset in DATASETS
    ]
    return Plan(name="ngram-gates", train=[], search=search)


def zero_name(dataset: str) -> str:
    """The zero-row n-gram copy of a dataset's final model, seed 42."""
    return replace(best_setting(dataset, min_seeds=len(FINAL_SEEDS)), seed=42).name + "-ngzero"


def spec_from_info(name: str) -> TrainSpec:
    """The TrainSpec of a trained sweep model, from its training record."""
    info = json.loads((SWEEP_INFO_DIR / f"{name}.json").read_text())
    return TrainSpec(
        base=info["base"],
        scale=info["scale"],
        dataset=info["dataset"],
        lr=info["lr"],
        seed=info["seed"],
        target=info["target"],
        extras=tuple(info["extras"]),
        full_epochs=info["full_epochs"],
        ngram_orders=tuple(info.get("ngram_orders", ())),
        ngram_k=info.get("ngram_k", 0),
        ngram_lr_scale=info.get("ngram_lr_scale", 1.0),
    )


def ngram_settings(dataset: str) -> dict[str, tuple[TrainSpec, list[float]]]:
    """Each n-gram setting of a dataset with dev scores for seeds 42 and 43: its seed-42 spec and NDCG@10s."""
    found: dict[str, dict[str, float]] = {}
    for record in map(json.loads, NGRAM_SCORES.read_text().splitlines()):
        model = record["model"]
        if record["eval_dataset"] != dataset or record["split"] != "dev" or not record["metrics"]:
            continue
        if "-ng" not in model or model.endswith("-ngzero") or record["exit_code"] != 0:
            continue
        found.setdefault(setting_of(model), {})[model] = record["metrics"]["ndcg"]["10"]
    out = {}
    for setting, by_model in found.items():
        spec = spec_from_info(next(iter(by_model)))
        names = [replace(spec, seed=seed).name for seed in NGRAM_SEEDS]
        if all(name in by_model for name in names):
            out[setting] = (replace(spec, seed=42), [by_model[name] for name in names])
    return out


def best_ngram_setting(dataset: str, lr_scale: float | None = None) -> TrainSpec:
    """The n-gram setting of a dataset with the best mean dev NDCG@10 over seeds 42 and 43.

    With `lr_scale`, only the settings with that n-gram learning-rate scale.
    """
    settings = [
        (spec, scores)
        for spec, scores in ngram_settings(dataset).values()
        if lr_scale is None or spec.ngram_lr_scale == lr_scale
    ]
    return max(settings, key=lambda item: sum(item[1]) / len(item[1]))[0]


def ngram_grid2() -> Plan:
    """Round 2: each dataset's best round-1 setting with a lower n-gram learning rate, or with a larger K."""
    train = []
    for dataset in NGRAM_ORDER:
        best = best_ngram_setting(dataset, lr_scale=1.0)
        variants = [replace(best, ngram_lr_scale=scale) for scale in NGRAM_LR_SCALES]
        variants.append(replace(best, ngram_k=NGRAM_BIG_K))
        train += [replace(v, seed=seed) for seed in NGRAM_SEEDS for v in variants]
    search = [SearchSpec(model=t.name, eval_dataset=t.dataset, split="dev", tag=NGRAM_TAG) for t in train]
    return Plan(name="ngram-grid2", train=train, search=search)


def ngram_confirm(datasets: tuple[str, ...]) -> Plan:
    """The best n-gram setting of each dataset with seeds 44-46 too, scored on dev."""
    train = [replace(best_ngram_setting(d), seed=seed) for d in datasets for seed in FINAL_SEEDS]
    search = [SearchSpec(model=t.name, eval_dataset=t.dataset, split="dev", tag=NGRAM_TAG) for t in train]
    return Plan(name=f"ngram-confirm-{'-'.join(datasets)}", train=train, search=search)


def ngram_final(datasets: tuple[str, ...]) -> Plan:
    """The best n-gram setting of each dataset (seeds 42-46) on the test queries of every dataset."""
    train = [replace(best_ngram_setting(d), seed=seed) for d in datasets for seed in FINAL_SEEDS]
    search = [
        SearchSpec(model=t.name, eval_dataset=other, split="test", tag=NGRAM_TAG) for t in train for other in DATASETS
    ]
    # The baselines on the other datasets' test queries, for the forgetting comparison.
    for dataset in datasets:
        best = best_setting(dataset, min_seeds=len(FINAL_SEEDS))
        search += [
            SearchSpec(model=replace(best, seed=seed).name, eval_dataset=other, split="test", tag=NGRAM_TAG)
            for seed in FINAL_SEEDS
            for other in DATASETS
            if other != dataset
        ]
    return Plan(name=f"ngram-final-{'-'.join(datasets)}", train=train, search=search)


def write(plan: Plan) -> None:
    """Write one plan file."""
    PLANS_DIR.mkdir(exist_ok=True)
    path = PLANS_DIR / f"{plan.name}.json"
    path.write_text(json.dumps(asdict(plan), indent=2) + "\n")
    logger.info(f"{path}: {len(plan.train)} training runs, {len(plan.search)} search runs")


def plan_builders(datasets: tuple[str, ...], top: int) -> dict[str, Callable[[], Plan]]:
    """Every plan by name. Plans that depend on earlier results read them when called."""
    return {
        "grid-mean": lambda: grid("mean"),
        "grid-mean-hi": lambda: grid("mean", HIGH_LRS, "grid-mean-hi"),
        "grid-mean-fiqa-top": lambda: grid("mean", FIQA_TOP_LRS, "grid-mean-fiqa-top", CLS_STARTS, ("fiqa",)),
        "grid-cls": lambda: grid("cls", CLS_LRS, "grid-cls", CLS_STARTS, CLS_ORDER),
        "baselines": baselines,
        "confirm": lambda: confirm(datasets, top),
        "extras": lambda: extras(datasets),
        "final": lambda: final(datasets),
        "cross": lambda: cross(datasets),
        "replay": lambda: replay(datasets),
        "ngram-baselines": ngram_baselines,
        "ngram-baselines-old": lambda: ngram_baselines(NGRAM_OLD_TAG),
        "ngram-grid": ngram_grid,
        "ngram-grid2": ngram_grid2,
        "ngram-gates": ngram_gates,
        "ngram-confirm": lambda: ngram_confirm(datasets),
        "ngram-final": lambda: ngram_final(datasets),
    }


def main() -> None:
    """Write the requested plans."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("plans", nargs="+", choices=sorted(plan_builders((), 0)))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), help="datasets of the result-based plans")
    parser.add_argument("--top", type=int, default=3, help="settings per dataset, for confirm")
    args = parser.parse_args()
    builders = plan_builders(tuple(args.datasets), args.top)
    for name in args.plans:
        write(builders[name]())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
