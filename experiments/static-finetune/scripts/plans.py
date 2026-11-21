"""Write the sweep plans (JSON) that sweep.py runs.

grid: starting model × learning rate × seed for one target, scored on dev queries of the train
dataset. FiQA gets one seed: its runs are ~10× longer, and its seed spread in the first
experiment was 0.001 NDCG@10.

baselines: the first experiment's models 2, 3, 4 and 4b (seeds 42 and 43) on dev queries, so
the sweep has a dev reference; and bge with mean pooling on test queries, which shows how good
the mean-pooled target can be.
"""

import json
import logging
from argparse import ArgumentParser
from dataclasses import asdict, replace

from common import DATASETS, EXPERIMENT_DIR
from sweep import Plan, SearchSpec, TrainSpec
from sweep_finetune import SWEEP_INFO_DIR
from sweep_report import load_scores, rows_for

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


def write(plan: Plan) -> None:
    """Write one plan file."""
    PLANS_DIR.mkdir(exist_ok=True)
    path = PLANS_DIR / f"{plan.name}.json"
    path.write_text(json.dumps(asdict(plan), indent=2) + "\n")
    logger.info(f"{path}: {len(plan.train)} training runs, {len(plan.search)} search runs")


def main() -> None:
    """Write the requested plans."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "plans",
        nargs="+",
        choices=(
            "grid-mean",
            "grid-cls",
            "grid-mean-hi",
            "baselines",
            "confirm",
            "extras",
            "final",
            "cross",
            "grid-mean-fiqa-top",
            "replay",
        ),
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), help="for confirm")
    parser.add_argument("--top", type=int, default=3, help="for confirm")
    args = parser.parse_args()
    for name in args.plans:
        if name == "confirm":
            write(confirm(tuple(args.datasets), args.top))
        elif name == "extras":
            write(extras(tuple(args.datasets)))
        elif name == "final":
            write(final(tuple(args.datasets)))
        elif name == "cross":
            write(cross(tuple(args.datasets)))
        elif name == "replay":
            write(replay(tuple(args.datasets)))
        elif name == "baselines":
            write(baselines())
        elif name == "grid-mean-hi":
            write(grid("mean", HIGH_LRS, name))
        elif name == "grid-mean-fiqa-top":
            write(grid("mean", FIQA_TOP_LRS, name, CLS_STARTS, ("fiqa",)))
        elif name == "grid-cls":
            write(grid("cls", CLS_LRS, name, CLS_STARTS, CLS_ORDER))
        else:
            write(grid(name.removeprefix("grid-")))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
