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
from dataclasses import asdict

from common import DATASETS, EXPERIMENT_DIR
from sweep import Plan, SearchSpec, TrainSpec

logger = logging.getLogger(__name__)

PLANS_DIR = EXPERIMENT_DIR / "plans"
STARTS = (("potion", 1.0), ("potion", 10.5), ("distill", 1.0))
LRS = (1e-3, 1e-2, 3e-2)
# The SciFact grid was best at its top learning rate (3e-2), so the grid is extended upward.
HIGH_LRS = (1e-1, 3e-1)
# The CLS grid: unscaled potion is left out, because on SciFact it scored like potion ÷10.5 at a
# 10× lower learning rate; 1e-1 is added, the best SciFact learning rate of the mean grid.
CLS_STARTS = (("potion", 10.5), ("distill", 1.0))
CLS_LRS = (1e-3, 1e-2, 3e-2, 1e-1)
# Short SciFact and NFCorpus runs first; FiQA waits for its CLS targets.
CLS_ORDER = ("scifact", "nfcorpus", "fiqa")
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


def write(plan: Plan) -> None:
    """Write one plan file."""
    PLANS_DIR.mkdir(exist_ok=True)
    path = PLANS_DIR / f"{plan.name}.json"
    path.write_text(json.dumps(asdict(plan), indent=2) + "\n")
    logger.info(f"{path}: {len(plan.train)} training runs, {len(plan.search)} search runs")


def main() -> None:
    """Write the requested plans."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("plans", nargs="+", choices=("grid-mean", "grid-cls", "grid-mean-hi", "baselines"))
    args = parser.parse_args()
    for name in args.plans:
        if name == "baselines":
            write(baselines())
        elif name == "grid-mean-hi":
            write(grid("mean", HIGH_LRS, name))
        elif name == "grid-cls":
            write(grid("cls", CLS_LRS, name, CLS_STARTS, CLS_ORDER))
        else:
            write(grid(name.removeprefix("grid-")))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
