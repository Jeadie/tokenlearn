"""Paths and helpers shared by the static fine-tuning experiment scripts."""

import json
import shutil
from importlib.metadata import version
from pathlib import Path
from typing import Any

from model2vec import StaticModel

# Large artifacts live outside /tmp (macOS purges it) and outside git.
MODELS_DIR = Path.home() / "Github/tokenlearn/models/static-finetune"
FEATURIZED_DIR = Path.home() / "Github/tokenlearn/data/static-finetune/featurized"
EXPERIMENT_DIR = Path(__file__).resolve().parent.parent

DATASETS = ("scifact", "nfcorpus", "fiqa")
SEEDS = (42, 43, 44, 45, 46)
TEACHER = "BAAI/bge-base-en-v1.5"
POTION = "minishlab/potion-base-8M"
DISTILLED = "bge-base-en-v1.5-distill-256"


def library_versions() -> dict[str, str]:
    """Versions of the libraries that decide the model's numbers."""
    return {
        name: version(name)
        for name in ("model2vec", "torch", "lightning", "sentence-transformers", "transformers", "tokenizers", "datasets")
    }


def save_atomically(model: StaticModel, out: Path, info: dict[str, Any]) -> None:
    """Save to `<out>.tmp`, write train_info.json, then rename, so a watcher never sees a partial model."""
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_pretrained(str(tmp))
    (tmp / "train_info.json").write_text(json.dumps(info, indent=2) + "\n")
    tmp.rename(out)
    # Keep a copy in git; the weights stay out of it.
    copy = EXPERIMENT_DIR / "train_info" / f"{out.name}.json"
    copy.write_text(json.dumps(info, indent=2) + "\n")
