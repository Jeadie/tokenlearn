"""Write each dataset's chosen n-gram models (seeds 42-46) without their n-gram tensors.

The token rows stay as they were trained together with the n-gram rows. Searching these copies
shows how much of a change (in domain, and on the other datasets) comes from the n-gram rows and
how much from the token rows.
"""

import json
import logging
import shutil

from common import DATASETS
from plans import FINAL_SEEDS, best_ngram_setting, drop_name
from safetensors import safe_open
from safetensors.numpy import load_file, save_file
from sweep_finetune import SWEEP_MODELS_DIR

logger = logging.getLogger(__name__)


def main() -> None:
    """Write one copy per model; skip those that exist."""
    for dataset in DATASETS:
        for seed in FINAL_SEEDS:
            name = best_ngram_setting(dataset).name.removesuffix("-s42") + f"-s{seed}"
            out = SWEEP_MODELS_DIR / drop_name(name)
            if out.exists():
                logger.info(f"{out.name} exists; skipping")
                continue
            source = SWEEP_MODELS_DIR / name
            shutil.copytree(source, out)
            weights = out / "model.safetensors"
            with safe_open(str(weights), framework="numpy") as f:
                metadata = f.metadata()
            tensors = {k: v for k, v in load_file(str(weights)).items() if not k.startswith("ngram_")}
            save_file(tensors, str(weights), metadata=metadata)
            config = json.loads((out / "config.json").read_text())
            config.pop("ngram_orders", None)
            config.pop("ngram_count", None)
            (out / "config.json").write_text(json.dumps(config, indent=4) + "\n")
            logger.info(f"{out.name}: {len(tensors)} tensors kept")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
