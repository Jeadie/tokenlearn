"""Distill model 2: a plain Model2Vec static model from BAAI/bge-base-en-v1.5 (no training)."""

import json
import logging
import time
from argparse import ArgumentParser
from dataclasses import asdict, dataclass

from common import MODELS_DIR, library_versions, save_atomically
from model2vec.distill import distill

logger = logging.getLogger(__name__)


@dataclass
class DistillInfo:
    teacher: str
    pca_dims: int
    vocabulary: str
    quantize_to: str
    wall_time_s: float
    libraries: dict[str, str]


def main() -> None:
    """Distill the teacher into a static model and record how it was made."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--pca-dims", type=int, default=256)
    parser.add_argument("--name", default="bge-base-en-v1.5-distill-256")
    args = parser.parse_args()

    out = MODELS_DIR / args.name
    if out.exists():
        logger.info(f"{out} exists; skipping")
        return

    start = time.time()
    # Default vocabulary: the teacher tokenizer's own; no corpus words are added.
    model = distill(model_name=args.teacher, pca_dims=args.pca_dims, quantize_to="float32")
    info = DistillInfo(
        teacher=args.teacher,
        pca_dims=args.pca_dims,
        vocabulary="teacher tokenizer (default)",
        quantize_to="float32",
        wall_time_s=round(time.time() - start, 1),
        libraries=library_versions(),
    )
    save_atomically(model, out, {"distill": asdict(info)})
    logger.info(json.dumps(asdict(info), indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
