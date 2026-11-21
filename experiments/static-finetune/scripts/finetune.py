"""Fine-tune a Model2Vec model on a featurized corpus: tokenlearn/train.py with a seed.

Same steps and hyperparameters as `tokenlearn.train --model2vec-model-name`; the only
addition is `random_seed`, which sets the train/validation split, head init and batch order.
"""

import logging
import time
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from typing import Any

import lightning.pytorch as pl
import model2vec.train.base as m2v_base
import torch
from common import DISTILLED, FEATURIZED_DIR, MODELS_DIR, POTION, library_versions, save_atomically
from lightning.pytorch.callbacks import EarlyStopping
from model2vec import StaticModel
from model2vec.train import StaticModelForSimilarity

from tokenlearn.utils import collect_means_and_texts

# tokenlearn/train.py defaults.
_BATCH_SIZE = 256
_LEARNING_RATE = 1e-3

# Short names used in model directory names -> where the base model loads from.
BASES: dict[str, str] = {
    "potion-base-8M": POTION,
    DISTILLED: str(MODELS_DIR / DISTILLED),
}


logger = logging.getLogger(__name__)


class _RecordingTrainer(pl.Trainer):
    """model2vec builds its Trainer internally; keep a handle to read epochs and losses after fit."""

    last: "_RecordingTrainer | None" = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        _RecordingTrainer.last = self


@dataclass
class TrainInfo:
    base_model: str
    base_model_source: str
    dataset: str
    seed: int
    n_featurized: int
    n_train: int
    n_val: int
    epochs_run: int
    best_val_loss: float | None
    final_val_loss: float | None
    batch_size: int
    learning_rate: float
    device: str
    torch_threads: int
    wall_time_s: float
    libraries: dict[str, str]
    # model2vec reloads the last checkpoint (default ModelCheckpoint), not the best-val one.
    saved_weights: str = "last epoch checkpoint (model2vec default)"


def main() -> None:
    """Fine-tune one start model on one featurized corpus, with one seed."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, choices=sorted(BASES))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    name = f"{args.base}-ft-{args.dataset}-s{args.seed}"
    out = MODELS_DIR / name
    if out.exists():
        logger.info(f"{out} exists; skipping")
        return
    torch.set_num_threads(args.threads)
    m2v_base.pl.Trainer = _RecordingTrainer  # type: ignore[misc]

    start = time.time()
    texts, vectors = collect_means_and_texts(FEATURIZED_DIR / args.dataset)
    model = StaticModel.from_pretrained(path=BASES[args.base], quantize_to="float32")
    trainable = StaticModelForSimilarity.from_static_model(
        model=model,
        out_dim=vectors.shape[1],
        n_layers=0,
        freeze_weights=True,
    )
    trainable.fit(
        X=texts,
        y=torch.from_numpy(vectors),
        batch_size=_BATCH_SIZE,
        device="cpu",
        learning_rate=_LEARNING_RATE,
        random_seed=args.seed,
    )

    trainer = _RecordingTrainer.last
    assert trainer is not None
    early_stopping = next(c for c in trainer.callbacks if isinstance(c, EarlyStopping))
    final_val = trainer.callback_metrics.get("val_loss")
    info = TrainInfo(
        base_model=args.base,
        base_model_source=BASES[args.base],
        dataset=args.dataset,
        seed=args.seed,
        n_featurized=len(texts),
        n_train=len(trainer.train_dataloader.dataset),  # type: ignore[union-attr]
        n_val=len(trainer.val_dataloaders.dataset),  # type: ignore[union-attr]
        epochs_run=trainer.current_epoch,
        best_val_loss=float(early_stopping.best_score),
        final_val_loss=float(final_val) if final_val is not None else None,
        batch_size=_BATCH_SIZE,
        learning_rate=_LEARNING_RATE,
        device="cpu",
        torch_threads=args.threads,
        wall_time_s=round(time.time() - start, 1),
        libraries=library_versions(),
    )
    save_atomically(trainable.to_static_model(), out, {"finetune": asdict(info)})
    logger.info(asdict(info))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
