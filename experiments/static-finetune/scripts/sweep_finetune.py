"""Fine-tune one sweep setting: like finetune.py, with the knobs that the first run held fixed.

Differences from finetune.py (whose defaults reproduce the first experiment):
- `--lr`: the Adam learning rate.
- `--scale`: divide the start model's vectors by this factor first. Search results of the start
  model do not change (each text vector is normalized), but Adam's fixed-size steps turn the
  vectors more.
- `--target`: train toward bge's mean token vector (`mean`) or its CLS embedding (`cls`).
- `--extra`: more featurized texts (for example `trainq`, `sents`), added to the training split only.
- Early stopping with `--min-delta` (default 0) and `--patience` (default 10), and the learning
  rate scheduler with PyTorch's default threshold (1e-4 relative) instead of model2vec's 3%.
- The checkpoint with the lowest validation loss is kept, not the last one.
- `--full-epochs N`: train on 100% of the corpus for exactly N epochs, with no hold-out.
- The validation split is `train_test_split(test_size=0.1, random_state=seed)` over corpus
  documents, so every setting with the same seed holds out the same documents.
- Validation loss and learning rate are recorded for every epoch.
"""

import json
import logging
import shutil
import time
from argparse import ArgumentParser
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import model2vec.train.base as m2v_base
import model2vec.train.lightning_modules as m2v_modules
import numpy as np
import torch
from common import DISTILLED, EXPERIMENT_DIR, FEATURIZED_DIR, MODELS_DIR, POTION, library_versions
from featurize_texts import output_dir
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from model2vec import StaticModel
from model2vec.train import StaticModelForSimilarity
from sklearn.model_selection import train_test_split

from tokenlearn.utils import collect_means_and_texts

logger = logging.getLogger(__name__)

SWEEP_MODELS_DIR = MODELS_DIR / "sweep"
SWEEP_INFO_DIR = EXPERIMENT_DIR / "train_info_sweep"
BATCH_SIZE = 256
BASES: dict[str, str] = {"potion": POTION, "distill": str(MODELS_DIR / DISTILLED)}


@dataclass
class EpochRecord:
    epoch: int
    val_loss: float | None
    train_loss: float | None
    lr: float


@dataclass
class SweepTrainInfo:
    name: str
    base: str
    base_source: str
    scale: float
    target: str
    dataset: str
    extras: list[str]
    lr: float
    seed: int
    full_epochs: int | None
    patience: int
    min_delta: float
    lr_threshold: float
    max_epochs: int
    n_corpus: int
    n_train: int
    n_val: int
    epochs_run: int
    best_epoch: int | None
    best_val_loss: float | None
    wall_time_s: float
    torch_threads: int
    libraries: dict[str, str]
    curve: list[EpochRecord] = field(default_factory=list)


def model_name(args: Any) -> str:
    """The directory name of one sweep model; it encodes every setting."""
    scale = f"-x{args.scale:g}" if args.scale != 1 else ""
    extras = "".join(f"+{e}" for e in sorted(args.extra))
    full = f"-full{args.full_epochs}" if args.full_epochs else ""
    return f"sw-{args.base}{scale}-{args.target}-{args.dataset}{extras}-lr{args.lr:g}{full}-s{args.seed}"


class _CurveRecorder(Callback):
    """Record validation loss, last train loss and learning rate after every validation."""

    def __init__(self) -> None:
        self.curve: list[EpochRecord] = []

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Append one record; skip Lightning's sanity check before training."""
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        self.curve.append(
            EpochRecord(
                epoch=trainer.current_epoch,
                val_loss=float(metrics["val_loss"]) if "val_loss" in metrics else None,
                train_loss=float(metrics["train_loss"]) if "train_loss" in metrics else None,
                lr=float(trainer.optimizers[0].param_groups[0]["lr"]),
            )
        )


class _SweepTrainer(pl.Trainer):
    """model2vec builds its Trainer internally; swap in this run's early stopping and checkpointing."""

    last: "_SweepTrainer | None" = None
    min_delta: float = 0.0
    keep_best: bool = True
    recorder: _CurveRecorder | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        callbacks = []
        for callback in kwargs.get("callbacks") or []:
            if isinstance(callback, EarlyStopping):
                callback = EarlyStopping(
                    monitor="val_loss", mode="min", patience=callback.patience, min_delta=self.min_delta
                )
            callbacks.append(callback)
        if self.keep_best:
            # model2vec reloads `checkpoint_callback.best_model_path`; this makes it the lowest val loss.
            callbacks.append(ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1))
        _SweepTrainer.recorder = _CurveRecorder()
        callbacks.append(_SweepTrainer.recorder)
        kwargs["callbacks"] = callbacks
        kwargs["enable_progress_bar"] = False
        super().__init__(*args, **kwargs)
        _SweepTrainer.last = self


def use_scheduler_threshold(threshold: float) -> None:
    """Replace model2vec's optimizer setup: same Adam and ReduceLROnPlateau, with this threshold."""

    def configure_optimizers(self: m2v_modules.StaticLightningModule) -> Any:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6, threshold=threshold, threshold_mode="rel"
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

    m2v_modules.StaticLightningModule.configure_optimizers = configure_optimizers  # type: ignore[method-assign]


def load(dataset: str, source: str, target: str) -> tuple[list[str], np.ndarray]:
    """Featurized texts and targets; rows with a NaN target (empty text) are dropped."""
    return collect_means_and_texts(output_dir(dataset, source, target))


def save(model: StaticModel, info: SweepTrainInfo) -> None:
    """Save atomically (`<name>.tmp`, then rename) and keep a copy of the train info in git."""
    out = SWEEP_MODELS_DIR / info.name
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_pretrained(str(tmp))
    text = json.dumps(asdict(info), indent=2) + "\n"
    (tmp / "train_info.json").write_text(text)
    tmp.rename(out)
    SWEEP_INFO_DIR.mkdir(exist_ok=True)
    (SWEEP_INFO_DIR / f"{info.name}.json").write_text(text)


def main() -> None:
    """Fine-tune one sweep setting and save the model with its training record."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, choices=sorted(BASES))
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--target", default="mean", choices=("mean", "cls"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--extra", action="append", default=[], choices=("trainq", "sents"))
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--lr-threshold", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--full-epochs", type=int, default=None)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    name = model_name(args)
    if (SWEEP_MODELS_DIR / name).exists():
        logger.info(f"{name} exists; skipping")
        return
    torch.set_num_threads(args.threads)
    use_scheduler_threshold(args.lr_threshold)
    _SweepTrainer.min_delta = args.min_delta
    _SweepTrainer.keep_best = args.full_epochs is None
    m2v_base.pl.Trainer = _SweepTrainer  # type: ignore[misc]

    start = time.time()
    texts, vectors = load(args.dataset, "corpus", args.target)
    if args.full_epochs is None:
        x_train, x_val, y_train, y_val = train_test_split(texts, vectors, test_size=0.1, random_state=args.seed)
    else:
        # No hold-out. model2vec needs a validation set; a small sample of the training rows keeps it
        # running, and nothing reads its loss (no early stopping, the last epoch is kept).
        x_train, y_train = texts, vectors
        rng = np.random.default_rng(args.seed)
        pick = rng.choice(len(texts), size=min(512, len(texts)), replace=False)
        x_val, y_val = [texts[i] for i in pick], vectors[pick]
    for source in args.extra:
        extra_texts, extra_vectors = load(args.dataset, source, args.target)
        x_train = list(x_train) + extra_texts
        y_train = np.concatenate([y_train, extra_vectors])

    model = StaticModel.from_pretrained(path=BASES[args.base], quantize_to="float32")
    if args.scale != 1:
        model.embedding = model.embedding / args.scale
    trainable = StaticModelForSimilarity.from_static_model(
        model=model, out_dim=vectors.shape[1], n_layers=0, freeze_weights=True
    )
    trainable.fit(
        X=list(x_train),
        y=torch.from_numpy(np.asarray(y_train, dtype=np.float32)),
        X_val=list(x_val),
        y_val=torch.from_numpy(np.asarray(y_val, dtype=np.float32)),
        batch_size=BATCH_SIZE,
        device="cpu",
        learning_rate=args.lr,
        random_seed=args.seed,
        early_stopping_patience=None if args.full_epochs else args.patience,
        min_epochs=args.full_epochs,
        max_epochs=args.full_epochs or args.max_epochs,
    )

    trainer = _SweepTrainer.last
    recorder = _SweepTrainer.recorder
    assert trainer is not None and recorder is not None
    curve = recorder.curve
    scored = [r for r in curve if r.val_loss is not None]
    best = min(scored, key=lambda r: r.val_loss or 0.0) if scored and args.full_epochs is None else None
    info = SweepTrainInfo(
        name=name,
        base=args.base,
        base_source=BASES[args.base],
        scale=args.scale,
        target=args.target,
        dataset=args.dataset,
        extras=sorted(args.extra),
        lr=args.lr,
        seed=args.seed,
        full_epochs=args.full_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        lr_threshold=args.lr_threshold,
        max_epochs=args.max_epochs,
        n_corpus=len(texts),
        n_train=len(x_train),
        n_val=len(x_val),
        epochs_run=trainer.current_epoch,
        best_epoch=best.epoch if best else None,
        best_val_loss=best.val_loss if best else None,
        wall_time_s=round(time.time() - start, 1),
        torch_threads=args.threads,
        libraries=library_versions(),
        curve=curve,
    )
    save(trainable.to_static_model(), info)
    logger.info(json.dumps({k: v for k, v in asdict(info).items() if k != "curve"}))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
