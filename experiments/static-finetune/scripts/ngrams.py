"""Additive token n-grams for a Model2Vec static model.

A text's vector is the mean of its token rows and of the rows of the listed n-grams it contains,
normalized:

    tokens:  t1 t2 t3 t4    (no special tokens, UNK dropped, at most 512)
    rows:    U[t1] U[t2] U[t3] U[t4]  +  G[t1 t2] G[t2 t3] G[t3 t4] G[t1 t2 t3] ...
                                         (only n-grams in the list; repeats count each time)
    vector = normalize(sum(rows) / number of rows)

The list and the n-gram rows are saved in model.safetensors next to the token rows:
`ngram_ids` ([K, MAX_ORDER] int32, -1 after a shorter n-gram) and `ngram_embeddings` ([K, dim]).
model2vec-rs (branch `ngrams`) reads them. The Python model2vec library ignores them, so use
`NgramEncoder` to encode with them in Python.

Training puts the K n-gram rows below the V token rows of one table. A text's input ids are its
token ids, then V + row for each listed n-gram it contains, so model2vec's own mean pooling
gives the vector above. The n-gram rows start at zero: before training, the model gives the same
vectors as the model without n-grams.
"""

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from model2vec import StaticModel
from model2vec.train import StaticModelForSimilarity
from model2vec.train.dataset import TextDataset
from safetensors import safe_open
from safetensors.numpy import load_file, save_file
from tokenizers import Tokenizer

MAX_ORDER = 3
# model2vec's fit() tokenizes the first 10 characters per token of each text (its speed limit).
FIT_CHARS_PER_TOKEN = 10
MAX_TOKENS = 512


@dataclass
class NgramTable:
    """The listed n-grams; row i of the n-gram table belongs to `ngrams[i]`."""

    ngrams: list[tuple[int, ...]]
    rows: dict[tuple[int, ...], int] = field(init=False)
    orders: list[int] = field(init=False)

    def __post_init__(self) -> None:
        """Index the n-grams by their token ids."""
        self.rows = {ngram: row for row, ngram in enumerate(self.ngrams)}
        self.orders = sorted({len(ngram) for ngram in self.ngrams})

    def find(self, ids: Sequence[int]) -> list[int]:
        """Rows of the listed n-grams in `ids`, shortest n-grams first, then left to right (as model2vec-rs)."""
        found = []
        for order in self.orders:
            for start in range(len(ids) - order + 1):
                row = self.rows.get(tuple(ids[start : start + order]))
                if row is not None:
                    found.append(row)
        return found

    def ids_array(self) -> np.ndarray:
        """The `ngram_ids` tensor."""
        out = np.full((len(self.ngrams), MAX_ORDER), -1, dtype=np.int32)
        for row, ngram in enumerate(self.ngrams):
            out[row, : len(ngram)] = ngram
        return out

    @classmethod
    def from_ids_array(cls, ids: np.ndarray) -> "NgramTable":
        """Read an `ngram_ids` tensor."""
        return cls([tuple(int(i) for i in row if i >= 0) for row in ids])


def mine(token_lists: Iterable[Sequence[int]], orders: Sequence[int], k: int, min_count: int) -> NgramTable:
    """The `k` most frequent n-grams of the given orders that occur at least `min_count` times."""
    counts: Counter[tuple[int, ...]] = Counter()
    for ids in token_lists:
        for order in orders:
            counts.update(zip(*(ids[offset:] for offset in range(order)), strict=False))
    kept = sorted(((c, g) for g, c in counts.items() if c >= min_count), key=lambda cg: (-cg[0], cg[1]))
    return NgramTable([g for _, g in kept[:k]])


def unk_id(tokenizer: Tokenizer) -> int | None:
    """The id of the tokenizer's unknown token, as model2vec-rs finds it."""
    spec = json.loads(tokenizer.to_str())
    return tokenizer.token_to_id(spec.get("model", {}).get("unk_token") or "[UNK]")


def fit_token_ids(tokenizer: Tokenizer, texts: Sequence[str]) -> list[list[int]]:
    """Token ids as model2vec's fit() makes them (its `_prepare_dataset`)."""
    chars = MAX_TOKENS * FIT_CHARS_PER_TOKEN
    out: list[list[int]] = []
    for start in range(0, len(texts), 1024):
        batch = [text[:chars] for text in texts[start : start + 1024]]
        out.extend(e.ids[:MAX_TOKENS] for e in tokenizer.encode_batch_fast(batch, add_special_tokens=False))
    return out


def without(ids: Sequence[int], drop: int | None) -> list[int]:
    """`ids` without the unknown token, as model2vec-rs pools them."""
    return [i for i in ids if i != drop]


class ScaledRows(torch.nn.Module):
    """An embedding table whose rows from `start` on are multiplied by `scale`.

    Adam moves each stored value by about the learning rate per step, so a row multiplied by
    `scale` moves `scale` times as fast as the other rows: a learning rate of its own.
    """

    def __init__(self, inner: torch.nn.Embedding, start: int, scale: float) -> None:
        """Wrap `inner`."""
        super().__init__()
        self.inner = inner
        factor = torch.ones((inner.num_embeddings, 1))
        factor[start:] = scale
        self.register_buffer("factor", factor)

    @property
    def weight(self) -> torch.Tensor:
        """The rows as stored (model2vec reads the device from this)."""
        return self.inner.weight

    def effective(self) -> torch.Tensor:
        """The rows as the model uses them."""
        return self.inner.weight * self.factor

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Look up rows, scaled."""
        return self.inner(ids) * self.factor[ids]


class NgramModelForSimilarity(StaticModelForSimilarity):
    """model2vec's trainable model, with n-gram ids appended to each text's token ids."""

    table: NgramTable
    n_tokens: int
    drop_id: int | None
    # The n-gram rows' learning rate, as a fraction of the token rows' learning rate.
    lr_scale: float = 1.0
    # nn.Embedding, or ScaledRows around it (which has the parts model2vec uses: forward and `weight`).
    embeddings: torch.nn.Module

    def _initialize(self) -> None:
        """model2vec's initialization; then scale the n-gram rows when they have a learning rate of their own."""
        super()._initialize()
        if self.lr_scale != 1.0:
            assert isinstance(self.embeddings, torch.nn.Embedding)
            self.embeddings = ScaledRows(self.embeddings, self.n_tokens, self.lr_scale)

    def _prepare_dataset(self, X: list[str], y: torch.Tensor, max_length: int = MAX_TOKENS) -> TextDataset:
        dataset = super()._prepare_dataset(X, y, max_length)
        dataset.tokenized_texts = [
            ids + [self.n_tokens + row for row in self.table.find(without(ids, self.drop_id))]
            for ids in dataset.tokenized_texts
        ]
        return dataset


def trainable(model: StaticModel, table: NgramTable, out_dim: int, lr_scale: float = 1.0) -> NgramModelForSimilarity:
    """A trainable model whose table holds the token rows, then one zero row per listed n-gram."""
    out = NgramModelForSimilarity.from_static_model(model=model, out_dim=out_dim, n_layers=0, freeze_weights=True)
    if model.weights is not None or model.token_mapping is not None:
        raise ValueError("n-grams need a model without token weights or a token mapping")
    out.lr_scale = lr_scale
    out.table = table
    out.n_tokens = model.embedding.shape[0]
    out.drop_id = unk_id(model.tokenizer)
    zeros = torch.zeros((len(table.ngrams), out.vectors.shape[1]), dtype=out.vectors.dtype)
    # fit() rebuilds the embeddings and token weights from these two.
    out.vectors = torch.cat([out.vectors, zeros])
    out.token_mapping = torch.nn.Parameter(torch.arange(len(out.vectors), dtype=torch.int64), requires_grad=False)
    return out


def save_pretrained(model: NgramModelForSimilarity, path: Path) -> None:
    """Save like model2vec's to_static_model().save_pretrained(), with the n-gram tensors added."""
    with torch.no_grad():
        rows = model.embeddings.effective() if isinstance(model.embeddings, ScaledRows) else model.embeddings.weight
        vectors = rows.cpu().numpy() * torch.sigmoid(model.w).cpu().numpy()[:, None]
    static = StaticModel(vectors=vectors[: model.n_tokens], tokenizer=model.tokenizer, normalize=model.normalize)
    static.save_pretrained(str(path))
    weights = path / "model.safetensors"
    with safe_open(str(weights), framework="numpy") as f:
        metadata = f.metadata()
    tensors = load_file(str(weights))
    tensors["ngram_ids"] = model.table.ids_array()
    tensors["ngram_embeddings"] = vectors[model.n_tokens :].astype(tensors["embeddings"].dtype)
    save_file(tensors, str(weights), metadata=metadata)
    config = json.loads((path / "config.json").read_text())
    config["ngram_orders"] = model.table.orders
    config["ngram_count"] = len(model.table.ngrams)
    (path / "config.json").write_text(json.dumps(config, indent=4) + "\n")


def with_ngram_rows(source: Path, out: Path, table: NgramTable, rows: np.ndarray) -> None:
    """Copy a saved model to `out`, with the given n-gram list and rows."""
    out.mkdir(parents=True)
    for name in ("tokenizer.json", "config.json", "modules.json", "README.md"):
        if (source / name).exists():
            (out / name).write_bytes((source / name).read_bytes())
    weights = source / "model.safetensors"
    with safe_open(str(weights), framework="numpy") as f:
        metadata = f.metadata()
    tensors = {k: v for k, v in load_file(str(weights)).items() if not k.startswith("ngram_")}
    tensors["ngram_ids"] = table.ids_array()
    tensors["ngram_embeddings"] = rows.astype(tensors["embeddings"].dtype)
    save_file(tensors, str(out / "model.safetensors"), metadata=metadata)


class NgramEncoder:
    """Encode texts as model2vec-rs does, n-gram rows included. The reference for checking spiced."""

    def __init__(self, path: Path) -> None:
        """Load a saved model, with its n-gram tensors when it has them."""
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        tensors = load_file(str(path / "model.safetensors"))
        self.embeddings = tensors["embeddings"].astype(np.float64)
        self.table = NgramTable.from_ids_array(tensors["ngram_ids"]) if "ngram_ids" in tensors else NgramTable([])
        self.ngram_embeddings = tensors.get("ngram_embeddings", np.zeros((0, self.embeddings.shape[1]))).astype(
            np.float64
        )
        config = path / "config.json"
        self.normalize = json.loads(config.read_text()).get("normalize", True) if config.exists() else True
        self.drop_id = unk_id(self.tokenizer)
        lengths = sorted(len(token.encode()) for token in self.tokenizer.get_vocab(with_added_tokens=False))
        self.median_token_length = lengths[len(lengths) // 2]

    def token_ids(self, text: str, max_length: int = MAX_TOKENS) -> list[int]:
        """Characters cut to max_length × the median token length, tokens without UNK, cut to max_length."""
        ids = self.tokenizer.encode(text[: max_length * self.median_token_length], add_special_tokens=False).ids
        return without(ids, self.drop_id)[:max_length]

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """One vector per text."""
        out = np.zeros((len(texts), self.embeddings.shape[1]))
        for i, text in enumerate(texts):
            ids = self.token_ids(text)
            rows = self.table.find(ids)
            total = self.embeddings[ids].sum(axis=0) + self.ngram_embeddings[rows].sum(axis=0)
            vector = total / max(len(ids) + len(rows), 1)
            if self.normalize:
                vector = vector / max(np.linalg.norm(vector), 1e-12)
            out[i] = vector
        return out
