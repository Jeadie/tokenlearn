"""Featurize texts with the teacher, for one of two targets.

Targets:
- mean: the mean of bge's token vectors without CLS and SEP (tokenlearn's target, through
  `tokenlearn.featurize`).
- cls: bge's own sentence embedding (CLS pooling, then normalization), which is what bge uses
  for search.

Sources:
- corpus: the pinned corpus `text` column (the documents that are searched).
- trainq: the pinned train-query texts from fetch_dev_qrels.py (never a dev or test query).
- sents: sentences of the corpus documents, a fixed random sample of at most 2 per document.
- c4: the pinned sample of C4 texts that potion-base-8M was trained on (fetch_c4.py), for replay.
  It does not depend on the dataset.

Writes FEATURIZED_DIR/<dataset>-<source>-<target> as a datasets directory with `text` and
`embedding` columns, the format that `tokenlearn.utils.collect_means_and_texts` reads.
"""

import json
import logging
import multiprocessing
import random
import re
import time
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from common import FEATURIZED_DIR, TEACHER, library_versions
from datasets import Dataset
from make_spicepods import DATA_DIR
from sentence_transformers import SentenceTransformer

from tokenlearn.featurize import _truncate_text, featurize

logger = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_MIN_SENTENCE_WORDS = 4
_SENTENCES_PER_DOC = 2
# MPS memory grows with every new batch shape and is not given back inside a process, so the
# CLS target is computed in chunks, each in a new process.
_CLS_CHUNK = 4096


@dataclass
class FeaturizeTextsInfo:
    dataset: str
    source: str
    target: str
    rows_in: int
    rows_out: int
    teacher: str
    max_length: int
    device: str
    wall_time_s: float
    libraries: dict[str, str]


def output_dir(dataset: str, source: str, target: str) -> Path:
    """Where the featurized texts go. The original mean-target corpus keeps its first location."""
    if source == "corpus" and target == "mean":
        return FEATURIZED_DIR / dataset
    if source == "c4":
        return FEATURIZED_DIR / f"c4-{target}"
    return FEATURIZED_DIR / f"{dataset}-{source}-{target}"


def load_texts(dataset: str, source: str) -> list[str]:
    """The non-empty texts of one source."""
    if source == "trainq":
        texts = pq.read_table(DATA_DIR / dataset / "trainq.parquet").column("text").to_pylist()
    elif source == "c4":
        texts = pq.read_table(DATA_DIR / "c4" / "texts.parquet").column("text").to_pylist()
    else:
        texts = pq.read_table(DATA_DIR / dataset / "corpus.parquet").column("text").to_pylist()
    texts = [t for t in texts if t and t.strip()]
    if source != "sents":
        return texts
    rng = random.Random(0)
    sentences = []
    for doc in texts:
        candidates = [s.strip() for s in _SENTENCE_END.split(doc) if len(s.split()) >= _MIN_SENTENCE_WORDS]
        sentences += rng.sample(candidates, min(_SENTENCES_PER_DOC, len(candidates)))
    return sentences


def teacher(device: str, max_length: int) -> SentenceTransformer:
    """The teacher, with its maximum length set."""
    model = SentenceTransformer(TEACHER, device=device)
    model.max_seq_length = max_length
    model.tokenizer.model_max_length = max_length
    return model


def encode_chunk(job: tuple[list[str], int, int]) -> np.ndarray:
    """Embed one chunk on MPS with the teacher's own pooling (runs in a child process)."""
    texts, batch_size, max_length = job
    model = teacher("mps", max_length)
    return model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)


def featurize_cls(texts: list[str], out: Path, batch_size: int, max_length: int) -> None:
    """Embed each text, cut to the teacher's max length, with the teacher's own pooling."""
    tokenizer = teacher("cpu", max_length).tokenizer
    truncated = [_truncate_text(tokenizer, t) for t in texts]
    jobs = [(truncated[i : i + _CLS_CHUNK], batch_size, max_length) for i in range(0, len(truncated), _CLS_CHUNK)]
    context = multiprocessing.get_context("spawn")
    chunks = []
    for job in jobs:
        # A new executor per chunk: its process exits after the chunk, and frees the MPS memory.
        with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
            chunks.append(pool.submit(encode_chunk, job).result())
    vectors = np.concatenate(chunks)
    Dataset.from_dict({"text": truncated, "embedding": vectors.astype(np.float32)}).save_to_disk(str(out))


def main() -> None:
    """Featurize one (dataset, source, target)."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source", required=True, choices=("corpus", "trainq", "sents", "c4"))
    parser.add_argument("--target", required=True, choices=("mean", "cls"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    out = output_dir(args.dataset, args.source, args.target)
    if (out / "featurize_info.json").exists():
        logger.info(f"{out} exists; skipping")
        return
    start = time.time()
    texts = load_texts(args.dataset, args.source)
    # Longest first: batches of similar length need less padding. Each text is embedded alone.
    texts.sort(key=len, reverse=True)
    if args.target == "cls":
        featurize_cls(texts, out, args.batch_size, args.max_length)
    else:
        featurize(
            Dataset.from_dict({"text": texts}),
            teacher("mps", args.max_length),
            str(out),
            max_rows=10_000_000,
            batch_size=1024,
            text_key="text",
            max_length=args.max_length,
        )
    info = FeaturizeTextsInfo(
        dataset=args.dataset,
        source=args.source,
        target=args.target,
        rows_in=len(texts),
        rows_out=len(Dataset.load_from_disk(str(out))),
        teacher=TEACHER,
        max_length=args.max_length,
        device="mps",
        wall_time_s=round(time.time() - start, 1),
        libraries=library_versions(),
    )
    (out / "featurize_info.json").write_text(json.dumps(asdict(info), indent=2) + "\n")
    logger.info(json.dumps(asdict(info)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
