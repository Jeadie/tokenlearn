"""Featurize one MTEB corpus with the bge teacher, using tokenlearn.featurize semantics.

The corpus file is the same Hugging Face parquet the spicepods read, so the static models
train on exactly the text spiced embeds (the `text` column; no title).
"""

import json
import logging
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from pathlib import Path

import requests  # type: ignore[import-untyped]
from common import FEATURIZED_DIR, TEACHER, library_versions
from datasets import Dataset
from sentence_transformers import SentenceTransformer

from tokenlearn.featurize import featurize

PARQUET_URL = "https://huggingface.co/api/datasets/mteb/{ds}/parquet/corpus/corpus/0.parquet"
SHARD_LIST_URL = "https://huggingface.co/api/datasets/mteb/{ds}/parquet"


logger = logging.getLogger(__name__)


@dataclass
class FeaturizeInfo:
    dataset: str
    source_url: str
    resolved_url: str
    corpus_shards: list[str]
    rows_in: int
    rows_out: int
    teacher: str
    max_length: int
    key: str
    device: str
    libraries: dict[str, str]


def download_corpus(ds: str, dest: Path) -> tuple[str, str]:
    """Download the corpus parquet; return (source URL, URL it redirected to)."""
    url = PARQUET_URL.format(ds=ds)
    response = requests.get(url, timeout=600)
    response.raise_for_status()
    dest.write_bytes(response.content)
    # Drop the signed CDN query string; the path alone identifies the file.
    return url, response.url.split("?", 1)[0]


def main() -> None:
    """Featurize one corpus with the teacher and record the source and settings."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    out = FEATURIZED_DIR / args.dataset
    if (out / "dataset_info.json").exists():
        logger.info(f"{out} exists; skipping")
        return

    raw_dir = FEATURIZED_DIR.parent / "corpus"
    raw_dir.mkdir(parents=True, exist_ok=True)
    parquet = raw_dir / f"{args.dataset}.parquet"
    source_url, resolved_url = download_corpus(args.dataset, parquet)
    # spiced reads shard 0 only; record every shard so a multi-shard corpus is visible.
    shards = requests.get(SHARD_LIST_URL.format(ds=args.dataset), timeout=60).json()
    corpus_shards = shards.get("corpus", {}).get("corpus", [])

    dataset = Dataset.from_parquet(str(parquet))
    # Sorting by length cuts padding: about 2x faster on MPS. Each text's target does not
    # depend on its batch, and training shuffles, so the order has no other effect.
    lengths = [len(text) for text in dataset["text"]]
    dataset = dataset.add_column("_len", lengths).sort("_len", reverse=True).remove_columns("_len")
    model = SentenceTransformer(TEACHER, device="mps")
    featurize(
        dataset,
        model,
        str(out),
        max_rows=10_000_000,
        # Texts handed to one `encode` call; sentence-transformers batches them by 32 inside.
        batch_size=args.batch_size,
        text_key="text",
        max_length=args.max_length,
    )

    rows_out = len(Dataset.load_from_disk(str(out)))
    info = FeaturizeInfo(
        dataset=args.dataset,
        source_url=source_url,
        resolved_url=resolved_url,
        corpus_shards=corpus_shards,
        rows_in=len(dataset),
        rows_out=rows_out,
        teacher=TEACHER,
        max_length=args.max_length,
        key="text",
        device="mps",
        libraries=library_versions(),
    )
    (out / "featurize_info.json").write_text(json.dumps(asdict(info), indent=2) + "\n")
    logger.info(json.dumps(asdict(info), indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
