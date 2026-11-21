"""Pin a fixed sample of general-domain texts for replay: the C4 texts potion-base-8M was trained on.

Downloads one shard of minishlab/tokenlearn-c4-en-bge-base-en-v1.5 at a pinned revision, takes a
fixed random sample of its texts, and writes DATA_DIR/c4/texts.parquet with its SHA-256 in
DATA_DIR/c4/SHA256SUMS. Only the texts are used; featurize_texts.py computes their targets the
same way as the corpus targets.
"""

import hashlib
import io
import logging
import urllib.request

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from make_spicepods import DATA_DIR

logger = logging.getLogger(__name__)

REPO = "minishlab/tokenlearn-c4-en-bge-base-en-v1.5"
REVISION = "3a6f3de041c410059e683316334efa05eae90795"
SHARD = "10m/train-00000-of-00089.parquet"
SAMPLE = 20_000


def main() -> None:
    """Download the shard, sample its texts and pin them."""
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{SHARD}"
    with urllib.request.urlopen(url) as response:
        shard = pq.read_table(io.BytesIO(response.read()), columns=["text"])
    texts = [t for t in shard.column("text").to_pylist() if t and t.strip()]
    pick = np.random.default_rng(0).choice(len(texts), size=SAMPLE, replace=False)
    out = DATA_DIR / "c4"
    out.mkdir(exist_ok=True)
    pq.write_table(pa.table({"text": [texts[i] for i in sorted(pick)]}), out / "texts.parquet")
    digest = hashlib.sha256((out / "texts.parquet").read_bytes()).hexdigest()
    (out / "SHA256SUMS").write_text(f"{digest}  texts.parquet\n# {url}\n")
    logger.info(f"{SAMPLE} of {len(texts)} texts from {url}; sha256 {digest}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
