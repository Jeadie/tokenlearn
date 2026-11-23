"""Write the gate G3 models: each dataset's final model (seed 42) with n-gram rows that are all zero.

A zero row adds nothing to the sum and one to the count, so each text vector only changes length.
Search results must then be the same as the base model's: any difference comes from the spiced
build, not from the n-gram rows. The n-grams are the 64,000 most frequent 2- and 3-grams of the
dataset's corpus.
"""

import logging

import ngrams
import numpy as np
from common import DATASETS
from model2vec import StaticModel
from plans import zero_name
from sweep_finetune import SWEEP_MODELS_DIR, load

logger = logging.getLogger(__name__)

ORDERS = (2, 3)
K = 64_000


def main() -> None:
    """Write one zero-row model per dataset; skip those that exist."""
    for dataset in DATASETS:
        name = zero_name(dataset)
        out = SWEEP_MODELS_DIR / name
        if out.exists():
            logger.info(f"{name} exists; skipping")
            continue
        source = SWEEP_MODELS_DIR / name.removesuffix("-ngzero")
        model = StaticModel.from_pretrained(str(source))
        texts, _ = load(dataset, "corpus", "mean")
        drop = ngrams.unk_id(model.tokenizer)
        ids = [ngrams.without(i, drop) for i in ngrams.fit_token_ids(model.tokenizer, texts)]
        table = ngrams.mine(ids, ORDERS, K, min_count=2)
        rows = np.zeros((len(table.ngrams), model.embedding.shape[1]), dtype=np.float32)
        ngrams.with_ngram_rows(source, out, table, rows)
        logger.info(f"{name}: {len(table.ngrams)} zero n-gram rows")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
