"""Pin the non-test query sets: dev qrels to choose between settings, and train queries as extra training text.

NFCorpus and FiQA have a `dev` split; it is the dev set. SciFact has only `train` and `test`, so its
train queries are split in two by a hash of the query id: one half is the dev set, the other half
is training text. Training text never includes a dev or test query.

Writes, per dataset, to the pinned MTEB directory:
- qrels_dev.parquet: the dev qrels (same columns as qrels_test.parquet).
- trainq.parquet: `_id`, `text` of the queries that may be used as training text.
and SHA256SUMS.sweep with the checksum of each file.
"""

import hashlib
import io
import json
import logging
import urllib.request
from dataclasses import asdict, dataclass

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from common import DATASETS, EXPERIMENT_DIR
from make_spicepods import DATA_DIR

logger = logging.getLogger(__name__)

PARQUET_URL = "https://huggingface.co/api/datasets/mteb/{dataset}/parquet/default/{split}/0.parquet"
# The qrels split that the dev set comes from.
DEV_SOURCE = {"scifact": "train", "nfcorpus": "dev", "fiqa": "dev"}


@dataclass
class QuerySets:
    dataset: str
    dev_source: str
    dev_queries: int
    trainq_queries: int
    test_queries: int
    dev_test_overlap: int
    trainq_dev_or_test_overlap: int


def fetch_qrels(dataset: str, split: str) -> pa.Table:
    """Download one qrels split of an MTEB dataset."""
    with urllib.request.urlopen(PARQUET_URL.format(dataset=dataset, split=split)) as response:
        return pq.read_table(io.BytesIO(response.read()))


def in_dev_half(query_id: str) -> bool:
    """A fixed, seed-free split of SciFact's train queries."""
    return hashlib.sha256(query_id.encode()).digest()[0] % 2 == 0


def query_ids(qrels: pa.Table) -> set[str]:
    """The distinct query ids of a qrels table."""
    return {str(q) for q in qrels.column("query-id").to_pylist()}


def main() -> None:
    """Write the dev qrels and the train-query texts for every dataset."""
    sums = []
    report = []
    for dataset in DATASETS:
        data = DATA_DIR / dataset
        test_ids = query_ids(pq.read_table(data / "qrels_test.parquet"))
        source = fetch_qrels(dataset, DEV_SOURCE[dataset])
        source = source.set_column(
            source.schema.get_field_index("query-id"), "query-id", pc.cast(source.column("query-id"), pa.string())
        )
        if dataset == "scifact":
            mask = pa.array([in_dev_half(q) for q in source.column("query-id").to_pylist()])
            dev = source.filter(mask)
            trainq_ids = query_ids(source.filter(pc.invert(mask)))
        else:
            dev = source
            trainq_ids = query_ids(fetch_qrels(dataset, "train"))
        dev_ids = query_ids(dev)
        trainq_ids -= dev_ids | test_ids

        queries = pq.read_table(data / "queries.parquet")
        trainq = queries.filter(pc.is_in(pc.cast(queries.column("_id"), pa.string()), pa.array(sorted(trainq_ids))))
        trainq = trainq.select(["_id", "text"])

        pq.write_table(dev, data / "qrels_dev.parquet")
        pq.write_table(trainq, data / "trainq.parquet")
        for name in ("qrels_dev.parquet", "trainq.parquet"):
            digest = hashlib.sha256((data / name).read_bytes()).hexdigest()
            sums.append(f"{digest}  {dataset}/{name}")
        report.append(
            QuerySets(
                dataset=dataset,
                dev_source=DEV_SOURCE[dataset],
                dev_queries=len(dev_ids),
                trainq_queries=trainq.num_rows,
                test_queries=len(test_ids),
                dev_test_overlap=len(dev_ids & test_ids),
                trainq_dev_or_test_overlap=len(
                    query_ids(trainq.rename_columns(["query-id", "text"])) & (dev_ids | test_ids)
                ),
            )
        )
    (DATA_DIR / "SHA256SUMS.sweep").write_text("\n".join(sums) + "\n")
    (EXPERIMENT_DIR / "query_sets.json").write_text(json.dumps([asdict(r) for r in report], indent=2) + "\n")
    for r in report:
        logger.info(json.dumps(asdict(r)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
