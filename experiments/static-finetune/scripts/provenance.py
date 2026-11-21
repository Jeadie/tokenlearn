"""Record the Hugging Face revisions this experiment used, in provenance.json."""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from common import DATASETS, EXPERIMENT_DIR, POTION, TEACHER, library_versions
from huggingface_hub import HfApi

logger = logging.getLogger(__name__)


@dataclass
class DatasetRevision:
    repo: str
    main_sha: str
    # The spicepods and featurize read the auto-converted parquet on this ref.
    convert_parquet_sha: str | None


@dataclass
class Provenance:
    recorded_at: str
    models: dict[str, str]
    datasets: dict[str, DatasetRevision]
    libraries: dict[str, str]


def main() -> None:
    """Record the revisions of the models and datasets, and the library versions."""
    api = HfApi()
    datasets: dict[str, DatasetRevision] = {}
    for ds in DATASETS:
        repo = f"mteb/{ds}"
        refs = api.list_repo_refs(repo, repo_type="dataset")
        convert = next((r.target_commit for r in refs.converts if r.name == "parquet"), None)
        datasets[ds] = DatasetRevision(
            repo=repo,
            main_sha=api.dataset_info(repo).sha or "",
            convert_parquet_sha=convert,
        )
    provenance = Provenance(
        recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        models={repo: api.model_info(repo).sha or "" for repo in (TEACHER, POTION)},
        datasets=datasets,
        libraries=library_versions(),
    )
    (EXPERIMENT_DIR / "provenance.json").write_text(json.dumps(asdict(provenance), indent=2) + "\n")
    logger.info(json.dumps(asdict(provenance), indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
