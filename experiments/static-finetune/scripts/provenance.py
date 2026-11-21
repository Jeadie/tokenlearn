"""Record the Hugging Face revisions this experiment used, in provenance.json."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from huggingface_hub import HfApi

from common import DATASETS, EXPERIMENT_DIR, POTION, TEACHER, library_versions


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
    print(json.dumps(asdict(provenance), indent=2))


if __name__ == "__main__":
    main()
