"""Featurize every corpus (one at a time on MPS) and run every fine-tune (several at a time on CPU).

Fine-tunes start as soon as their corpus is featurized, in priority order. Finished models are
skipped, so the script can be re-run after a failure.
"""

import os
import subprocess
import sys
import threading
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

from common import DATASETS, DISTILLED, EXPERIMENT_DIR, FEATURIZED_DIR, MODELS_DIR, SEEDS

SCRIPTS = Path(__file__).resolve().parent
LOGS = EXPERIMENT_DIR / "logs"
ENV = {**os.environ, "PYTHONPATH": str(EXPERIMENT_DIR.parent.parent)}


@dataclass(frozen=True)
class FinetuneJob:
    base: str
    dataset: str
    seed: int

    @property
    def name(self) -> str:
        return f"{self.base}-ft-{self.dataset}-s{self.seed}"


def featurized(dataset: str) -> bool:
    return (FEATURIZED_DIR / dataset / "featurize_info.json").exists()


def featurizing_elsewhere(dataset: str) -> bool:
    """True while another process (e.g. one started by hand) featurizes this corpus."""
    pattern = f"featurize_corpus.py --dataset {dataset}"
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True, check=False).returncode == 0


def featurize_all() -> None:
    for dataset in DATASETS:
        while featurizing_elsewhere(dataset):
            time.sleep(10)
        if featurized(dataset):
            continue
        with open(LOGS / f"featurize_{dataset}.log", "a") as log:
            subprocess.run(
                [sys.executable, str(SCRIPTS / "featurize_corpus.py"), "--dataset", dataset],
                stdout=log, stderr=subprocess.STDOUT, env=ENV, check=False,
            )


def jobs_in_priority_order() -> list[FinetuneJob]:
    """Dataset order first (scifact is needed first), then model 4 before 4b, then seed."""
    return [
        FinetuneJob(base, dataset, seed)
        for dataset in DATASETS
        for base in ("potion-base-8M", DISTILLED)
        for seed in SEEDS
    ]


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2, help="torch CPU threads per fine-tune")
    args = parser.parse_args()

    (LOGS / "train").mkdir(parents=True, exist_ok=True)
    featurizer = threading.Thread(target=featurize_all, daemon=True)
    featurizer.start()

    pending = [job for job in jobs_in_priority_order() if not (MODELS_DIR / job.name).exists()]
    running: dict[FinetuneJob, subprocess.Popen[bytes]] = {}
    failed: list[FinetuneJob] = []
    while pending or running:
        for job, proc in list(running.items()):
            if proc.poll() is not None:
                del running[job]
                status = "ok" if proc.returncode == 0 and (MODELS_DIR / job.name).exists() else "FAILED"
                if status == "FAILED":
                    failed.append(job)
                print(f"{time.strftime('%H:%M:%S')} {status} {job.name}", flush=True)
        for job in [j for j in pending if featurized(j.dataset)]:
            if len(running) >= args.workers:
                break
            pending.remove(job)
            log = open(LOGS / "train" / f"{job.name}.log", "w")
            running[job] = subprocess.Popen(
                [sys.executable, str(SCRIPTS / "finetune.py"), "--base", job.base, "--dataset", job.dataset,
                 "--seed", str(job.seed), "--threads", str(args.threads)],
                stdout=log, stderr=subprocess.STDOUT, env=ENV,
            )
            print(f"{time.strftime('%H:%M:%S')} start {job.name}", flush=True)
        if not featurizer.is_alive() and pending and not running:
            # A corpus failed to featurize; its jobs can never start.
            failed.extend(pending)
            print("featurize failed for: " + ", ".join(sorted({j.dataset for j in pending})), flush=True)
            break
        time.sleep(5)
    print("failed: " + (", ".join(j.name for j in failed) or "none"), flush=True)


if __name__ == "__main__":
    main()
