"""Run one sweep plan: train models, then score each one with `testoperator run search`.

A plan (JSON, from plans.py) lists training runs and search runs. A training run calls
sweep_finetune.py. A search run starts when its model exists. Each search run gets its own
spiced ports, and its metric@k table goes to results/sweep/scores.jsonl. Its per-query metrics go
to the OTLP collector, as in the first experiment. Finished work is skipped, so a plan can be
run again after a failure.
"""

import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from common import EXPERIMENT_DIR
from make_spicepods import Model, models, render
from run_benchmarks import Job, run_job, sha256
from sweep_finetune import SWEEP_MODELS_DIR

logger = logging.getLogger(__name__)

SCRIPTS = Path(__file__).resolve().parent
SWEEP_SPICEPODS = EXPERIMENT_DIR / "spicepods_sweep"
SCORES_PATH = EXPERIMENT_DIR / "results" / "sweep" / "scores.jsonl"
LOGS = EXPERIMENT_DIR / "logs" / "sweep"
ENV = {**os.environ, "PYTHONPATH": str(EXPERIMENT_DIR.parent.parent)}
POLL_SECONDS = 5
# A failed search run is tried this many times in all. testoperator fails a run when the host
# is too busy for its latency check, even when the metric table is complete.
SEARCH_ATTEMPTS = 3


@dataclass(frozen=True)
class TrainSpec:
    base: str
    dataset: str
    lr: float
    seed: int
    scale: float = 1.0
    target: str = "mean"
    extras: tuple[str, ...] = ()
    full_epochs: int | None = None

    @property
    def name(self) -> str:
        """The model directory name, the same as sweep_finetune.model_name."""
        scale = f"-x{self.scale:g}" if self.scale != 1 else ""
        extras = "".join(f"+{e}" for e in sorted(self.extras))
        full = f"-full{self.full_epochs}" if self.full_epochs else ""
        return f"sw-{self.base}{scale}-{self.target}-{self.dataset}{extras}-lr{self.lr:g}{full}-s{self.seed}"

    def command(self, threads: int) -> list[str]:
        """The sweep_finetune.py command line."""
        cmd = [sys.executable, str(SCRIPTS / "sweep_finetune.py"), "--base", self.base, "--dataset", self.dataset]
        cmd += ["--lr", str(self.lr), "--seed", str(self.seed), "--scale", str(self.scale), "--target", self.target]
        cmd += ["--threads", str(threads)]
        for extra in self.extras:
            cmd += ["--extra", extra]
        if self.full_epochs:
            cmd += ["--full-epochs", str(self.full_epochs)]
        return cmd


@dataclass(frozen=True)
class SearchSpec:
    # A sweep model name, or a label of make_spicepods.models() (the first experiment's models),
    # or "hf-mean[bge-base-en-v1.5]" for bge with mean pooling.
    model: str
    eval_dataset: str
    split: str  # "dev" or "test"

    @property
    def run_name(self) -> str:
        """The spicepod name, and the run directory name."""
        return f"{self.model}-{self.split}"


@dataclass
class Plan:
    name: str
    train: list[TrainSpec]
    search: list[SearchSpec]


def load_plan(path: Path) -> Plan:
    """Read a plan file."""
    raw = json.loads(path.read_text())
    train = [TrainSpec(**{**t, "extras": tuple(t.get("extras", ()))}) for t in raw["train"]]
    return Plan(name=raw["name"], train=train, search=[SearchSpec(**s) for s in raw["search"]])


def spec_model(spec: SearchSpec) -> Model:
    """The embedding model of a search run."""
    if spec.model.startswith("sw-"):
        path = SWEEP_MODELS_DIR / spec.model
        return Model(
            label=spec.model,
            source=f"model2vec:{path}",
            family="sweep",
            params=(("embed_max_token_length", 512),),
            local_path=str(path),
        )
    if spec.model == "hf-mean[bge-base-en-v1.5]":
        return Model(
            label=spec.model,
            source="huggingface:huggingface.co/BAAI/bge-base-en-v1.5",
            family="1-mean",
            params=(("truncate", "END"), ("pooling", "mean")),
            hf_model="BAAI/bge-base-en-v1.5",
        )
    labels = {m.label: m for m in models()}
    if spec.model not in labels:
        raise SystemExit(f"unknown model {spec.model}")
    return labels[spec.model]


def search_job(spec: SearchSpec) -> Job:
    """Write the spicepod of one search run and return its job."""
    model = spec_model(spec)
    qrels = "qrels_dev.parquet" if spec.split == "dev" else "qrels_test.parquet"
    path = SWEEP_SPICEPODS / spec.split / spec.eval_dataset / f"{spec.run_name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(spec.run_name, spec.eval_dataset, model, qrels))
    return Job(
        name=spec.run_name,
        eval_dataset=spec.eval_dataset,
        spicepod_path=str(path),
        family=model.family,
        local_path=model.local_path,
        hf_model=model.hf_model,
    )


_ROW = re.compile(r"^\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$")


def parse_metrics(log: Path) -> dict[str, dict[int, float]] | None:
    """The metric@k table that testoperator prints, or None when there is none."""
    rows: dict[str, dict[int, float]] = {"ndcg": {}, "recall": {}, "mrr": {}, "precision": {}}
    in_table = False
    for line in log.read_text(errors="replace").splitlines():
        if line.startswith("Retrieval metrics @k"):
            in_table = True
            continue
        if in_table:
            match = _ROW.match(line)
            if not match:
                if rows["ndcg"]:
                    break
                continue
            k = int(match.group(1))
            for metric, value in zip(rows, match.groups()[1:], strict=True):
                rows[metric][k] = float(value)
    return rows if rows["ndcg"] else None


def scored_runs() -> set[str]:
    """Search runs that already have a score from a passed run."""
    if not SCORES_PATH.exists():
        return set()
    lines = SCORES_PATH.read_text().splitlines()
    return {r["run_name"] for r in map(json.loads, lines) if r["exit_code"] == 0}


async def train(spec: TrainSpec, threads: int) -> bool:
    """Run one training; True when the model exists afterwards."""
    if (SWEEP_MODELS_DIR / spec.name).exists():
        return True
    LOGS.mkdir(parents=True, exist_ok=True)
    logger.info(f"[train start] {spec.name}")
    with (LOGS / f"{spec.name}.log").open("w") as log:
        process = await asyncio.create_subprocess_exec(
            *spec.command(threads), stdout=log, stderr=subprocess.STDOUT, env=ENV, cwd=SCRIPTS
        )
        code = await process.wait()
    ok = code == 0 and (SWEEP_MODELS_DIR / spec.name).exists()
    logger.info(f"[train {'done ' if ok else 'FAIL '}] {spec.name}")
    return ok


async def search(spec: SearchSpec, slot: int, args: argparse.Namespace, sha: str, version: str) -> bool:
    """Run one search run, with retries; append its score."""
    job = search_job(spec)
    code = -1
    for attempt in range(1, SEARCH_ATTEMPTS + 1):
        if attempt > 1:
            (job.out_dir / "stdout.log").rename(job.out_dir / f"stdout.attempt{attempt - 1}.log")
        ok = await run_job(job, slot, args, sha, version)
        code = json.loads((job.out_dir / "run.json").read_text())["exit_code"]
        if ok:
            break
    metrics = parse_metrics(job.out_dir / "stdout.log")
    record = {**asdict(spec), "run_name": spec.run_name, "exit_code": code, "attempts": attempt, "metrics": metrics}
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SCORES_PATH.open("a") as out:
        out.write(json.dumps(record) + "\n")
    ndcg10 = metrics["ndcg"].get(10) if metrics else None
    logger.info(f"[search done] exit={code} ndcg@10={ndcg10} {spec.run_name} ({spec.eval_dataset})")
    return code == 0


def ready(spec: SearchSpec) -> bool:
    """True when the model of a search run can be loaded now."""
    return not spec.model.startswith("sw-") or (SWEEP_MODELS_DIR / spec.model).is_dir()


async def run_plan(plan: Plan, args: argparse.Namespace) -> int:
    """Train and search with bounded concurrency; return the number of failures."""
    sha = sha256(args.testoperator)
    version = subprocess.run([str(args.spiced), "--version"], capture_output=True, text=True, check=True).stdout.strip()
    done = scored_runs()
    train_queue = [t for t in plan.train if not (SWEEP_MODELS_DIR / t.name).exists()]
    search_queue = [s for s in plan.search if s.run_name not in done]
    failed_models: set[str] = set()
    logger.info(f"plan {plan.name}: {len(train_queue)} to train, {len(search_queue)} to search")

    training: dict[asyncio.Task[bool], TrainSpec] = {}
    searching: dict[asyncio.Task[bool], tuple[SearchSpec, int]] = {}
    free_slots = list(range(args.search_workers))
    failures = 0
    while train_queue or search_queue or training or searching:
        while train_queue and len(training) < args.train_workers:
            train_spec = train_queue.pop(0)
            training[asyncio.create_task(train(train_spec, args.threads))] = train_spec
        search_queue = [s for s in search_queue if s.model not in failed_models]
        for search_spec in [s for s in search_queue if ready(s)]:
            if not free_slots:
                break
            search_queue.remove(search_spec)
            slot = free_slots.pop(0)
            searching[asyncio.create_task(search(search_spec, slot, args, sha, version))] = (search_spec, slot)
        pending = list(training) + list(searching)
        if not pending:
            if search_queue:
                failures += len(search_queue)
                logger.info(f"no model for: {[s.run_name for s in search_queue]}")
            break
        done_tasks, _ = await asyncio.wait(pending, timeout=POLL_SECONDS, return_when=asyncio.FIRST_COMPLETED)
        for task in done_tasks:
            if task in training:
                finished = training.pop(task)
                if not task.result():
                    failures += 1
                    failed_models.add(finished.name)
            else:
                _, slot = searching.pop(task)
                free_slots.append(slot)
                failures += 0 if task.result() else 1
    logger.info(f"plan {plan.name} finished, {failures} failures")
    return failures


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--train-workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2, help="torch CPU threads per training run")
    parser.add_argument("--search-workers", type=int, default=2)
    parser.add_argument("--port-base", type=int, default=19100)
    parser.add_argument("--testoperator", type=Path, default=Path.home() / ".spice/bin/testoperator-search-ports")
    parser.add_argument("--spiced", type=Path, default=Path.home() / ".spice/bin/spiced")
    parser.add_argument("--ready-wait", type=int, default=7200)
    parser.add_argument("--timeout", type=int, default=4 * 3600)
    parser.add_argument("--per-query-flag", default="--per-query-all-k")
    parser.add_argument("--http-port-flag", default="--spiced-http-port")
    parser.add_argument("--flight-port-flag", default="--spiced-flight-port")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    sys.exit(1 if asyncio.run(run_plan(load_plan((args := parse_args()).plan), args)) else 0)
