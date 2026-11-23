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
from dataclasses import asdict, dataclass, field
from pathlib import Path

from common import EXPERIMENT_DIR
from featurize_texts import output_dir
from make_spicepods import Model, models, render
from run_benchmarks import Job, run_job, sha256
from sweep_finetune import SWEEP_MODELS_DIR, ngram_suffix

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
    ngram_orders: tuple[int, ...] = ()
    ngram_k: int = 0
    ngram_lr_scale: float = 1.0

    @property
    def name(self) -> str:
        """The model directory name, the same as sweep_finetune.model_name."""
        scale = f"-x{self.scale:g}" if self.scale != 1 else ""
        extras = "".join(f"+{e}" for e in sorted(self.extras))
        full = f"-full{self.full_epochs}" if self.full_epochs else ""
        ngram = ngram_suffix(self.ngram_orders, self.ngram_k, self.ngram_lr_scale)
        return f"sw-{self.base}{scale}-{self.target}-{self.dataset}{extras}-lr{self.lr:g}{full}{ngram}-s{self.seed}"

    def data_ready(self) -> bool:
        """True when every featurized dataset this run reads exists."""
        sources = ("corpus", *self.extras)
        return all(output_dir(self.dataset, source, self.target).is_dir() for source in sources) and all(
            (output_dir(self.dataset, source, self.target) / "featurize_info.json").exists()
            for source in sources
            if (source, self.target) != ("corpus", "mean")
        )

    def command(self, threads: int) -> list[str]:
        """The sweep_finetune.py command line."""
        cmd = [sys.executable, str(SCRIPTS / "sweep_finetune.py"), "--base", self.base, "--dataset", self.dataset]
        cmd += ["--lr", str(self.lr), "--seed", str(self.seed), "--scale", str(self.scale), "--target", self.target]
        cmd += ["--threads", str(threads)]
        for extra in self.extras:
            cmd += ["--extra", extra]
        if self.full_epochs:
            cmd += ["--full-epochs", str(self.full_epochs)]
        if self.ngram_k:
            cmd += ["--ngram-orders", ",".join(map(str, self.ngram_orders)), "--ngram-k", str(self.ngram_k)]
            cmd += ["--ngram-lr-scale", str(self.ngram_lr_scale)]
        return cmd


@dataclass(frozen=True)
class SearchSpec:
    # A sweep model name, or a label of make_spicepods.models() (the first experiment's models),
    # or "hf-mean[bge-base-en-v1.5]" for bge with mean pooling.
    model: str
    eval_dataset: str
    split: str  # "dev" or "test"
    # Names the spiced build when it is not the one of the first sweep, so its runs get their own names.
    tag: str = ""

    @property
    def run_name(self) -> str:
        """The spicepod name, and the run directory name."""
        return f"{self.model}-{self.split}" + (f"-{self.tag}" if self.tag else "")


@dataclass
class Plan:
    name: str
    train: list[TrainSpec]
    search: list[SearchSpec]


def load_plan(path: Path) -> Plan:
    """Read a plan file."""
    raw = json.loads(path.read_text())
    train = [
        TrainSpec(**{**t, "extras": tuple(t.get("extras", ())), "ngram_orders": tuple(t.get("ngram_orders", ()))})
        for t in raw["train"]
    ]
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


def scored_runs(scores: Path) -> set[tuple[str, str]]:
    """(run name, eval dataset) of search runs that already have a score from a passed run."""
    if not scores.exists():
        return set()
    lines = scores.read_text().splitlines()
    return {(r["run_name"], r["eval_dataset"]) for r in map(json.loads, lines) if r["exit_code"] == 0}


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
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    with args.scores.open("a") as out:
        out.write(json.dumps(record) + "\n")
    ndcg10 = metrics["ndcg"].get(10) if metrics else None
    logger.info(f"[search done] exit={code} ndcg@10={ndcg10} {spec.run_name} ({spec.eval_dataset})")
    return code == 0


def ready(spec: SearchSpec) -> bool:
    """True when the model of a search run can be loaded now."""
    return not spec.model.startswith("sw-") or (SWEEP_MODELS_DIR / spec.model).is_dir()


@dataclass
class Runner:
    """The queues and running tasks of one plan."""

    args: argparse.Namespace
    sha: str
    version: str
    train_queue: list[TrainSpec]
    search_queue: list[SearchSpec]
    training: dict[asyncio.Task[bool], TrainSpec] = field(default_factory=dict)
    searching: dict[asyncio.Task[bool], tuple[SearchSpec, int]] = field(default_factory=dict)
    free_slots: list[int] = field(default_factory=list)
    failed_models: set[str] = field(default_factory=set)
    failures: int = 0

    def start_ready(self) -> None:
        """Start the training and search runs whose inputs exist, up to the worker limits."""
        for train_spec in [t for t in self.train_queue if t.data_ready()]:
            if len(self.training) >= self.args.train_workers:
                break
            self.train_queue.remove(train_spec)
            self.training[asyncio.create_task(train(train_spec, self.args.threads))] = train_spec
        self.search_queue = [s for s in self.search_queue if s.model not in self.failed_models]
        for search_spec in [s for s in self.search_queue if ready(s)]:
            if not self.free_slots:
                break
            self.search_queue.remove(search_spec)
            slot = self.free_slots.pop(0)
            task = asyncio.create_task(search(search_spec, slot, self.args, self.sha, self.version))
            self.searching[task] = (search_spec, slot)

    def finish(self, task: asyncio.Task[bool]) -> None:
        """Record one finished task."""
        if task in self.training:
            spec = self.training.pop(task)
            if not task.result():
                self.failures += 1
                self.failed_models.add(spec.name)
        else:
            _, slot = self.searching.pop(task)
            self.free_slots.append(slot)
            self.failures += 0 if task.result() else 1


async def run_plan(plan: Plan, args: argparse.Namespace) -> int:
    """Train and search with bounded concurrency; return the number of failures."""
    done = scored_runs(args.scores)
    version = subprocess.run([str(args.spiced), "--version"], capture_output=True, text=True, check=True).stdout.strip()
    if args.expect_version and args.expect_version not in version and not args.train_only:
        raise SystemExit(f"{args.spiced} is {version!r}, not the expected build {args.expect_version!r}")
    runner = Runner(
        args=args,
        sha=sha256(args.testoperator),
        version=version,
        train_queue=[t for t in plan.train if not (SWEEP_MODELS_DIR / t.name).exists()],
        search_queue=[] if args.train_only else [s for s in plan.search if (s.run_name, s.eval_dataset) not in done],
        free_slots=list(range(args.search_workers)),
    )
    logger.info(f"plan {plan.name}: {len(runner.train_queue)} to train, {len(runner.search_queue)} to search")
    while runner.train_queue or runner.search_queue or runner.training or runner.searching:
        runner.start_ready()
        pending = list(runner.training) + list(runner.searching)
        if not pending and runner.train_queue:
            # Featurized data for the remaining runs is still being made.
            await asyncio.sleep(POLL_SECONDS * 12)
            continue
        if not pending:
            runner.failures += len(runner.search_queue)
            logger.info(f"no model for: {[s.run_name for s in runner.search_queue]}")
            break
        done_tasks, _ = await asyncio.wait(pending, timeout=POLL_SECONDS, return_when=asyncio.FIRST_COMPLETED)
        for task in done_tasks:
            runner.finish(task)
    logger.info(f"plan {plan.name} finished, {runner.failures} failures")
    return runner.failures


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
    parser.add_argument("--scores", type=Path, default=SCORES_PATH, help="where search scores are appended")
    parser.add_argument("--expect-version", default="", help="refuse to search unless `spiced --version` has this")
    parser.add_argument("--train-only", action="store_true", help="train the plan's models; run no searches")
    parser.add_argument("--ready-wait", type=int, default=7200)
    parser.add_argument("--timeout", type=int, default=4 * 3600)
    parser.add_argument("--per-query-flag", default="--per-query-all-k")
    parser.add_argument("--http-port-flag", default="--spiced-http-port")
    parser.add_argument("--flight-port-flag", default="--spiced-flight-port")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    sys.exit(1 if asyncio.run(run_plan(load_plan((args := parse_args()).plan), args)) else 0)
