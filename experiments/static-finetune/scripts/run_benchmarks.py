"""Run `testoperator run search` for every spicepod in the manifest, many at a time.

Each concurrent run gets its own spiced HTTP and Flight ports. A run starts once its local model
directory exists, so this can run while models are still training. A run whose `run.json` records
exit code 0 is skipped, so the script can be restarted.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = EXPERIMENT_DIR / "manifest.json"
RUNS_DIR = EXPERIMENT_DIR / "results" / "runs"

OTLP_ENDPOINT = "http://127.0.0.1:4317"
POLL_SECONDS = 15


@dataclass(frozen=True)
class Job:
    name: str
    eval_dataset: str
    spicepod_path: str
    family: str
    local_path: str | None
    hf_model: str | None

    @property
    def out_dir(self) -> Path:
        return RUNS_DIR / self.eval_dataset / self.name

    def model_ready(self) -> bool:
        # Models are saved to `<name>.tmp` and renamed, so the directory exists only when complete.
        return self.local_path is None or Path(self.local_path).is_dir()

    def succeeded(self) -> bool:
        path = self.out_dir / "run.json"
        return path.exists() and json.loads(path.read_text()).get("exit_code") == 0


@dataclass(frozen=True)
class Ports:
    http: int
    flight: int


@dataclass
class RunRecord:
    name: str
    eval_dataset: str
    family: str
    command: list[str]
    ports: Ports
    testoperator_sha256: str
    spiced_version: str
    started_at: float
    finished_at: float | None = None
    exit_code: int | None = None


@dataclass
class Scheduler:
    pending: list[Job]
    free_slots: list[int]
    # Hugging Face models no run has loaded yet: one cold download at a time per model.
    cold_hf_models_running: set[str] = field(default_factory=set)
    warm_hf_models: set[str] = field(default_factory=set)

    def next_job(self) -> Job | None:
        for job in self.pending:
            if not job.model_ready():
                continue
            if job.hf_model and job.hf_model not in self.warm_hf_models:
                if job.hf_model in self.cold_hf_models_running:
                    continue
                self.cold_hf_models_running.add(job.hf_model)
            self.pending.remove(job)
            return job
        return None

    def finish(self, job: Job, ok: bool) -> None:
        if job.hf_model:
            self.cold_hf_models_running.discard(job.hf_model)
            if ok:
                self.warm_hf_models.add(job.hf_model)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jobs(eval_datasets: set[str] | None, families: set[str] | None, names: set[str] | None) -> list[Job]:
    jobs = []
    for run in json.loads(MANIFEST_PATH.read_text()):
        model = run["model"]
        job = Job(
            name=run["name"],
            eval_dataset=run["eval_dataset"],
            spicepod_path=run["spicepod_path"],
            family=model["family"],
            local_path=model["local_path"],
            hf_model=model["hf_model"],
        )
        if eval_datasets and job.eval_dataset not in eval_datasets:
            continue
        if families and job.family not in families:
            continue
        if names and job.name not in names:
            continue
        jobs.append(job)
    # bge runs are the slowest (transformer on every corpus row), so start them first.
    return sorted(jobs, key=lambda job: (job.family != "1", job.eval_dataset != "scifact"))


def ports_for(slot: int, base: int) -> Ports:
    return Ports(http=base + slot, flight=base + 100 + slot)


async def run_job(job: Job, slot: int, args: argparse.Namespace, testoperator_sha: str, spiced_version: str) -> bool:
    ports = ports_for(slot, args.port_base)
    command = [
        str(args.testoperator),
        "run",
        "search",
        "--spicepod-path",
        job.spicepod_path,
        "--spiced-path",
        str(args.spiced),
        "--ready-wait",
        str(args.ready_wait),
        "--disable-progress-bars",
        "--otlp-endpoint",
        OTLP_ENDPOINT,
        args.per_query_flag,
        args.http_port_flag,
        str(ports.http),
        args.flight_port_flag,
        str(ports.flight),
    ]
    job.out_dir.mkdir(parents=True, exist_ok=True)
    record = RunRecord(
        name=job.name,
        eval_dataset=job.eval_dataset,
        family=job.family,
        command=command,
        ports=ports,
        testoperator_sha256=testoperator_sha,
        spiced_version=spiced_version,
        started_at=time.time(),
    )
    env = {key: value for key, value in os.environ.items() if key != "SPICEAI_BENCHMARK_METRICS_KEY"}
    print(f"[start] slot={slot} {job.eval_dataset}/{job.name}", flush=True)
    with (job.out_dir / "stdout.log").open("w") as log:
        process = await asyncio.create_subprocess_exec(
            *command, stdout=log, stderr=asyncio.subprocess.STDOUT, cwd=job.out_dir, env=env
        )
        try:
            record.exit_code = await asyncio.wait_for(process.wait(), timeout=args.timeout)
        except TimeoutError:
            process.kill()
            await process.wait()
            record.exit_code = -9
            log.write(f"\n[run_benchmarks] killed after {args.timeout}s timeout\n")
    record.finished_at = time.time()
    (job.out_dir / "run.json").write_text(json.dumps(asdict(record), indent=2) + "\n")
    elapsed = record.finished_at - record.started_at
    print(f"[done ] slot={slot} exit={record.exit_code} {elapsed:.0f}s {job.eval_dataset}/{job.name}", flush=True)
    return record.exit_code == 0


async def main(args: argparse.Namespace) -> int:
    eval_datasets = set(args.eval_dataset) if args.eval_dataset else None
    families = set(args.family) if args.family else None
    names = set(args.name) if args.name else None
    jobs = [job for job in load_jobs(eval_datasets, families, names) if args.force or not job.succeeded()]
    print(f"{len(jobs)} runs to do", flush=True)

    testoperator_sha = sha256(args.testoperator)
    spiced_version = subprocess.run(
        [str(args.spiced), "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()

    scheduler = Scheduler(pending=jobs, free_slots=list(range(args.workers)))
    running: dict[asyncio.Task[bool], tuple[Job, int]] = {}
    failures: list[str] = []
    while scheduler.pending or running:
        while scheduler.free_slots and (job := scheduler.next_job()) is not None:
            slot = scheduler.free_slots.pop(0)
            task = asyncio.create_task(run_job(job, slot, args, testoperator_sha, spiced_version))
            running[task] = (job, slot)
        if not running:
            # Nothing runnable yet: wait for models to appear.
            await asyncio.sleep(POLL_SECONDS)
            continue
        done, _ = await asyncio.wait(running, timeout=POLL_SECONDS, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            job, slot = running.pop(task)
            ok = task.result()
            scheduler.finish(job, ok)
            scheduler.free_slots.append(slot)
            if not ok:
                failures.append(f"{job.eval_dataset}/{job.name}")
    print(f"Finished. {len(failures)} failed: {failures}", flush=True)
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testoperator", type=Path, default=Path.home() / ".spice/bin/testoperator-search-ports")
    parser.add_argument("--spiced", type=Path, default=Path.home() / ".spice/bin/spiced")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--port-base", type=int, default=18100)
    parser.add_argument("--ready-wait", type=int, default=7200)
    parser.add_argument("--timeout", type=int, default=4 * 3600, help="Wall-clock limit per run, in seconds.")
    parser.add_argument("--eval-dataset", action="append", help="Only these eval datasets (repeatable).")
    parser.add_argument("--family", action="append", help="Only these model families: 1, 2, 3, 4, 4b (repeatable).")
    parser.add_argument("--name", action="append", help="Only these spicepod names (repeatable).")
    parser.add_argument("--force", action="store_true", help="Rerun runs that already succeeded.")
    # Flag names of the testoperator build; kept configurable until that PR settles.
    parser.add_argument("--per-query-flag", default="--per-query-all-k")
    parser.add_argument("--http-port-flag", default="--spiced-http-port")
    parser.add_argument("--flight-port-flag", default="--spiced-flight-port")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args())))
