"""Gates G1 and G2 at the level of single text vectors, through spiced's /v1/embeddings.

- G2: an n-gram model with random (non-zero) n-gram rows. spiced's vectors must equal NgramEncoder's
  (the Python reference of the pooling in ngrams.py): cosine >= 0.99999 for every text. The same
  vectors without n-gram rows are reported too, to show that the rows change the vectors.
- G1: a model without n-grams. The new spiced build must give the same vectors as the old build.

The texts are FiQA corpus texts, plus texts that test the edges: longer than 512 tokens, with
unknown tokens, and with repeated n-grams. Writes results/ngrams/spiced_check.json.
"""

import json
import logging
import subprocess
import time
import urllib.request
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from pathlib import Path

import ngrams
import numpy as np
from common import EXPERIMENT_DIR, MODELS_DIR
from model2vec import StaticModel
from plans import zero_name
from sweep_finetune import SWEEP_MODELS_DIR, load

logger = logging.getLogger(__name__)

CHECK_DIR = MODELS_DIR / "ngram-check"
OUT_PATH = EXPERIMENT_DIR / "results" / "ngrams" / "spiced_check.json"
EDGE_TEXTS = [
    "hello",
    "emoji 🤖🤖 and ☃ snowman with ☃☃ unknown tokens",
    "the interest rate the interest rate the interest rate",
    " ".join(["market"] * 2000),
    "Stock options " * 400,
]


@dataclass
class Comparison:
    name: str
    spiced: str
    reference: str
    texts: int
    min_cosine: float
    median_cosine: float
    max_abs_diff: float


def random_model(seed: int) -> Path:
    """FiQA's final model (seed 42) with its zero-row n-grams replaced by random rows of a similar size."""
    out = CHECK_DIR / f"random-rows-s{seed}"
    if out.exists():
        return out
    zero = SWEEP_MODELS_DIR / zero_name("fiqa")
    encoder = ngrams.NgramEncoder(zero)
    scale = float(np.median(np.linalg.norm(encoder.embeddings, axis=1))) / np.sqrt(encoder.embeddings.shape[1])
    rows = np.random.default_rng(seed).normal(0, scale, encoder.ngram_embeddings.shape).astype(np.float32)
    ngrams.with_ngram_rows(zero, out, encoder.table, rows)
    return out


def spiced_vectors(spiced: Path, model: Path, texts: list[str], port: int) -> np.ndarray:
    """Start spiced with one model2vec embedding model, embed the texts, stop spiced."""
    work = CHECK_DIR / f"spicepod-{port}"
    work.mkdir(parents=True, exist_ok=True)
    (work / "spicepod.yaml").write_text(
        "version: v1\nkind: Spicepod\nname: ngram-check\nembeddings:\n"
        f"  - from: model2vec:{model}\n    name: m\n    params:\n      embed_max_token_length: 512\n"
    )
    with (work / "spiced.log").open("w") as log:
        process = subprocess.Popen(
            [str(spiced), "spicepod.yaml", "--http", f"127.0.0.1:{port}", "--flight", f"127.0.0.1:{port + 50}"],
            cwd=work,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    try:
        for _ in range(300):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/ready") as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            time.sleep(1)
        else:
            raise SystemExit(f"spiced not ready; see {work / 'spiced.log'}")
        out = []
        for start in range(0, len(texts), 64):
            body = json.dumps({"model": "m", "input": texts[start : start + 64]}).encode()
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/embeddings", body, {"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(request) as response:
                out += [row["embedding"] for row in json.load(response)["data"]]
        return np.asarray(out, dtype=np.float64)
    finally:
        process.terminate()
        process.wait(timeout=60)


def compare(name: str, spiced: str, reference: str, a: np.ndarray, b: np.ndarray) -> Comparison:
    """Cosines and the largest coordinate difference of two sets of unit vectors."""
    cos = (a * b).sum(axis=1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
    result = Comparison(
        name=name,
        spiced=spiced,
        reference=reference,
        texts=len(a),
        min_cosine=float(cos.min()),
        median_cosine=float(np.median(cos)),
        max_abs_diff=float(np.abs(a - b).max()),
    )
    logger.info(json.dumps(asdict(result)))
    return result


def version(spiced: Path) -> str:
    """`spiced --version`."""
    return subprocess.run([str(spiced), "--version"], capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    """Run both checks and write the results."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--new", type=Path, default=Path.home() / ".spice/bin/spiced")
    parser.add_argument("--old", type=Path, default=Path.home() / ".spice/bin/spiced_60eaf209cd")
    parser.add_argument("--port", type=int, default=19600)
    args = parser.parse_args()

    corpus, _ = load("fiqa", "corpus", "mean")
    texts = corpus[:400] + EDGE_TEXTS
    noisy = random_model(seed=0)
    new_noisy = spiced_vectors(args.new, noisy, texts, args.port)
    plain = SWEEP_MODELS_DIR / zero_name("fiqa").removesuffix("-ngzero")
    results = [
        compare("G2", version(args.new), "NgramEncoder", new_noisy, ngrams.NgramEncoder(noisy).encode(texts)),
        compare(
            "G2 rows matter",
            version(args.new),
            "model2vec without n-grams",
            new_noisy,
            StaticModel.from_pretrained(str(plain)).encode(texts).astype(np.float64),
        ),
        compare(
            "G1",
            version(args.new),
            version(args.old),
            spiced_vectors(args.new, plain, texts, args.port),
            spiced_vectors(args.old, plain, texts, args.port),
        ),
    ]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps([asdict(r) for r in results], indent=2) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
