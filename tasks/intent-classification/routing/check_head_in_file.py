"""Does the head baked into the GGUF answer the same as the head in Python?

Sends the first N test rows to a llama.cpp server whose model file carries cls.output
(truncate_gguf.py --head) and compares the returned logits with the Python head applied to
the states captured earlier from the plain file (capture_llama.py): same label, softmax
confidence within a tolerance, and the wall time per request.

    uv run python check_head_in_file.py --head runs/latency/head_llama32.npz --states runs/latency/llama32_test --n 50
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from latency_bench import TEMPLATE, post

ROOT = Path(__file__).parent


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--head", required=True)
    ap.add_argument("--states", required=True)
    ap.add_argument("--rows", default="data/banking77/test.jsonl")
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()
    h = np.load(ROOT / args.head)
    labels = json.loads(str(h["labels"]))
    z = np.load(str(ROOT / args.states) + ".layers.npz")
    x = z[next(k for k in z.files if k.endswith("_prompt_last"))][: args.n].astype(
        np.float64
    )
    local = ((x - h["scaler_mean"]) / h["scaler_scale"]) @ h["coef"].T.astype(
        np.float64
    ) + h["intercept"]
    rows = [
        json.loads(line)
        for line in (ROOT / args.rows).read_text().splitlines()
        if line.strip()
    ][: args.n]
    gold = [json.loads(r["output"])["intent"] for r in rows]
    served, times = [], []
    for r in rows:
        t = time.perf_counter()
        body = post(
            f"{args.url}/embeddings",
            {"content": TEMPLATE.format(text=r["input"]), "embd_normalize": -1},
        )
        times.append((time.perf_counter() - t) * 1000)
        served.append(body[0]["embedding"][0])
    served = np.asarray(served, dtype=np.float64)
    assert served.shape == local.shape, (served.shape, local.shape)
    p_local, p_served = softmax(local), softmax(served)
    same_label = int((p_local.argmax(1) == p_served.argmax(1)).sum())
    conf_diff = np.abs(p_local.max(1) - p_served.max(1))
    acc_served = float(
        np.mean([labels[i] == g for i, g in zip(p_served.argmax(1), gold)])
    )
    acc_local = float(
        np.mean([labels[i] == g for i, g in zip(p_local.argmax(1), gold)])
    )
    out = {
        "rows": args.n,
        "logit_abs_diff_max": float(np.abs(served - local).max()),
        "logit_rel_l2_max": float(
            (
                np.linalg.norm(served - local, axis=1) / np.linalg.norm(local, axis=1)
            ).max()
        ),
        "same_label": same_label,
        "confidence_abs_diff_max": float(conf_diff.max()),
        "accuracy_served": acc_served,
        "accuracy_python": acc_local,
        "client_ms_p50": float(np.median(times)),
        "client_ms_p95": float(np.percentile(times, 95)),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
