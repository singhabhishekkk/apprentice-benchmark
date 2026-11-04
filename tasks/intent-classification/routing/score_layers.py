"""Hidden states at several depths and positions, for the probe-layer experiment.

The literature (Azaria and Mitchell 2023; Kossen et al. 2024; Orgad et al. 2024; CH-Wang et
al. 2023) finds the "will this be correct" signal at half to three-quarters depth and on the
answer tokens, not at the final norm where score_rows.py reads it. This pass saves, per row:
  layers 8, 16, 24, 32 of 32 (outputs of those decoder layers, residual stream, pre-norm)
  x positions: last prompt token, last answer token, mean over answer tokens
Greedy answers are regenerated here (cheap, batched); samples are not needed.

Writes <out>.layers.npz (float16 arrays [rows, 2560] keyed L{layer}_{position}) and
<out>.layers.jsonl (one object per row with correct, format_valid, answer).

Usage: uv run python score_layers.py --adapter runs/banking77/adapter --rows data/banking77/test.jsonl --out runs/banking77/test_layers
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.generate import batch_generate
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.sample_utils import make_sampler

from score_rows import MAX_TOKENS, MODEL_ID, parse_intent

LAYERS = (8, 16, 24, 32)  # of 32; "L32" is the last layer's output before the final norm
BATCH = 32


def hidden_at(model: Any, tokens: list[int]) -> dict[int, mx.array]:
    # Mirrors Qwen3_5TextModel.__call__ (mlx_lm/models/qwen3_5.py) with no cache, keeping the
    # residual stream after the chosen layers. Masks are built the same way the model does.
    inner = model.language_model.model if hasattr(model, "language_model") else model.model
    h = inner.embed_tokens(mx.array(tokens)[None])
    fa_mask = create_attention_mask(h, None) if inner.fa_idx is not None else None
    ssm_mask = create_ssm_mask(h, None) if inner.ssm_idx is not None else None
    out: dict[int, mx.array] = {}
    for i, layer in enumerate(inner.layers, start=1):
        h = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=None)
        if i in LAYERS:
            out[i] = h[0]
    mx.eval(*out.values())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="path stem; .layers.npz and .layers.jsonl are added")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.rows.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    model, tokenizer = load(MODEL_ID, adapter_path=str(args.adapter) if args.adapter else None)
    tok = cast(Any, tokenizer)

    def prompt_tokens(text: str) -> list[int]:
        s = tok.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        return tok.encode(s, add_special_tokens=False)

    prompts = [prompt_tokens(r["input"]) for r in rows]
    greedy = make_sampler(temp=0.0)
    feats: dict[str, list[np.ndarray]] = {f"L{L}_{p}": [] for L in LAYERS for p in ("prompt_last", "answer_last", "answer_mean")}
    labels: list[dict[str, Any]] = []
    t0 = time.time()
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        chunk_prompts = prompts[start:start + BATCH]
        g = batch_generate(model, tok, chunk_prompts, max_tokens=MAX_TOKENS, return_token_ids=True, sampler=greedy)
        for i, row in enumerate(chunk):
            ids = list((g.token_ids or [[]])[i])
            n_p = len(chunk_prompts[i])
            hs = hidden_at(model, chunk_prompts[i] + ids)
            for L, h in hs.items():
                h32 = h.astype(mx.float32)
                ans = h32[n_p:] if ids else h32[n_p - 1:]
                feats[f"L{L}_prompt_last"].append(np.asarray(h32[n_p - 1]))
                feats[f"L{L}_answer_last"].append(np.asarray(ans[-1]))
                feats[f"L{L}_answer_mean"].append(np.asarray(mx.mean(ans, axis=0)))
            answer = parse_intent(g.texts[i])
            gold = parse_intent(row["output"])
            labels.append({"answer": answer, "correct": int(answer is not None and answer == gold),
                           "format_valid": int(answer is not None)})
        print(f"{start + len(chunk)}/{len(rows)}  {time.time() - t0:.0f}s  acc so far "
              f"{sum(r['correct'] for r in labels) / len(labels):.3f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(args.out) + ".layers.npz", **{k: np.stack(v).astype(np.float16) for k, v in feats.items()})
    with open(str(args.out) + ".layers.jsonl", "w", encoding="utf-8") as f:
        for r in labels:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"rows": len(labels), "accuracy": sum(r["correct"] for r in labels) / len(labels),
                      "keys": sorted(feats), "seconds": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
