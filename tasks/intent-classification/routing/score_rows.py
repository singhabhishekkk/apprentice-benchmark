"""Score every row with the fine-tuned model and record each candidate routing signal.

Per row, with one model load:
  greedy answer with per-token logprobs   -> seq_logprob, mean_logprob, min_token_logprob
  N samples at temperature T              -> agreement with the greedy answer, modal answer
  hidden state at the last PROMPT token   -> h_pre  (prefill probe: decides before decoding)
  hidden state at the last ANSWER token   -> h_post (post-answer probe)
  correctness against gold                -> correct (closed set: the intent string matches)

Hidden states go to <out>.h_pre.npy / <out>.h_post.npy (float32, row order = the JSONL).
Everything else is one JSON object per line in <out>.

Usage: uv run python score_rows.py --adapter runs/banking77/adapter --rows data/banking77/test.jsonl --out runs/banking77/test_scores.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.generate import batch_generate
from mlx_lm.sample_utils import make_sampler

MODEL_ID = "mlx-community/Qwen3.5-4B-OptiQ-4bit"  # same base the product trains (local.py LOCAL_MODEL_ID)
MAX_TOKENS = 48  # {"intent": "..."} is under 20 tokens; this only caps a model that never stops
BATCH = 32


def parse_intent(text: str) -> str | None:
    try:
        obj = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and isinstance(obj.get("intent"), str):
        return obj["intent"]
    return None


def parse_fields(text: str) -> dict[str, str] | None:
    """Extraction tasks: a flat JSON object of string fields, or None when malformed."""
    try:
        obj = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and all(isinstance(v, (str, int, float)) or v is None for v in obj.values()):
        return {k: ("" if v is None else str(v)) for k, v in obj.items()}
    return None


def _norm(value: str) -> str:
    return " ".join(value.strip().lower().split())


def fields_correct(answer: dict[str, str] | None, gold: dict[str, str] | None) -> tuple[int, dict[str, int]]:
    """"Wrong" for extraction, frozen in the plan: schema invalid, a required (gold) field
    missing, or any field's value not matching gold after whitespace and case normalisation.
    Returns (all_correct, per-field correctness)."""
    if answer is None or gold is None:
        return 0, {}
    per = {k: int(k in answer and _norm(answer[k]) == _norm(v)) for k, v in gold.items()}
    return int(all(per.values())), per


def field_min_logprobs(tok: Any, ids: list[int], lps: list[float], answer: dict[str, str] | None) -> dict[str, float]:
    """Per-field minimum token logprob: map each generated token to its character span in the
    decoded text, then take the minimum over the tokens that overlap each field's value. An
    approximation (byte-level tokenizers split inside values), stated as such in RESULTS."""
    if answer is None or not ids:
        return {}
    spans: list[tuple[int, int, float]] = []
    text = ""
    for tid, lp in zip(ids, lps):
        piece = tok.decode([tid])
        spans.append((len(text), len(text) + len(piece), lp))
        text += piece
    out: dict[str, float] = {}
    for key, value in answer.items():
        needle = json.dumps(value, ensure_ascii=False)[1:-1]  # as it appears inside the JSON string
        start = text.find(needle) if needle else -1
        if start < 0:
            continue
        end = start + len(needle)
        overlapping = [lp for s, e, lp in spans if s < end and e > start]
        if overlapping:
            out[key] = float(min(overlapping))
    return out


def last_hidden(model: Any, tokens: list[int]) -> np.ndarray:
    # qwen3_5.Model -> language_model -> .model returns the final-norm hidden states
    # (mlx_lm/models/qwen3_5.py: `return self.norm(hidden_states)`), lm_head not applied.
    inner = model.language_model.model if hasattr(model, "language_model") else model.model
    h = inner(mx.array(tokens)[None])
    mx.eval(h)
    return np.asarray(h[0, -1].astype(mx.float32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, default=None, help="omit to score the base model")
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--kind", choices=("intent", "fields"), default="intent",
                    help="intent: closed-set label under 'intent'; fields: flat JSON of string fields")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                    help="generation cap; 48 fits an intent label, a four-field receipt needs about 256 "
                         "(the first SROIE pass at 48 cut every answer mid-address)")
    args = ap.parse_args()
    max_tokens = args.max_tokens

    rows = [json.loads(l) for l in args.rows.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    model, tokenizer = load(MODEL_ID, adapter_path=str(args.adapter) if args.adapter else None)
    tok = cast(Any, tokenizer)

    def prompt_tokens(text: str) -> list[int]:
        # Same chat shape the product's held-out scorer uses (_mlx_eval.py): one user turn,
        # thinking off so the answer is the JSON and nothing else.
        s = tok.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        return tok.encode(s, add_special_tokens=False)

    prompts = [prompt_tokens(r["input"]) for r in rows]
    greedy = make_sampler(temp=0.0)
    sampled = make_sampler(temp=args.temperature)
    t0 = time.time()

    out_rows: list[dict[str, Any]] = []
    h_pre: list[np.ndarray] = []
    h_post: list[np.ndarray] = []
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        chunk_prompts = prompts[start:start + BATCH]
        g = batch_generate(model, tok, chunk_prompts, max_tokens=max_tokens,
                           return_token_ids=True, return_logprobs=True, sampler=greedy)
        rep_prompts = [p for p in chunk_prompts for _ in range(args.samples)]
        s = batch_generate(model, tok, rep_prompts, max_tokens=max_tokens, sampler=sampled)
        for i, row in enumerate(chunk):
            text = g.texts[i]
            lps = (g.logprobs or [[]])[i]
            ids = (g.token_ids or [[]])[i]
            if args.kind == "intent":
                answer: Any = parse_intent(text)
                gold: Any = parse_intent(row["output"])
                correct = int(answer is not None and answer == gold)
                extra: dict[str, Any] = {}
                sample_answers: list[Any] = [parse_intent(t) for t in s.texts[i * args.samples:(i + 1) * args.samples]]
            else:
                answer = parse_fields(text)
                gold = parse_fields(row["output"])
                correct, per_field = fields_correct(answer, gold)
                fmin = field_min_logprobs(tok, list(ids), list(lps), answer)
                extra = {"per_field": per_field, "field_min_logprob": fmin,
                         "field_min_logprob_min": min(fmin.values()) if fmin else -math.inf,
                         "n_fields_gold": len(gold) if gold else 0}
                sample_answers = [parse_fields(t) for t in s.texts[i * args.samples:(i + 1) * args.samples]]
            agree = sum(1 for a in sample_answers if a is not None and a == answer) / max(1, args.samples)
            hashable = [json.dumps(a, sort_keys=True) if isinstance(a, dict) else a for a in sample_answers]
            modal = max({a for a in hashable if a is not None} or {None}, key=lambda a: hashable.count(a))
            modal_share = hashable.count(modal) / max(1, args.samples) if modal is not None else 0.0
            h_pre.append(last_hidden(model, chunk_prompts[i]))
            h_post.append(last_hidden(model, chunk_prompts[i] + list(ids)))
            out_rows.append({
                "input": row["input"], "gold": gold, "answer": answer, "raw": text,
                "correct": correct,
                "format_valid": int(answer is not None),
                **extra,
                "n_tokens": len(lps),
                "seq_logprob": float(sum(lps)) if lps else -math.inf,
                "mean_logprob": float(sum(lps) / len(lps)) if lps else -math.inf,
                "min_token_logprob": float(min(lps)) if lps else -math.inf,
                "agreement": agree, "modal": modal, "modal_share": modal_share,
                "samples": sample_answers,
            })
        done = start + len(chunk)
        print(f"{done}/{len(rows)}  {time.time() - t0:.0f}s  acc so far "
              f"{sum(r['correct'] for r in out_rows) / len(out_rows):.3f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    np.save(args.out.with_suffix(".h_pre.npy"), np.stack(h_pre))
    np.save(args.out.with_suffix(".h_post.npy"), np.stack(h_post))
    print(json.dumps({"rows": len(out_rows), "accuracy": sum(r["correct"] for r in out_rows) / len(out_rows),
                      "format_valid": sum(r["format_valid"] for r in out_rows) / len(out_rows),
                      "seconds": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
