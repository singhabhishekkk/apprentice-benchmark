"""Shared output parsing and scoring. Every runner and notebook imports this file.

Two answer kinds:
  json   - one JSON object, scored by field-level F1 (exact value match per key)
  label  - one short string, scored by case-insensitive exact match

Pure standard library, so notebooks and verify.py need no extra install.
"""

from __future__ import annotations

import json
import re

TASK_KIND = {
    "json-extraction": "json",
    "receipt-extraction": "json",
    "contract-clause-extraction": "json",
    "document-type-classification": "label",
}


def strip_reasoning(raw: str) -> str:
    """Drop <think> blocks (including an unclosed one) and markdown code fences."""
    raw = re.sub(r"<think>.*?(</think>|$)", "", raw or "", flags=re.DOTALL)
    return re.sub(r"```(?:json)?", "", raw)


def parse_output(raw: str, kind: str):
    """Return the parsed answer: a dict (json kind), a string (label kind), or None."""
    text = strip_reasoning(raw).strip()
    if kind == "label":
        return text.splitlines()[0].strip().strip("\"'").strip() if text else ""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        return json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        return None


def _expected_obj(expected: str, kind: str):
    if kind == "label":
        return expected
    try:
        return json.loads(expected)
    except json.JSONDecodeError:
        return None


def score_parsed(expected: str, parsed, kind: str) -> dict:
    """Score a parsed answer. Returns {"score": 0..1, "fields": {key: 0|1}, "feedback": str}."""
    exp = _expected_obj(expected, kind)
    if kind == "label":
        ok = isinstance(parsed, str) and parsed.strip().casefold() == str(exp).strip().casefold()
        return {
            "score": float(ok),
            "fields": {},
            "feedback": "exact match" if ok else f"expected exactly {expected!r}; got {parsed!r}",
        }
    if not isinstance(exp, dict):
        return {"score": 0.0, "fields": {}, "feedback": "gold output is not a JSON object; row should be cleaned"}
    if parsed is None:
        return {"score": 0.0, "fields": {k: 0 for k in exp}, "feedback": "output is not valid JSON; emit a single JSON object only"}
    fields = {k: int(k in parsed and parsed[k] == v) for k, v in exp.items()}
    tp = sum(fields.values())
    precision = tp / len(parsed) if parsed else 0.0
    recall = tp / len(exp) if exp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    missing = [k for k in exp if k not in parsed]
    wrong = [k for k in exp if k in parsed and parsed[k] != exp[k]]
    extra = [k for k in parsed if k not in exp]
    parts = [f"{label}: {names}" for label, names in
             (("missing fields", missing), ("wrong values for", wrong), ("unexpected extra fields", extra)) if names]
    return {"score": f1, "fields": fields, "feedback": "; ".join(parts) or "all fields correct"}


def score_output(expected: str, raw: str, kind: str) -> dict:
    """Parse a raw model completion and score it. The one entry point for runners and notebooks."""
    return score_parsed(expected, parse_output(raw, kind), kind)
