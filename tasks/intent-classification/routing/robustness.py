"""E6 robustness: split seeds, calibration size, and a simulated shift in the intent mix.

Signals and the binary event come from run_signals.load_signals; SGR and evaluation from sgr.py.

Usage: uv run python robustness.py --scores runs/banking77/test_scores.jsonl \
           --probe-layers runs/banking77/probe_layers_conf.jsonl \
           --signals min_token_logprob,agreement,L16_answer_last --target-risk 0.05 --delta 0.05
       uv run python robustness.py --self-check
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from run_signals import _fmt, load_signals
from sgr import evaluate, sgr

SEEDS = (42, 7, 2026)
CAL_SIZES = (400, 200, 100, 50)


def _cell(score: np.ndarray, event: np.ndarray, cal_idx: np.ndarray, test_idx: np.ndarray, r: float, d: float):
    cal = sgr(score[cal_idx], event[cal_idx], r, d)
    return cal, evaluate(score[test_idx], event[test_idx], cal["theta"])


def _acc(ev: dict) -> str:
    lo, hi = ev["acceptance_wilson95"]
    return f"{ev['acceptance_rate']:.3f} ({lo:.3f} to {hi:.3f})"


def _kept(ev: dict) -> str:
    return f"{ev['errors_accepted']}/{ev['accepted']}, {_fmt(ev['error_rate_accepted'])}, CP95 {_fmt(ev['error_rate_accepted_cp95_upper'])}"


def study_seeds(signals: dict, event: np.ndarray, r: float, d: float) -> str:
    out = [
        "### (a) Split seeds",
        "",
        "| signal | seed | theta | calibrate bound | evaluate acceptance (Wilson 95%) | evaluate error among kept (errors/kept, rate, CP95 upper) |",
        "|---|---|---|---|---|---|",
    ]
    spread = []
    for name, score in signals.items():
        rates = []
        for seed in SEEDS:
            _, cal_idx, test_idx = np.array_split(np.random.default_rng(seed).permutation(len(event)), 3)
            cal, ev = _cell(score, event, cal_idx, test_idx, r, d)
            rates.append(ev["acceptance_rate"])
            out.append(f"| {name} | {seed} | {_fmt(cal['theta'], '.4g')} | {_fmt(cal['risk_bound'])} | {_acc(ev)} | {_kept(ev)} |")
        spread.append(f"- {name}: evaluate acceptance across seeds, min {min(rates):.3f}, max {max(rates):.3f}")
    return "\n".join([*out, "", *spread])


def study_cal_size(signals: dict, event: np.ndarray, r: float, d: float, seed: int = 42) -> str:
    _, cal_idx, test_idx = np.array_split(np.random.default_rng(seed).permutation(len(event)), 3)
    sizes = [len(cal_idx)] + [n for n in CAL_SIZES if n < len(cal_idx)]
    out = [
        f"### (b) Calibration size (seed {seed}; evaluate third untouched, {len(test_idx)} rows)",
        "",
        "| signal | calibrate rows | theta | calibrate bound | evaluate acceptance (Wilson 95%) | evaluate error among kept (errors/kept, rate, CP95 upper) |",
        "|---|---|---|---|---|---|",
    ]
    for name, score in signals.items():
        for n in sizes:
            subset = np.random.default_rng(seed).choice(cal_idx, size=n, replace=False)
            cal, ev = _cell(score, event, subset, test_idx, r, d)
            out.append(f"| {name} | {n} | {_fmt(cal['theta'], '.4g')} | {_fmt(cal['risk_bound'])} | {_acc(ev)} | {_kept(ev)} |")
    return "\n".join(out)


def study_shift(signals: dict, event: np.ndarray, gold: list[str], r: float, d: float) -> str:
    intents = sorted(set(gold))
    half_a = set(intents[0::2])
    in_a = np.array([g in half_a for g in gold])
    halves = {"A": np.flatnonzero(in_a), "B": np.flatnonzero(~in_a)}
    out = [
        "### (c) Simulated distribution shift (not production drift)",
        "",
        (
            f"Intents sorted and split by alternating index: A = {len(half_a)} intents, {len(halves['A'])} rows; "
            f"B = {len(intents) - len(half_a)} intents, {len(halves['B'])} rows. "
            "Calibrate on every row of one half, evaluate on every row of the other."
        ),
        "",
        "| signal | calibrate on | evaluate on | calibrate bound | evaluate acceptance (Wilson 95%) | evaluate error among kept (errors/kept, rate, CP95 upper) | realised error within bound? |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, score in signals.items():
        for src, dst in (("A", "B"), ("B", "A")):
            cal, ev = _cell(score, event, halves[src], halves[dst], r, d)
            rate = ev["error_rate_accepted"]
            verdict = "no rows kept" if rate is None else ("yes" if rate <= cal["risk_bound"] else "NO")
            out.append(f"| {name} | {src} | {dst} | {_fmt(cal['risk_bound'])} | {_acc(ev)} | {_kept(ev)} | {verdict} |")
    return "\n".join(out)


def report(scores: Path, probe_layers: Path | None, names: list[str], r: float, d: float) -> str:
    rows, event, signals = load_signals(scores, None, probe_layers)
    missing = [n for n in names if n not in signals]
    assert not missing, f"unknown signals {missing}; available: {sorted(signals)}"
    chosen = {n: signals[n] for n in names}
    gold = [str(row["gold"]) for row in rows]
    header = f"Rows {len(rows)}, target risk {r}, delta {d}, event = correct AND format_valid."
    sections = [study_seeds(chosen, event, r, d), study_cal_size(chosen, event, r, d), study_shift(chosen, event, gold, r, d)]
    return "\n\n".join([header, *sections])


def _self_check() -> None:
    rng = np.random.default_rng(0)
    n = 3000
    intents = [f"intent_{i:02d}" for i in range(77)]
    conf = rng.uniform(0, 1, n)
    gold = rng.choice(intents, n)
    in_a = np.isin(gold, sorted(intents)[0::2])
    # Half A: a sharp planted signal (above 0.3 almost always right). Half B: confidence carries no
    # information and accuracy is 50%, so calibrating on B must keep zero rows, and a threshold
    # learned on A must break on B.
    p_right = np.where(in_a, np.where(conf > 0.3, 0.995, 0.3), 0.5)
    conf = np.where(in_a, conf, conf * 0.5)
    correct = (rng.uniform(0, 1, n) < p_right).astype(int)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp, "scores.jsonl")
        path.write_text(
            "".join(
                json.dumps({"gold": str(g), "correct": int(c), "format_valid": 1, "min_token_logprob": float(np.log(p + 1e-9)),
                            "seq_logprob": 0.0, "mean_logprob": 0.0, "agreement": float(p), "modal_share": float(p)}) + "\n"
                for g, c, p in zip(gold, correct, conf)
            )
        )
        text = report(path, None, ["min_token_logprob", "agreement"], 0.05, 0.05)
    print(text)
    for title in ("### (a)", "### (b)", "### (c)"):
        assert title in text
    assert "| agreement | B | A |" in text and "no rows kept" in text, "study (c) must report a half with zero kept rows"
    assert "| NO |" in text, "a threshold calibrated on A must break on B"
    assert "| agreement | 42 | inf |" not in text, "the planted signal must keep rows on the seed-42 split"
    print("robustness self-check passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", type=Path)
    ap.add_argument("--probe-layers", type=Path)
    ap.add_argument("--signals", default="min_token_logprob,agreement")
    ap.add_argument("--target-risk", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        return _self_check()
    if not args.scores:
        ap.error("--scores is required")
    print(report(args.scores, args.probe_layers, args.signals.split(","), args.target_risk, args.delta))


if __name__ == "__main__":
    main()
