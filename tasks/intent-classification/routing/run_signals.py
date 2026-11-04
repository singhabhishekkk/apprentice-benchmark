"""Compare confidence signals with one shared explore / calibrate / evaluate split.

Every signal is oriented "higher = more trustworthy". The event is correct AND format_valid
(a malformed answer is wrong). Explore AUROC is for picking a signal; SGR runs on the
calibrate third; the frozen threshold is applied to the untouched evaluate third.

Usage: uv run python run_signals.py --scores test_scores.jsonl [--probe probe_conf.jsonl]
           [--target-risk 0.05 --delta 0.05 --seed 42]
       uv run python run_signals.py --self-check
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from probe import auroc
from sgr import evaluate, sgr

FRONTIER_USD_PER_1K = 2.42  # Opus 5 on Banking77, OpenRouter Banking77 post
RAW_SIGNALS = ["seq_logprob", "mean_logprob", "min_token_logprob", "agreement", "modal_share"]


def _column(rows: list[dict], key: str) -> np.ndarray:
    return np.array([-math.inf if r[key] is None else float(r[key]) for r in rows])


def _ecdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Mid-rank percentile of each value within the reference (explore) rows."""
    ref = np.sort(reference)
    return (np.searchsorted(ref, values, "left") + np.searchsorted(ref, values, "right")) / (2 * len(ref))


def _fmt(x: float | None, spec: str = ".3f") -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def load_signals(
    scores: Path, probe: Path | None, probe_layers: Path | None = None
) -> tuple[list[dict], np.ndarray, dict[str, np.ndarray]]:
    """Rows, the binary event (correct AND format_valid), and every split-independent signal."""
    with open(scores) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    event = np.array([int(r["correct"]) and int(r["format_valid"]) for r in rows])
    signals = {key: _column(rows, key) for key in RAW_SIGNALS}
    if probe is not None and probe.exists():
        with open(probe) as f:
            probe_rows = [json.loads(line) for line in f if line.strip()]
        assert len(probe_rows) == len(rows), f"{probe} has {len(probe_rows)} rows, {scores} has {len(rows)}"
        signals |= {key: _column(probe_rows, key) for key in ("probe_pre", "probe_post")}
    if probe_layers is not None:
        with open(probe_layers) as f:
            layer_rows = [json.loads(line) for line in f if line.strip()]
        assert len(layer_rows) == len(rows), f"{probe_layers} has {len(layer_rows)} rows, {scores} has {len(rows)}"
        signals |= {key: _column(layer_rows, key) for key in layer_rows[0]}
    return rows, event, signals


def report(
    scores: Path, probe: Path | None, target_risk: float, delta: float, seed: int, probe_layers: Path | None = None
) -> tuple[str, dict]:
    rows, event, signals = load_signals(scores, probe, probe_layers)
    explore, calibrate, test = np.array_split(np.random.default_rng(seed).permutation(len(rows)), 3)
    # Rank-average uses percentiles from the explore third only, so the score is frozen before calibration.
    signals["min_token_logprob_x_agreement"] = (
        _ecdf(signals["min_token_logprob"][explore], signals["min_token_logprob"])
        + _ecdf(signals["agreement"][explore], signals["agreement"])
    ) / 2

    lines = [
        (
            f"Rows {len(rows)}: explore {len(explore)}, calibrate {len(calibrate)}, evaluate {len(test)}. "
            f"Target risk {target_risk}, delta {delta}, seed {seed}."
        ),
        (
            f"Evaluate third: accuracy {event[test].mean():.3f}, "
            f"format_valid rate {np.mean([rows[i]['format_valid'] for i in test]):.3f}."
        ),
        "",
        (
            "| signal | explore AUROC | calibrate accepted/rows | calibrate bound | evaluate acceptance (Wilson 95%) "
            "| evaluate error among kept (count, rate, CP95 upper) | error among deferred "
            "| saved $/1k vs always-frontier |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    results = {}
    for name, score in signals.items():
        cal = sgr(score[calibrate], event[calibrate], target_risk, delta)
        ev = evaluate(score[test], event[test], cal["theta"])
        results[name] = {"calibrate": cal, "evaluate": ev}
        lo, hi = ev["acceptance_wilson95"]
        lines.append(
            f"| {name} | {_fmt(auroc(score[explore], event[explore]))} | {cal['accepted']}/{cal['calibrate_rows']} "
            f"| {_fmt(cal['risk_bound'])} | {_fmt(ev['acceptance_rate'])} ({lo:.3f} to {hi:.3f}) "
            f"| {ev['errors_accepted']}, {_fmt(ev['error_rate_accepted'])}, {_fmt(ev['error_rate_accepted_cp95_upper'])} "
            f"| {_fmt(ev['error_rate_deferred'])} | {ev['acceptance_rate'] * FRONTIER_USD_PER_1K:.2f} |"
        )
    lines += [
        "",
        (
            f"Saved $/1k = evaluate acceptance rate x ${FRONTIER_USD_PER_1K} per 1k requests (Opus 5, OpenRouter "
            "Banking77 post). Small model cost assumed zero on a Mac, electricity not counted."
        ),
    ]
    return "\n".join(lines), results


def _self_check() -> None:
    smoke = Path(__file__).parent / "runs/smoke/probe_scores.jsonl"
    text, results = report(smoke, None, 0.05, 0.05, 42)
    print(text)
    assert "| signal |" in text and "probe_pre" not in results
    assert len(results) == len(RAW_SIGNALS) + 1
    assert all(r["calibrate"]["accepted"] == 0 and r["evaluate"]["accepted"] == 0 for r in results.values())
    print("self-check passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", type=Path)
    ap.add_argument("--probe", type=Path)
    ap.add_argument("--probe-layers", type=Path, help="probe_layers.py output; every key becomes a signal")
    ap.add_argument("--target-risk", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        return _self_check()
    if not args.scores:
        ap.error("--scores is required")
    print(report(args.scores, args.probe, args.target_risk, args.delta, args.seed, args.probe_layers)[0])


if __name__ == "__main__":
    main()
