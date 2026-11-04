"""Selection with Guaranteed Risk (SGR), Geifman and El-Yaniv 2017, arXiv 1705.08500.

Algorithm 1 with the Lemma 3.1 bound. Confidence convention: HIGHER confidence means MORE
trustworthy; a row is accepted (kept by the small model) when confidence >= theta.
`correct` is the frozen binary event (1 = right, 0 = wrong), plan/routing-decision-research.md s7.

Usage: uv run python sgr.py rows.jsonl [--target-risk 0.05 --delta 0.05 --seed 42]
       uv run python sgr.py --self-check
"""

import argparse
import json
import math

import numpy as np


def _log_binom_cdf(k: int, n: int, p: float) -> float:
    """log P[X <= k] for X ~ Bin(n, p), summed in log space (stable for n in the thousands)."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 0.0 if k >= n else -math.inf
    lp, lq, lgn = math.log(p), math.log1p(-p), math.lgamma(n + 1)
    terms = [lgn - math.lgamma(j + 1) - math.lgamma(n - j + 1) + j * lp + (n - j) * lq for j in range(k + 1)]
    top = max(terms)
    return top + math.log(sum(math.exp(t - top) for t in terms))


def binomial_upper_bound(errors: int, n: int, delta: float) -> float:
    """B*(r, delta, S) of Lemma 3.1: the b solving eq. (4), sum_{j<=errors} C(n,j) b^j (1-b)^(n-j) = delta.

    The CDF falls monotonically in b, so bisect. This is also the exact one-sided
    Clopper-Pearson upper limit at level 1 - delta.
    """
    if n == 0 or errors >= n:
        return 1.0
    lo, hi, log_delta = 0.0, 1.0, math.log(delta)
    for _ in range(100):
        mid = (lo + hi) / 2
        if _log_binom_cdf(errors, n, mid) > log_delta:
            lo = mid
        else:
            hi = mid
    return hi


# Ten cuts at 10% coverage steps: the same number of tests as the paper's binary search on a
# ~1,000-row sample (ceil(log2 1027) = 11), so the same delta / k per cut, without the
# path dependence. Forty cuts was tried first and its delta / 40 per cut made every seed
# keep nothing: this adapter sits at the 5% edge, where the union-bound cost decides.
GRID_CUTS = 10


def sgr(confidence: np.ndarray, correct: np.ndarray, target_risk: float, delta: float) -> dict:
    """Algorithm 1. Theorem 3.2: with prob >= 1 - delta, risk among accepted <= the returned bound.

    Rows are sorted by confidence descending, so a cut at z accepts the top z rows. The paper
    sorts so that a passing bound sets z_max = z; with a descending sort the same step is
    z_min = z (accept more). Each of the k = ceil(log2 m) steps uses delta / k (union bound).
    """
    order = np.argsort(-confidence, kind="stable")
    conf, wrong = confidence[order], 1 - correct[order].astype(int)
    m = len(conf)
    # Grid scan instead of the paper's binary search (changed 2026-11-04 after the Banking77
    # robustness run). The binary search assumes risk is monotone in coverage; on real data it
    # is not, and the search walked past the feasible region on one seed in three and at two
    # of four calibration sizes while a neighbouring size kept 58%. A fixed grid of k cuts,
    # each tested at delta / k, keeps Theorem 3.2's union-bound argument exactly (k tests,
    # each valid with probability 1 - delta / k) and cannot miss a feasible cut on the grid.
    k = min(GRID_CUTS, m)
    cuts = sorted({max(1, round(m * i / k)) for i in range(1, k + 1)})
    k = len(cuts)
    best = {"theta": math.inf, "bound": 1.0, "accepted": 0, "errors": 0}  # accept nothing if no cut passes
    for z in cuts:
        theta = float(conf[z - 1])
        accepted = int(np.sum(conf >= theta))  # ties at theta are accepted, matching evaluate()
        errors = int(wrong[:accepted].sum())
        bound = binomial_upper_bound(errors, accepted, delta / k)  # eq. (4) at delta / k
        if bound < target_risk and accepted > best["accepted"]:
            best = {"theta": theta, "bound": bound, "accepted": accepted, "errors": errors}
    n_acc = best["accepted"]
    return {
        "theta": best["theta"],
        "risk_bound": best["bound"],
        "empirical_selective_risk": best["errors"] / n_acc if n_acc else 0.0,  # eq. (1), empirical
        "coverage": n_acc / m,
        "accepted": n_acc,
        "errors_accepted": best["errors"],
        "calibrate_rows": m,
        "steps": k,
    }


def _wilson(x: int, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = x / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def evaluate(confidence: np.ndarray, correct: np.ndarray, theta: float) -> dict:
    """Apply a frozen theta to untouched rows."""
    keep = confidence >= theta
    wrong = 1 - correct.astype(int)
    n, n_acc = len(confidence), int(keep.sum())
    err_acc, err_def = int(wrong[keep].sum()), int(wrong[~keep].sum())
    return {
        "evaluate_rows": n,
        "accepted": n_acc,
        "acceptance_rate": n_acc / n if n else 0.0,
        "acceptance_wilson95": _wilson(n_acc, n),
        "errors_accepted": err_acc,
        "error_rate_accepted": err_acc / n_acc if n_acc else None,
        "error_rate_accepted_cp95_upper": binomial_upper_bound(err_acc, n_acc, 0.05),
        "error_rate_deferred": err_def / (n - n_acc) if n - n_acc else None,
    }


def run(confidence: np.ndarray, correct: np.ndarray, target_risk: float, delta: float, seed: int) -> dict:
    idx = np.random.default_rng(seed).permutation(len(confidence))
    explore, calibrate, test = np.array_split(idx, 3)
    cal = sgr(confidence[calibrate], correct[calibrate], target_risk, delta)
    return {
        "target_risk": target_risk,
        "delta": delta,
        "seed": seed,
        "explore_rows": len(explore),
        "calibrate": cal,
        "evaluate": evaluate(confidence[test], correct[test], cal["theta"]),
    }


def _self_check() -> None:
    assert 0.045 < binomial_upper_bound(0, 59, 0.05) < 0.05  # 1 - 0.05**(1/59) = 0.0495
    assert abs(binomial_upper_bound(0, 1, 0.05) - 0.95) < 1e-9  # 1 - b = 0.05
    assert 0.5 < binomial_upper_bound(5, 10, 0.5) < 0.6
    assert binomial_upper_bound(3, 3, 0.05) == 1.0 and binomial_upper_bound(0, 0, 0.05) == 1.0
    for m in (3000, 30000):
        rng = np.random.default_rng(0)
        conf = rng.uniform(0, 1, m)
        correct = (rng.uniform(0, 1, m) < conf).astype(int)
        out = run(conf, correct, 0.05, 0.05, 42)
        cal, ev = out["calibrate"], out["evaluate"]
        print(f"--- synthetic m={m}")
        print(json.dumps(out, indent=2))
        assert cal["risk_bound"] >= cal["empirical_selective_risk"]
        if m == 3000:
            # 1,000 calibration rows of U(0,1) confidence cannot certify 0.05: the best cut
            # (top 50 rows, 0 errors) bounds at 0.10 under delta/k. SGR must accept nothing.
            assert cal["accepted"] == 0 and ev["accepted"] == 0
            continue
        # With ten coverage cuts, the top 10% of U(0,1) confidence carries about 5% error,
        # so 0.05 is not certifiable on the grid; 0.10 is, and the realised error must sit
        # under the bound on the untouched third.
        out = run(conf, correct, 0.10, 0.05, 42)
        cal, ev = out["calibrate"], out["evaluate"]
        assert 0 < cal["accepted"] and cal["risk_bound"] < 0.10
        print(f"evaluate-third realised error among accepted: {ev['error_rate_accepted']:.4f}")
        assert ev["error_rate_accepted"] <= cal["risk_bound"]
    print("self-check passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows", nargs="?", help="JSONL with fields confidence (float), correct (0/1)")
    ap.add_argument("--target-risk", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        return _self_check()
    if not args.rows:
        ap.error("rows path required")
    with open(args.rows) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    conf = np.array([float(r["confidence"]) for r in rows])
    correct = np.array([int(r["correct"]) for r in rows])
    print(json.dumps(run(conf, correct, args.target_risk, args.delta, args.seed)))


if __name__ == "__main__":
    main()
