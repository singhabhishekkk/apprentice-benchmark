"""Hidden-state probe: logistic regression on h_pre / h_post predicting `correct`.

Fit on --train rows (never seen by the fine-tune), write P(correct) for every --apply row.
Sidecars follow score_rows.py: <jsonl stem>.h_pre.npy / .h_post.npy, same row order.

Usage: uv run python probe.py --train probe_scores.jsonl --apply test_scores.jsonl --out probe_conf.jsonl
       uv run python probe.py --self-check
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney U), ties get average ranks. NaN if one class is absent."""
    label = np.asarray(label).astype(bool)
    n_pos, n_neg = int(label.sum()), int((~label).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]  # 1-based average rank per row
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fit_probe(x: np.ndarray, y: np.ndarray):
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight=None)).fit(x, y)


def _load(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with open(path) as f:
        y = np.array([int(json.loads(line)["correct"]) for line in f if line.strip()])
    hidden = {name: np.load(path.with_suffix(f".{name}.npy")) for name in ("h_pre", "h_post")}
    for name, h in hidden.items():
        assert len(h) == len(y), f"{name} has {len(h)} rows, {path} has {len(y)}"
    return y, hidden


def _self_check() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 16)).astype(np.float32)
    w = rng.normal(size=16)
    y = (x @ w + rng.normal(scale=1.0, size=200) > 0).astype(int)  # planted linear signal
    model = fit_probe(x[:100], y[:100])
    held_out = auroc(model.predict_proba(x[100:])[:, 1], y[100:])
    print(f"held-out AUROC on planted signal: {held_out:.4f}")
    assert held_out > 0.9
    assert auroc(np.array([0.1, 0.2, 0.3, 0.4]), np.array([0, 0, 1, 1])) == 1.0
    assert auroc(np.array([0.5, 0.5]), np.array([0, 1])) == 0.5  # tie
    print("self-check passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path)
    ap.add_argument("--apply", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        return _self_check()
    if not (args.train and args.apply and args.out):
        ap.error("--train, --apply and --out are required")
    y_train, h_train = _load(args.train)
    y_apply, h_apply = _load(args.apply)
    print(f"train rows {len(y_train)}, positive rate {y_train.mean():.4f}")
    probs = {}
    for name in ("h_pre", "h_post"):
        model = fit_probe(h_train[name], y_train)
        print(f"{name}: train AUROC {auroc(model.predict_proba(h_train[name])[:, 1], y_train):.4f}")
        probs[name] = model.predict_proba(h_apply[name])[:, 1]
    with open(args.out, "w") as f:
        f.writelines(
            json.dumps({"probe_pre": float(pre), "probe_post": float(post)}) + "\n"
            for pre, post in zip(probs["h_pre"], probs["h_post"])
        )
    print(f"wrote {len(y_apply)} rows to {args.out}")


if __name__ == "__main__":
    main()
