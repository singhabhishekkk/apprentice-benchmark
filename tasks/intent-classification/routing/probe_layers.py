"""Per-layer, per-position hidden-state probes, C chosen by 5-fold CV on the train rows only.

Input pair per rows file: <stem>.layers.npz (float16 [rows, 2560] per key `L{layer}_{position}`)
and <stem>.layers.jsonl (one object per row with `correct`, same order). No metric is ever
computed on --apply rows: run_signals.py does the explore / calibrate / evaluate split there.

Usage: uv run python probe_layers.py --train runs/banking77/probe_layers --apply runs/banking77/test_layers \
           --out runs/banking77/probe_layers_conf.jsonl --seed 42
       uv run python probe_layers.py --self-check
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from probe import auroc

C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0)


def _model(c: float):
    # Scaler inside the pipeline, so each CV fold standardizes on its own training rows.
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000, class_weight=None))


def _load(stem: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with open(stem.with_name(stem.name + ".layers.jsonl")) as f:
        y = np.array([int(json.loads(line)["correct"]) for line in f if line.strip()])
    with np.load(stem.with_name(stem.name + ".layers.npz")) as npz:
        hidden = {key: npz[key].astype(np.float32) for key in npz.files}
    for key, h in hidden.items():
        assert len(h) == len(y), f"{key} has {len(h)} rows, labels have {len(y)}"
    return y, hidden


def cv_select(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, float, list[float]]:
    """Best C by pooled out-of-fold AUROC; returns (C, AUROC, bootstrap 95% interval over the OOF predictions)."""
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(x, y))
    best_c, best_auc, best_oof = C_GRID[0], -1.0, np.zeros(len(y))
    for c in C_GRID:
        oof = np.zeros(len(y))
        for fit_idx, held_idx in folds:
            oof[held_idx] = _model(c).fit(x[fit_idx], y[fit_idx]).predict_proba(x[held_idx])[:, 1]
        auc = auroc(oof, y)
        if auc > best_auc:
            best_c, best_auc, best_oof = c, auc, oof
    rng = np.random.default_rng(seed)
    boots = [auroc(best_oof[i], y[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(1000))]
    return best_c, best_auc, list(np.nanpercentile(boots, [2.5, 97.5]))


def run(train: Path, apply: Path, out: Path, seed: int) -> list[dict]:
    y_train, h_train = _load(train)
    _, h_apply = _load(apply)
    keys = sorted(set(h_train) & set(h_apply))
    results, probs = [], {}
    for key in keys:
        c, auc, (lo, hi) = cv_select(h_train[key], y_train, seed)
        probs[key] = _model(c).fit(h_train[key], y_train).predict_proba(h_apply[key])[:, 1]
        results.append({"key": key, "C": c, "cv_auroc": auc, "lo": lo, "hi": hi})
    results.sort(key=lambda r: -r["cv_auroc"])
    print(f"train rows {len(y_train)}, positive rate {y_train.mean():.4f}, keys {len(keys)}")
    print("\n| key | chosen C | train 5-fold AUROC (bootstrap 95%) | positive rate |\n|---|---|---|---|")
    for r in results:
        print(
            f"| {r['key']} | {r['C']} | {r['cv_auroc']:.3f} ({r['lo']:.3f} to {r['hi']:.3f}) | {y_train.mean():.3f} |"
        )
    n_apply = len(next(iter(h_apply.values())))
    with open(out, "w") as f:
        f.writelines(json.dumps({key: float(probs[key][i]) for key in keys}) + "\n" for i in range(n_apply))
    return results


def _self_check() -> None:
    import run_signals

    rng = np.random.default_rng(0)
    w = rng.normal(size=16)
    with tempfile.TemporaryDirectory() as tmp:
        for stem, n in (("train", 200), ("apply", 100)):
            x = rng.normal(size=(n, 16))
            y = (x @ w + rng.normal(scale=1.0, size=n) > 0).astype(int)
            weak = 0.3 * x + rng.normal(size=(n, 16))  # same signal, buried in noise
            arrays = {"L8_answer_last": x, "L4_prompt_last": rng.normal(size=(n, 16)), "L6_answer_mean": weak}
            np.savez(Path(tmp, f"{stem}.layers.npz"), **{k: v.astype(np.float16) for k, v in arrays.items()})
            Path(tmp, f"{stem}.layers.jsonl").write_text("".join(json.dumps({"correct": int(v)}) + "\n" for v in y))
        out = Path(tmp, "conf.jsonl")
        results = run(Path(tmp, "train"), Path(tmp, "apply"), out, 42)
        assert results[0]["key"] == "L8_answer_last", "planted key must rank first"
        noise = next(r for r in results if r["key"] == "L4_prompt_last")
        assert noise["cv_auroc"] < 0.65, f"noise key CV AUROC {noise['cv_auroc']:.3f}"
        lines = out.read_text().splitlines()
        assert len(lines) == 100 and set(json.loads(lines[0])) == {r["key"] for r in results}

        # run_signals.py picks every --probe-layers key up as a signal (4-row smoke file).
        smoke = Path(run_signals.__file__).parent / "runs/smoke/probe_scores.jsonl"
        layer_conf = Path(tmp, "smoke_layers.jsonl")
        layer_conf.write_text("".join(json.dumps({"L8_answer_last": p}) + "\n" for p in (0.1, 0.2, 0.3, 0.4)))
        _, signals = run_signals.report(smoke, None, 0.05, 0.05, 42, probe_layers=layer_conf)
        assert "L8_answer_last" in signals
    run_signals._self_check()
    print("probe_layers self-check passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path)
    ap.add_argument("--apply", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        return _self_check()
    if not (args.train and args.apply and args.out):
        ap.error("--train, --apply and --out are required")
    run(args.train, args.apply, args.out, args.seed)


if __name__ == "__main__":
    main()
