"""Refit the published classifier head and save it in the product's head.npz layout.

classifier_tier.py printed the accuracy but kept no weights. This refits the same model
(StandardScaler + multinomial LogisticRegression, C by the same 5-fold search, seed 42) on
the same hidden states, checks it reproduces the published test accuracy, and writes the
file the model card describes: coef, intercept, scaler_mean, scaler_scale, labels, plus the
adapter hash so a serving process can refuse a head that does not match its adapter.

Usage: uv run python export_head.py --train runs/b/train_layers runs/b/probe_layers
           --train-rows data/b/train.jsonl data/b/probe.jsonl --test runs/b/test_layers
           --test-rows data/b/test.jsonl --adapter runs/b/adapter --out hf/repo/head_L32.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from classifier_tier import gold_labels, load

KEY = "L32_prompt_last"
MODEL_ID = "mlx-community/Qwen3.5-4B-OptiQ-4bit"


def tree_sha256(directory: Path) -> str:
    """Same recipe as the SDK's _tree_sha256: sorted relative path + bytes per file."""
    digest = hashlib.sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(directory)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, nargs="+", required=True)
    ap.add_argument("--train-rows", type=Path, nargs="+", required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--test-rows", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X_parts, y_train = [], []
    for stem, rows in zip(args.train, args.train_rows):
        feats, _ = load(stem)
        X_parts.append(feats[KEY])
        y_train.extend(gold_labels(rows))
    X_train, y = np.concatenate(X_parts), np.array(y_train)
    X_test, y_test = load(args.test)[0][KEY], np.array(gold_labels(args.test_rows))

    scaler = StandardScaler().fit(X_train)
    Xs = scaler.transform(X_train)
    best_c, best_cv = 1.0, -1.0
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for C in (0.01, 0.03, 0.1, 0.3, 1.0):
        accs = [(LogisticRegression(C=C, max_iter=3000).fit(Xs[tr], y[tr]).predict(Xs[va]) == y[va]).mean()
                for tr, va in skf.split(Xs, y)]
        if float(np.mean(accs)) > best_cv:
            best_c, best_cv = C, float(np.mean(accs))
    clf = LogisticRegression(C=best_c, max_iter=3000).fit(Xs, y)
    coef, intercept = clf.coef_.astype(np.float32), clf.intercept_.astype(np.float32)
    labels = [str(c) for c in clf.classes_]

    # Score the saved arithmetic, not sklearn's, so the file is what the number describes.
    logits = (X_test - scaler.mean_) / scaler.scale_ @ coef.T + intercept
    acc = float((np.asarray(labels)[logits.argmax(1)] == y_test).mean())

    adapter_dir = args.adapter
    # The hash covers the adapter files that ship, not the training checkpoints beside them.
    shipped = Path(args.out).parent / "_adapter_hash_tmp"
    shipped.mkdir(parents=True, exist_ok=True)
    for name in ("adapter_config.json", "adapters.safetensors"):
        (shipped / name).write_bytes((adapter_dir / name).read_bytes())
    adapter_sha256 = tree_sha256(shipped)
    for p in shipped.iterdir():
        p.unlink()
    shipped.rmdir()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, coef=coef, intercept=intercept, scaler_mean=scaler.mean_.astype(np.float32),
             scaler_scale=scaler.scale_.astype(np.float32), labels=json.dumps(labels, ensure_ascii=False),
             C=best_c, cv_accuracy=best_cv, layer=32, base_model=MODEL_ID,
             adapter_sha256=adapter_sha256, key="intent")
    print(json.dumps({"rows": int(len(y)), "classes": len(labels), "C": best_c,
                      "cv_accuracy": round(best_cv, 4), "test_accuracy": round(acc, 4),
                      "adapter_sha256": adapter_sha256, "out": str(args.out)}))


if __name__ == "__main__":
    main()
