"""E3: a classifier head on the fine-tuned model's prefill hidden state, no decoding.

TRACER replaces an LLM classifier with classical ML on text embeddings. The version inside
our own stack: the fine-tuned model's hidden state at the last prompt token already encodes
the task, so a 77-way logistic head on it answers in one matmul, before any token is
generated. Question: does it match the generative answer's accuracy, and can SGR gate it
with its own max-probability as the confidence?

Inputs: <stem>.layers.npz + <stem>.layers.jsonl from score_layers.py for the head's training
rows (probe rows plus the fine-tune's own training rows) and for the test rows, plus the
original rows files for the gold labels.
Output: a Markdown table per layer: head accuracy on test rows vs the generative answer's
accuracy on the same rows, and a JSONL of head confidences for run_signals (--probe-layers).

Usage: uv run python classifier_tier.py --train runs/banking77/probe_layers runs/banking77/train_layers \
    --train-rows data/banking77/probe.jsonl data/banking77/train.jsonl \
    --test runs/banking77/test_layers --test-rows data/banking77/test.jsonl \
    --out runs/banking77/classifier_conf.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

POSITIONS = ("prompt_last",)  # the prefill position is the point; answer positions need decoding


def gold_labels(rows_path: Path) -> list[str]:
    out = []
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(json.loads(line)["output"])["intent"])
    return out


def load(stem: Path) -> tuple[dict[str, np.ndarray], list[dict]]:
    z = np.load(str(stem) + ".layers.npz")
    feats = {k: z[k].astype(np.float32) for k in z.files}
    labels = [json.loads(l) for l in open(str(stem) + ".layers.jsonl", encoding="utf-8") if l.strip()]
    return feats, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, nargs="+", required=True)
    ap.add_argument("--train-rows", type=Path, nargs="+", required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--test-rows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    assert len(args.train) == len(args.train_rows)

    train_feats: dict[str, list[np.ndarray]] = {}
    train_gold: list[str] = []
    for stem, rows in zip(args.train, args.train_rows):
        feats, _ = load(stem)
        gold = gold_labels(rows)
        assert len(gold) == next(iter(feats.values())).shape[0], (stem, len(gold))
        for k, v in feats.items():
            train_feats.setdefault(k, []).append(v)
        train_gold.extend(gold)
    test_feats, test_labels = load(args.test)
    test_gold = gold_labels(args.test_rows)
    assert len(test_gold) == len(test_labels)
    gen_correct = np.array([r["correct"] for r in test_labels])
    y_train = np.array(train_gold)
    y_test = np.array(test_gold)

    rows_out: list[dict] = [{} for _ in test_gold]
    print(f"head training rows {len(y_train)}, classes {len(set(train_gold))}, test rows {len(y_test)}")
    print(f"generative answer accuracy on test rows: {gen_correct.mean():.4f}\n")
    print("| layer, position | C | head CV accuracy (train) | head accuracy (test) | generative accuracy | head confidence = max p |")
    print("|---|---|---|---|---|---|")
    for key in sorted(train_feats):
        if not any(key.endswith(p) for p in POSITIONS):
            continue
        X_train = np.concatenate(train_feats[key])
        X_test = test_feats[key]
        scaler = StandardScaler().fit(X_train)
        Xs, Xt = scaler.transform(X_train), scaler.transform(X_test)
        best_c, best_cv = None, -1.0
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        for C in (0.01, 0.03, 0.1, 0.3, 1.0):
            accs = []
            for tr, va in skf.split(Xs, y_train):
                clf = LogisticRegression(C=C, max_iter=3000).fit(Xs[tr], y_train[tr])
                accs.append((clf.predict(Xs[va]) == y_train[va]).mean())
            cv = float(np.mean(accs))
            if cv > best_cv:
                best_c, best_cv = C, cv
        clf = LogisticRegression(C=best_c, max_iter=3000).fit(Xs, y_train)
        proba = clf.predict_proba(Xt)
        pred = clf.classes_[proba.argmax(1)]
        acc = float((pred == y_test).mean())
        conf = proba.max(1)
        for i in range(len(y_test)):
            rows_out[i][f"head_{key}"] = float(conf[i])
            rows_out[i][f"head_{key}_correct"] = int(pred[i] == y_test[i])
        print(f"| {key} | {best_c} | {best_cv:.4f} | {acc:.4f} | {gen_correct.mean():.4f} | written |")

    with args.out.open("w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {args.out}: per-row head confidence and head correctness per layer")


if __name__ == "__main__":
    main()
