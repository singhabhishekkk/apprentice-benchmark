"""Phase 0 spike: golden dataset (CSV) -> DSPy GEPA prompt optimization -> eval report.

Success criterion (plan_fable.md, Phase 0): optimized prompt beats the baseline
prompt on a held-out split of the gold set.

Usage:
    python gepa_optimize.py --data golden.csv

The CSV needs two columns: one input, one expected output (JSON extraction task
recommended first — deterministic metric, per plan).
"""

from __future__ import annotations

import argparse
import json
import random

import dspy

import checks


def load_dataset(path: str, input_col: str, output_col: str) -> list[dspy.Example]:
    import csv

    examples = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            examples.append(
                dspy.Example(text=row[input_col], expected=row[output_col]).with_inputs("text")
            )
    return examples


def _metric(kind: str):
    def metric(example: dspy.Example, prediction, trace=None, pred_name=None, pred_trace=None):
        """GEPA-compatible: 5-arg signature, returns score + text feedback for reflection.

        Parsing and scoring live in checks.py, shared with eval_run.py, verify.py and the notebooks.
        """
        result = checks.score_output(str(example.expected), str(getattr(prediction, "extracted", "")), kind)
        return dspy.Prediction(score=result["score"], feedback=result["feedback"])

    return metric


json_field_metric = _metric("json")
exact_match_metric = _metric("label")


class Extract(dspy.Signature):
    """Extract the structured fields from the text as a JSON object."""

    text: str = dspy.InputField()
    extracted: str = dspy.OutputField(desc="valid JSON object with the extracted fields")


class ExtractExact(dspy.Signature):
    """Return the exact answer text."""

    text: str = dspy.InputField()
    extracted: str = dspy.OutputField(desc="answer text only")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV with golden input/output pairs")
    parser.add_argument("--input-col", default="input")
    parser.add_argument("--output-col", default="output")
    parser.add_argument("--model", default="openai/gpt-4o-mini", help="student LM")
    parser.add_argument("--reflection-model", default="openai/gpt-4o", help="GEPA reflection LM")
    parser.add_argument("--metric", choices=["json", "exact"], default="json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--api-base",
        default=None,
        help="override API base for --model/--reflection-model (e.g. an OpenAI-compatible gateway)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for --api-base (defaults to whatever the LM provider reads from its own env var)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Evaluate() parallelism. Lower reduces burst load on rate-limited gateways (default 4).",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=30,
        help="Evaluate() error budget before the whole run is cancelled (default 30, well above dspy's default of a few).",
    )
    args = parser.parse_args()

    lm_kwargs: dict[str, str] = {}
    if args.api_base:
        lm_kwargs["api_base"] = args.api_base
    if args.api_key:
        lm_kwargs["api_key"] = args.api_key

    dspy.configure(lm=dspy.LM(args.model, **lm_kwargs))

    examples = load_dataset(args.data, args.input_col, args.output_col)
    random.Random(args.seed).shuffle(examples)
    split = int(len(examples) * 0.7)
    trainset, devset = examples[:split], examples[split:]
    print(f"dataset: {len(examples)} rows -> train {len(trainset)}, held-out {len(devset)}")

    metric = exact_match_metric if args.metric == "exact" else json_field_metric
    program = dspy.Predict(ExtractExact if args.metric == "exact" else Extract)
    evaluate = dspy.Evaluate(
        devset=devset,
        metric=metric,
        display_progress=True,
        num_threads=args.num_threads,
        max_errors=args.max_errors,
    )

    baseline_score = evaluate(program)
    print(f"\nbaseline score: {baseline_score}")

    optimizer = dspy.GEPA(
        metric=metric,
        auto="light",
        reflection_lm=dspy.LM(args.reflection_model, **lm_kwargs),
    )
    optimized = optimizer.compile(program, trainset=trainset)

    optimized_score = evaluate(optimized)
    print(f"\nbaseline:  {baseline_score}")
    print(f"optimized: {optimized_score}")

    optimized.save("optimized_program.json")
    print("saved optimized program -> optimized_program.json")


if __name__ == "__main__":
    main()
