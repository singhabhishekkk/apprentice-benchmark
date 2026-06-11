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


def load_dataset(path: str, input_col: str, output_col: str) -> list[dspy.Example]:
    import csv

    examples = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            examples.append(
                dspy.Example(text=row[input_col], expected=row[output_col]).with_inputs("text")
            )
    return examples


def json_field_metric(
    example: dspy.Example, prediction, trace=None, pred_name=None, pred_trace=None
):
    """Tier-1 deterministic metric for JSON extraction: schema-valid + field-level F1.

    GEPA-compatible: accepts the 5-arg signature and returns score + text feedback
    (feedback is Actionable Side Information GEPA's reflection uses to evolve prompts).
    """
    try:
        expected = json.loads(example.expected)
    except json.JSONDecodeError:
        return dspy.Prediction(
            score=0.0, feedback="gold output is not valid JSON; row should be cleaned"
        )
    try:
        actual = json.loads(prediction.extracted)
    except (json.JSONDecodeError, AttributeError):
        return dspy.Prediction(
            score=0.0, feedback="output is not valid JSON; emit a single JSON object only"
        )
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return dspy.Prediction(
            score=0.0, feedback="output must be a JSON object, not a list or scalar"
        )

    true_positives = sum(1 for k, v in expected.items() if actual.get(k) == v)
    precision = true_positives / len(actual) if actual else 0.0
    recall = true_positives / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    missing = [k for k in expected if k not in actual]
    wrong = [k for k in expected if k in actual and actual[k] != expected[k]]
    extra = [k for k in actual if k not in expected]
    parts = []
    if missing:
        parts.append(f"missing fields: {missing}")
    if wrong:
        parts.append(f"wrong values for: {wrong}")
    if extra:
        parts.append(f"unexpected extra fields: {extra}")
    feedback = "; ".join(parts) if parts else "all fields correct"
    return dspy.Prediction(score=f1, feedback=feedback)


class Extract(dspy.Signature):
    """Extract the structured fields from the text as a JSON object."""

    text: str = dspy.InputField()
    extracted: str = dspy.OutputField(desc="valid JSON object with the extracted fields")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV with golden input/output pairs")
    parser.add_argument("--input-col", default="input")
    parser.add_argument("--output-col", default="output")
    parser.add_argument("--model", default="openai/gpt-4o-mini", help="student LM")
    parser.add_argument("--reflection-model", default="openai/gpt-4o", help="GEPA reflection LM")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dspy.configure(lm=dspy.LM(args.model))

    examples = load_dataset(args.data, args.input_col, args.output_col)
    random.Random(args.seed).shuffle(examples)
    split = int(len(examples) * 0.7)
    trainset, devset = examples[:split], examples[split:]
    print(f"dataset: {len(examples)} rows -> train {len(trainset)}, held-out {len(devset)}")

    program = dspy.Predict(Extract)
    evaluate = dspy.Evaluate(devset=devset, metric=json_field_metric, display_progress=True)

    baseline_score = evaluate(program)
    print(f"\nbaseline score: {baseline_score}")

    optimizer = dspy.GEPA(
        metric=json_field_metric,
        auto="light",
        reflection_lm=dspy.LM(args.reflection_model),
    )
    optimized = optimizer.compile(program, trainset=trainset)

    optimized_score = evaluate(optimized)
    print(f"\nbaseline:  {baseline_score}")
    print(f"optimized: {optimized_score}")

    optimized.save("optimized_program.json")
    print("saved optimized program -> optimized_program.json")


if __name__ == "__main__":
    main()
