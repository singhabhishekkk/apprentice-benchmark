"""Score or generate predictions against manually downloaded BIRD dev data."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import threading
import urllib.error
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

Status = Literal["match", "mismatch", "error", "timeout"]


def readonly_connection(database: Path) -> sqlite3.Connection:
    uri = f"{database.resolve().as_uri()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection


def execute(database: Path, sql: str, timeout: float) -> tuple[Status, frozenset[tuple[Any, ...]] | None]:
    connection = readonly_connection(database)
    timed_out = threading.Event()

    def interrupt() -> None:
        timed_out.set()
        connection.interrupt()

    timer = threading.Timer(timeout, interrupt)
    timer.start()
    try:
        rows = frozenset(tuple(row) for row in connection.execute(sql).fetchall())
        return "match", rows
    except sqlite3.Error:
        return ("timeout" if timed_out.is_set() else "error"), None
    finally:
        timer.cancel()
        connection.close()


def score_sql(database: Path, predicted: str, gold: str, timeout: float) -> Status:
    predicted_status, predicted_rows = execute(database, predicted, timeout)
    if predicted_status != "match":
        return predicted_status
    gold_status, gold_rows = execute(database, gold, timeout)
    if gold_status != "match":
        raise RuntimeError(f"gold SQL failed with {gold_status}: {gold}")
    return "match" if predicted_rows == gold_rows else "mismatch"


def find_layout(bird_dir: Path) -> tuple[Path, Path]:
    candidates = (bird_dir, bird_dir / "dev_20240627")
    for root in candidates:
        dev_json = root / "dev.json"
        databases = root / "dev_databases"
        if dev_json.is_file() and databases.is_dir():
            return dev_json, databases
    expected = " or ".join(str(root / "dev.json") for root in candidates)
    db_expected = " or ".join(str(root / "dev_databases") for root in candidates)
    raise SystemExit(f"BIRD dev layout incomplete. Missing: {expected}; {db_expected}")


def load_dev(dev_json: Path, databases: Path) -> list[dict[str, Any]]:
    with dev_json.open(encoding="utf-8") as handle:
        tasks: list[dict[str, Any]] = json.load(handle)
    missing = []
    for task in tasks:
        database = databases / str(task.get("db_id")) / f"{task.get('db_id')}.sqlite"
        if not database.is_file():
            missing.append(str(database))
    if missing:
        raise SystemExit("BIRD dev layout incomplete. Missing:\n" + "\n".join(missing))
    return tasks


def schema_ddl(database: Path) -> str:
    # sqlite3's context manager only ends the transaction; close explicitly so
    # generation mode does not leak one file handle per question.
    connection = readonly_connection(database)
    try:
        rows = connection.execute(
            "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL "
            "AND type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    finally:
        connection.close()
    return "\n".join(str(row[0]) + ";" for row in rows)


def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()[1:-1]
        if lines and lines[0].strip().lower() == "sql":
            lines = lines[1:]
        return "\n".join(lines).strip()
    return text


def generate_sql(api_base: str, model: str, prompt: str) -> str:
    payload = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}
    ).encode()
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions", data=payload, headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.load(response)
    except urllib.error.URLError as error:
        raise RuntimeError(f"generation request failed: {error.reason}") from error
    return strip_fences(str(result["choices"][0]["message"]["content"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bird-dir", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--predictions", type=Path)
    mode.add_argument("--api-base")
    parser.add_argument("--model")
    parser.add_argument("--generated-output", type=Path, default=Path("predictions.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("report.json"))
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timeout <= 0:
        raise SystemExit("--timeout must be greater than 0")
    if args.api_base and not args.model:
        raise SystemExit("--model is required with --api-base")

    dev_json, databases = find_layout(args.bird_dir)
    tasks = load_dev(dev_json, databases)
    if args.predictions:
        with args.predictions.open(encoding="utf-8") as handle:
            predictions = {str(row["question_id"]): str(row["sql"]) for row in map(json.loads, handle)}
    else:
        predictions = {}
        with args.generated_output.open("w", encoding="utf-8") as output:
            for task in tasks:
                db_id = str(task["db_id"])
                prompt = (
                    f"Schema:\n{schema_ddl(databases / db_id / f'{db_id}.sqlite')}\n\n"
                    f"External knowledge/evidence:\n{task.get('evidence', '')}\n\n"
                    f"Question:\n{task['question']}\n\nOutput ONLY the SQL."
                )
                sql = generate_sql(args.api_base, args.model, prompt)
                question_id = str(task["question_id"])
                predictions[question_id] = sql
                output.write(json.dumps({"question_id": task["question_id"], "sql": sql}) + "\n")
                output.flush()

    counts: Counter[str] = Counter()
    difficulty: dict[str, Counter[str]] = {}
    for task in tasks:
        question_id = str(task["question_id"])
        level = str(task.get("difficulty", "unknown")).lower()
        level_counts = difficulty.setdefault(level, Counter())
        prediction = predictions.get(question_id)
        if prediction is None:
            status: Status = "error"
        else:
            db_id = str(task["db_id"])
            gold = task.get("SQL", task.get("sql"))
            if not isinstance(gold, str):
                raise RuntimeError(f"question {question_id} missing gold SQL/ sql field")
            status = score_sql(databases / db_id / f"{db_id}.sqlite", prediction, gold, args.timeout)
        counts[status] += 1
        level_counts[status] += 1

    total = len(tasks)
    breakdown = {
        level: {"ex": values["match"] / sum(values.values()), "correct": values["match"], "total": sum(values.values())}
        for level, values in sorted(difficulty.items())
    }
    report = {
        "overall_ex": counts["match"] / total if total else 0.0,
        "correct": counts["match"],
        "total": total,
        "breakdown": breakdown,
        "errors": counts["error"],
        "timeouts": counts["timeout"],
        "mismatches": counts["mismatch"],
        "config": {"model": args.model, "timestamp": datetime.now(UTC).isoformat(), "timeout": args.timeout},
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Overall EX: {report['overall_ex']:.6f} ({counts['match']}/{total})")
    for level, values in breakdown.items():
        print(f"{level}: EX {values['ex']:.6f} ({values['correct']}/{values['total']})")
    print(f"errors={counts['error']} timeouts={counts['timeout']} mismatches={counts['mismatch']}")
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
