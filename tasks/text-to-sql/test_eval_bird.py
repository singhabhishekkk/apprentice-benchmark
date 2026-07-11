from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from eval_bird import execute, find_layout, load_dev, score_sql


@pytest.fixture
def bird_dev(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "dev_20240627"
    database_dir = root / "dev_databases" / "tiny"
    database_dir.mkdir(parents=True)
    database = database_dir / "tiny.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")
        connection.executemany("INSERT INTO items VALUES (?, ?)", [(1, "one"), (2, "two")])
    (root / "dev.json").write_text(
        json.dumps(
            [{"question_id": 1, "db_id": "tiny", "question": "names?", "SQL": "SELECT name FROM items", "difficulty": "simple"}]
        ),
        encoding="utf-8",
    )
    return tmp_path, database


def test_layout_and_scoring_paths(bird_dev: tuple[Path, Path]) -> None:
    bird_dir, database = bird_dev
    dev_json, databases = find_layout(bird_dir)
    assert len(load_dev(dev_json, databases)) == 1
    assert score_sql(database, "SELECT name FROM items", "SELECT name FROM items", 1) == "match"
    assert score_sql(database, "SELECT name FROM items WHERE id = 1", "SELECT name FROM items", 1) == "mismatch"
    assert score_sql(database, "SELECT missing FROM items", "SELECT name FROM items", 1) == "error"
    assert score_sql(database, "SELECT name FROM items ORDER BY id DESC", "SELECT name FROM items ORDER BY id", 1) == "match"


def test_timeout_scores_zero(bird_dev: tuple[Path, Path]) -> None:
    _, database = bird_dev
    endless = "WITH RECURSIVE loop(x) AS (VALUES(1) UNION ALL SELECT x + 1 FROM loop) SELECT sum(x) FROM loop"
    assert score_sql(database, endless, "SELECT name FROM items", 0.01) == "timeout"


@pytest.mark.parametrize(
    "sql",
    ["INSERT INTO items(name) VALUES ('bad')", "UPDATE items SET name = 'bad'", "DROP TABLE items"],
)
def test_database_is_read_only(bird_dev: tuple[Path, Path], sql: str) -> None:
    _, database = bird_dev
    status, _ = execute(database, sql, 1)
    assert status == "error"
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT name FROM items ORDER BY id").fetchall() == [("one",), ("two",)]
