import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(os.environ["FEEDBACK_DB"]) if "FEEDBACK_DB" in os.environ else Path(__file__).resolve().parent / "feedback.db"
MAX_ROWS = 10000
LABELS = ("REAL", "FAKE", "NOT_NOTE")


class FeedbackFull(Exception):
    pass


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with closing(_connect()) as conn, conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                predicted TEXT NOT NULL,
                actual TEXT NOT NULL,
                confidence REAL,
                quality TEXT,
                comment TEXT NOT NULL DEFAULT ''
            )
            """
        )


def add_feedback(predicted: str, actual: str, confidence=None, quality=None, comment: str = "") -> int:
    with closing(_connect()) as conn, conn:
        count = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        if count >= MAX_ROWS:
            raise FeedbackFull
        cur = conn.execute(
            "INSERT INTO feedback (created_at, predicted, actual, confidence, quality, comment) VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), predicted, actual, confidence, quality, comment),
        )
        return cur.lastrowid


def list_feedback() -> list[dict]:
    with closing(_connect()) as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM feedback ORDER BY id")]


def summary() -> dict:
    rows = list_feedback()
    wrong = [r for r in rows if r["predicted"] != r["actual"]]
    return {"total": len(rows), "correct": len(rows) - len(wrong), "incorrect": len(wrong)}


init_db()
