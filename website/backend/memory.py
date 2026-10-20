import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import numpy as np

DB_PATH = Path(os.environ["MEMORY_DB"]) if "MEMORY_DB" in os.environ else Path(__file__).resolve().parent / "memory.db"
MAX_ROWS = 5000
EMBEDDING_DIM = 2048
SIMILARITY_OVERRIDE_THRESHOLD = 0.93
SIMILARITY_BLEND_THRESHOLD = 0.86

_cache_lock = Lock()
# stage -> list of (id, label, normalized_vector)
_memory_cache: dict[int, list[tuple[int, str, np.ndarray]]] = {1: [], 2: []}


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with closing(_connect()) as conn, conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                stage INTEGER NOT NULL,
                label TEXT NOT NULL,
                note_hash TEXT DEFAULT '',
                embedding BLOB NOT NULL,
                approved INTEGER NOT NULL DEFAULT 1,
                comment TEXT DEFAULT ''
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_stage ON memory (stage, approved)")
    _reload_cache()


def _reload_cache() -> None:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT id, stage, label, embedding FROM memory WHERE approved = 1 ORDER BY id"
        ).fetchall()

    new_cache: dict[int, list[tuple[int, str, np.ndarray]]] = {1: [], 2: []}
    for r in rows:
        vec = np.frombuffer(r["embedding"], dtype=np.float32)
        if len(vec) == EMBEDDING_DIM:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            new_cache.setdefault(r["stage"], []).append((r["id"], r["label"], vec))

    global _memory_cache
    with _cache_lock:
        _memory_cache = new_cache


def normalize_vector(v: np.ndarray) -> np.ndarray:
    v = v.flatten().astype(np.float32)
    norm = np.linalg.norm(v)
    return (v / norm) if norm > 0 else v


def add_memory(
    embedding: np.ndarray,
    stage: int,
    label: str,
    note_hash: str = "",
    approved: bool = True,
    comment: str = "",
) -> int:
    """Store a corrected note embedding into the memory index."""
    v = normalize_vector(embedding)
    if len(v) != EMBEDDING_DIM:
        raise ValueError(f"Expected {EMBEDDING_DIM}-D embedding, got {len(v)}")

    with closing(_connect()) as conn, conn:
        count = conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
        if count >= MAX_ROWS:
            # Drop oldest unapproved or oldest record
            conn.execute("DELETE FROM memory WHERE id = (SELECT id FROM memory ORDER BY approved ASC, id ASC LIMIT 1)")

        cur = conn.execute(
            """
            INSERT INTO memory (created_at, stage, label, note_hash, embedding, approved, comment)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                stage,
                label.upper(),
                note_hash,
                v.tobytes(),
                1 if approved else 0,
                comment,
            ),
        )
        mem_id = cur.lastrowid

    if approved:
        with _cache_lock:
            _memory_cache.setdefault(stage, []).append((mem_id, label.upper(), v))
    return mem_id


def lookup_memory(embedding: np.ndarray, stage: int) -> dict | None:
    """Find the closest verified memory for this stage using cosine similarity."""
    v = normalize_vector(embedding)
    if len(v) != EMBEDDING_DIM:
        return None

    with _cache_lock:
        candidates = list(_memory_cache.get(stage, []))

    if not candidates:
        return None

    # Stack vectors into matrix and compute dot product
    ids = [c[0] for c in candidates]
    labels = [c[1] for c in candidates]
    mat = np.stack([c[2] for c in candidates])  # (N, 2048)
    sims = mat.dot(v)  # (N,)

    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])

    if best_sim < SIMILARITY_BLEND_THRESHOLD:
        return None

    is_override = best_sim >= SIMILARITY_OVERRIDE_THRESHOLD
    return {
        "matched": True,
        "memory_id": ids[best_idx],
        "label": labels[best_idx],
        "similarity": round(best_sim, 3),
        "action": "override" if is_override else "blend",
    }


def list_memories() -> list[dict]:
    with closing(_connect()) as conn:
        rows = conn.execute("SELECT id, created_at, stage, label, note_hash, approved, comment FROM memory ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]


def review_memory(memory_id: int, approved: bool) -> bool:
    with closing(_connect()) as conn, conn:
        cur = conn.execute("UPDATE memory SET approved = ? WHERE id = ?", (1 if approved else 0, memory_id))
        updated = cur.rowcount > 0
    if updated:
        _reload_cache()
    return updated


def delete_memory(memory_id: int) -> bool:
    with closing(_connect()) as conn, conn:
        cur = conn.execute("DELETE FROM memory WHERE id = ?", (memory_id,))
        deleted = cur.rowcount > 0
    if deleted:
        _reload_cache()
    return deleted


init_db()
