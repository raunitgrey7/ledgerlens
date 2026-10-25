"""SQLite persistence for processed documents and review outcomes.

SQLite needs no extra service, survives restarts and is fast enough for a review
queue measured in thousands of documents. Results are stored as JSON text so the
schema does not have to change when the extraction record grows.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    filename TEXT,
    doc_type TEXT NOT NULL,
    decision TEXT NOT NULL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL,
    result_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reviews (
    document_id TEXT PRIMARY KEY REFERENCES documents(id),
    reviewed_at TEXT NOT NULL,
    reviewer TEXT,
    fields_total INTEGER NOT NULL,
    fields_corrected INTEGER NOT NULL,
    line_items_corrected INTEGER NOT NULL,
    corrections_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS documents_status ON documents(status, created_at);
"""


@dataclass
class StoredDocument:
    id: str
    created_at: str
    filename: str | None
    doc_type: str
    decision: str
    confidence: float
    status: str
    result: dict[str, object]


class Storage:
    """Thread-safe wrapper around one SQLite connection."""

    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def insert_document(self, doc_id: str, filename: str | None, result: dict[str, object]) -> StoredDocument:
        status = "accepted" if result["decision"] == "accepted" else "pending"
        row = StoredDocument(
            doc_id,
            datetime.now(UTC).isoformat(timespec="seconds"),
            filename,
            str(result["doc_type"]),
            str(result["decision"]),
            float(result["document_confidence"]),
            status,
            result,
        )
        with self._lock:
            self._conn.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row.id,
                    row.created_at,
                    row.filename,
                    row.doc_type,
                    row.decision,
                    row.confidence,
                    row.status,
                    json.dumps(result, default=str),
                ),
            )
            self._conn.commit()
        return row

    def get_document(self, doc_id: str) -> StoredDocument | None:
        with self._lock:
            cursor = self._conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
            found = cursor.fetchone()
        return self._to_document(found) if found else None

    def list_documents(self, status: str | None, limit: int, offset: int) -> list[StoredDocument]:
        query = (
            "SELECT * FROM documents"
            + (" WHERE status = ?" if status else "")
            + " ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?"
        )
        params: tuple[object, ...] = (status, limit, offset) if status else (limit, offset)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [self._to_document(row) for row in rows]

    def count(self, status: str | None = None) -> int:
        query = "SELECT COUNT(*) FROM documents" + (" WHERE status = ?" if status else "")
        with self._lock:
            return int(self._conn.execute(query, (status,) if status else ()).fetchone()[0])

    def record_review(
        self,
        doc_id: str,
        reviewer: str | None,
        fields_total: int,
        fields_corrected: int,
        line_items_corrected: int,
        corrections: dict[str, object],
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO reviews VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    doc_id,
                    datetime.now(UTC).isoformat(timespec="seconds"),
                    reviewer,
                    fields_total,
                    fields_corrected,
                    line_items_corrected,
                    json.dumps(corrections, default=str),
                ),
            )
            self._conn.execute("UPDATE documents SET status = 'reviewed' WHERE id = ?", (doc_id,))
            self._conn.commit()

    def review_totals(self) -> dict[str, int]:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(fields_total), 0), COALESCE(SUM(fields_corrected), 0), "
                "COALESCE(SUM(line_items_corrected), 0) FROM reviews"
            ).fetchone()
        return {
            "reviews": int(row[0]),
            "fields_total": int(row[1]),
            "fields_corrected": int(row[2]),
            "line_items_corrected": int(row[3]),
        }

    def accepted_field_count(self) -> int:
        with self._lock:
            rows = self._conn.execute("SELECT result_json FROM documents WHERE status = 'accepted'").fetchall()
        return sum(len(json.loads(row[0]).get("fields", [])) for row in rows)

    @staticmethod
    def _to_document(row: sqlite3.Row) -> StoredDocument:
        return StoredDocument(
            row["id"],
            row["created_at"],
            row["filename"],
            row["doc_type"],
            row["decision"],
            row["confidence"],
            row["status"],
            json.loads(row["result_json"]),
        )
