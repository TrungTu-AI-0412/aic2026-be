import sqlite3
from contextlib import contextmanager
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    job_id TEXT PRIMARY KEY,
    entity TEXT NOT NULL,
    manifest_path TEXT NOT NULL,
    collection_name TEXT NOT NULL,
    feature_profile TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT,
    progress_completed INTEGER,
    progress_total INTEGER,
    error TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


@contextmanager
def _connect(db_path: str):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute(_SCHEMA)
        yield connection
        connection.commit()
    finally:
        connection.close()


def create_job(
    db_path: str,
    job_id: str,
    entity: str,
    manifest_path: str,
    collection_name: str,
    feature_profile: str,
) -> None:
    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO ingestion_jobs (
                job_id, entity, manifest_path, collection_name, feature_profile, status
            ) VALUES (?, ?, ?, ?, ?, 'queued')
            """,
            (job_id, entity, manifest_path, collection_name, feature_profile),
        )


def collection_name_exists(db_path: str, collection_name: str) -> bool:
    with _connect(db_path) as connection:
        row = connection.execute(
            "SELECT 1 FROM ingestion_jobs WHERE collection_name = ? AND status != 'failed'",
            (collection_name,),
        ).fetchone()
        return row is not None


def get_job(db_path: str, job_id: str) -> sqlite3.Row | None:
    with _connect(db_path) as connection:
        return connection.execute(
            "SELECT * FROM ingestion_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()


def list_jobs(db_path: str) -> list[sqlite3.Row]:
    with _connect(db_path) as connection:
        return connection.execute(
            "SELECT * FROM ingestion_jobs ORDER BY created_at DESC"
        ).fetchall()


def update_job(
    db_path: str,
    job_id: str,
    *,
    status: str | None = None,
    stage: str | None = None,
    progress_completed: int | None = None,
    progress_total: int | None = None,
    error: str | None = None,
) -> None:
    fields = []
    values = []

    if status is not None:
        fields.append("status = ?")
        values.append(status)
    if stage is not None:
        fields.append("stage = ?")
        values.append(stage)
    if progress_completed is not None:
        fields.append("progress_completed = ?")
        values.append(progress_completed)
    if progress_total is not None:
        fields.append("progress_total = ?")
        values.append(progress_total)
    if error is not None:
        fields.append("error = ?")
        values.append(error)

    if not fields:
        return

    values.append(job_id)
    with _connect(db_path) as connection:
        connection.execute(
            f"UPDATE ingestion_jobs SET {', '.join(fields)} WHERE job_id = ?",
            values,
        )
