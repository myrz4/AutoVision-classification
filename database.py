import csv
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATABASE_PATH = PROJECT_DIR / "data" / "inspections.db"


def get_database_path():
    configured_path = os.getenv("AUTOVISION_DB_PATH")
    return Path(configured_path).expanduser() if configured_path else DEFAULT_DATABASE_PATH


@contextmanager
def _connect():
    database_path = get_database_path()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=30000")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_database():
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS inspections (
                inspection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('Pass', 'Fail')),
                confidence_score REAL,
                operator_id TEXT,
                image_path TEXT,
                defect_class TEXT
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_inspections_timestamp "
            "ON inspections(timestamp)"
        )


def insert_inspection(
    status,
    confidence_score,
    timestamp=None,
    operator_id=None,
    image_path=None,
    defect_class=None,
):
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO inspections (
                timestamp, status, confidence_score, operator_id, image_path, defect_class
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (timestamp, status, confidence_score, operator_id, image_path, defect_class),
        )
        return cursor.lastrowid


def fetch_recent_inspections(limit=5):
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM inspections ORDER BY timestamp DESC, inspection_id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def fetch_all_inspections():
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM inspections ORDER BY timestamp DESC, inspection_id DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def fetch_today_inspections():
    today = datetime.now(timezone.utc).date()
    start = datetime.combine(today, datetime.min.time(), tzinfo=timezone.utc).isoformat()
    end = datetime.combine(today + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc).isoformat()
    with _connect() as connection:
        rows = connection.execute(
            "SELECT status FROM inspections WHERE timestamp >= ? AND timestamp < ?",
            (start, end),
        ).fetchall()
    return [dict(row) for row in rows]


def import_inspections_csv(csv_path):
    init_database()
    imported = 0
    skipped = 0

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as csv_file:
        rows = csv.DictReader(csv_file)
        with _connect() as connection:
            for row in rows:
                status = (row.get("status") or "").strip()
                timestamp_value = (row.get("timestamp") or "").strip()
                confidence_value = (row.get("confidence_score") or "").strip()
                inspection_id = (row.get("inspection_id") or row.get("id") or "").strip()

                if status not in {"Pass", "Fail"} or not timestamp_value:
                    skipped += 1
                    continue

                try:
                    parsed_timestamp = datetime.fromisoformat(
                        timestamp_value.replace("Z", "+00:00")
                    )
                    if parsed_timestamp.tzinfo is None:
                        parsed_timestamp = parsed_timestamp.replace(tzinfo=timezone.utc)
                    timestamp = parsed_timestamp.astimezone(timezone.utc).isoformat()
                    confidence = float(confidence_value) if confidence_value else None
                    operator_id = (row.get("operator_id") or "").strip() or None
                except (ValueError, TypeError):
                    skipped += 1
                    continue

                values = (timestamp, status, confidence, operator_id)
                if inspection_id:
                    try:
                        source_id = int(inspection_id)
                    except ValueError:
                        skipped += 1
                        continue
                    cursor = connection.execute(
                        """
                        INSERT OR IGNORE INTO inspections (
                            inspection_id, timestamp, status, confidence_score, operator_id
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (source_id, *values),
                    )
                else:
                    cursor = connection.execute(
                        """
                        INSERT INTO inspections (
                            timestamp, status, confidence_score, operator_id
                        ) VALUES (?, ?, ?, ?)
                        """,
                        values,
                    )

                imported += cursor.rowcount
                skipped += 1 - cursor.rowcount

    return {"imported": imported, "skipped": skipped}