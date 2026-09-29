"""SQLite layer: builds the database from the cleaned CSVs and runs validated, read-only SQL."""
import re
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "fitbit.db"
if not DB_PATH.exists():  # read-only hosts: fall back to a temp location
    try:
        (ROOT / "data").mkdir(exist_ok=True)
        (ROOT / "data" / ".w").touch(); (ROOT / "data" / ".w").unlink()
    except OSError:
        import tempfile
        DB_PATH = Path(tempfile.gettempdir()) / "fitbit.db"
BLOCKED = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|vacuum)\b", re.I)


class QueryError(ValueError):
    """Raised when a query is unsafe or fails."""


def build_db(force: bool = False) -> Path:
    """Create data/fitbit.db (tables: daily, hourly) from the merged CSVs."""
    if DB_PATH.exists() and not force:
        return DB_PATH
    try:
        daily = pd.read_csv(ROOT / "data" / "fitbit_daily_merged.csv", parse_dates=["Date"])
        hourly = pd.read_csv(ROOT / "data" / "fitbit_hourly_merged.csv", parse_dates=["Hour"])
    except FileNotFoundError as e:
        raise RuntimeError(f"Missing data file: {e.filename}. Put the merged CSVs in /data.") from e
    with sqlite3.connect(DB_PATH) as con:
        daily.to_sql("daily", con, if_exists="replace", index=False)
        hourly.to_sql("hourly", con, if_exists="replace", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS ix_daily ON daily(Id, Date)")
        con.execute("CREATE INDEX IF NOT EXISTS ix_hourly ON hourly(Id, Hour)")
    return DB_PATH


def validate(sql: str) -> str:
    """Allow a single read-only SELECT/WITH statement."""
    s = (sql or "").strip().rstrip(";").strip()
    if not s:
        raise QueryError("Query is empty.")
    if ";" in s:
        raise QueryError("Only one statement is allowed.")
    if not re.match(r"^(select|with)\b", s, re.I):
        raise QueryError("Only SELECT / WITH queries are allowed.")
    if BLOCKED.search(s):
        raise QueryError("Write or admin keywords are not allowed (read-only mode).")
    return s


def run_query(sql: str, limit: int = 5000) -> pd.DataFrame:
    """Run validated SQL on a read-only connection."""
    s = validate(sql)
    try:
        with sqlite3.connect(f"file:{build_db()}?mode=ro", uri=True) as con:
            return pd.read_sql_query(s, con).head(limit)
    except sqlite3.Error as e:
        raise QueryError(f"SQL error: {e}") from e
