"""Build dashboard.duckdb from the parquet file."""

import duckdb
from pathlib import Path

PARQUET_PATH = Path(__file__).parent.parent.parent / "notebooks" / "data" / "dashboard_all_in_one.parquet"
DB_PATH = Path(__file__).parent.parent / "data" / "dashboard.duckdb"


def main():
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"Parquet not found: {PARQUET_PATH}")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Delete old file to avoid bloat (DuckDB doesn't reclaim space from DROP TABLE)
    if DB_PATH.exists():
        DB_PATH.unlink()

    con = duckdb.connect(str(DB_PATH))
    con.execute(f"CREATE TABLE dashboard AS SELECT * FROM read_parquet('{PARQUET_PATH}')")

    count = con.execute("SELECT COUNT(*) FROM dashboard").fetchone()[0]
    cols = con.execute(
        "SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'dashboard'"
    ).fetchone()[0]
    con.close()

    print(f"Built {DB_PATH}: {count:,} rows, {cols} columns")


if __name__ == "__main__":
    main()
