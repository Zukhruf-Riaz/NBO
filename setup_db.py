"""
setup_db.py
-----------
One-time script: reads all CSV files from the Tables/ folder
and loads them into a local SQLite database (nbo_ads.db).

Run once before starting the agent:
    python setup_db.py
"""

import sqlite3
import csv
import os
import glob

# ── Configuration ────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR = os.path.join(BASE_DIR, "Tables")
DB_PATH    = os.path.join(BASE_DIR, "nbo_ads.db")
# ─────────────────────────────────────────────────────────────────────────────


def infer_col_type(values: list[str]) -> str:
    """
    Peek at up to 100 non-empty sample values and decide whether
    the column should be stored as REAL, INTEGER, or TEXT.
    """
    samples = [v for v in values if v.strip() != ""][:100]
    if not samples:
        return "TEXT"

    int_count  = 0
    real_count = 0
    for v in samples:
        try:
            int(v)
            int_count += 1
            real_count += 1
        except ValueError:
            try:
                float(v)
                real_count += 1
            except ValueError:
                pass

    ratio = len(samples)
    if int_count == ratio:
        return "INTEGER"
    if real_count == ratio:
        return "REAL"
    return "TEXT"


def csv_to_table(conn: sqlite3.Connection, csv_path: str) -> str:
    """
    Load a single CSV file into a SQLite table.
    Table name = CSV filename without extension (lowercased).
    Returns the table name on success.
    """
    table_name = os.path.splitext(os.path.basename(csv_path))[0].lower()
    # Replace spaces / hyphens in file names with underscores
    table_name = table_name.replace(" ", "_").replace("-", "_")

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        headers = next(reader)
        # Strip whitespace from column names and make lowercase
        headers = [h.strip().lower().replace(" ", "_") for h in headers]

        rows = list(reader)

    if not rows:
        print(f"  [SKIP] {table_name} — no data rows")
        return table_name

    # Infer column types from first 100 rows
    col_types = []
    for col_idx in range(len(headers)):
        col_values = [row[col_idx] if col_idx < len(row) else "" for row in rows[:100]]
        col_types.append(infer_col_type(col_values))

    # Build CREATE TABLE statement
    col_defs = ", ".join(
        f'"{h}" {t}' for h, t in zip(headers, col_types)
    )
    create_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({col_defs});'

    cursor = conn.cursor()
    # Drop existing table so re-runs are idempotent
    cursor.execute(f'DROP TABLE IF EXISTS "{table_name}";')
    cursor.execute(create_sql)

    # Insert rows — handle rows that are shorter than the header
    placeholders = ", ".join("?" * len(headers))
    insert_sql = f'INSERT INTO "{table_name}" VALUES ({placeholders});'

    clean_rows = []
    for row in rows:
        # Pad short rows, truncate long rows
        padded = (row + [""] * len(headers))[: len(headers)]
        # Convert empty strings to None so SQLite stores NULL
        padded = [None if v.strip() == "" else v.strip() for v in padded]
        clean_rows.append(padded)

    cursor.executemany(insert_sql, clean_rows)
    conn.commit()

    print(f"  [OK] {table_name:<30} {len(rows):>7,} rows  |  {len(headers)} columns")
    return table_name


def main():
    # Discover all CSV files in Tables/
    csv_files = sorted(glob.glob(os.path.join(TABLES_DIR, "*.csv")))
    if not csv_files:
        print(f"ERROR: No CSV files found in {TABLES_DIR}")
        return

    print(f"\nDatabase : {DB_PATH}")
    print(f"Source   : {TABLES_DIR}")
    print(f"Files    : {len(csv_files)} CSV files found\n")
    print("-" * 60)

    conn = sqlite3.connect(DB_PATH)

    loaded_tables = []
    for csv_path in csv_files:
        try:
            t = csv_to_table(conn, csv_path)
            loaded_tables.append(t)
        except Exception as exc:
            print(f"  [ERR] {os.path.basename(csv_path)} — {exc}")

    conn.close()

    print("-" * 60)
    print(f"\nDone. {len(loaded_tables)} tables loaded into {DB_PATH}\n")
    print("Tables available:")
    for t in loaded_tables:
        print(f"  - {t}")
    print("\nYou can now run:  python ads_agent.py\n")


if __name__ == "__main__":
    main()
