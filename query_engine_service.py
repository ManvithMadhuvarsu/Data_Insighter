import os
import re
from typing import Any, Dict, Optional

import pandas as pd

from access_control_service import row_policies
from dataset_runtime import load_accessible_dataframe


FORBIDDEN_SQL_PATTERNS = (
    r'\bINSERT\b',
    r'\bUPDATE\b',
    r'\bDELETE\b',
    r'\bDROP\b',
    r'\bALTER\b',
    r'\bCREATE\b',
    r'\bATTACH\b',
    r'\bCOPY\b',
    r'\bPRAGMA\b',
    r'\bCALL\b',
)


def _load_duckdb():
    try:
        import duckdb
    except ImportError as exc:
        raise ValueError('DuckDB is not installed in this environment.') from exc
    return duckdb


def _sanitize_sql(sql: str) -> str:
    statement = (sql or '').strip().rstrip(';')
    if not statement:
        raise ValueError('Write a SQL query before running the workbench.')
    lowered = statement.lower()
    if not (lowered.startswith('select') or lowered.startswith('with')):
        raise ValueError('Only SELECT or WITH queries are allowed in the workbench.')
    for pattern in FORBIDDEN_SQL_PATTERNS:
        if re.search(pattern, statement, flags=re.IGNORECASE):
            raise ValueError('Only read-only SQL is allowed in the workbench.')
    return statement


def _relation_sql(filepath: str) -> Optional[str]:
    extension = os.path.splitext(filepath)[1].lower()
    safe_path = filepath.replace("'", "''")
    if extension == '.parquet':
        return f"read_parquet('{safe_path}')"
    if extension == '.csv':
        return f"read_csv_auto('{safe_path}', HEADER=TRUE)"
    if extension == '.tsv':
        return f"read_csv_auto('{safe_path}', HEADER=TRUE, delim='\\t')"
    if extension in {'.json', '.jsonl'}:
        return f"read_json_auto('{safe_path}')"
    return None


def execute_dataset_sql(
    dataset_record: Dict[str, Any],
    username: str,
    sql: str,
    limit: int = 200,
) -> Dict[str, Any]:
    duckdb = _load_duckdb()
    statement = _sanitize_sql(sql)
    limit = max(1, min(int(limit or 200), 2000))
    connection = duckdb.connect(database=':memory:')
    connection.execute("PRAGMA threads=4")

    filepath = dataset_record['stored_path']
    relation_sql = _relation_sql(filepath)
    use_dataframe = dataset_record.get('owner') != username and bool(row_policies(dataset_record))
    if relation_sql and not use_dataframe:
        connection.execute(f'CREATE OR REPLACE VIEW dataset AS SELECT * FROM {relation_sql}')
        engine = 'duckdb_file_scan'
    else:
        frame = load_accessible_dataframe(filepath, dataset_record, username)
        connection.register('dataset', frame)
        engine = 'duckdb_dataframe'

    preview = connection.execute(
        f"SELECT * FROM ({statement}) AS dataset_query LIMIT {limit}"
    ).fetchdf()
    total_rows = connection.execute(
        f"SELECT COUNT(*) AS row_count FROM ({statement}) AS dataset_query"
    ).fetchone()[0]

    preview = preview.where(pd.notna(preview), None)
    return {
        'engine': engine,
        'columns': preview.columns.tolist(),
        'rows': preview.to_dict(orient='records'),
        'row_count': int(total_rows),
        'returned_rows': int(len(preview)),
        'sql': statement,
    }
