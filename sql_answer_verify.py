#!/usr/bin/env python3
"""
quick_sqlite_runner.py
Run ad‑hoc SQL against chatbot22.db from VS Code.

Usage examples (from terminal or VS Code’s Run panel):

    python quick_sqlite_runner.py                     # runs the demo query
    python quick_sqlite_runner.py "SELECT count(*) FROM messages;"
    python quick_sqlite_runner.py "INSERT INTO logs(event) VALUES (?);" "User‑test"
"""

import sys
import sqlite3
from pathlib import Path
from typing import Sequence, Any, Optional

try:
    import pandas as pd   # Handy but optional
except ImportError:
    pd = None


DB_PATH = Path("chatbot22.db")      # adjust if the file lives elsewhere
DEFAULT_QUERY = """
SELECT g_ntee_letter, SUM(grant_amount) AS total_grant_amount FROM RRNAResearchGrants WHERE f_name LIKE '%Seva Foundation%' GROUP BY g_ntee_letter ORDER BY total_grant_amount DESC LIMIT 1
"""

def run_query(sql: str, params: Optional[Sequence[Any]] = None,
              *, as_df: bool = True):
    """
    Execute `sql` with optional parameters.
    If `as_df` is True and pandas is installed, return a DataFrame;
    otherwise return the raw cursor.fetchall() list.
    """
    params = params or ()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row  # get dict‑like rows
        cur = conn.cursor()
        cur.execute(sql, params)

        # For SELECT statements, fetch results
        if sql.lstrip().upper().startswith("SELECT"):
            rows = cur.fetchall()
            if as_df and pd:
                return pd.DataFrame(rows)        # nice tabular view
            return [dict(r) for r in rows]       # plain Python list

        # For modifying statements, commit and report rows affected
        conn.commit()
        return f"{cur.rowcount} row(s) affected."


def main(argv: Sequence[str] = sys.argv[1:]):
    """
    If the user passes arguments, treat arg[0] as SQL and the rest as parameters.
    Otherwise run DEFAULT_QUERY.
    """
    if argv:
        sql = argv[0]
        params = argv[1:]
    else:
        sql = DEFAULT_QUERY
        params = ()

    result = run_query(sql, params)
    if pd and isinstance(result, pd.DataFrame):
        # Pretty‑print DataFrame to console
        print(result.to_string(index=False))
    else:
        print(result)


if __name__ == "__main__":
    main()
