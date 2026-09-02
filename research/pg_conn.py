"""PostgreSQL conninfo for motivation scripts. Password comes from the environment."""
from __future__ import annotations

import os


def pg_conninfo(default_port: str = "55432") -> str:
    password = os.environ.get("PGPASSWORD")
    if not password:
        raise SystemExit("set PGPASSWORD")
    return (
        f"host={os.environ.get('PGHOST', '127.0.0.1')} "
        f"port={os.environ.get('PGPORT', default_port)} "
        f"dbname={os.environ.get('PGDATABASE', 'hybrid_vector')} "
        f"user={os.environ.get('PGUSER', 'postgres')} "
        f"password={password}"
    )
