#!/usr/bin/env python3
"""Online clone of the Amazon-10M PGDATA for the r44 replica.

Keeps one session open from pg_backup_start to pg_backup_stop. Does not
change 55437 GUCs, pg_hba, or shared_buffers. Source stays mounted.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import os
from pathlib import Path

EXCLUDE = (
    "postmaster.pid",
    "postmaster.opts",
    "pg_dynshmem/***",
    "pg_notify/***",
    "pg_serial/***",
    "pg_snapshots/***",
    "pg_stat_tmp/***",
    "pg_subtrans/***",
)


def _rsync(src: Path, dest: Path, *, extra: list[str] | None = None) -> None:
    cmd = [
        "sudo",
        "-n",
        "rsync",
        "-aH",
        "--numeric-ids",
        "--delete",
        "--info=progress2",
    ]
    for pattern in EXCLUDE:
        cmd.extend(["--exclude", pattern])
    if extra:
        cmd.extend(extra)
    cmd.extend([f"{src}/", f"{dest}/"])
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _write(path: Path, text: str) -> None:
    subprocess.run(
        ["sudo", "-n", "tee", str(path)],
        input=text.encode("utf-8"),
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(["sudo", "-n", "chown", "999:999", str(path)], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=None)
    parser.add_argument("--dest", type=Path, default=None)
    parser.add_argument("--pgport", type=int, default=55437)
    args = parser.parse_args()

    try:
        src = args.src or Path(os.environ["TABLE10_PGDATA"])
        dest = args.dest or Path(os.environ["R44_PGDATA"])
        password = os.environ["PGPASSWORD"]
    except KeyError as exc:
        print(f"set {exc.args[0]}", file=sys.stderr)
        return 2

    import psycopg

    src_ok = subprocess.run(
        ["sudo", "-n", "test", "-f", str(src / "PG_VERSION")]
    ).returncode == 0
    if not src_ok:
        print(f"source PGDATA missing: {src}", file=sys.stderr)
        return 2
    subprocess.run(["sudo", "-n", "mkdir", "-p", str(dest)], check=True)
    subprocess.run(["sudo", "-n", "chown", "999:999", str(dest)], check=True)
    subprocess.run(["sudo", "-n", "chmod", "700", str(dest)], check=True)

    conninfo = (
        f"host=127.0.0.1 port={args.pgport} dbname=hybrid_vector "
        "user=%s password=%s" % (os.environ.get("PGUSER", "postgres"), password)
    )
    with psycopg.connect(conninfo, autocommit=True) as conn:
        cur = conn.cursor()
        cur.execute("SELECT pg_is_in_recovery()")
        if cur.fetchone()[0]:
            print("source is in recovery; refuse clone", file=sys.stderr)
            return 2
        print("BACKUP_START", flush=True)
        cur.execute("SELECT pg_backup_start(%s, true)", ("r44-clone",))
        start_lsn = cur.fetchone()[0]
        print(f"start_lsn={start_lsn}", flush=True)
        try:
            _rsync(src, dest)
        except Exception:
            try:
                cur.execute("SELECT pg_backup_stop(false)")
            except Exception:
                pass
            raise
        cur.execute("SELECT lsn, labelfile, spcmapfile FROM pg_backup_stop(false)")
        stop_lsn, labelfile, spcmapfile = cur.fetchone()
        print(f"stop_lsn={stop_lsn}", flush=True)
    _write(dest / "backup_label", labelfile)
    if spcmapfile:
        _write(dest / "tablespace_map", spcmapfile)
    print("RSYNC_WAL", flush=True)
    wal_cmd = [
        "sudo",
        "-n",
        "rsync",
        "-aH",
        "--numeric-ids",
        "--info=progress2",
        f"{src}/pg_wal/",
        f"{dest}/pg_wal/",
    ]
    print("RUN", " ".join(wal_cmd), flush=True)
    subprocess.run(wal_cmd, check=True)
    subprocess.run(["sudo", "-n", "rm", "-f", str(dest / "postmaster.pid")], check=True)
    for name in (
        "pg_dynshmem",
        "pg_notify",
        "pg_serial",
        "pg_snapshots",
        "pg_stat_tmp",
        "pg_subtrans",
    ):
        path = dest / name
        subprocess.run(["sudo", "-n", "mkdir", "-p", str(path)], check=True)
        subprocess.run(["sudo", "-n", "chown", "999:999", str(path)], check=True)
        subprocess.run(["sudo", "-n", "chmod", "700", str(path)], check=True)
    version = subprocess.check_output(
        ["sudo", "-n", "cat", str(dest / "PG_VERSION")], text=True
    ).strip()
    if version != "16":
        print(f"cloned major {version} != 16", file=sys.stderr)
        return 1
    print(f"cloned:{dest}:pg{version}:start={start_lsn}:stop={stop_lsn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
