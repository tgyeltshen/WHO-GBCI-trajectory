"""Single entry point for Project 3."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone

from study import (
    LOGS,
    analyse_all,
    download_public_data,
    ensure_directories,
    process_all,
    render_outputs,
)


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {message}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["download", "process", "analyse", "report", "all"], nargs="?", default="all")
    args = parser.parse_args()
    ensure_directories()
    logfile = LOGS / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
    frames = {}
    analyses = {}
    try:
        if args.stage in {"download", "all"}:
            log("Downloading and verifying public source snapshots")
            paths = download_public_data()
            log(f"Verified {len(paths)} raw files")
        if args.stage in {"process", "analyse", "report", "all"}:
            log("Processing source data and building audits")
            frames = process_all()
        if args.stage in {"analyse", "report", "all"}:
            log("Running analyses with available required inputs")
            analyses = analyse_all(frames)
        if args.stage in {"report", "all"}:
            log("Rendering tables, figures, and execution report")
            render_outputs(frames, analyses)
        message = f"Stage {args.stage} completed"
        logfile.write_text(message + "\n", encoding="utf-8")
        log(message)
        return 0
    except Exception as exc:
        detail = "".join(traceback.format_exception(exc))
        logfile.write_text(detail, encoding="utf-8")
        print(detail, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
