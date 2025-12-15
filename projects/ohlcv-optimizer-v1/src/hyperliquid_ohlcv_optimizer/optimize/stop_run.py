from __future__ import annotations

import argparse
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_run_dir(*, run: str | None, run_dir: str | None, latest: bool) -> Path:
    root = _project_root()
    runs_root = root / "runs"

    if run_dir:
        p = Path(run_dir)
        return p if p.is_absolute() else (runs_root / p)

    if latest:
        if not runs_root.exists():
            raise SystemExit(f"runs folder not found: {runs_root}")
        dirs = [p for p in runs_root.iterdir() if p.is_dir()]
        if not dirs:
            raise SystemExit(f"no runs found under: {runs_root}")
        dirs.sort(key=lambda p: p.name, reverse=True)
        return dirs[0]

    if run:
        return runs_root / run

    raise SystemExit("Provide --run, --run-dir, or --latest")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--latest", action="store_true")
    p.add_argument("--reason", default="stop")
    args = p.parse_args()

    run_dir = _resolve_run_dir(run=str(args.run) if args.run else None, run_dir=str(args.run_dir) if args.run_dir else None, latest=bool(args.latest))
    run_dir.mkdir(parents=True, exist_ok=True)

    stop_flag = run_dir / "stop.flag"
    stop_flag.write_text(str(args.reason), encoding="utf-8")
    print(f"Wrote stop flag: {stop_flag}")


if __name__ == "__main__":
    main()
