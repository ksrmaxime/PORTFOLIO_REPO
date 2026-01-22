# scripts/fedlex_run_all.py
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from portfolio_repo.paths import find_repo_root


def run(cmd: list[str], cwd: Path) -> None:
    print("\n==>", " ".join(cmd))
    p = subprocess.run(cmd, cwd=str(cwd))
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--limit", type=int, default=None, help="Optional: only N laws for testing")
    ap.add_argument("--langs", default="de,fr,it")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--target-chars", type=int, default=10000)
    ap.add_argument("--overlap-chars", type=int, default=300)
    args = ap.parse_args()

    root = find_repo_root(args.repo_root)

    # Step 1
    cmd1 = [sys.executable, "scripts/fedlex_build_registry.py"]
    if args.limit:
        cmd1 += ["--limit", str(args.limit)]
    run(cmd1, cwd=root)

    # Step 2
    cmd2 = [
        sys.executable,
        "scripts/fedlex_download_xml.py",
        "--langs",
        args.langs,
    ]
    if args.limit:
        cmd2 += ["--max-laws", str(args.limit)]
    if args.resume:
        cmd2 += ["--resume"]
    run(cmd2, cwd=root)

    # Step 3
    cmd3 = [
        sys.executable,
        "scripts/fedlex_build_law_chunks.py",
        "--target-chars",
        str(args.target_chars),
        "--overlap-chars",
        str(args.overlap_chars),
        "--only-ok",
    ]
    run(cmd3, cwd=root)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
