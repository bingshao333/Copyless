#!/usr/bin/env python3
"""Copy article folders from the unpacked arXiv source tree into a sample subset.

This helper increments the contents of a "sample" tree until it reaches the
requested number of article directories (depth=3) without deleting any existing
content. It simply preserves the relative path inside the unpacked tree, so it
works for any subject prefix (not only `cs*`).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterable, Set


def iter_article_dirs(root: Path) -> Iterable[Path]:
    """Yield relative paths for depth=3 article directories under *root*."""
    for batch in sorted(root.iterdir()):
        if not batch.is_dir():
            continue
        for shard in sorted(batch.iterdir()):
            if not shard.is_dir():
                continue
            for paper in sorted(shard.iterdir()):
                if paper.is_dir():
                    yield paper.relative_to(root)


def count_existing(dest: Path) -> tuple[int, Set[Path]]:
    rel_paths: Set[Path] = set()
    total = 0
    for rel in iter_article_dirs(dest):
        rel_paths.add(rel)
        total += 1
    return total, rel_paths


def copy_missing(src: Path, dest: Path, target: int, report_every: int = 100) -> int:
    existing_count, existing_rel = count_existing(dest)
    if existing_count >= target:
        print(f"Already at target: existing={existing_count} ≥ target={target}")
        return 0

    needed = target - existing_count
    copied = 0
    skipped = 0

    for rel in iter_article_dirs(src):
        if rel in existing_rel:
            skipped += 1
            continue
        src_dir = src / rel
        dest_dir = dest / rel
        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copytree(src_dir, dest_dir)
        except FileExistsError:
            continue
        except Exception as exc:  # pragma: no cover - best-effort copy
            print(f"[error] failed to copy {src_dir} -> {dest_dir}: {exc}", file=sys.stderr)
            continue
        existing_rel.add(rel)
        copied += 1
        if copied % report_every == 0:
            print(
                f"Progress: copied {copied} / {needed} (total {existing_count + copied}/{target})"
            )
        if copied >= needed:
            break

    print(
        f"Done. copied={copied} skipped={skipped} final_total={existing_count + copied}"
    )
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grow sample subset of arXiv sources")
    parser.add_argument("--src", type=Path, required=True, help="Unpacked source root")
    parser.add_argument(
        "--dst", type=Path, required=True, help="Destination/sample root to grow"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=10_000,
        help="Desired number of article folders in the destination (default: 10000)",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=200,
        help="Print progress after copying this many folders (default: 200)",
    )
    args = parser.parse_args()
    if not args.src.exists():
        parser.error(f"Source root {args.src} does not exist")
    args.dst.mkdir(parents=True, exist_ok=True)
    return args


def main() -> None:
    args = parse_args()
    copy_missing(args.src, args.dst, args.target, args.report_every)


if __name__ == "__main__":
    main()
