#!/usr/bin/env python3
"""Bulk extract .tar archives containing PDFs/LaTeX sources.

Usage:
    python scripts/extract_archives.py --input /root/Copyless/data/pdf \
        --output /root/Copyless/data/pdf_extracted

The script scans the input directory recursively for ``.tar`` archives and
extracts each archive into a dedicated subdirectory under the output path. The
folder name is derived from the archive filename (without the ``.tar`` suffix).
Existing folders will be skipped unless ``--overwrite`` is supplied.
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


def extract_archive(archive: Path, output_root: Path, overwrite: bool = False) -> None:
    target_dir = output_root / archive.stem
    if target_dir.exists():
        if not overwrite:
            print(f"[skip] {archive} -> {target_dir} (already exists)")
            return
    else:
        target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r") as tar:
        tar.extractall(path=target_dir)
    print(f"[done] {archive} -> {target_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract .tar archives for Copyless pipeline")
    parser.add_argument("--input", required=True, type=Path, help="Directory containing .tar archives")
    parser.add_argument(
        "--output", required=True, type=Path, help="Directory to store extracted files"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract archives even if target directory exists",
    )
    args = parser.parse_args()

    input_dir: Path = args.input
    output_dir: Path = args.output

    if not input_dir.exists():
        raise SystemExit(f"Input directory {input_dir} does not exist")
    output_dir.mkdir(parents=True, exist_ok=True)

    archives = sorted(p for p in input_dir.rglob("*.tar") if p.is_file())
    if not archives:
        print(f"No .tar archives found under {input_dir}")
        return

    for archive in archives:
        try:
            extract_archive(archive, output_dir, overwrite=args.overwrite)
        except tarfile.TarError as exc:
            print(f"[error] Failed to extract {archive}: {exc}")


if __name__ == "__main__":
    main()
