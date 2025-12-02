#!/usr/bin/env python3
"""Unpack arXiv source trees into a flat directory for ingestion.

This script scans an input directory for ``.gz`` source archives and ``.pdf``
files. The archives are expanded into per-paper folders under the output root,
mirroring the relative directory structure of the input tree. PDFs are copied
verbatim to the matching location in the output tree.
"""
from __future__ import annotations

import argparse
import gzip
import os
import shutil
import tarfile
import zlib
from pathlib import Path
from typing import Iterable


def iter_sources(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in {".gz", ".pdf"}:
            yield path


def ensure_removed(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def safe_extract(tar: tarfile.TarFile, target: Path) -> None:
    target = target.resolve()
    target_str = str(target)
    for member in tar.getmembers():
        member_path = (target / member.name).resolve()
        member_str = str(member_path)
        if not (
            member_str == target_str
            or member_str.startswith(target_str + os.sep)
        ):
            raise RuntimeError(f"Unsafe member path: {member.name}")
    tar.extractall(path=target)


def handle_pdf(
    source: Path,
    input_root: Path,
    dest_root: Path,
    overwrite: bool,
    verbose: bool,
) -> bool:
    dest = dest_root / source.relative_to(input_root)
    if dest.exists():
        if not overwrite:
            if verbose:
                print(f"[skip] {source} -> {dest} (exists)")
            return False
        ensure_removed(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    if verbose:
        print(f"[copy] {source} -> {dest}")
    return True


def handle_gz(
    source: Path,
    input_root: Path,
    dest_root: Path,
    overwrite: bool,
    verbose: bool,
    failures: list[str],
) -> str:
    rel = source.relative_to(input_root)
    dest = (dest_root / rel).with_suffix("")
    if dest.suffix == ".tar":
        dest = dest.with_suffix("")
    if dest.exists():
        if not overwrite:
            if verbose:
                print(f"[skip] {source} -> {dest} (exists)")
            return "skip"
        ensure_removed(dest)
    tar_error: BaseException | None = None
    try:
        with tarfile.open(source, "r:*") as tar:
            dest.mkdir(parents=True, exist_ok=True)
            safe_extract(tar, dest)
            if verbose:
                print(f"[extract] {source} -> {dest}")
            return "extract"
    except (tarfile.TarError, EOFError, zlib.error, OSError) as exc:
        tar_error = exc
        if dest.exists():
            ensure_removed(dest)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(source, "rb") as comp, open(dest, "wb") as raw:
            shutil.copyfileobj(comp, raw)
        if verbose:
            if tar_error:
                print(f"[gunzip] {source} -> {dest} (tar fallback: {tar_error})")
            else:
                print(f"[gunzip] {source} -> {dest}")
        return "gunzip"
    except (OSError, zlib.error, EOFError) as exc:
        if dest.exists():
            ensure_removed(dest)
        reason = tar_error or exc
        failures.append(f"{source}: {reason}")
        if verbose:
            print(f"[error] {source}: {reason}")
        return "error"


def main(args: argparse.Namespace) -> None:
    total_copied = total_extracted = total_gunzipped = total_failed = 0
    verbose = not args.quiet
    failures: list[str] = []
    for source in iter_sources(args.input):
        if source.suffix.lower() == ".pdf":
            if handle_pdf(source, args.input, args.output, args.overwrite, verbose):
                total_copied += 1
        else:
            action = handle_gz(
                source,
                args.input,
                args.output,
                args.overwrite,
                verbose,
                failures,
            )
            if action == "extract":
                total_extracted += 1
            elif action == "gunzip":
                total_gunzipped += 1
            elif action == "error":
                total_failed += 1
    print(
        "Done. copied={copied} extracted={extracted} gunzipped={gunzipped} failed={failed}".format(
            copied=total_copied,
            extracted=total_extracted,
            gunzipped=total_gunzipped,
            failed=total_failed,
        )
    )
    if failures and verbose:
        print("Failures:")
        for item in failures:
            print(f"  - {item}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unpack arXiv sources")
    parser.add_argument("--input", required=True, type=Path, help="Input directory")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-file log messages",
    )
    args = parser.parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input directory {args.input} does not exist")
    args.output.mkdir(parents=True, exist_ok=True)
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
