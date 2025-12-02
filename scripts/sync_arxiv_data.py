#!/usr/bin/env python3
"""Incremental同步 arXiv 数据集的辅助脚本。

特点：
- 支持选择同步 latex 源文件 (src) 与 pdf。
- 使用 rsync 的 --partial/--append-verify 选项，可断点续传。
- 默认源目录为已经挂载到本机的 `/mnt/data/corpus/DocAI/arXiv_2023-12`。

示例：
  python scripts/sync_arxiv_data.py --components src pdf
  python scripts/sync_arxiv_data.py --remote /mnt/data/corpus/DocAI/arXiv_2023-12 \
    --target-root data --components src --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

DEFAULT_REMOTE_ROOT = Path("/mnt/data/corpus/DocAI/arXiv_2023-12")
DEFAULT_TARGET_ROOT = Path("data")
VALID_COMPONENTS = {
    "src": "arXiv_2023-12_src",
    "pdf": "arXiv_2023-12_pdf",
}


def build_rsync_command(source: Path, target: Path, dry_run: bool, extra_opts: Iterable[str]) -> List[str]:
    cmd = [
        "rsync",
        "-avh",
        "--info=progress2",
        "--partial",
        "--append-verify",
        "--no-inc-recursive",
    ]
    cmd.extend(extra_opts)
    if dry_run:
        cmd.append("--dry-run")
    cmd.extend([f"{source}/", str(target)])
    return cmd


def ensure_prerequisites(remote_root: Path, components: Iterable[str]) -> None:
    if shutil.which("rsync") is None:
        raise SystemExit("rsync 未安装，请先执行：sudo apt-get install rsync")

    if not remote_root.exists():
        raise SystemExit(f"远程目录 {remote_root} 不存在，请确认挂载可用。")

    for comp in components:
        subdir = remote_root / comp
        if not subdir.exists():
            raise SystemExit(f"未找到 {comp} 数据目录：{subdir}")


def sync_components(remote_root: Path, target_root: Path, components: Iterable[str], dry_run: bool) -> None:
    extra_opts = ["--delete-delay", "--prune-empty-dirs"]
    for comp in components:
        subdir_name = VALID_COMPONENTS[comp]
        source = remote_root / comp
        target = target_root / subdir_name
        target.mkdir(parents=True, exist_ok=True)

        cmd = build_rsync_command(source, target, dry_run=dry_run, extra_opts=extra_opts)
        print("➜ 同步", comp, "->", target)
        print(" ", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"同步 {comp} 失败，退出码 {exc.returncode}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="增量同步 arXiv 数据集")
    parser.add_argument(
        "--remote",
        type=Path,
        default=DEFAULT_REMOTE_ROOT,
        help="已挂载的远程根目录 (包含 src/ 与 pdf/)",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=DEFAULT_TARGET_ROOT,
        help="同步到的本地根目录 (默认 data)",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        choices=sorted(VALID_COMPONENTS.keys()),
        default=["src"],
        help="需要同步的组件，可多选 (默认: src)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅查看将执行的操作，不实际复制",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    components = args.components

    ensure_prerequisites(args.remote, components)
    sync_components(args.remote, args.target_root, components, dry_run=args.dry_run)

    print("✅ 同步完成" if not args.dry_run else "ℹ️ dry-run 完成，未实际写入")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(exc.code if isinstance(exc.code, int) else 1)