"""在有 git 的机器上盖章推理代码版本，供远程无 .git 环境读取。

用法::

    python3 -m infer.scripts.stamp_git_revision
    python3 -m infer.scripts.stamp_git_revision --repo /path/to/repo

生成 ``infer/INFER_GIT_REVISION.json`` 后，随代码一起 rsync 到推理端即可。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stamp infer git revision for remote machines")
    parser.add_argument(
        "--repo",
        type=Path,
        default=None,
        help="仓库根目录（默认：本文件所在仓库）",
    )
    args = parser.parse_args(argv)

    # 保证从任意 cwd 都能 import infer
    repo_root = args.repo
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from infer.recording.git_info import format_infer_git_info, stamp_infer_git_revision

    try:
        info = stamp_infer_git_revision(repo_root)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"[OK] stamped: {info.get('infer_git_stamp_path')}")
    print(f"[OK] {format_infer_git_info(info)}")
    print(json.dumps({k: v for k, v in info.items() if k != "infer_git_stamp_path"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
