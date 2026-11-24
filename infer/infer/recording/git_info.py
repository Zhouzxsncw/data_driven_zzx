"""采集推理代码版本信息，写入 meta 便于复现。

远程推理端常无 ``.git``（rsync 常 ``--exclude '.git/'``），因此采用回退链：

1. 环境变量 ``INFER_GIT_COMMIT``（可选 ``INFER_GIT_BRANCH`` / ``INFER_GIT_DIRTY``）
2. 本机 ``git``（成功时顺带刷新 stamp 文件，方便下次同步到远程）
3. ``infer/INFER_GIT_REVISION.json`` stamp（随代码一起拷到远程）
4. 字段为 None（``infer_git_source=unknown``）

部署前可显式盖章::

    python3 -m infer.scripts.stamp_git_revision
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# infer/recording/git_info.py → parents[2] = 仓库根目录
_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
STAMP_RELATIVE_PATH = Path("infer") / "INFER_GIT_REVISION.json"

_ENV_COMMIT = "INFER_GIT_COMMIT"
_ENV_BRANCH = "INFER_GIT_BRANCH"
_ENV_DIRTY = "INFER_GIT_DIRTY"


def stamp_path(repo_root: Optional[Path] = None) -> Path:
    root = Path(repo_root) if repo_root is not None else _DEFAULT_REPO_ROOT
    return root / STAMP_RELATIVE_PATH


def _empty_info(*, source: str = "unknown") -> Dict[str, Any]:
    return {
        "infer_git_commit": None,
        "infer_git_commit_short": None,
        "infer_git_branch": None,
        "infer_git_dirty": None,
        "infer_git_source": source,
    }


def _with_short(commit: Optional[str], **extra: Any) -> Dict[str, Any]:
    short = commit[:7] if commit else None
    out: Dict[str, Any] = {
        "infer_git_commit": commit,
        "infer_git_commit_short": short,
        "infer_git_branch": extra.get("infer_git_branch"),
        "infer_git_dirty": extra.get("infer_git_dirty"),
        "infer_git_source": extra.get("infer_git_source", "unknown"),
    }
    return out


def _run_git(repo_root: Path, *args: str, timeout: float = 2.0) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("git %s failed: %s", " ".join(args), exc)
        return None
    if completed.returncode != 0:
        logger.debug(
            "git %s exit=%s stderr=%s",
            " ".join(args),
            completed.returncode,
            (completed.stderr or "").strip(),
        )
        return None
    return (completed.stdout or "").strip() or None


def _parse_dirty_env(raw: Optional[str]) -> Optional[bool]:
    if raw is None or raw.strip() == "":
        return None
    val = raw.strip().lower()
    if val in ("1", "true", "yes", "dirty"):
        return True
    if val in ("0", "false", "no", "clean"):
        return False
    return None


def _from_env() -> Optional[Dict[str, Any]]:
    commit = (os.environ.get(_ENV_COMMIT) or "").strip() or None
    if not commit:
        return None
    branch = (os.environ.get(_ENV_BRANCH) or "").strip() or None
    dirty = _parse_dirty_env(os.environ.get(_ENV_DIRTY))
    return _with_short(
        commit,
        infer_git_branch=branch,
        infer_git_dirty=dirty,
        infer_git_source="env",
    )


def _from_git(repo_root: Path) -> Optional[Dict[str, Any]]:
    # 无 .git 时跳过，避免无意义的 git 报错刷屏
    if not (repo_root / ".git").exists():
        return None
    commit = _run_git(repo_root, "rev-parse", "HEAD")
    if not commit:
        return None
    branch = _run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    dirty: Optional[bool] = None
    status = _run_git(repo_root, "status", "--porcelain")
    if status is not None:
        dirty = bool(status)
    return _with_short(
        commit,
        infer_git_branch=branch,
        infer_git_dirty=dirty,
        infer_git_source="git",
    )


def _from_stamp(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("读取 git stamp 失败 (%s): %s", path, exc)
        return None
    if not isinstance(payload, dict):
        return None
    commit = payload.get("infer_git_commit") or payload.get("commit")
    if not isinstance(commit, str) or not commit.strip():
        return None
    commit = commit.strip()
    branch = payload.get("infer_git_branch") or payload.get("branch")
    dirty = payload.get("infer_git_dirty")
    if dirty is not None and not isinstance(dirty, bool):
        dirty = _parse_dirty_env(str(dirty))
    return _with_short(
        commit,
        infer_git_branch=branch if isinstance(branch, str) else None,
        infer_git_dirty=dirty if isinstance(dirty, bool) else None,
        infer_git_source="stamp",
    )


def write_infer_git_stamp(
    info: Dict[str, Any],
    *,
    repo_root: Optional[Path] = None,
    path: Optional[Path] = None,
) -> Optional[Path]:
    """把版本信息写入 stamp 文件；失败返回 None，不抛异常。"""
    commit = info.get("infer_git_commit")
    if not commit:
        return None
    out = path if path is not None else stamp_path(repo_root)
    payload = {
        "infer_git_commit": commit,
        "infer_git_commit_short": info.get("infer_git_commit_short") or str(commit)[:7],
        "infer_git_branch": info.get("infer_git_branch"),
        "infer_git_dirty": info.get("infer_git_dirty"),
        "stamped_at": datetime.now().astimezone().isoformat(),
        "stamped_from": info.get("infer_git_source") or "manual",
    }
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        logger.warning("写入 git stamp 失败 (%s): %s", out, exc)
        return None
    return out


def stamp_infer_git_revision(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """从本机 git 盖章；无 git 时抛 RuntimeError。"""
    root = Path(repo_root) if repo_root is not None else _DEFAULT_REPO_ROOT
    info = _from_git(root)
    if info is None:
        raise RuntimeError(
            f"无法从 {root} 读取 git（无 .git 或 git 不可用）。"
            f"远程机请先在有仓库的机器上运行: python3 -m infer.scripts.stamp_git_revision"
        )
    written = write_infer_git_stamp(info, repo_root=root)
    if written is None:
        raise RuntimeError(f"git 可读但无法写入 stamp: {stamp_path(root)}")
    info = dict(info)
    info["infer_git_stamp_path"] = str(written)
    return info


def get_infer_git_info(
    repo_root: Optional[Path] = None,
    *,
    refresh_stamp: bool = True,
) -> Dict[str, Any]:
    """返回推理代码版本元数据；失败时字段为 None，不抛异常。

    Keys:
      infer_git_commit / infer_git_commit_short / infer_git_branch / infer_git_dirty
      infer_git_source: env | git | stamp | unknown
    """
    root = Path(repo_root) if repo_root is not None else _DEFAULT_REPO_ROOT

    info = _from_env()
    if info is not None:
        return info

    info = _from_git(root)
    if info is not None:
        if refresh_stamp:
            write_infer_git_stamp(info, repo_root=root)
        return info

    info = _from_stamp(stamp_path(root))
    if info is not None:
        return info

    return _empty_info(source="unknown")


def format_infer_git_info(info: Optional[Dict[str, Any]] = None) -> str:
    """人类可读的一行摘要，供 run.log 打印。"""
    data = info if info is not None else get_infer_git_info()
    commit = data.get("infer_git_commit_short") or data.get("infer_git_commit") or "unknown"
    branch = data.get("infer_git_branch") or "?"
    dirty = data.get("infer_git_dirty")
    if dirty is True:
        status = "dirty"
    elif dirty is False:
        status = "clean"
    else:
        status = "unknown"
    source = data.get("infer_git_source") or "unknown"
    return f"sha={commit} branch={branch} status={status} source={source}"
