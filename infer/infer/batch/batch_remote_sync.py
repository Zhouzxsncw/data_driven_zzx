"""批量推理结束后，将实验目录通过 SSH 私钥同步到远程服务器。"""

from __future__ import annotations

import logging
import os
import posixpath
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from infer.config_loader import BatchRemoteSyncConfig

if TYPE_CHECKING:
    from infer.ui.pedal_input import PedalInputHub

logger = logging.getLogger(__name__)

_DEFAULT_PRIVATE_KEY = Path("~/.ssh/id_rsa")
_PRETRAINED_MODEL_DIR = "pretrained_model"
_BATCH_STAGING_DIR = "_batch_staging"
_MAX_SYNC_FILE_BYTES = 1024**3  # 1 GiB，单文件超过则不上传


def _is_under_dir(path: Path, dir_name: str) -> bool:
    return dir_name in path.parts


def _is_under_pretrained_model(path: Path) -> bool:
    return _is_under_dir(path, _PRETRAINED_MODEL_DIR)


def skip_sync_reason(path: Path) -> Optional[str]:
    """
    若文件不应上传，返回跳过原因；否则返回 None。

    规则:
    - pretrained_model/ 仅 .json
    - 任意路径下单文件 > 1 GiB 跳过（如 training_state/optimizer_state.safetensors）
    - training_state/ 内 .json 等小文件正常上传
    - _batch_staging/ 为推理临时目录，不上传
    """
    if _is_under_dir(path, _BATCH_STAGING_DIR):
        return "batch_staging"
    if _is_under_pretrained_model(path):
        if path.suffix.lower() == ".json":
            return None
        return "pretrained_model_non_json"
    try:
        size = path.stat().st_size
    except OSError:
        return "stat_failed"
    if size > _MAX_SYNC_FILE_BYTES:
        return "file_too_large"
    return None


def should_sync_file(path: Path) -> bool:
    """远程同步文件筛选。"""
    return skip_sync_reason(path) is None


def collect_sync_files(local_dir: Path) -> Tuple[List[Path], List[Path]]:
    """扫描待上传/跳过的文件列表（绝对路径）。"""
    local_dir = local_dir.expanduser().resolve()
    to_upload: List[Path] = []
    skipped: List[Path] = []
    for root, _, files in os.walk(local_dir):
        for name in files:
            path = Path(root) / name
            if should_sync_file(path):
                to_upload.append(path)
            else:
                skipped.append(path)
    to_upload.sort()
    skipped.sort()
    return to_upload, skipped


def _format_rel_path(local_dir: Path, path: Path) -> str:
    return path.relative_to(local_dir).as_posix()


def _human_bytes(n: int) -> str:
    size = float(max(0, n))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def _upload_bucket(rel_path: str) -> str:
    """按目录归类，便于终端摘要展示。"""
    parts = Path(rel_path).parts
    if _PRETRAINED_MODEL_DIR in parts:
        return "pretrained_model/（仅 .json）"
    for part in parts:
        if part.startswith("action-step_"):
            return "action-step_*"
        if part == "ood-language":
            return "ood-language/"
    return "其他（summary / 配置等）"


def build_sync_summary(
    local_dir: Path,
    to_upload: List[Path],
    skipped: List[Path],
) -> Dict[str, Any]:
    """生成上传摘要（程序自动校验，不需人工逐文件核对）。"""
    local_dir = local_dir.expanduser().resolve()
    buckets: Dict[str, Dict[str, int]] = {}
    upload_bytes = 0
    for path in to_upload:
        rel = _format_rel_path(local_dir, path)
        bucket = _upload_bucket(rel)
        size = path.stat().st_size
        upload_bytes += size
        entry = buckets.setdefault(bucket, {"count": 0, "bytes": 0})
        entry["count"] += 1
        entry["bytes"] += size

    skipped_bytes = sum(p.stat().st_size for p in skipped)
    skipped_paths = [_format_rel_path(local_dir, p) for p in skipped]
    skipped_names = [Path(p).name for p in skipped_paths]
    skipped_files = [
        {
            "path": _format_rel_path(local_dir, p),
            "bytes": p.stat().st_size,
            "reason": skip_sync_reason(p) or "unknown",
        }
        for p in skipped
    ]

    validation_errors: List[str] = []
    for path in to_upload:
        reason = skip_sync_reason(path)
        if reason is not None:
            validation_errors.append(
                f"{_format_rel_path(local_dir, path)} ({reason})"
            )

    return {
        "upload_count": len(to_upload),
        "upload_bytes": upload_bytes,
        "skipped_count": len(skipped),
        "skipped_bytes": skipped_bytes,
        "buckets": buckets,
        "skipped_paths": skipped_paths,
        "skipped_names": skipped_names,
        "skipped_files": skipped_files,
        "validation_ok": not validation_errors,
        "validation_errors": validation_errors,
    }


def write_sync_manifest(
    manifest_path: Path,
    local_dir: Path,
    to_upload: List[Path],
    skipped: List[Path],
    summary: Dict[str, Any],
) -> Path:
    """完整文件清单写入日志（供事后审计，不在终端刷屏）。"""
    import json

    manifest_path = manifest_path.expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    local_dir = local_dir.expanduser().resolve()
    with manifest_path.open("w", encoding="utf-8") as fp:
        fp.write(
            json.dumps(
                {
                    "type": "remote_sync_manifest_header",
                    "upload_count": summary["upload_count"],
                    "skipped_count": summary["skipped_count"],
                    "validation_ok": summary["validation_ok"],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        for path in to_upload:
            rel = _format_rel_path(local_dir, path)
            fp.write(
                json.dumps(
                    {
                        "action": "upload",
                        "path": rel,
                        "bytes": path.stat().st_size,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        for path in skipped:
            rel = _format_rel_path(local_dir, path)
            fp.write(
                json.dumps(
                    {
                        "action": "skip",
                        "path": rel,
                        "bytes": path.stat().st_size,
                        "reason": skip_sync_reason(path) or "unknown",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    logger.info("远程同步清单已写入: %s", manifest_path)
    return manifest_path


def _append_remote_sync_log(
    output_base_dir: Path,
    event: str,
    **payload: Any,
) -> None:
    """远程同步详细记录仅写入 batch_infer_log.jsonl（不刷终端）。"""
    from infer.batch.batch_infer_utils import append_phase_log

    safe_payload = {k: v for k, v in payload.items() if v is not None}
    append_phase_log(
        output_base_dir,
        "remote_sync",
        {"event": event, **safe_payload},
    )


def _log_remote_sync_plan(
    output_base_dir: Path,
    local_dir: Path,
    prep: Dict[str, Any],
    sync_cfg: BatchRemoteSyncConfig,
    *,
    manifest_path: Optional[Path] = None,
) -> None:
    """详细同步计划写入 batch_infer_log.jsonl + logger 摘要（终端不展示文件列表）。"""
    local_dir = local_dir.expanduser().resolve()
    summary = prep["summary"]
    to_upload = prep["to_upload"]

    logger.info(
        "[REMOTE][PLAN] local=%s remote=%s upload=%d (%s) skip_files=%d (%s) manifest=%s",
        local_dir,
        prep["remote_target"],
        summary["upload_count"],
        _human_bytes(summary["upload_bytes"]),
        summary["skipped_count"],
        _human_bytes(summary["skipped_bytes"]),
        manifest_path,
    )

    upload_files = [
        {"path": _format_rel_path(local_dir, p), "bytes": p.stat().st_size}
        for p in to_upload
    ]
    skipped_files = summary.get("skipped_files", [])
    _append_remote_sync_log(
        output_base_dir,
        "plan",
        local_dir=str(local_dir),
        remote_target=prep["remote_target"],
        remote_dest=prep["remote_dest"],
        host=sync_cfg.host,
        port=sync_cfg.port,
        username=sync_cfg.username,
        private_key_path=str(prep["key_path"]),
        manifest_path=str(manifest_path) if manifest_path else None,
        upload_count=summary["upload_count"],
        upload_bytes=summary["upload_bytes"],
        skipped_count=summary["skipped_count"],
        skipped_bytes=summary["skipped_bytes"],
        buckets=summary["buckets"],
        validation_ok=summary["validation_ok"],
        upload_files=upload_files,
        skipped_files=skipped_files,
    )


def _log_remote_sync_transfer(
    output_base_dir: Path,
    event: str,
    **payload: Any,
) -> None:
    logger.info("[REMOTE][TRANSFER] %s", event)
    _append_remote_sync_log(output_base_dir, event, **payload)


def _ensure_sync_validation_ok(summary: Dict[str, Any]) -> None:
    if summary["validation_ok"]:
        return
    for err in summary["validation_errors"]:
        logger.error("[REMOTE][PLAN] validation_error: %s", err)
    sample = ", ".join(summary["validation_errors"][:3])
    raise RuntimeError(
        "远程同步校验失败：存在不应上传的文件"
        f"（例如 {sample}）。请检查 collect_sync_files 逻辑。"
    )


def _assert_remote_sync_enabled(sync_cfg: BatchRemoteSyncConfig) -> None:
    if not sync_cfg.enabled:
        raise RuntimeError("remote_sync.enabled=false，远程同步已关闭")


def resolve_experiment_bundle_dir(ckpt_dir: Path) -> Path:
    """
    批量推理结果包根目录（与 batch_infer 输出目录一致）。

    只同步当前 checkpoint step 目录，不上传整个 checkpoints/ 父目录。

    例: .../checkpoints/010000/pretrained_model → .../checkpoints/010000
        .../infer/609/0423-leftonly/5000/pretrained_model → .../5000
    """
    from infer.batch.batch_infer_utils import resolve_batch_output_dir

    return resolve_batch_output_dir(ckpt_dir)


def resolve_remote_bundle_relpath(ckpt_dir: Path) -> str:
    """
    远程目录在 remote_dir 下的相对路径（POSIX）。

    上传源仍是当前 step 目录（pretrained_model 同级），远程用
    「模型 run 名 / step」区分，避免不同模型共用 2000/5000 时互相覆盖。

    例:
      .../0423-0416-leftonly-c50-a50-vla/5000/pretrained_model
        → 0423-0416-leftonly-c50-a50-vla/5000
      .../0423-0416-leftonly-c50-a50-vla/2000/pretrained_model
        → 0423-0416-leftonly-c50-a50-vla/2000
      .../checkpoints/010000/pretrained_model
        → checkpoints/010000
    """
    step_dir = resolve_experiment_bundle_dir(ckpt_dir)
    model_run = step_dir.parent.name
    if model_run and model_run not in (".", ".."):
        return posixpath.join(model_run, step_dir.name)
    return step_dir.name


def _resolved_private_key(sync_cfg: BatchRemoteSyncConfig) -> Path:
    key = (
        sync_cfg.private_key_path.expanduser().resolve()
        if sync_cfg.private_key_path is not None
        else _DEFAULT_PRIVATE_KEY.expanduser().resolve()
    )
    if not key.is_file():
        raise FileNotFoundError(f"SSH 私钥不存在: {key}")
    return key


def _remote_destination(sync_cfg: BatchRemoteSyncConfig, bundle_name: str) -> str:
    base = sync_cfg.remote_dir.strip().rstrip("/")
    if not base:
        raise ValueError("remote_sync.remote_dir 不能为空")
    return posixpath.join(base, bundle_name)


def _validate_sync_config(sync_cfg: BatchRemoteSyncConfig) -> None:
    if not sync_cfg.enabled:
        return
    missing: List[str] = []
    if not sync_cfg.host:
        missing.append("host")
    if not sync_cfg.username:
        missing.append("username")
    if not sync_cfg.remote_dir:
        missing.append("remote_dir")
    if missing:
        raise ValueError(f"remote_sync 已启用但缺少配置: {', '.join(missing)}")
    _resolved_private_key(sync_cfg)


def _sync_with_rsync(
    local_dir: Path,
    sync_cfg: BatchRemoteSyncConfig,
    remote_dest: str,
    key_path: Path,
    files_to_upload: List[Path],
) -> Dict[str, Any]:
    if shutil.which("rsync") is None:
        raise RuntimeError("未找到 rsync，请安装 rsync")
    if not files_to_upload:
        raise RuntimeError("没有可上传的文件（请检查实验目录是否为空）")

    ssh_parts = [
        "ssh",
        "-p",
        str(int(sync_cfg.port)),
        "-i",
        str(key_path),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=no",
    ]
    target = f"{sync_cfg.username}@{sync_cfg.host}:{remote_dest}/"
    rel_paths = [_format_rel_path(local_dir, p) for p in files_to_upload]

    list_file: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".rsync-files",
            delete=False,
        ) as fp:
            for rel in rel_paths:
                fp.write(rel + "\n")
            list_file = Path(fp.name)

        cmd = [
            "rsync",
            "-a",
            "--info=progress2",
            "--files-from",
            str(list_file),
            "-e",
            " ".join(ssh_parts),
            f"{local_dir}/",
            target,
        ]
        logger.debug("[REMOTE][TRANSFER] rsync cmd: %s", " ".join(cmd))
        print("[REMOTE] 上传中，请稍候…")
        t0 = time.time()
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        rsync_log = {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout.rstrip() if proc.stdout else "",
            "stderr": proc.stderr.rstrip() if proc.stderr else "",
        }
        if proc.returncode != 0:
            raise RuntimeError(f"rsync 失败 (exit={proc.returncode})")
        elapsed = time.time() - t0
        logger.info(
            "[REMOTE][TRANSFER] rsync done: %d files in %.2fs",
            len(files_to_upload),
            elapsed,
        )
        return {
            "method": "rsync",
            "elapsed_sec": round(elapsed, 2),
            "remote_dest": remote_dest,
            "private_key_path": str(key_path),
            "files": len(files_to_upload),
            "skipped_pretrained_model_files": 0,
            "rsync_log": rsync_log,
        }
    finally:
        if list_file is not None:
            try:
                list_file.unlink(missing_ok=True)
            except OSError:
                pass


def _sftp_mkdir_p(sftp: Any, remote_path: str) -> None:
    remote_path = remote_path.replace("\\", "/").rstrip("/")
    if not remote_path:
        return
    is_abs = remote_path.startswith("/")
    parts = [p for p in remote_path.split("/") if p]
    cur = ""
    for part in parts:
        if is_abs:
            cur = f"{cur}/{part}" if cur else f"/{part}"
        else:
            cur = f"{cur}/{part}" if cur else part
        try:
            sftp.stat(cur)
        except OSError:
            sftp.mkdir(cur)


def _sync_with_paramiko(
    local_dir: Path,
    sync_cfg: BatchRemoteSyncConfig,
    remote_dest: str,
    key_path: Path,
    files_to_upload: List[Path],
    skipped: List[Path],
) -> Dict[str, Any]:
    try:
        import paramiko
    except ImportError as exc:
        raise RuntimeError(
            "rsync 失败且未安装 paramiko。请安装 rsync，或: pip install paramiko"
        ) from exc

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    connect_kwargs: Dict[str, Any] = {
        "hostname": sync_cfg.host,
        "port": int(sync_cfg.port),
        "username": sync_cfg.username,
        "timeout": 30,
        "key_filename": str(key_path),
        "allow_agent": False,
        "look_for_keys": False,
    }

    logger.info(
        "[REMOTE][TRANSFER] SFTP %s@%s:%s files=%d",
        sync_cfg.username,
        sync_cfg.host,
        sync_cfg.port,
        len(files_to_upload),
    )
    print("[REMOTE] 上传中，请稍候…")
    t0 = time.time()
    client.connect(**connect_kwargs)
    file_count = 0
    byte_count = 0
    uploaded_files: List[Dict[str, Any]] = []
    try:
        sftp = client.open_sftp()
        try:
            _sftp_mkdir_p(sftp, remote_dest)
            for local_file in files_to_upload:
                rel = local_file.relative_to(local_dir)
                remote_file = posixpath.join(remote_dest, rel.as_posix())
                remote_parent = posixpath.dirname(remote_file)
                if remote_parent and remote_parent != ".":
                    _sftp_mkdir_p(sftp, remote_parent)
                size = local_file.stat().st_size
                sftp.put(str(local_file), remote_file)
                file_count += 1
                byte_count += size
                uploaded_files.append(
                    {
                        "path": rel.as_posix(),
                        "remote": remote_file,
                        "bytes": size,
                    }
                )
        finally:
            sftp.close()
    finally:
        client.close()

    elapsed = time.time() - t0
    logger.info(
        "[REMOTE][TRANSFER] SFTP done: %d files (%s) in %.2fs",
        file_count,
        _human_bytes(byte_count),
        elapsed,
    )
    return {
        "method": "paramiko_sftp",
        "elapsed_sec": round(elapsed, 2),
        "remote_dest": remote_dest,
        "private_key_path": str(key_path),
        "files": file_count,
        "bytes": byte_count,
        "skipped_pretrained_model_files": len(skipped),
        "uploaded_files": uploaded_files,
    }


def _prepare_remote_sync(
    sync_cfg: BatchRemoteSyncConfig,
    ckpt_dir: Path,
    *,
    sync_source_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """扫描、校验并打包同步所需信息（用户无需参与）。"""
    _assert_remote_sync_enabled(sync_cfg)
    _validate_sync_config(sync_cfg)
    local_dir = (
        sync_source_dir.expanduser().resolve()
        if sync_source_dir is not None
        else resolve_experiment_bundle_dir(ckpt_dir)
    )
    if not local_dir.is_dir():
        raise FileNotFoundError(f"实验目录不存在: {local_dir}")
    key_path = _resolved_private_key(sync_cfg)
    bundle_relpath = resolve_remote_bundle_relpath(ckpt_dir)
    remote_dest = _remote_destination(sync_cfg, bundle_relpath)
    to_upload, skipped = collect_sync_files(local_dir)
    summary = build_sync_summary(local_dir, to_upload, skipped)
    _ensure_sync_validation_ok(summary)
    return {
        "local_dir": local_dir,
        "bundle_name": bundle_relpath,
        "remote_dest": remote_dest,
        "remote_target": f"{sync_cfg.username}@{sync_cfg.host}:{remote_dest}/",
        "key_path": key_path,
        "to_upload": to_upload,
        "skipped": skipped,
        "summary": summary,
        "upload_file_count": summary["upload_count"],
        "upload_bytes": summary["upload_bytes"],
        "skipped_file_count": summary["skipped_count"],
        "skipped_bytes": summary["skipped_bytes"],
        "validation_ok": summary["validation_ok"],
    }


def sync_experiment_bundle(
    sync_cfg: BatchRemoteSyncConfig,
    ckpt_dir: Path,
    *,
    manifest_dir: Optional[Path] = None,
    prepared: Optional[Dict[str, Any]] = None,
    output_base_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """将 ckpt_dir 对应实验包目录同步到远程服务器（SSH 私钥认证）。"""
    _assert_remote_sync_enabled(sync_cfg)
    prep = prepared or _prepare_remote_sync(sync_cfg, ckpt_dir)
    local_dir = prep["local_dir"]
    remote_dest = prep["remote_dest"]
    key_path = prep["key_path"]
    to_upload = prep["to_upload"]
    skipped = prep["skipped"]
    summary = prep["summary"]
    remote_target = prep["remote_target"]

    if manifest_dir is None or output_base_dir is None:
        from infer.batch.batch_infer_utils import resolve_batch_output_dir

        resolved_output = resolve_batch_output_dir(ckpt_dir)
        if manifest_dir is None:
            manifest_dir = resolved_output
        if output_base_dir is None:
            output_base_dir = resolved_output

    manifest_path = write_sync_manifest(
        manifest_dir / "remote_sync_manifest.jsonl",
        local_dir,
        to_upload,
        skipped,
        summary,
    )
    # 详细 plan 由 maybe_sync 在确认前写入；此处仅记录传输阶段日志。

    result: Dict[str, Any] = {
        "local_dir": str(local_dir),
        "remote_dest": remote_dest,
        "host": sync_cfg.host,
        "username": sync_cfg.username,
        "private_key_path": str(key_path),
        "files": len(to_upload),
        "upload_bytes": summary["upload_bytes"],
        "skipped_pretrained_model_files": sum(
            1 for p in skipped if _is_under_pretrained_model(p)
        ),
        "skipped_bytes": summary["skipped_bytes"],
        "manifest_path": str(manifest_path),
        "validation_ok": summary["validation_ok"],
    }

    try:
        result.update(
            _sync_with_rsync(local_dir, sync_cfg, remote_dest, key_path, to_upload)
        )
        result["skipped_pretrained_model_files"] = len(skipped)
    except RuntimeError as exc:
        logger.warning("rsync 失败，尝试 paramiko SFTP: %s", exc)
        result.update(
            _sync_with_paramiko(
                local_dir, sync_cfg, remote_dest, key_path, to_upload, skipped,
            )
        )

    _log_remote_sync_transfer(
        output_base_dir,
        "complete",
        method=result.get("method"),
        elapsed_sec=result.get("elapsed_sec"),
        files=result.get("files"),
        bytes=result.get("bytes", summary["upload_bytes"]),
        remote_dest=remote_dest,
        rsync_log=result.get("rsync_log"),
        uploaded_files=result.get("uploaded_files"),
    )
    logger.info(
        "[REMOTE][TRANSFER] complete method=%s elapsed=%ss files=%s",
        result.get("method"),
        result.get("elapsed_sec"),
        result.get("files"),
    )
    print(f"[REMOTE] 上传完成（{result.get('elapsed_sec')}s）")
    return result


def _print_remote_upload_prompt(prep: Dict[str, Any]) -> None:
    """上传前仅展示最少信息；用户唯一操作是 Y/N。"""
    print(f"\n[REMOTE] 实验结果将上传到: {prep['remote_target']}")
    print(
        f"  约 {_human_bytes(int(prep['upload_bytes']))} "
        f"（{prep['upload_file_count']} 个文件）"
    )


def confirm_remote_sync_transfer(
    sync_cfg: BatchRemoteSyncConfig,
    ckpt_dir: Path,
    *,
    pedal_hub: Optional["PedalInputHub"] = None,
    prepared: Optional[Dict[str, Any]] = None,
    output_base_dir: Optional[Path] = None,
) -> bool:
    """唯一人工步骤：是否开始上传（Y/N）。"""
    from infer.batch.batch_infer_utils import log_batch_interaction, resolve_batch_output_dir
    from infer.ui.pedal_input import ask_yes_no_pedal

    _assert_remote_sync_enabled(sync_cfg)
    prep = prepared or _prepare_remote_sync(
        sync_cfg,
        ckpt_dir,
        sync_source_dir=output_base_dir or resolve_batch_output_dir(ckpt_dir),
    )
    if output_base_dir is None:
        output_base_dir = resolve_batch_output_dir(ckpt_dir)

    _print_remote_upload_prompt(prep)
    confirmed = ask_yes_no_pedal(pedal_hub, "是否开始上传？")
    try:
        log_batch_interaction(
            output_base_dir,
            "remote_sync_confirm",
            confirmed=confirmed,
            remote_target=prep.get("remote_target"),
            local_dir=str(prep.get("local_dir")),
            upload_count=prep.get("upload_file_count"),
            upload_bytes=prep.get("upload_bytes"),
            skipped_count=prep.get("skipped_file_count"),
            skipped_bytes=prep.get("skipped_bytes"),
        )
    except Exception as exc:
        logger.warning("记录远程同步交互日志失败: %s", exc)
    if not confirmed:
        logger.info("[REMOTE] user declined upload")
        print("[REMOTE] 已跳过上传。")
    return confirmed


def maybe_sync_experiment_to_remote(
    sync_cfg: BatchRemoteSyncConfig,
    ckpt_dir: Path,
    *,
    summary_path: Optional[Path] = None,
    pedal_hub: Optional["PedalInputHub"] = None,
    output_base_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """若配置启用则询问 Y/N 后同步；失败时记录日志但不中断主流程。"""
    if not sync_cfg.enabled:
        return None
    try:
        from infer.batch.batch_infer_utils import resolve_batch_output_dir

        log_dir = (
            output_base_dir.expanduser().resolve()
            if output_base_dir is not None
            else resolve_batch_output_dir(ckpt_dir)
        )
        prep = _prepare_remote_sync(sync_cfg, ckpt_dir, sync_source_dir=log_dir)
        _log_remote_sync_plan(
            log_dir,
            prep["local_dir"],
            prep,
            sync_cfg,
        )
        if not confirm_remote_sync_transfer(
            sync_cfg,
            ckpt_dir,
            pedal_hub=pedal_hub,
            prepared=prep,
            output_base_dir=log_dir,
        ):
            skipped = {
                "skipped": True,
                "reason": "user_declined",
                "local_dir": str(prep["local_dir"]),
                "bundle_name": prep["bundle_name"],
                "remote_dest": prep["remote_dest"],
                "remote_target": prep["remote_target"],
                "upload_file_count": prep["upload_file_count"],
                "upload_bytes": prep["upload_bytes"],
                "skipped_file_count": prep["skipped_file_count"],
                "skipped_bytes": prep["skipped_bytes"],
            }
            _log_remote_sync_transfer(
                log_dir,
                "declined",
                **{k: v for k, v in skipped.items() if k != "skipped"},
            )
            if summary_path is not None and summary_path.is_file():
                import json

                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary["remote_sync"] = skipped
                summary_path.write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )
            return skipped
        _log_remote_sync_transfer(
            log_dir,
            "start",
            remote_target=prep["remote_target"],
            upload_count=prep["upload_file_count"],
            upload_bytes=prep["upload_bytes"],
        )
        result = sync_experiment_bundle(
            sync_cfg,
            ckpt_dir,
            manifest_dir=summary_path.parent if summary_path else log_dir,
            prepared=prep,
            output_base_dir=log_dir,
        )
        if summary_path is not None and summary_path.is_file():
            import json

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["remote_sync"] = {
                **result,
                "config": {
                    k: (str(v) if isinstance(v, Path) else v)
                    for k, v in asdict(sync_cfg).items()
                },
            }
            summary.pop("remote_sync_error", None)
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        return result
    except Exception as exc:
        logger.exception("[REMOTE][ERROR] 远程同步失败: %s", exc)
        try:
            from infer.batch.batch_infer_utils import resolve_batch_output_dir

            err_log_dir = (
                output_base_dir.expanduser().resolve()
                if output_base_dir is not None
                else resolve_batch_output_dir(ckpt_dir)
            )
            _log_remote_sync_transfer(
                err_log_dir,
                "error",
                error=str(exc),
            )
        except Exception as log_exc:
            logger.warning("记录远程同步错误日志失败: %s", log_exc)
        print("[REMOTE][ERROR] 上传失败，请联系老师。", flush=True)
        if summary_path is not None and summary_path.is_file():
            import json

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["remote_sync_error"] = str(exc)
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        return None
