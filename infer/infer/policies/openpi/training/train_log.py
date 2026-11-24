"""Helpers to persist OpenPI training logs / hyperparameters for multi-run comparison."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


TRAIN_LOG_NAME = "train.log"
TRAIN_CONFIG_NAME = "train_config.json"
TRAIN_HARDWARE_NAME = "train_hardware.json"
RUN_META_NAME = "run_meta.json"
SOURCE_YAML_ENV = "OPENPI_TRAIN_YAML"


def to_jsonable(obj: Any) -> Any:
    """Recursively convert configs / pathlib / nested structures into JSON-serializable values."""
    try:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        if isinstance(obj, dict):
            return {str(k): to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_jsonable(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        if hasattr(obj, "__fspath__"):
            return str(obj)
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, (bytes, bytearray)):
            return obj.decode("utf-8", errors="replace")
        item = getattr(obj, "item", None)
        if callable(item):
            try:
                return to_jsonable(item())
            except Exception:
                pass
        return repr(obj)
    except Exception as exc:  # never let config dump crash training
        return f"<unserializable {type(obj).__name__}: {exc}>"


def resolve_source_yaml(source_yaml: str | Path | None = None) -> Path | None:
    if source_yaml:
        path = Path(source_yaml)
        return path if path.is_file() else None
    env = os.environ.get(SOURCE_YAML_ENV, "").strip()
    if env:
        path = Path(env)
        return path if path.is_file() else None
    return None


def copy_source_yaml(output_dir: Path | str, source_yaml: str | Path | None = None) -> Path | None:
    """Copy user YAML into the run dir after checkpoint wipe; keep basename + train.yaml alias."""
    src = resolve_source_yaml(source_yaml)
    if src is None:
        return None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dst = output_dir / src.name
    shutil.copy2(src, dst)
    alias = output_dir / "train.yaml"
    if dst.resolve() != alias.resolve():
        shutil.copy2(src, alias)
    return dst


def save_run_meta(
    output_dir: Path | str,
    *,
    source_yaml: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    src = resolve_source_yaml(source_yaml)
    meta: dict[str, Any] = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "argv": list(sys.argv),
        "config_file": str(src.resolve()) if src else None,
        "config_basename": src.name if src else None,
    }
    if extra:
        meta.update(extra)
    path = output_dir / RUN_META_NAME
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)
        f.write("\n")
    return path


def save_train_config(
    output_dir: Path | str,
    config: Any,
    *,
    source_yaml: str | Path | None = None,
) -> Path:
    """Write resolved training hyperparameters to ``train_config.json`` under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    src = resolve_source_yaml(source_yaml)
    payload = {
        "_meta": {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "config_file": str(src.resolve()) if src else None,
            "config_basename": src.name if src else None,
            "note": "Resolved effective TrainConfig after YAML overrides on the named preset.",
        },
        "params": to_jsonable(config),
    }
    path = output_dir / TRAIN_CONFIG_NAME
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
        f.write("\n")
    return path


def save_train_hardware(output_dir: Path | str, extra: dict[str, Any] | None = None) -> Path:
    """Write a small hardware / launcher manifest next to train.log."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "argv": list(sys.argv),
    }
    if extra:
        payload.update(extra)
    path = output_dir / TRAIN_HARDWARE_NAME
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
        f.write("\n")
    return path


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        item = getattr(value, "item", None)
        if callable(item):
            value = item()
        return float(value)
    except Exception:
        return None


def setup_train_file_logging(output_dir: Path | str, *, append: bool = True) -> Path:
    """Attach a FileHandler that writes to ``{output_dir}/train.log`` (console handlers kept)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / TRAIN_LOG_NAME

    logger = logging.getLogger()
    if logger.level == logging.NOTSET or logger.level > logging.INFO:
        logger.setLevel(logging.INFO)
    resolved = str(log_path.resolve())
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == resolved:
            return log_path

    file_handler = logging.FileHandler(log_path, mode="a" if append else "w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    if logger.handlers and logger.handlers[0].formatter is not None:
        file_handler.setFormatter(logger.handlers[0].formatter)
    else:
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    logger.addHandler(file_handler)
    return log_path


def format_metrics_line(
    step: int,
    *,
    loss: float | None = None,
    grad_norm: float | None = None,
    lr: float | None = None,
    update_s: float | None = None,
    data_s: float | None = None,
    epoch: float | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Build a LeRobot-compatible metrics line: ``step:N loss:... grdn:... lr:...``."""
    parts = [f"step:{int(step)}"]
    epoch_f = _as_float(epoch)
    loss_f = _as_float(loss)
    grad_f = _as_float(grad_norm)
    lr_f = _as_float(lr)
    updt_f = _as_float(update_s)
    data_f = _as_float(data_s)
    if epoch_f is not None:
        parts.append(f"epch:{epoch_f:.4g}")
    if loss_f is not None:
        parts.append(f"loss:{loss_f:.6g}")
    if grad_f is not None:
        parts.append(f"grdn:{grad_f:.6g}")
    if lr_f is not None:
        parts.append(f"lr:{lr_f:.6g}")
    if updt_f is not None:
        parts.append(f"updt_s:{updt_f:.6g}")
    if data_f is not None:
        parts.append(f"data_s:{data_f:.6g}")
    if extra:
        for key, value in extra.items():
            value_f = _as_float(value)
            if value_f is not None:
                parts.append(f"{key}:{value_f:.6g}")
    return " ".join(parts)


def init_train_log(
    output_dir: Path | str,
    config: Any,
    *,
    append: bool = True,
    hardware_extra: dict[str, Any] | None = None,
    source_yaml: str | Path | None = None,
) -> Path:
    """
    Prepare run artifacts (call **after** checkpoint-dir overwrite wipe):

    - ``train.log``
    - ``train_config.json`` (resolved effective TrainConfig)
    - ``train_hardware.json``
    - ``run_meta.json``
    - copied source YAML (basename + ``train.yaml`` alias)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    src = resolve_source_yaml(source_yaml)

    log_path = setup_train_file_logging(output_dir, append=append)
    yaml_dst = None
    try:
        yaml_dst = copy_source_yaml(output_dir, src)
    except Exception:
        logging.exception("Failed to copy source yaml into output dir")

    try:
        config_path = save_train_config(output_dir, config, source_yaml=src)
    except Exception:
        logging.exception("Failed to save %s", TRAIN_CONFIG_NAME)
        config_path = output_dir / TRAIN_CONFIG_NAME

    hw_extra = dict(hardware_extra or {})
    if src is not None:
        hw_extra.setdefault("config_file", str(src.resolve()))
        hw_extra.setdefault("config_basename", src.name)
    try:
        hw_path = save_train_hardware(output_dir, hw_extra)
    except Exception:
        logging.exception("Failed to save %s", TRAIN_HARDWARE_NAME)
        hw_path = output_dir / TRAIN_HARDWARE_NAME

    try:
        meta_path = save_run_meta(
            output_dir,
            source_yaml=src,
            extra={
                "framework": hw_extra.get("framework"),
                "exp_name": getattr(config, "exp_name", None),
                "config_name": getattr(config, "name", None),
                "checkpoint_dir": str(getattr(config, "checkpoint_dir", output_dir)),
            },
        )
    except Exception:
        logging.exception("Failed to save %s", RUN_META_NAME)
        meta_path = output_dir / RUN_META_NAME

    logging.info("Output dir: %s", output_dir)
    logging.info("Train log: %s", log_path)
    logging.info("Train config: %s", config_path)
    logging.info("Train hardware: %s", hw_path)
    logging.info("Run meta: %s", meta_path)
    if yaml_dst is not None:
        logging.info("Source yaml copied: %s", yaml_dst)
    logging.info("cfg.steps=%s", getattr(config, "num_train_steps", None))
    try:
        logging.info("%s", json.dumps(to_jsonable(config), indent=2, ensure_ascii=False))
    except Exception:
        logging.exception("Failed to dump train config into train.log")
    return log_path
