"""日志初始化。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> logging.Logger:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    return logging.getLogger("infer")


def setup_test_human_logger(
    log_path: Path,
    *,
    human_log_level: str = "INFO",
    file_log_level: Optional[str] = None,
) -> logging.Logger:
    """为单次 test 创建 run.log 专用 logger（native 引擎）。"""
    human_lvl = getattr(logging, (human_log_level or "INFO").upper(), logging.INFO)
    file_lvl = getattr(
        logging, (file_log_level or human_log_level or "INFO").upper(), logging.INFO
    )
    test_logger = logging.getLogger(f"infer.test.{log_path.parent.name}")
    test_logger.setLevel(min(human_lvl, file_lvl))
    test_logger.handlers.clear()
    test_logger.propagate = False

    plain = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(human_lvl)
    stream_handler.setFormatter(_TTYLossFormatter(plain, sys.stdout))
    test_logger.addHandler(stream_handler)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(file_lvl)
    file_handler.setFormatter(plain)
    test_logger.addHandler(file_handler)
    return test_logger


class _TTYLossFormatter(logging.Formatter):
    """LOSS 行只在终端上色。run.log 继续用普通 Formatter。"""

    def __init__(self, inner: logging.Formatter, stream: object) -> None:
        super().__init__()
        self._inner = inner
        self._stream = stream

    def format(self, record: logging.LogRecord) -> str:
        text = self._inner.format(record)
        if not getattr(self._stream, "isatty", lambda: False)():
            return text
        message = record.getMessage()
        if not message.startswith("LOSS "):
            return text
        from infer.recording.ckpt_loss_notice import color_for_message

        color = {
            "red": "\033[31;1m",
            "yellow": "\033[33;1m",
            "green": "\033[32;1m",
        }.get(color_for_message(message), "")
        if not color:
            return text
        return f"{color}{text}\033[0m"
