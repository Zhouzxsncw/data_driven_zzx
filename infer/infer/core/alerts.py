"""推理期间告警收集与持久化。"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from infer.core.types import AlertEvent, AlertLevel

logger = logging.getLogger(__name__)


class AlertManager:
    """统一告警：内存缓冲 + JSONL 落盘 + 日志输出。"""

    def __init__(self, enabled: bool = True, alerts_path: Optional[Path] = None) -> None:
        self.enabled = enabled
        self.alerts_path = alerts_path
        self.events: List[AlertEvent] = []
        self._fp = None
        if enabled and alerts_path is not None:
            alerts_path.parent.mkdir(parents=True, exist_ok=True)
            self._fp = open(alerts_path, "a", encoding="utf-8")

    def emit(
        self,
        level: AlertLevel,
        code: str,
        message: str,
        **context: Any,
    ) -> None:
        if not self.enabled:
            return
        ev = AlertEvent(
            level=level,
            code=code,
            message=message,
            context=context,
            wall_time=time.time(),
        )
        self.events.append(ev)
        log_fn = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.ERROR: logger.error,
            AlertLevel.CRITICAL: logger.critical,
        }[level]
        log_fn("[ALERT:%s] %s %s", code, message, context or "")

        if self._fp is not None:
            row = {
                "level": level.value,
                "code": code,
                "message": message,
                "context": context,
                "wall_time": ev.wall_time,
            }
            self._fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._fp.flush()

    def scan_machine_flow(
        self,
        events: List[Dict[str, Any]],
        *,
        sync_warn_rate_threshold: float = 0.30,
    ) -> None:
        """从 machine_flow 事件批量提取告警。"""
        sync_warns = sum(
            1 for e in events
            if e.get("type") == "sync_warning_time_diff_too_large"
            or e.get("event") == "sync_warning_time_diff_too_large"
        )
        if sync_warns > 0:
            self.emit(
                AlertLevel.WARNING,
                "SYNC_WARNING",
                f"检测到 {sync_warns} 次 sync 告警",
                count=sync_warns,
            )
            total = len(events) or 1
            rate = sync_warns / total
            if rate > sync_warn_rate_threshold:
                self.emit(
                    AlertLevel.WARNING,
                    "HIGH_SYNC_WARN_RATE",
                    f"sync 告警占比 {rate:.1%} 超过阈值 {sync_warn_rate_threshold:.0%}",
                    rate=rate,
                    threshold=sync_warn_rate_threshold,
                    sync_warnings=sync_warns,
                    total_events=total,
                )

        for e in events:
            et = e.get("type") or e.get("event")
            if et == "inference_error":
                self.emit(AlertLevel.ERROR, "INFERENCE_ERROR", "推理失败", payload=e)
            elif et == "post_action_invalid":
                self.emit(AlertLevel.ERROR, "INVALID_ACTION", "模型输出 NaN/Inf", payload=e)
            elif et == "infer_loop_fatal":
                self.emit(AlertLevel.CRITICAL, "INFER_FATAL", "推理主循环异常退出", payload=e)

    def close(self) -> None:
        if self._fp is not None:
            try:
                self._fp.close()
            except OSError:
                pass
            self._fp = None

    def summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for ev in self.events:
            counts[ev.level.value] = counts.get(ev.level.value, 0) + 1
        return counts
