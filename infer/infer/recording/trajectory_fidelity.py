"""Optional observation only recorder. No hardware, ROS, torch or numpy dependency."""
from __future__ import annotations

import csv
import json
import queue
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


SCHEMAS = {
    "commands": ["time_sec", "command_id", "arm", "chunk_id", "step_index",
                 "send_start_sec", "send_end_sec", "return_code", "error", "follow",
                 *[f"q{i}_rad" for i in range(7)], *[f"sdk_q{i}_deg" for i in range(7)]],
    "feedback": ["time_sec", "sample_id", "arm", "source", "query_start_sec",
                 "query_end_sec", "return_code", "error", *[f"q{i}_rad" for i in range(7)]],
    "chunks": ["time_sec", "chunk_id", "step_index", "reference_offset_sec",
               "state_wall_sec", "image_wall_sec", *[f"q{i}" for i in range(16)]],
    "events": ["time_sec", "event", "chunk_id", "details_json"],
    "gripper": ["time_sec", "source", "chunk_id", "left", "right", "details_json"],
}


class TrajectoryFidelityRecorder:
    """Nonblocking producers, bounded queue, one background writer.

    Constructor failures abort the requested recording before hardware starts.
    Runtime recording failures are counted and never change robot commands.
    A run is usable only after inspecting run_meta.json quality fields.
    """

    def __init__(self, directory, metadata=None, *, capacity=8192):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.origin_ns = time.monotonic_ns()
        self.meta = {
            "schema_version": 1, "run_id": self.directory.parent.name,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "origin_monotonic_ns": self.origin_ns, "origin_wall_ns": time.time_ns(),
            "position_unit": "radian", "sdk_position_unit": "degree",
            "joint_convention": "TongBot J7 physical minus 120 degrees",
            "feedback_time_semantics": "host per-arm SDK query interval; no hardware sample timestamp",
            "source_metadata": metadata or {}, "closed": False, "run_returned": False,
        }
        self._queue = queue.Queue(maxsize=capacity)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._closed = False
        self._ids = Counter()
        self._accepted = Counter()
        self._dropped = Counter()
        self._written = Counter()
        self._errors = []
        self._files = {}
        self._writers = {}
        try:
            for name, fields in SCHEMAS.items():
                f = (self.directory / f"{name}.csv").open("w", newline="", encoding="utf-8")
                self._files[name] = f
                self._writers[name] = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                self._writers[name].writeheader()
                f.flush()
            self._write_meta()
        except BaseException:
            for f in self._files.values():
                f.close()
            raise
        self._thread = threading.Thread(target=self._write_loop, name="trajectory-csv", daemon=True)
        self._thread.start()

    def seconds(self, ns=None):
        return ((time.monotonic_ns() if ns is None else ns) - self.origin_ns) / 1e9

    def next_id(self, kind):
        with self._lock:
            self._ids[kind] += 1
            return self._ids[kind]

    def emit(self, stream, **row):
        try:
            row.setdefault("time_sec", self.seconds())
            with self._lock:
                if self._closed:
                    self._dropped[stream] += 1
                    return
                try:
                    self._queue.put_nowait((stream, row))
                    self._accepted[stream] += 1
                except queue.Full:
                    self._dropped[stream] += 1
        except Exception as exc:
            with self._lock:
                self._dropped[stream] += 1
                if len(self._errors) < 10:
                    self._errors.append(repr(exc))

    def event(self, name, *, chunk_id=None, **details):
        self.emit("events", event=name, chunk_id=chunk_id,
                  details_json=json.dumps(details, default=str, ensure_ascii=False))

    def _write_loop(self):
        last_flush = time.monotonic()
        while not self._stop.is_set() or not self._queue.empty():
            try:
                stream, row = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._writers[stream].writerow(row)
                self._written[stream] += 1
                if time.monotonic() - last_flush > 1:
                    for f in self._files.values():
                        f.flush()
                    last_flush = time.monotonic()
            except Exception as exc:
                if len(self._errors) < 10:
                    self._errors.append(repr(exc))

    def _write_meta(self):
        p = self.directory / "run_meta.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        tmp.replace(p)

    def close(self, *, run_returned=False):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._stop.set()
        self._thread.join()
        for f in self._files.values():
            try:
                f.close()
            except Exception as exc:
                self._errors.append(repr(exc))
        self.meta.update(closed=True, run_returned=run_returned,
                         accepted=dict(self._accepted), written=dict(self._written),
                         dropped=dict(self._dropped), errors=list(self._errors),
                         ended_utc=datetime.now(timezone.utc).isoformat())
        self.meta["recording_complete"] = (
            not self._errors and not any(self._dropped.values())
            and self._accepted == self._written
        )
        try:
            self._write_meta()
        except Exception:
            # The initial metadata remains closed=False, explicitly incomplete.
            import logging
            logging.getLogger(__name__).exception("轨迹记录元数据写入失败")
