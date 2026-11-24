"""Ctrl+. 打点后即时追问"这次推理评测结果"，并把答案与客观信号配成一条记录。

三条硬约束（都在单测里钉住）：

1. **不阻塞实时循环**：``submit()`` 只入队就返回（调用方是主循环里的打点处理，
   也可能是任意线程），提问与 stdin 读取发生在独立线程；推理控制循环、其它热键
   （外骨骼接管 / Delete）都不会被一个问题卡住。
2. **没回答也要落一条**：无终端 / 中途结束 / 连按溢出分别记 ``no_stdin`` /
   ``aborted`` / ``burst``。默认**不超时、一直等输入**；只有显式把
   ``timeout_sec`` 配成正数时才会记 ``timeout``。否则人工成功率的分会被悄悄丢掉，
   把"没评测"读成"0% 成功"——现场已经出现过一次这种假 0%。
3. **人工标签 × 客观信号同条**：记录里带上这次推理的 model/ik/flow 等分段耗时，
   后续能直接统计"IK 截断的那批 chunk，人眼认为真失败的占多少"。
"""

from __future__ import annotations

import queue
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

# 追问默认收文字描述；数字档位仍能解析，但不在提示里展示
VERDICT_LABELS: Dict[str, str] = {
    "success": "成功",
    "partial": "部分完成",
    "fail": "失败",
    "described": "描述",
}

# 输入 token → 判定；匹配前统一小写、去空白
_VERDICT_TOKENS: Dict[str, str] = {
    "1": "success",
    "y": "success",
    "s": "success",
    "ok": "success",
    "yes": "success",
    "success": "success",
    "成功": "success",
    "2": "partial",
    "p": "partial",
    "partial": "partial",
    "部分": "partial",
    "部分完成": "partial",
    "未完成": "partial",
    "3": "fail",
    "n": "fail",
    "f": "fail",
    "no": "fail",
    "fail": "fail",
    "失败": "fail",
    "0": "skipped",
    "skip": "skipped",
    "跳过": "skipped",
}

# 只有这三档进"有效评测"分母；文字描述 / skipped / timeout / … 不进成功率
VALID_VERDICTS = ("success", "partial", "fail")

# 没回答的几种口径，统计时合并成"无有效作答"
NO_ANSWER_VERDICTS = ("skipped", "timeout", "no_stdin", "aborted", "burst")

# 追问期间随附的客观信号键（与 engine 写入的 timing entry 键名一致）
CONTEXT_TIMING_KEYS = (
    "preprocess_sec",
    "model_sec",
    "postprocess_sec",
    "infer_sec",
    "fk_sec",
    "flow_sec",
    "convert_sec",
    "ik_sec",
    "server_infer_sec",
    "network_sec",
)

PROMPT_HINT = (
    "请用文字描述本次表现（例如：夹爪提前闭合、轨迹发抖、物体没夹稳）"
    "  提示：可对照 PI0.5 的结果作为参考。"
)
PROMPT_WAIT_HINT = "输入描述后回车进入下一步；空行=跳过。无超时，会一直等待输入。"


class StdinUnavailable(RuntimeError):
    """当前进程没有可用终端，追问无法进行（不算错误，只降级记录）。"""


def parse_eval_answer(text: Optional[str]) -> Dict[str, Any]:
    """把一行回答拆成 verdict + remark。默认按文字描述收下，不猜成功/失败。

    ``""`` → skipped；``"夹爪提前闭合"`` → described；仍认 ``成功`` / ``失败``
    等整词以及开头的 ``1``/``2``/``3``（兼容旧输入，但提示里不再给选项）。
    """
    raw = ("" if text is None else str(text)).strip()
    if not raw:
        return {"verdict": "skipped", "remark": "", "raw_answer": raw, "parse_error": False}
    head, _, tail = raw.partition(" ")
    verdict = _VERDICT_TOKENS.get(head.strip().lower())
    remark = tail.strip()
    if verdict is None:
        verdict = _VERDICT_TOKENS.get(raw.lower())
        remark = "" if verdict is not None else raw
    if verdict is None:
        return {"verdict": "described", "remark": raw, "raw_answer": raw, "parse_error": False}
    return {"verdict": verdict, "remark": remark, "raw_answer": raw, "parse_error": False}


def timing_context(timing_entry: Any) -> Dict[str, float]:
    """从 engine 本次推理的 timing entry 里挑出可配对的客观分段耗时（秒）。"""
    if not isinstance(timing_entry, dict):
        return {}
    return {
        key: float(timing_entry[key])
        for key in CONTEXT_TIMING_KEYS
        if isinstance(timing_entry.get(key), (int, float))
    }


def compute_eval_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总计数；一个有效判定都没有时 rate=None（而不是 0.0）。"""
    counts: Dict[str, int] = {}
    for row in rows:
        verdict = str(row.get("verdict") or "unparsed")
        counts[verdict] = counts.get(verdict, 0) + 1
    valid = {key: counts.get(key, 0) for key in VALID_VERDICTS}
    n_valid = sum(valid.values())
    return {
        "asked": len(rows),
        "answered_valid": n_valid,
        "counts": counts,
        "verdict_counts": valid,
        # 分母只算有效判定；只有 success 记正例（partial 不算成功）
        "success_rate": round(valid["success"] / n_valid, 4) if n_valid else None,
        "partial_inclusive_success_rate": (
            round((valid["success"] + valid["partial"]) / n_valid, 4) if n_valid else None
        ),
        "described": counts.get("described", 0),
        "unparsed": counts.get("unparsed", 0),
        "no_answer": sum(counts.get(key, 0) for key in NO_ANSWER_VERDICTS),
        "marks": [
            {
                "marker_index": row.get("marker_index"),
                "inference_frame_index": row.get("inference_frame_index"),
                "task_segment_index": row.get("task_segment_index"),
                "verdict": row.get("verdict"),
                "remark": row.get("remark") or "",
                "wall_time": row.get("asked_at_wall"),
                "timing": row.get("timing") or {},
            }
            for row in rows
        ],
    }


def format_eval_line(stats: Dict[str, Any]) -> str:
    """人读一眼就懂的单行汇总（run 结束时打印，也进 run.log）。"""
    rate = stats.get("success_rate")
    rate_txt = "未评测" if rate is None else f"{rate * 100:.1f}%"
    counts = stats.get("verdict_counts") or {}
    return (
        f"人工评测 {stats.get('asked', 0)} 次追问"
        f"（有效 {stats.get('answered_valid', 0)}）："
        f"成功 {counts.get('success', 0)} / 部分 {counts.get('partial', 0)}"
        f" / 失败 {counts.get('fail', 0)}；成功率 {rate_txt}"
        f"；描述 {stats.get('described', 0)}"
        f"；未答或跳过 {stats.get('no_answer', 0)}"
        f"；无法解析 {stats.get('unparsed', 0)}"
    )


def _read_line_with_timeout(timeout_sec: float) -> Optional[str]:
    """有终端就等一行。``timeout_sec <= 0`` 时一直等到输入，不超时。

    stdin 不是 tty（被管道/重定向）时抛 ``StdinUnavailable``，由调用方降级记
    ``no_stdin`` —— 后台 nohup 跑批不该被一个永远读不到的问题卡住。
    """
    import select
    import sys

    if not sys.stdin.isatty():
        raise StdinUnavailable("stdin 不是终端")
    timeout = float(timeout_sec)
    wait_forever = timeout <= 0
    deadline = None if wait_forever else time.monotonic() + timeout
    while True:
        if deadline is None:
            wait = None
        else:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            wait = min(0.2, remaining)
        ready, _, _ = select.select([sys.stdin], [], [], wait)
        if not ready:
            continue
        line = sys.stdin.readline()
        if line == "":  # EOF：stdin 已被关掉
            raise StdinUnavailable("stdin 已关闭")
        return line


class HumanEvalPrompter:
    """单线程串行追问：一次只问一个问题，多按的 Ctrl+. 排队等答。

    ``read_line`` / ``write`` 可注入，便于在没有终端的单测里跑完整流程。
    """

    #: 排队上限：现场连按也不该堆一堆没答的问题，超出的那条直接记 burst
    MAX_PENDING = 8

    def __init__(
        self,
        *,
        timeout_sec: float = 0.0,
        on_result: Callable[[Dict[str, Any]], None],
        read_line: Optional[Callable[[float], Optional[str]]] = None,
        write: Optional[Callable[[str], None]] = None,
        interactive: bool = True,
    ) -> None:
        self.timeout_sec = float(timeout_sec)
        self.on_result = on_result
        self._read_line = read_line or _read_line_with_timeout
        self._write = write or (lambda text: print(text, flush=True))
        self._interactive = bool(interactive)
        self._pending: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._asked_count = 0
        self._no_stdin_recorded = False
        self._lock = threading.Lock()

    @property
    def disabled(self) -> bool:
        return not self._interactive

    def submit(self, context: Dict[str, Any]) -> None:
        """打点处调用：只入队，绝不等待答案。"""
        if not self._interactive:
            # 已经知道这里没有终端：只补一条说明，之后不再重复落条刷屏
            with self._lock:
                first = not self._no_stdin_recorded
                self._no_stdin_recorded = True
            if first:
                self._emit(context, "no_stdin")
            return
        if self._pending.qsize() >= self.MAX_PENDING:
            self._emit(context, "burst")
            return
        if self._thread is None or not self._thread.is_alive():
            with self._lock:
                if self._thread is None or not self._thread.is_alive():
                    self._thread = threading.Thread(
                        target=self._run, name="human-eval-prompt", daemon=True
                    )
                    self._thread.start()
        self._pending.put(context)

    def shutdown(self, wait_sec: float = 0.0) -> None:
        """run 结束时把还没问的补齐成 aborted，保证"问过几次"与落盘条数一致。"""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and wait_sec > 0:
            # 正在作答的那条值得等一下；无超时时不能按 timeout_sec 无限 join
            join_cap = (
                float(wait_sec)
                if self.timeout_sec <= 0
                else min(float(wait_sec), self.timeout_sec + 1.0)
            )
            thread.join(timeout=join_cap)
        self._drain_pending_as_aborted()

    def _drain_pending_as_aborted(self) -> None:
        while True:
            try:
                context = self._pending.get_nowait()
            except queue.Empty:
                break
            self._emit(context, "aborted")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                context = self._pending.get(timeout=0.1)
            except queue.Empty:
                continue
            self._ask_one(context)

    def _ask_one(self, context: Dict[str, Any]) -> None:
        self._print_prompt(context)
        asked_at = time.time()
        t0 = time.perf_counter()
        try:
            answer = self._read_line(self.timeout_sec)
        except StdinUnavailable:
            self._interactive = False
            self._no_stdin_recorded = True
            self._emit(context, "no_stdin", asked_at=asked_at)
            return
        except Exception:
            # 终端异常不能让推理挂掉，降级后不再追问
            self._interactive = False
            self._no_stdin_recorded = True
            self._emit(context, "no_stdin", asked_at=asked_at)
            return
        waited = time.perf_counter() - t0
        if answer is None:
            if self.timeout_sec <= 0:
                # 默认一直等输入；注入的 read_line 若返回 None，按跳过而不是 timeout
                self._emit(context, "skipped", asked_at=asked_at, waited_sec=waited)
                return
            self._emit(context, "timeout", asked_at=asked_at, waited_sec=waited)
            self._write("[EVAL] 超时未回答，本条记为 timeout（不进成功率的分子分母）")
            return
        parsed = parse_eval_answer(answer)
        self._emit(
            context,
            parsed["verdict"],
            remark=parsed["remark"],
            raw_answer=parsed["raw_answer"],
            parse_error=parsed["parse_error"],
            asked_at=asked_at,
            waited_sec=waited,
        )

    def _print_prompt(self, context: Dict[str, Any]) -> None:
        self._write("")
        self._write(
            "[EVAL] 刚按 {key} 标记：推理 #{frame}（marker #{marker}）—— 请用文字描述这次结果".format(
                key=context.get("key") or "Ctrl+.",
                frame=context.get("inference_frame_index"),
                marker=context.get("marker_index"),
            )
        )
        self._write(f"[EVAL]   {PROMPT_HINT}")
        if self.timeout_sec <= 0:
            self._write(f"[EVAL]   {PROMPT_WAIT_HINT}")
        else:
            self._write(
                f"[EVAL]   输入描述后回车进入下一步；空行=跳过。"
                f"（{self.timeout_sec:g}s 未回答记 timeout，不进成功率）"
            )

    def _emit(
        self,
        context: Dict[str, Any],
        verdict: str,
        *,
        remark: str = "",
        raw_answer: str = "",
        parse_error: bool = False,
        asked_at: Optional[float] = None,
        waited_sec: float = 0.0,
    ) -> None:
        with self._lock:
            self._asked_count += 1
            index = self._asked_count
        row = dict(context)
        row.update(
            {
                "type": "inference_human_eval_mark",
                "eval_index": int(index),
                "verdict": verdict,
                "verdict_label": VERDICT_LABELS.get(verdict, verdict),
                "remark": remark,
                "raw_answer": raw_answer,
                "parse_error": bool(parse_error),
                "asked_at_wall": float(asked_at if asked_at is not None else time.time()),
                "iso_time": datetime.now().astimezone().isoformat(),
                "answer_wait_sec": round(float(waited_sec), 3),
                "timeout_sec": float(self.timeout_sec),
            }
        )
        self.on_result(row)
