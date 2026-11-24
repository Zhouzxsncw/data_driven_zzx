"""Ctrl+. 打点后追问人工评测结果：解析、统计、非阻塞落库的单元测试。

覆盖三段：
  recording/human_eval        parse_eval_answer / timing_context / compute_eval_stats
                              / HumanEvalPrompter（注入 read_line，不需要终端）
  core/engine                 _ask_human_eval / _record_human_eval / _drain_human_eval_marks
                              （SimpleNamespace 绑定，不构造真实引擎）
  recording/experiment_index  _human_eval_entry 精简摘要
"""

from __future__ import annotations

import json
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import infer.core.engine as engine_module
from infer.core.engine import InferEngine
from infer.recording.experiment_index import _human_eval_entry
from infer.recording.human_eval import (
    HumanEvalPrompter,
    StdinUnavailable,
    compute_eval_stats,
    format_eval_line,
    parse_eval_answer,
    timing_context,
)


class ParseEvalAnswerTest(unittest.TestCase):
    def test_number_and_letter_tokens(self):
        cases = {
            "1": "success",
            " y ": "success",
            "ok": "success",
            "成功": "success",
            "2": "partial",
            "p": "partial",
            "未完成": "partial",
            "3": "fail",
            "n": "fail",
            "失败": "fail",
            "0": "skipped",
            "": "skipped",
            None: "skipped",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_eval_answer(text)["verdict"], expected)

    def test_remark_after_verdict_is_kept(self):
        out = parse_eval_answer("2 夹爪提前闭合")
        self.assertEqual(out["verdict"], "partial")
        self.assertEqual(out["remark"], "夹爪提前闭合")
        self.assertEqual(out["raw_answer"], "2 夹爪提前闭合")
        self.assertFalse(out["parse_error"])

    def test_unknown_words_are_kept_as_description(self):
        out = parse_eval_answer("抖得厉害")
        self.assertEqual(out["verdict"], "described")
        self.assertFalse(out["parse_error"])
        # 人的原话一个字都不该丢
        self.assertEqual(out["remark"], "抖得厉害")

    def test_freeform_description_is_kept(self):
        out = parse_eval_answer("夹爪提前闭合，轨迹发抖")
        self.assertEqual(out["verdict"], "described")
        self.assertEqual(out["remark"], "夹爪提前闭合，轨迹发抖")
        self.assertFalse(out["parse_error"])


class TimingContextTest(unittest.TestCase):
    def test_picks_known_keys_only(self):
        ctx = timing_context(
            {
                "frame": 12,
                "preprocess_sec": 0.01,
                "model_sec": 0.32,
                "ik_sec": 0.0096,
                "network_sec": 0.03,
                "something_else": 1.0,
            }
        )
        self.assertEqual(
            ctx,
            {
                "preprocess_sec": 0.01,
                "model_sec": 0.32,
                "ik_sec": 0.0096,
                "network_sec": 0.03,
            },
        )

    def test_non_dict_gives_empty(self):
        self.assertEqual(timing_context(None), {})
        self.assertEqual(timing_context("x"), {})


def _mark(verdict, frame=1, remark=""):
    return {
        "verdict": verdict,
        "inference_frame_index": frame,
        "marker_index": frame,
        "task_segment_index": 0,
        "remark": remark,
        "asked_at_wall": 0.0,
        "timing": {},
    }


class ComputeEvalStatsTest(unittest.TestCase):
    def test_rates_over_valid_verdicts_only(self):
        stats = compute_eval_stats(
            [
                _mark("success", 1),
                _mark("success", 2),
                _mark("partial", 3, "差一点"),
                _mark("fail", 4),
                _mark("timeout", 5),
                _mark("skipped", 6),
            ]
        )
        self.assertEqual(stats["asked"], 6)
        # 分母只算 success/partial/fail 这三档有效判定
        self.assertEqual(stats["answered_valid"], 4)
        self.assertAlmostEqual(stats["success_rate"], 0.5)
        self.assertAlmostEqual(stats["partial_inclusive_success_rate"], 0.75)
        self.assertEqual(stats["no_answer"], 2)
        self.assertEqual(stats["marks"][3]["inference_frame_index"], 4)

    def test_no_valid_answer_gives_none_not_zero(self):
        """只有超时/跳过时必须返回 None，否则"没评测"会被读成"0% 成功"。"""
        stats = compute_eval_stats([_mark("timeout"), _mark("no_stdin")])
        self.assertIsNone(stats["success_rate"])
        self.assertIsNone(stats["partial_inclusive_success_rate"])
        self.assertEqual(stats["answered_valid"], 0)
        self.assertIn("未评测", format_eval_line(stats))

    def test_described_answers_are_not_in_success_rate(self):
        stats = compute_eval_stats(
            [
                _mark("described", 1, "夹爪提前闭合"),
                _mark("success", 2),
                _mark("fail", 3),
            ]
        )
        self.assertEqual(stats["described"], 1)
        self.assertEqual(stats["answered_valid"], 2)
        self.assertAlmostEqual(stats["success_rate"], 0.5)
        self.assertIn("描述 1", format_eval_line(stats))


class _Recorder:
    """收 prompter 回调结果；用条件变量等待"第 N 条已到"，不靠 sleep。"""

    def __init__(self) -> None:
        self.rows = []
        self._cond = threading.Condition()

    def __call__(self, row):
        with self._cond:
            self.rows.append(row)
            self._cond.notify_all()

    def wait(self, n=1, timeout=2.0):
        with self._cond:
            return self._cond.wait_for(lambda: len(self.rows) >= n, timeout=timeout)


class HumanEvalPrompterTest(unittest.TestCase):
    def _prompter(self, answers, recorder, **kw):
        it = iter(answers)
        return HumanEvalPrompter(
            on_result=recorder,
            read_line=lambda timeout: next(it),
            write=lambda text: None,
            **kw,
        )

    def test_answer_is_recorded_with_context(self):
        rec = _Recorder()
        self._prompter(["2 夹爪提前闭合"], rec).submit(
            {
                "marker_index": 7,
                "inference_frame_index": 473,
                "timing": {"ik_sec": 0.0096},
            }
        )
        self.assertTrue(rec.wait(), "追问结果没有回调")
        row = rec.rows[0]
        self.assertEqual(row["verdict"], "partial")
        self.assertEqual(row["verdict_label"], "部分完成")
        self.assertEqual(row["remark"], "夹爪提前闭合")
        self.assertEqual(row["marker_index"], 7)
        self.assertEqual(row["inference_frame_index"], 473)
        self.assertEqual(row["timing"], {"ik_sec": 0.0096})
        self.assertEqual(row["type"], "inference_human_eval_mark")
        self.assertEqual(row["eval_index"], 1)

    def test_timeout_is_still_recorded_when_configured(self):
        rec = _Recorder()
        self._prompter([None], rec, timeout_sec=12.0).submit({"marker_index": 1})
        self.assertTrue(rec.wait())
        self.assertEqual(rec.rows[0]["verdict"], "timeout")
        self.assertEqual(rec.rows[0]["remark"], "")

    def test_wait_forever_does_not_record_timeout(self):
        rec = _Recorder()
        self._prompter([None], rec, timeout_sec=0.0).submit({"marker_index": 1})
        self.assertTrue(rec.wait())
        self.assertEqual(rec.rows[0]["verdict"], "skipped")

    def test_prompt_asks_for_description(self):
        lines = []
        rec = _Recorder()
        HumanEvalPrompter(
            timeout_sec=0.0,
            on_result=rec,
            read_line=lambda timeout: "夹爪提前闭合",
            write=lines.append,
        ).submit(
            {
                "marker_index": 7,
                "inference_frame_index": 473,
                "key": "Ctrl+.",
            }
        )
        self.assertTrue(rec.wait(), "追问结果没有回调")
        text = "\n".join(lines)
        self.assertIn("请用文字描述", text)
        self.assertIn("提示", text)
        self.assertIn("PI0.5", text)
        self.assertIn("一直等待", text)
        self.assertNotIn("1=成功", text)
        self.assertNotIn("未回答记 timeout", text)
        self.assertEqual(rec.rows[0]["verdict"], "described")
        self.assertEqual(rec.rows[0]["remark"], "夹爪提前闭合")

    def test_unusable_stdin_records_and_disables_asking(self):
        rec = _Recorder()

        def boom(timeout):
            raise StdinUnavailable("无终端")

        prompter = self._prompter([], rec)
        prompter._read_line = boom
        prompter.submit({"marker_index": 1})
        self.assertTrue(rec.wait())
        self.assertEqual(rec.rows[0]["verdict"], "no_stdin")
        self.assertTrue(prompter.disabled)
        # 已知没终端：再按不再打扰，也不再落条
        prompter.submit({"marker_index": 2})
        self.assertFalse(rec.wait(2, 0.3))
        self.assertEqual(len(rec.rows), 1)

    def test_non_interactive_prompter_records_no_stdin_without_thread(self):
        rec = _Recorder()
        prompter = HumanEvalPrompter(
            on_result=rec,
            read_line=lambda timeout: "1",
            write=lambda text: None,
            interactive=False,
        )
        prompter.submit({"marker_index": 1})
        self.assertTrue(rec.wait())
        self.assertEqual(rec.rows[0]["verdict"], "no_stdin")
        self.assertIsNone(prompter._thread)

    def test_pending_questions_are_aborted_on_shutdown(self):
        """排队没问的问题必须补齐成 aborted，"追问了几次"才对得上落盘条数。"""
        rec = _Recorder()
        prompter = self._prompter([], rec)
        # 直接构造"已入队但没轮到问"的状态，避免依赖线程时序
        prompter._pending.put({"marker_index": 1})
        prompter._pending.put({"marker_index": 2})
        prompter.shutdown()
        self.assertTrue(rec.wait(2))
        self.assertEqual({r["verdict"] for r in rec.rows}, {"aborted"})
        self.assertEqual({r["marker_index"] for r in rec.rows}, {1, 2})

    def test_overflow_queue_is_recorded_as_burst(self):
        rec = _Recorder()
        prompter = self._prompter([], rec)
        for i in range(prompter.MAX_PENDING):
            prompter._pending.put({"marker_index": i})
        prompter.submit({"marker_index": 999})
        self.assertTrue(rec.wait())
        self.assertEqual(rec.rows[0]["verdict"], "burst")
        self.assertEqual(rec.rows[0]["marker_index"], 999)
        prompter.shutdown()


class EngineHumanEvalWiringTest(unittest.TestCase):
    """engine 这几个方法只碰列出的属性，可用 SimpleNamespace 绑定验证。"""

    def _engine(self, *, enabled=True, timeout=12.0, prompter=None, qsize=64):
        eng = SimpleNamespace(
            cfg=SimpleNamespace(
                session=SimpleNamespace(
                    eval_prompt_on_marker=enabled,
                    eval_prompt_timeout_sec=timeout,
                )
            ),
            _human_eval_prompter=prompter,
            _human_eval_rows=[],
            _human_eval_lock=threading.Lock(),
            _human_eval_write_queue=queue.Queue(maxsize=qsize),
            _infer_timing_log=[],
        )
        # 用真实现而不是替身，确保 timing 取的是最近一次推理的条目
        eng._last_infer_timing_entry = lambda: InferEngine._last_infer_timing_entry(eng)
        eng._record_human_eval = lambda row: InferEngine._record_human_eval(eng, row)
        return eng

    def test_disabled_switch_creates_no_prompter(self):
        eng = self._engine(enabled=False)
        InferEngine._ask_human_eval(eng, {"marker_index": 1})
        self.assertIsNone(eng._human_eval_prompter)

    def test_first_press_builds_prompter_with_configured_timeout(self):
        built = {}

        def fake_prompter(**kwargs):
            built.update(kwargs)
            return SimpleNamespace(
                disabled=False,
                submit=lambda ctx: None,
                timeout_sec=kwargs.get("timeout_sec"),
            )

        eng = self._engine(timeout=0.0)
        with mock.patch.object(engine_module, "HumanEvalPrompter", fake_prompter):
            InferEngine._ask_human_eval(eng, {"marker_index": 1})
        self.assertAlmostEqual(built["timeout_sec"], 0.0)
        self.assertTrue(callable(built["on_result"]))
        self.assertAlmostEqual(eng._human_eval_prompter.timeout_sec, 0.0)

    def test_submit_carries_marker_and_latest_timing(self):
        submitted = []
        prompter = SimpleNamespace(
            disabled=False, submit=lambda ctx: submitted.append(ctx)
        )
        eng = self._engine(prompter=prompter)
        eng._infer_timing_log.append(
            {"frame": 9, "model_sec": 0.31, "ik_sec": 0.0096, "network_sec": 0.03}
        )
        InferEngine._ask_human_eval(
            eng,
            {
                "marker_index": 3,
                "inference_frame_index": 473,
                "task_segment_index": 1,
                "key": "ctrl+.",
                "source": "pynput",
                "wall_time": 111.0,
            },
        )
        self.assertEqual(len(submitted), 1)
        ctx = submitted[0]
        self.assertEqual(ctx["marker_index"], 3)
        self.assertEqual(ctx["inference_frame_index"], 473)
        self.assertEqual(ctx["key"], "ctrl+.")
        self.assertEqual(ctx["marker_wall_time"], 111.0)
        # 人工判定与当次推理的客观分段耗时配成同一条
        self.assertEqual(
            ctx["timing"],
            {"model_sec": 0.31, "ik_sec": 0.0096, "network_sec": 0.03},
        )

    def test_no_inference_yet_still_asks_with_empty_timing(self):
        submitted = []
        eng = self._engine(
            prompter=SimpleNamespace(
                disabled=False, submit=lambda ctx: submitted.append(ctx)
            )
        )
        InferEngine._ask_human_eval(eng, {"marker_index": 1})
        self.assertEqual(submitted[0]["timing"], {})

    def test_already_disabled_prompter_is_not_resubmitted(self):
        calls = []
        eng = self._engine(
            prompter=SimpleNamespace(
                disabled=True, submit=lambda ctx: calls.append(ctx)
            )
        )
        InferEngine._ask_human_eval(eng, {"marker_index": 1})
        self.assertEqual(calls, [])

    def test_record_appends_row_and_queues_write(self):
        eng = self._engine()
        row = {"verdict": "success", "marker_index": 1}
        InferEngine._record_human_eval(eng, row)
        self.assertEqual(eng._human_eval_rows, [row])
        self.assertIs(eng._human_eval_write_queue.get_nowait(), row)

    def test_record_writes_directly_when_queue_full(self):
        eng = self._engine(qsize=1)
        written = []
        eng._write_human_eval_row = lambda row: written.append(row)
        eng._human_eval_write_queue.put({"占位": 1})
        row = {"verdict": "fail", "marker_index": 2}
        InferEngine._record_human_eval(eng, row)
        self.assertEqual(written, [row])

    def test_drain_flushes_every_queued_row(self):
        eng = self._engine()
        written = []
        eng._write_human_eval_row = lambda row: written.append(row)
        eng._human_eval_write_queue.put({"i": 1})
        eng._human_eval_write_queue.put({"i": 2})
        InferEngine._drain_human_eval_marks(eng)
        self.assertEqual(written, [{"i": 1}, {"i": 2}])
        # 队列空时也必须安全返回
        InferEngine._drain_human_eval_marks(eng)
        self.assertEqual(len(written), 2)

    def test_stats_empty_when_no_press(self):
        self.assertEqual(InferEngine._human_eval_stats(self._engine()), {})

    def test_stats_from_recorded_rows(self):
        eng = self._engine()
        eng._human_eval_path = SimpleNamespace(name="inference_eval_marks.jsonl")
        eng._human_eval_rows = [_mark("success", 1), _mark("fail", 2)]
        stats = InferEngine._human_eval_stats(eng)
        self.assertEqual(stats["asked"], 2)
        self.assertAlmostEqual(stats["success_rate"], 0.5)
        self.assertEqual(stats["marks_file"], "inference_eval_marks.jsonl")


class HumanEvalEntryIndexTest(unittest.TestCase):
    def test_summary_is_compacted_for_index(self):
        stats = compute_eval_stats(
            [
                _mark("success", 1),
                _mark("fail", 8),
                _mark("partial", 9),
                _mark("timeout", 10),
            ]
        )
        entry = _human_eval_entry(stats)
        self.assertEqual(entry["asked"], 4)
        self.assertEqual(entry["answered_valid"], 3)
        self.assertAlmostEqual(entry["success_rate"], round(1 / 3, 4))
        self.assertEqual(entry["fail"], 1)
        self.assertEqual(entry["bad_frames"], [8, 9])
        # 明细不进索引，避免实验索引被撑大
        self.assertNotIn("marks", entry)

    def test_unanswered_run_has_null_rate(self):
        entry = _human_eval_entry(compute_eval_stats([_mark("timeout", 1)]))
        self.assertIsNone(entry["success_rate"])
        self.assertNotIn("bad_frames", entry)


class HumanEvalFilesEndToEndTest(unittest.TestCase):
    """追问 → 回调 → 主循环落盘 → 结束汇总，走真文件（临时目录）。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.rec = _Recorder()
        self.flow_events = []
        self.log_lines = []
        self.alerts = []
        eng = SimpleNamespace(
            _human_eval_path=self.tmp / "inference_eval_marks.jsonl",
            _human_eval_summary_path=self.tmp / "inference_eval_summary.json",
            _human_eval_rows=[],
            _human_eval_lock=threading.Lock(),
            _human_eval_write_queue=queue.Queue(maxsize=64),
            _infer_timing_log=[
                {"frame": 473, "model_sec": 0.31, "ik_sec": 0.0096, "flow_sec": 0.27}
            ],
        )
        eng._flow_event = lambda event, row: self.flow_events.append((event, row))
        eng._human_log = lambda level, msg: self.log_lines.append(msg)
        eng.alerts = SimpleNamespace(
            warning=lambda code, msg, context=None: self.alerts.append((code, msg))
        )
        eng._record_human_eval = lambda row: InferEngine._record_human_eval(eng, row)
        eng._last_infer_timing_entry = lambda: InferEngine._last_infer_timing_entry(eng)
        eng._write_human_eval_row = lambda row: InferEngine._write_human_eval_row(eng, row)
        eng._drain_human_eval_marks = lambda: InferEngine._drain_human_eval_marks(eng)
        eng._human_eval_stats = lambda: InferEngine._human_eval_stats(eng)
        eng._finalize_human_eval_stats = lambda: InferEngine._finalize_human_eval_stats(eng)
        answers = iter(["1", "3 撞桌沿"])
        eng._human_eval_prompter = HumanEvalPrompter(
            timeout_sec=1.0,
            # 回调链：既进引擎的排队，也让测试知道第 N 条已到
            on_result=lambda row: (eng._record_human_eval(row), self.rec(row)),
            read_line=lambda timeout: next(answers, None),
            write=lambda text: None,
        )
        self.eng = eng

    def _press(self, marker_index, frame):
        self.eng._human_eval_prompter.submit(
            {
                "marker_index": marker_index,
                "inference_frame_index": frame,
                "task_segment_index": 0,
                "key": "ctrl+.",
                "timing": timing_context(self.eng._last_infer_timing_entry()),
            }
        )

    def test_marks_and_summary_are_written(self):
        self._press(1, 473)
        self._press(2, 480)
        self.assertTrue(self.rec.wait(2), "两条答案没有都回到引擎")
        # 落盘只发生在主循环侧：drain 之前文件不该存在
        self.assertFalse(self.eng._human_eval_path.exists())

        self.eng._drain_human_eval_marks()
        lines = self.eng._human_eval_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        rows = [json.loads(line) for line in lines]
        self.assertEqual([r["verdict"] for r in rows], ["success", "fail"])
        self.assertEqual(rows[1]["remark"], "撞桌沿")
        self.assertEqual(rows[0]["timing"]["ik_sec"], 0.0096)
        self.assertEqual(
            [ev for ev, _ in self.flow_events],
            ["inference_human_eval_mark", "inference_human_eval_mark"],
        )

        stats = self.eng._finalize_human_eval_stats()
        self.assertEqual(stats["asked"], 2)
        self.assertAlmostEqual(stats["success_rate"], 0.5)
        summary = json.loads(
            self.eng._human_eval_summary_path.read_text(encoding="utf-8")
        )
        self.assertEqual(summary["verdict_counts"]["fail"], 1)
        self.assertEqual(summary["marks_file"], "inference_eval_marks.jsonl")
        self.assertTrue(any("[EVAL]" in line for line in self.log_lines))
        self.assertEqual(self.alerts, [])

    def test_unanswered_press_still_lands_in_marks(self):
        """没答（timeout）也必须有条，否则"追问过几次"从文件里根本看不出来。"""
        self.eng._human_eval_prompter.timeout_sec = 1.0
        self._press(1, 473)
        self._press(2, 480)
        self._press(3, 487)  # answers 已空 → read_line 返回 None → timeout
        self.assertTrue(self.rec.wait(3))
        self.eng._drain_human_eval_marks()
        stats = self.eng._finalize_human_eval_stats()
        self.assertEqual(stats["asked"], 3)
        self.assertEqual(stats["counts"].get("timeout"), 1)
        self.assertEqual(stats["answered_valid"], 2)
        self.assertAlmostEqual(stats["success_rate"], 0.5)

    def test_finalize_without_any_press_writes_nothing(self):
        self.assertEqual(self.eng._finalize_human_eval_stats(), {})
        self.assertFalse(self.eng._human_eval_summary_path.exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
