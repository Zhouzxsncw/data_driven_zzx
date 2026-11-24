#!/usr/bin/env python3
"""把 Cosmos3 DREAM 和真机推理录像按时间对齐，合成 REAL vs DREAM 对比视频。

每包一张短片，上下两行 × 三列：
  上 = AFTER（推理结束后 1.13s 真机：胸前 / 左腕 / 第三视角）
  下 = DREAM 对应三路（concat 裁开：front / left / right）
写到 <action-step>/dream_vs_real/chunks/pair_XXXX_after_vs_dream.mp4。

没有相机 mp4、或时间戳对不上录像时：REAL 用当前 cond → 下一包 cond 逐帧插值，
不要把时间钳到录像首/尾帧（那会整段冻住）。缺某一路相机时只补那一列，不要整行留黑。

用法：
  /data/cosmos3-infer/bin/python recipes/cosmos3/merge_dream_vs_real_video.py \
      --action-step /media/xk/T9_3/checkpoint/cosmos3_00091/iter_000003000/action-step_16

  /data/cosmos3-infer/bin/python recipes/cosmos3/merge_dream_vs_real_video.py --auto
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dream_compare_cosmos3_v2 import (  # noqa: E402
    FfmpegPipe,
    _fit,
    _vstack,
    annotate,
)

IMAGE_SIZE = 256
FPS = 15.0
DREAM_FRAMES = 17
TZ_OFFSET = 8 * 3600
CELL_W, CELL_H = 320, 240
GRID_W, GRID_H = CELL_W * 3, CELL_H * 2


def _iso_unix(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()


def _eval_rgb_and_out(ev: dict | None) -> tuple[float | None, float | None, float | None]:
    """一次推理记下的两个时刻：RGB 进模型、模型出结果。

    eval_log.wall_time 是 action chunk 执行完才写，晚大约一整包，不能当模型输出。
    模型输出 = timestamps.image + infer_timing.infer_time_sec。
    """
    if not ev:
        return None, None, None
    ts = ev.get("timestamps") or {}
    t_rgb = ts.get("image")
    t_rgb = float(t_rgb) if t_rgb not in (None, 0, 0.0) else None
    inf = (ev.get("infer_timing") or {}).get("infer_time_sec")
    t_out = (t_rgb + float(inf)) if t_rgb is not None and inf is not None else None
    t_logged = float(ev["wall_time"]) if ev.get("wall_time") is not None else None
    return t_rgb, t_out, t_logged


def _nearest_eval(evals: list[dict], t: float, max_dt: float = 4.0) -> dict | None:
    """latent.recorded_at ≈ 模型刚出结果，落在 [RGB进, wall_time] 里；不要按 RGB 时刻就近（会错配到下一包）。"""
    contained = None
    best = None
    best_dt = max_dt
    for e in evals:
        t_rgb, t_out, t_logged = _eval_rgb_and_out(e)
        if t_rgb is not None and t_logged is not None and (t_rgb - 0.05) <= t <= (t_logged + 0.05):
            contained = e
            break
        ref = t_out if t_out is not None else t_rgb
        if ref is None:
            continue
        dt = abs(ref - t)
        if dt < best_dt:
            best_dt = dt
            best = e
    return contained or best


def _concat_view(front: Image.Image, left: Image.Image | None, right: Image.Image | None) -> Image.Image:
    top = front.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    if left is None or right is None:
        return top
    half = IMAGE_SIZE // 2
    canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE + half))
    canvas.paste(top, (0, 0))
    canvas.paste(left.convert("RGB").resize((half, half), Image.BILINEAR), (0, IMAGE_SIZE))
    canvas.paste(right.convert("RGB").resize((half, half), Image.BILINEAR), (half, IMAGE_SIZE))
    return canvas


def _letterbox(im: Image.Image, w: int, h: int) -> Image.Image:
    im = im.convert("RGB")
    scale = min(w / im.width, h / im.height)
    nw = max(1, int(im.width * scale))
    nh = max(1, int(im.height * scale))
    im = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    canvas.paste(im, ((w - nw) // 2, (h - nh) // 2))
    return canvas


def _blank(w: int, h: int, text: str) -> Image.Image:
    return annotate(Image.new("RGB", (w, h), (16, 16, 16)), text)


def _panel(im: Image.Image | None, text: str, w: int = CELL_W, h: int = CELL_H) -> Image.Image:
    if im is None:
        return _blank(w, h, text)
    return annotate(_letterbox(im, w, h), text)


def _split_dream_concat(dream: Image.Image) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Cosmos concat：上 front/head 正方形，下左/右腕各一半。

    v2 dream mp4 顶部有 24px 标注条，裁 front 时跳过，避免 head 被条带顶歪。
    """
    im = dream.convert("RGB")
    w, h = im.size
    bar = 24 if h >= 48 else 0
    front_h = int(round(h * 2 / 3))
    front = im.crop((0, bar, w, front_h))
    if h <= front_h:
        return front, front, front
    half = w // 2
    left = im.crop((0, front_h, half, h))
    right = im.crop((half, front_h, w, h))
    return front, left, right


def _grid_3view(
    real_left: Image.Image | None,
    real_head: Image.Image | None,
    real_right: Image.Image | None,
    dream: Image.Image,
    *,
    chunk_idx: int,
    t: int,
    n_d: int,
) -> Image.Image:
    """列序：左腕 | head | 右腕。00091 录像文件名和 concat 对不上：
    cam_left_wrist.mp4 = concat front/head，cam_mid_chest.mp4 = concat 左腕。
    """
    df, dl, dr = _split_dream_concat(dream)
    tag = f"chunk{chunk_idx} t={t}/{n_d - 1}"
    top = [
        _panel(real_left, f"AFTER left {tag}"),
        _panel(real_head, f"AFTER head {tag}"),
        _panel(real_right, f"AFTER right {tag}"),
    ]
    bot = [
        _panel(dl, f"DREAM left {tag}"),
        _panel(df, f"DREAM head {tag}"),
        _panel(dr, f"DREAM right {tag}"),
    ]
    canvas = Image.new("RGB", (GRID_W, GRID_H), (0, 0, 0))
    for c, im in enumerate(top):
        canvas.paste(im, (c * CELL_W, 0))
    for c, im in enumerate(bot):
        canvas.paste(im, (c * CELL_W, CELL_H))
    return canvas


def _load_eval(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    for line in path.open():
        d = json.loads(line)
        if d.get("type") == "eval_inference_frame":
            rows.append(d)
    return rows


def _load_index(session: Path) -> list[dict]:
    p = session / "index.jsonl"
    rows = []
    if p.is_file():
        for line in p.open():
            rows.append(json.loads(line))
        rows.sort(key=lambda r: int(r.get("chunk_idx", 0)))
        return rows
    for pt in sorted(session.glob("chunk_*.pt")):
        idx = int(pt.stem.split("_")[1])
        rows.append({"chunk_idx": idx, "pt": pt.name, "recorded_at": None})
    return rows


def _session_meta(session: Path) -> dict:
    p = session / "session.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def _load_ts(jsonl: Path, camera: str) -> tuple[np.ndarray, np.ndarray]:
    ts, idx = [], []
    if not jsonl.is_file():
        return np.array([]), np.array([])
    for line in jsonl.open():
        d = json.loads(line)
        if d.get("camera") != camera:
            continue
        ts.append(float(d["image_timestamp"]))
        idx.append(int(d["video_frame_index"]))
    if not ts:
        return np.array([]), np.array([])
    order = np.argsort(ts)
    return np.asarray(ts, dtype=np.float64)[order], np.asarray(idx, dtype=np.int32)[order]


class VideoCursor:
    def __init__(self, mp4: Path) -> None:
        self.mp4 = mp4
        self.cap = cv2.VideoCapture(str(mp4))
        if not self.cap.isOpened():
            raise RuntimeError(f"打不开视频 {mp4}")
        self.pos = -1
        self.frame: Image.Image | None = None

    def at(self, frame_idx: int) -> Image.Image | None:
        if frame_idx < 0:
            return self.frame
        if frame_idx < self.pos:
            self.cap.release()
            self.cap = cv2.VideoCapture(str(self.mp4))
            self.pos = -1
        while self.pos < frame_idx:
            ok, arr = self.cap.read()
            self.pos += 1
            if not ok:
                return self.frame
            self.frame = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
        return self.frame

    def close(self) -> None:
        self.cap.release()


def _read_dream_mp4(path: Path) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, arr = cap.read()
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def _discover_sessions(latents: Path, t0: float | None, t1: float | None) -> list[Path]:
    out = []
    if not latents.is_dir():
        return out
    for s in sorted(latents.glob("session_*")):
        n = len(list(s.glob("chunk_*.pt")))
        if n == 0:
            continue
        meta = _session_meta(s)
        created = meta.get("created_at")
        if t0 is not None and created:
            cu = _iso_unix(created)
            if cu < t0 - 90 or (t1 is not None and cu > t1 + 90):
                continue
        out.append(s)
    return out


def _run_window(action_step: Path) -> tuple[float | None, float | None]:
    meta = json.loads((action_step / "meta.json").read_text()) if (action_step / "meta.json").is_file() else {}
    summ = json.loads((action_step / "inference_summary.json").read_text()) if (action_step / "inference_summary.json").is_file() else {}
    t0 = _iso_unix(meta["started_at"]) if meta.get("started_at") else None
    t1 = _iso_unix(summ["ended_at"]) if summ.get("ended_at") else None
    return t0, t1


def _nearest_idx(ts: np.ndarray, t: float) -> int:
    if ts.size == 0:
        return 0
    i = int(np.searchsorted(ts, t))
    i = min(max(i, 0), len(ts) - 1)
    if i > 0 and abs(ts[i - 1] - t) < abs(ts[i] - t):
        return i - 1
    return i


def _synth_ts(mp4: Path, t0: float, fps: float = FPS) -> tuple[np.ndarray, np.ndarray]:
    """没有 frame_timestamps.jsonl 时，按 started_at + 恒定 fps 对上 mp4 帧号。"""
    cap = cv2.VideoCapture(str(mp4))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if n <= 1:
        return np.array([]), np.array([])
    return t0 + np.arange(n, dtype=np.float64) / fps, np.arange(n, dtype=np.int32)


def _camera_frame_ids(
    ts: np.ndarray, fidx: np.ndarray, t0: float, n_d: int, fps: float, *, min_unique: int = 4
) -> list[int | None] | None:
    """时间戳落在录像区间内才取帧；钳到首/末帧会整段冻住，直接判失败。"""
    if ts.size == 0 or t0 is None:
        return None
    ids: list[int | None] = []
    for t in range(n_d):
        tt = t0 + t / fps
        if tt < ts[0] - 0.12 or tt > ts[-1] + 0.12:
            ids.append(None)
        else:
            ids.append(int(fidx[_nearest_idx(ts, tt)]))
    ok = [i for i in ids if i is not None]
    if len(set(ok)) < min_unique:
        return None
    return ids


def _load_cond_file(session: Path, idx: int) -> Image.Image | None:
    p = session / f"chunk_{idx:06d}_cond.jpg"
    return Image.open(p).convert("RGB") if p.is_file() else None


def _lerp_im(a: Image.Image | None, b: Image.Image | None, alpha: float) -> Image.Image | None:
    if a is None:
        return b
    if b is None:
        return a
    fa = np.asarray(a.convert("RGB"), dtype=np.float32)
    fb = np.asarray(b.convert("RGB").resize(a.size, Image.BILINEAR), dtype=np.float32)
    out = (1.0 - alpha) * fa + alpha * fb
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _cams_from_concat(split: tuple[Image.Image | None, Image.Image | None, Image.Image | None] | None) -> dict[str, Image.Image | None]:
    """concat (front, left, right) → 录像文件名。00091/00036 文件名都是反的：left_wrist 文件其实是 head。"""
    if not split:
        return {"cam_mid_chest": None, "cam_left_wrist": None, "cam_third_view": None}
    return {
        "cam_mid_chest": split[1],
        "cam_left_wrist": split[0],
        "cam_third_view": split[2],
    }


def _neighbor_cond(
    chunks: list[tuple[Path, dict]], i: int, delta: int, *, max_gap_s: float = 20.0
) -> Image.Image | None:
    """下一/上一包 cond。同一场推理里 session 被拆开时也要对上，否则单包 session 上排会冻住。"""
    j = i + delta
    if j < 0 or j >= len(chunks):
        return None
    _cs, crow = chunks[i]
    ns, nrow = chunks[j]
    t_i, t_j = crow.get("recorded_at"), nrow.get("recorded_at")
    if t_i and t_j:
        if abs(_iso_unix(str(t_j)) - _iso_unix(str(t_i))) > max_gap_s:
            return None
    return _load_cond_file(ns, int(nrow.get("chunk_idx", 0)))


def merge_run(
    action_step: Path,
    latents: Path | None = None,
    out_dir: Path | None = None,
    fps: float = FPS,
    *,
    all_sessions: bool = False,
    session_names: list[str] | None = None,
) -> Path | None:
    action_step = action_step.resolve()
    latents = (latents or action_step.parent / "vision_latents").resolve()
    t0, t1 = (None, None) if all_sessions else _run_window(action_step)
    sessions = _discover_sessions(latents, t0, t1)
    if session_names:
        want = set(session_names)
        sessions = [s for s in sessions if s.name in want]
    not_ready = [s for s in sessions if not (s / "dream_compare_v2" / "summary.json").is_file()]
    if not_ready:
        print(f"SKIP no dream_v2 {action_step.name}: {[s.name for s in not_ready]}", flush=True)
        sessions = [s for s in sessions if s not in not_ready]
    if not sessions:
        print(f"SKIP 无 latent session {action_step}", flush=True)
        return None

    chunks: list[tuple[Path, dict]] = []
    for s in sessions:
        for row in _load_index(s):
            chunks.append((s, row))
    eval_rows = _load_eval(action_step / "eval_log.jsonl")
    n = len(chunks)

    videos = action_step / "videos"
    cam_files = {
        "cam_mid_chest": videos / "cam_mid_chest.mp4",
        "cam_left_wrist": videos / "cam_left_wrist.mp4",
        "cam_third_view": videos / "cam_third_view.mp4",
    }
    have_video = any(p.is_file() for p in cam_files.values())
    ts_jsonl = videos / "frame_timestamps.jsonl"
    t_run0, _ = _run_window(action_step)
    ts_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, mp4 in cam_files.items():
        ts, fidx = _load_ts(ts_jsonl, name)
        if ts.size == 0 and mp4.is_file() and t_run0 is not None:
            ts, fidx = _synth_ts(mp4, t_run0, fps)
        ts_map[name] = (ts, fidx)

    out_dir = Path(out_dir) if out_dir else action_step / "dream_vs_real"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "chunks").mkdir(parents=True, exist_ok=True)

    cursors = {name: VideoCursor(p) for name, p in cam_files.items() if p.is_file()}
    compare_pipe: FfmpegPipe | None = None
    summary = []
    real_source = "camera_mp4" if have_video else "cond_jpg"

    try:
        for i, (session, row) in enumerate(chunks):
            idx = int(row.get("chunk_idx", i))
            dream_mp4 = session / "dream_compare_v2" / f"chunk_{idx:06d}_dream.mp4"
            dream_frames = _read_dream_mp4(dream_mp4) if dream_mp4.is_file() else []
            if not dream_frames:
                print(f"  skip chunk{idx} 无 dream mp4", flush=True)
                continue
            n_d = len(dream_frames)
            if compare_pipe is None:
                compare_pipe = FfmpegPipe(out_dir / "compare_real_vs_dream.mp4", fps, GRID_W, GRID_H)

            hint = _iso_unix(row["recorded_at"]) if row.get("recorded_at") else None
            ev = eval_rows[i] if i < len(eval_rows) else None
            if ev is not None and hint is not None:
                _rgb, _out, _ = _eval_rgb_and_out(ev)
                ref = _out if _out is not None else _rgb
                if ref is None or abs(ref - hint) > 5.0:
                    ev = _nearest_eval(eval_rows, hint) or ev
            elif ev is None and hint is not None:
                ev = _nearest_eval(eval_rows, hint)
            t_rgb, t_out, t_logged = _eval_rgb_and_out(ev)
            # DREAM 以进模型的那帧 RGB 为 t=0；REAL 按同一时间轴从录像取帧。
            t_start = t_rgb or t_out or hint or t_logged
            t_end = (t_start + n_d / fps) if t_start is not None else None

            cond = _load_cond_file(session, idx)
            cond_cams = _cams_from_concat(_split_dream_concat(cond) if cond is not None else None)
            next_im = _neighbor_cond(chunks, i, 1)
            prev_im = _neighbor_cond(chunks, i, -1)
            if next_im is not None:
                lerp_a, lerp_b = cond_cams, _cams_from_concat(_split_dream_concat(next_im))
            elif prev_im is not None:
                lerp_a, lerp_b = _cams_from_concat(_split_dream_concat(prev_im)), cond_cams
            else:
                lerp_a, lerp_b = cond_cams, cond_cams

            cam_ids = {
                name: _camera_frame_ids(ts, fidx, t_start, n_d, fps)
                for name, (ts, fidx) in ts_map.items()
            }

            chunk_pipe = FfmpegPipe(
                out_dir / "chunks" / f"pair_{i:04d}_after_vs_dream.mp4",
                fps, GRID_W, GRID_H,
            )
            cam_hits = {name: 0 for name in cam_files}
            for t, dream in enumerate(dream_frames):
                alpha = t / max(n_d - 1, 1)
                grabbed: dict[str, Image.Image | None] = {}
                for name in ("cam_mid_chest", "cam_left_wrist", "cam_third_view"):
                    ids = cam_ids.get(name)
                    im = None
                    if ids and ids[t] is not None and name in cursors:
                        im = cursors[name].at(ids[t])
                    if im is not None:
                        cam_hits[name] += 1
                    else:
                        im = _lerp_im(lerp_a.get(name), lerp_b.get(name), alpha)
                    grabbed[name] = im
                grid = _grid_3view(
                    grabbed.get("cam_mid_chest"),
                    grabbed.get("cam_left_wrist"),
                    grabbed.get("cam_third_view"),
                    dream,
                    chunk_idx=idx, t=t, n_d=n_d,
                )
                chunk_pipe.write(grid)
                compare_pipe.write(grid)
            chunk_pipe.close()

            n_cam_views = sum(1 for v in cam_hits.values() if v >= max(n_d // 2, 1))
            if n_cam_views == 3:
                src = "camera_mp4"
            elif n_cam_views:
                src = "camera+cond_lerp"
            else:
                src = "cond_lerp"
            summary.append({
                "i": i,
                "session": session.name,
                "chunk_idx": idx,
                "n_dream": n_d,
                "t_rgb_in": t_rgb,
                "t_model_out": t_out,
                "t_eval_logged": t_logged,
                "t_after_start": t_start,
                "t_after_end": t_end,
                "real_source": src,
                "cam_hits": cam_hits,
                "have_mid": bool(cam_files["cam_mid_chest"].is_file()),
                "have_left": bool(cam_files["cam_left_wrist"].is_file()),
                "have_third": bool(cam_files["cam_third_view"].is_file()),
                "per_chunk_mp4": f"chunks/pair_{i:04d}_after_vs_dream.mp4",
            })
            print(f"[{i + 1}/{n}] {session.name} chunk{idx} {src}", flush=True)
    finally:
        for c in cursors.values():
            c.close()
        if compare_pipe:
            compare_pipe.close()

    meta = json.loads((action_step / "meta.json").read_text()) if (action_step / "meta.json").is_file() else {}
    srcs = [c.get("real_source") for c in summary if c.get("real_source")]
    real_source = ",".join(sorted(set(srcs))) if srcs else real_source
    (out_dir / "summary.json").write_text(
        json.dumps({
            "action_step": str(action_step),
            "prompt": meta.get("task_prompt"),
            "ckpt": meta.get("ckpt_dir"),
            "n_action_steps": meta.get("n_action_steps"),
            "real_source": real_source,
            "have_camera_mp4": have_video,
            "n_pairs": len(summary),
            "layout": "2x3: left wrist | head | right; DREAM head in center (matches 00091 file/concat swap)",
            "chunks": summary,
        }, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"done {len(summary)} -> {out_dir} source={real_source}", flush=True)
    _publish_into_sessions(action_step, out_dir, summary)
    return out_dir


def _publish_into_sessions(action_step: Path, out_dir: Path, summary: list[dict]) -> None:
    """session 目录里全是 dream_compare_v2（纯 DREAM / cond 冻住），把 3 路 AFTER vs DREAM 拷过去。

    T9 可能是 exFAT，不能 symlink。
    """
    import shutil

    def _place(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            dst.symlink_to(src)
        except OSError:
            shutil.copy2(src, dst)

    latents = action_step.parent / "vision_latents"
    for row in summary:
        src = (out_dir / row["per_chunk_mp4"]).resolve()
        if not src.is_file():
            continue
        sess = latents / row["session"] / "dream_vs_real"
        _place(src, sess / src.name)
    big = out_dir / "compare_real_vs_dream.mp4"
    if big.is_file():
        _place(big.resolve(), action_step / "AFTER_vs_DREAM_3view.mp4")


def concat_mp4s(paths: list[Path], out: Path) -> None:
    if not paths:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    lst = out.with_suffix(".concat.txt")
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in paths), encoding="utf-8")
    subprocess.check_call(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
         "-c", "copy", str(out)],
    )


def auto_all(ckpt_root: Path) -> None:
    jobs = []
    for ds in ("cosmos3_00036", "cosmos3_00091"):
        root = ckpt_root / ds
        if not root.is_dir():
            continue
        for it in sorted(root.glob("iter_*")):
            for as_dir in sorted(it.glob("action-step_*")):
                if "aborted" in as_dir.name:
                    continue
                jobs.append(as_dir)
    for as_dir in jobs:
        print(f"\n===== {as_dir} =====", flush=True)
        merge_run(as_dir)

    # 只拼 960×480 三路 AFTER vs DREAM，不要把 session 里 192×576 的 cond-vs-dream 当成 REAL。
    for ds in ("cosmos3_00036", "cosmos3_00091"):
        real_clips = []
        for it in sorted((ckpt_root / ds).glob("iter_*")):
            for as_dir in sorted(it.glob("action-step_*")):
                if "aborted" in as_dir.name:
                    continue
                mp4 = as_dir / "dream_vs_real" / "compare_real_vs_dream.mp4"
                if mp4.is_file():
                    real_clips.append(mp4)
            orphan = it / "dream_vs_real" / "compare_real_vs_dream.mp4"
            if orphan.is_file() and orphan not in real_clips:
                real_clips.append(orphan)
        if real_clips:
            out = ckpt_root / ds / "AFTER_vs_DREAM_3view.mp4"
            try:
                concat_mp4s(real_clips, out)
                print(f"concat 3view {ds} n={len(real_clips)} -> {out}", flush=True)
            except subprocess.CalledProcessError:
                print(f"concat 3view {ds} 失败", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--action-step", default="")
    p.add_argument("--latents", default="")
    p.add_argument("--out", default="")
    p.add_argument("--auto", action="store_true")
    p.add_argument("--all-sessions", action="store_true",
                   help="该 iter 下所有有 latent 的 session 都按包出片，不按 eval_log 裁切")
    p.add_argument("--session", action="append", default=[],
                   help="只处理这些 session 名，可重复；需配 --all-sessions")
    p.add_argument("--ckpt-root", default="/media/xk/T9_3/checkpoint")
    p.add_argument("--fps", type=float, default=FPS)
    args = p.parse_args()
    if args.auto:
        auto_all(Path(args.ckpt_root))
        return
    if not args.action_step:
        raise SystemExit("需要 --action-step 或 --auto")
    merge_run(
        Path(args.action_step),
        Path(args.latents) if args.latents else None,
        Path(args.out) if args.out else None,
        fps=args.fps,
        all_sessions=args.all_sessions,
        session_names=args.session or None,
    )


if __name__ == "__main__":
    main()
