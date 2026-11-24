#!/usr/bin/env python3
"""00108 rgb_depth_head：训练帧 RGB+DEPTH 走与真机相同的 OpenpiPi05Runtime，写探针并对照 GT。

  PYTHONPATH=/home/xk/data_Driven/infer OPENPI_DISABLE_TORCH_COMPILE=1 \\
    /home/xk/VLA/openpi/cigai-agilex-train-openpi/.venv/bin/python \\
    infer/tools/offline/offline_pi05_rgb_depth_head_smoke.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_INFER_ROOT = Path(__file__).resolve().parents[2]
if str(_INFER_ROOT) not in sys.path:
    sys.path.insert(0, str(_INFER_ROOT))

from infer.policies.openpi_pi05_runtime import OpenpiPi05Runtime  # noqa: E402
from infer.recording.infer_probe import (  # noqa: E402
    InferProbeWriter,
    action_stats,
    client_frame_row,
    one_line,
    vision_from_obs,
)

DEFAULT_CKPT = Path(
    "/media/xk/T9_3/checkpoint/00108_pi05_a2npu8_ah50_rgb_depth_head_fixbase_step4000/pretrained_model"
)
DEFAULT_DS = Path("/home/xk/dataset/00108_20260922_RM_01_YJ")
PROMPT = (
    "With both hands, pick up the yellow banana and the yellow lemon and "
    "put them into the black box; then take the green vegetables and place them into the blue box."
)
CAMS = ("left", "front", "right")
VIDEO_KEYS = {
    "left": "observation.images.cam_left",
    "front": "observation.images.cam_mid",
    "right": "observation.images.cam_right",
}


def _read_png_u16(path: Path) -> np.ndarray:
    from PIL import Image

    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint16)


def _load_rgb_frames(root: Path, indices: List[int]) -> Dict[int, List[np.ndarray]]:
    """AV1 mp4：本机 OpenCV 解不开，用 PyAV/libdav1d。"""
    import av

    wanted = set(int(i) for i in indices)
    last = max(wanted)
    by_cam: Dict[str, Dict[int, np.ndarray]] = {cam: {} for cam in CAMS}
    for cam, key in VIDEO_KEYS.items():
        mp4 = root / "videos" / key / "chunk-000" / "file-000.mp4"
        container = av.open(str(mp4))
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        i = 0
        try:
            for frame in container.decode(stream):
                if i in wanted:
                    by_cam[cam][i] = frame.to_ndarray(format="rgb24")
                i += 1
                if i > last:
                    break
        finally:
            container.close()
        missing = [j for j in wanted if j not in by_cam[cam]]
        if missing:
            raise RuntimeError(f"{cam} 读不到 frame {missing} from {mp4}")
    return {idx: [by_cam[cam][idx] for cam in CAMS] for idx in indices}


def _depth_path(root: Path, episode: int, frame: int) -> Path:
    return (
        root
        / "depth"
        / "observation.depths.cam_mid"
        / "chunk-000"
        / f"episode-{episode:06d}"
        / f"frame_{frame:06d}.png"
    )


def _pick_frames(parquet: Path, episode: int) -> List[Tuple[int, int, np.ndarray, np.ndarray]]:
    import pyarrow.parquet as pq

    table = pq.read_table(
        parquet, columns=["episode_index", "frame_index", "index", "observation.state", "action"]
    )
    ep = np.asarray(table["episode_index"].to_pylist())
    mask = ep == int(episode)
    fi = np.asarray(table["frame_index"].to_pylist())[mask]
    idx = np.asarray(table["index"].to_pylist())[mask]
    state = np.stack(table["observation.state"].to_pylist())[mask].astype(np.float32)
    action = np.stack(table["action"].to_pylist())[mask].astype(np.float32)
    n = int(fi.shape[0])
    locals_ = [0, min(41, n - 1), min(64, n - 1)]
    seen = []
    rows = []
    for loc in locals_:
        if loc in seen:
            continue
        seen.append(loc)
        rows.append((int(idx[loc]), int(fi[loc]), state[loc], action[loc]))
    return rows


def _chunk_stats(pred: np.ndarray, gt: np.ndarray) -> Dict[str, Any]:
    p0 = np.asarray(pred[0], dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    err = p0 - g
    return {
        "pred_t0": {
            "joints_mae": round(float(np.mean(np.abs(err[:14]))), 4),
            "grip_L": round(float(p0[14]), 4),
            "grip_R": round(float(p0[15]), 4),
            "grip_L_err": round(float(err[14]), 4),
            "grip_R_err": round(float(err[15]), 4),
            "vec": [round(float(x), 4) for x in p0[:16]],
        },
        "gt": {
            "grip_L": round(float(g[14]), 4),
            "grip_R": round(float(g[15]), 4),
        },
        "chunk": action_stats(pred),
    }


def main() -> int:
    ckpt = Path(os.environ.get("PI05_CKPT") or DEFAULT_CKPT)
    ds = Path(os.environ.get("PI05_DS") or DEFAULT_DS)
    out_dir = Path(
        os.environ.get("PI05_SMOKE_OUT")
        or "/tmp/pi05_00108_rgb_depth_head_smoke"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OPENPI_DISABLE_TORCH_COMPILE", "1")

    frames = _pick_frames(ds / "data/chunk-000/file-000.parquet", 0)
    rgb_by_idx = _load_rgb_frames(ds, [g for g, _, _, _ in frames])
    prompt = PROMPT

    print(f"ckpt={ckpt}", flush=True)
    print(f"ds={ds} frames={[(g, loc) for g, loc, _, _ in frames]}", flush=True)
    t_load = time.perf_counter()
    runtime = OpenpiPi05Runtime(
        ckpt_dir=ckpt,
        config_name="pi05_aloha_rgb_depth_head",
        repo_path=None,
        camera_names=list(CAMS),
        action_dim=16,
        state_dim=16,
        action_horizon=50,
        image_size=224,
        default_prompt=prompt,
        adapt_to_pi=False,
        delta_action_dims=[7, 7, -1, -1],
    )
    runtime.load()
    print(f"loaded_sec={time.perf_counter() - t_load:.1f}", flush=True)

    writer = InferProbeWriter(out_dir / "infer_probe.jsonl")
    summary: Dict[str, Any] = {
        "ckpt": str(ckpt),
        "dataset": str(ds),
        "config_name": "pi05_aloha_rgb_depth_head",
        "depth_obs_keys": list(runtime._depth_obs_keys),
        "use_modality_prompt_tokens": bool(
            getattr(getattr(runtime._train_config, "model", None), "use_modality_prompt_tokens", None)
        ),
        "rows": [],
    }
    try:
        for i, (global_i, local_i, state, gt) in enumerate(frames):
            images = rgb_by_idx[global_i]
            depth_file = _depth_path(ds, 0, local_i)
            depths: Optional[List[Optional[np.ndarray]]] = [
                None,
                _read_png_u16(depth_file),
                None,
            ]
            for with_depth in (True, False):
                t0 = time.perf_counter()
                obs = runtime.build_observation(
                    images,
                    state,
                    prompt,
                    depths=depths if with_depth else None,
                )
                raw, post = runtime.infer(obs)
                infer_sec = time.perf_counter() - t0
                vis = vision_from_obs(obs)
                timed = type("T", (), {})()
                timed.preprocess_sec = 0.0
                timed.model_sec = infer_sec
                timed.postprocess_sec = 0.0
                timed.infer_sec = infer_sec
                timed.server_timing = {}
                timed.ik = None
                timed.raw_action = raw
                timed.post_action = post
                timed.vision = vis
                row = client_frame_row(
                    frame=i * 2 + int(with_depth),
                    policy="pi05",
                    timed=timed,
                    state=state,
                    n_exec=50,
                )
                row["with_depth"] = bool(with_depth)
                row["episode"] = 0
                row["frame_index"] = local_i
                writer.write(row)
                line = one_line(row)
                print(line, flush=True)
                cmp_ = _chunk_stats(post, gt)
                summary["rows"].append(
                    {
                        "with_depth": bool(with_depth),
                        "frame_index": local_i,
                        "global_index": global_i,
                        "vision": vis,
                        "compare": cmp_,
                        "infer_sec": round(infer_sec, 3),
                        "finite": row.get("finite"),
                    }
                )
                print(
                    f"  with_depth={with_depth} t={local_i} "
                    f"mae_j={cmp_['pred_t0']['joints_mae']} "
                    f"gripL {cmp_['pred_t0']['grip_L']} vs {cmp_['gt']['grip_L']} "
                    f"gripR {cmp_['pred_t0']['grip_R']} vs {cmp_['gt']['grip_R']} "
                    f"finite={row.get('finite')} sec={infer_sec:.2f}",
                    flush=True,
                )
    finally:
        writer.close()

    pairs = [
        (summary["rows"][j], summary["rows"][j + 1])
        for j in range(0, len(summary["rows"]), 2)
        if j + 1 < len(summary["rows"])
    ]
    deltas = []
    for with_d, no_d in pairs:
        a = np.asarray(with_d["compare"]["pred_t0"]["vec"], dtype=np.float64)
        b = np.asarray(no_d["compare"]["pred_t0"]["vec"], dtype=np.float64)
        deltas.append(
            {
                "frame_index": with_d["frame_index"],
                "mae_j_with": with_d["compare"]["pred_t0"]["joints_mae"],
                "mae_j_rgb": no_d["compare"]["pred_t0"]["joints_mae"],
                "gripL_with": with_d["compare"]["pred_t0"]["grip_L"],
                "gripL_rgb": no_d["compare"]["pred_t0"]["grip_L"],
                "gripR_with": with_d["compare"]["pred_t0"]["grip_R"],
                "gripR_rgb": no_d["compare"]["pred_t0"]["grip_R"],
                "t0_l2_with_vs_rgb": round(float(np.linalg.norm(a - b)), 4),
            }
        )
    summary["with_vs_rgb"] = deltas
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {out_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
