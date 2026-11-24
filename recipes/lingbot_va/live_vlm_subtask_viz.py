#!/usr/bin/env python3
"""真机 live：三路 RealSense +（可选）瑞尔曼读状态 → π0.5 PaliGemma 生成当前 subtask 叠字可视化。

不发动作，只读相机 / 可选读臂。用来肉眼看 VLM 是否 reliable。

示例:
  # 默认用本机抽出的 PG（000490 或 pi05_fixed_paligemma）+ 00036 现场 yaml
  python3 /home/xk/VLA/lingbot-va-infer/live_vlm_subtask_viz.py

  # 连臂（只读关节，不下发）
  python3 /home/xk/VLA/lingbot-va-infer/live_vlm_subtask_viz.py --robot

  # 指定权重
  python3 ... --weight-dir /home/xk/VLA/openpi/models/pi05_fixed_paligemma
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

PROMPT = "Task: {task}\nWhat is the current subtask?\nSubtask:"
IMAGE_TOKEN_ID = 257152
N_IMAGE_TOKENS = 256

DEFAULT_YAML = Path(
    "/home/xk/VLA/infer/infer/config/lingbot/lingbot_va_tongbot_16d_00036_local.yaml"
)
DEFAULT_TASK = (
    "With both hands, pick up the yellow banana and the yellow lemon and put them "
    "into the black box; then take the green vegetables and place them into the blue box."
)
WEIGHT_CANDIDATES = [
    Path("/home/xk/VLA/openpi/models/pi05_fixed_paligemma"),
    Path("/home/xk/VLA/openpi/models/pi05_pg_only_000490"),
    Path("/home/xk/VLA/h100_ckpts/00036_pi05_16gpu/000490/pretrained_model"),
    Path("/home/xk/VLA/openpi/models/pi05_fixed"),
]


def _add_paths() -> Path:
    openpi = Path("/home/xk/VLA/openpi/cigai-agilex-train-openpi")
    infer_root = Path("/home/xk/VLA/infer")
    for p in (openpi, openpi / "src", infer_root):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    return openpi


def resolve_weight_dir(explicit: str) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not (p / "model.safetensors").is_file():
            raise SystemExit(f"没有 model.safetensors: {p}")
        return p
    for p in WEIGHT_CANDIDATES:
        if (p / "model.safetensors").is_file():
            return p
    raise SystemExit("找不到权重。先拉 pi05_fixed_paligemma 或用 --weight-dir")


def load_templates(config_yaml: Path) -> dict[str, str]:
    if not config_yaml.is_file():
        return {}
    return (yaml.safe_load(config_yaml.read_text()) or {}).get("subtask_templates") or {}


def closest_label(pred: str, templates: dict[str, str]) -> str | None:
    import re

    ws = re.compile(r"[^a-z0-9 ]+")

    def norm(t: str) -> str:
        return ws.sub(" ", t.lower().strip()).strip()

    p = norm(pred)
    if not p:
        return None
    best, best_d = None, 10**9
    for label, tmpl in templates.items():
        t = norm(tmpl.replace("{hand}", "arm"))
        if t in p or p in t:
            return label
        d = abs(len(p) - len(t)) + sum(a != b for a, b in zip(p, t, strict=False))
        if d < best_d:
            best, best_d = label, d
    return best


class SubtaskVLM:
    def __init__(self, weight_dir: Path, tokenizer_path: Path, device: str, task: str):
        from scripts.eval_e2_vlm_val import load_paligemma, load_sentencepiece

        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        if self.device.type == "cuda" and ":" not in device and device == "cuda":
            # leave room for cosmos on same GPU if needed — still use cuda:0
            self.device = torch.device("cuda:0")
        logging.info("load VLM %s -> %s", weight_dir, self.device)
        self.model = load_paligemma(weight_dir, self.device)
        self.sp = load_sentencepiece(tokenizer_path)
        self.task = task
        self.prompt = PROMPT.format(task=task)
        self.lock = threading.Lock()
        self.last_pred = ""
        self.last_label = None
        self.last_ms = 0.0
        self.last_error = ""
        self.n_infer = 0

    def _pixels(self, rgb: np.ndarray) -> torch.Tensor:
        import cv2

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.shape[-1] == 3 else rgb
        img = cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_CUBIC)
        arr = img.astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        # BGR->RGB for model
        arr = arr[:, :, ::-1].copy()
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        return t.to(device=self.device, dtype=dtype)

    @torch.no_grad()
    def generate(self, rgb: np.ndarray, max_new_tokens: int = 32) -> str:
        t0 = time.time()
        try:
            pixels = self._pixels(rgb)
            text_ids = list(self.sp.encode(self.prompt.strip(), add_bos=True)) + list(
                self.sp.encode("\n")
            )
            input_ids = torch.tensor(
                [[IMAGE_TOKEN_ID] * N_IMAGE_TOKENS + text_ids],
                device=self.device,
                dtype=torch.long,
            )
            attn = torch.ones_like(input_ids)
            out = self.model.generate(
                pixel_values=pixels,
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            new_ids = [
                int(i)
                for i in out[0, input_ids.shape[1] :].tolist()
                if 0 <= int(i) < self.sp.GetPieceSize()
            ]
            text = self.sp.decode(new_ids).strip().split("\n")[0].strip()
            err = ""
        except Exception as exc:  # noqa: BLE001
            text = ""
            err = f"{type(exc).__name__}: {exc}"
            logging.exception("generate failed")
        ms = (time.time() - t0) * 1000
        with self.lock:
            self.last_pred = text
            self.last_ms = ms
            self.last_error = err
            self.n_infer += 1
        return text


def draw_panel(
    images_rgb: list[np.ndarray],
    *,
    subtask: str,
    label: str | None,
    weight_name: str,
    infer_ms: float,
    n_infer: int,
    arm_line: str,
    err: str,
) -> np.ndarray:
    panels = []
    titles = ["front", "left", "right"]
    for i, rgb in enumerate(images_rgb[:3]):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (640, 480))
        cv2.putText(
            bgr,
            titles[i] if i < len(titles) else f"cam{i}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(bgr)
    while len(panels) < 3:
        panels.append(np.zeros((480, 640, 3), dtype=np.uint8))
    row = np.concatenate(panels, axis=1)

    banner = np.zeros((160, row.shape[1], 3), dtype=np.uint8)
    banner[:] = (32, 32, 32)
    pred = subtask if subtask else "(empty / immediate EOS — VLM not reliable for subtask)"
    color = (80, 220, 80) if subtask else (80, 80, 220)
    cv2.putText(banner, f"SUBTASK: {pred}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)
    cv2.putText(
        banner,
        f"label≈{label or '-'}   infer={infer_ms:.0f}ms  n={n_infer}   weight={weight_name}",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        banner,
        arm_line[:120] if arm_line else "arm: (not connected)",
        (20, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    if err:
        cv2.putText(banner, err[:100], (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return np.concatenate([banner, row], axis=0)


def main() -> None:
    p = argparse.ArgumentParser(description="Live VLM subtask + camera viz (Realman optional read-only)")
    p.add_argument("--config", default=str(DEFAULT_YAML))
    p.add_argument("--weight-dir", default="")
    p.add_argument("--tokenizer", default="")
    p.add_argument("--task-config", default="/home/xk/VLA/annotate_pipeline/configs/00036.yaml")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--interval", type=float, default=1.0, help="subtask 推理间隔秒")
    p.add_argument("--robot", action="store_true", help="连接瑞尔曼只读关节")
    p.add_argument("--cam", default="front", choices=["front", "left", "right"], help="送进 VLM 的相机")
    p.add_argument("--window", default="vlm_subtask_live")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    openpi = _add_paths()
    # eval_e2_vlm_val lives under openpi/scripts
    sys.path.insert(0, str(openpi / "scripts"))

    weight_dir = resolve_weight_dir(args.weight_dir)
    tok = Path(args.tokenizer) if args.tokenizer else weight_dir / "tokenizer.model"
    if not tok.is_file():
        tok = Path("/home/xk/VLA/openpi/models/pi05_fixed/tokenizer.model")
    if not tok.is_file():
        raise SystemExit(f"没有 tokenizer.model: {tok}")

    templates = load_templates(Path(args.task_config))
    task = DEFAULT_TASK
    raw_task = yaml.safe_load(Path(args.task_config).read_text()) if Path(args.task_config).is_file() else {}
    if isinstance(raw_task, dict) and raw_task.get("task"):
        task = str(raw_task["task"]).strip()

    vlm = SubtaskVLM(weight_dir, tok, args.device, task)

    from infer.config_loader import load_infer_config
    from infer.io.camera_tcp_sender import ManagedLocalCameraSource

    cfg = load_infer_config(args.config)
    cam_idx = {"front": 0, "left": 1, "right": 2}[args.cam]

    latest = {"images": None, "t": 0.0}
    lock = threading.Lock()

    def on_frame(timestamp, images, times, depths):  # noqa: ARG001
        with lock:
            latest["images"] = [im.copy() for im in images]
            latest["t"] = float(timestamp)

    cam = ManagedLocalCameraSource(cfg.camera_sender, on_frame=on_frame)
    if not cam.enabled:
        raise SystemExit("camera_sender 未启用或 transport!=local，检查 yaml")
    cam.start()

    robot = None
    arm_line = ""
    if args.robot:
        from infer.io.direct_robot_io import DirectTongGenesisRobot

        robot = DirectTongGenesisRobot(
            cfg.direct_robot,
            gripper_state_port=cfg.network.gripper_state_port,
            gripper_state_stale_sec=cfg.network.gripper_state_stale_sec,
        )
        robot.connect()
        logging.info("Realman connected (read-only)")

    stop = False
    last_infer_t = 0.0

    def infer_loop():
        nonlocal last_infer_t
        while not stop:
            time.sleep(0.05)
            with lock:
                imgs = latest["images"]
            if not imgs or time.time() - last_infer_t < args.interval:
                continue
            idx = min(cam_idx, len(imgs) - 1)
            pred = vlm.generate(imgs[idx])
            lab = closest_label(pred, templates) if pred else None
            with vlm.lock:
                vlm.last_label = lab
            last_infer_t = time.time()
            logging.info("subtask=%r label=%s ms=%.0f", pred, lab, vlm.last_ms)

    th = threading.Thread(target=infer_loop, daemon=True)
    th.start()

    logging.info("window=%s  q=quit  weight=%s", args.window, weight_dir)
    try:
        while True:
            with lock:
                imgs = latest["images"]
            if imgs is None:
                time.sleep(0.05)
                continue
            if robot is not None:
                try:
                    st = robot.read_state_15d()
                    arm_line = (
                        f"arm L0-2={st[0]:.1f},{st[1]:.1f},{st[2]:.1f}  "
                        f"R0-2={st[7]:.1f},{st[8]:.1f},{st[9]:.1f}  "
                        f"grip={st[14]:.2f}/{st[15]:.2f}"
                    )
                except Exception as exc:  # noqa: BLE001
                    arm_line = f"arm read err: {exc}"
            with vlm.lock:
                panel = draw_panel(
                    imgs,
                    subtask=vlm.last_pred,
                    label=vlm.last_label,
                    weight_name=weight_dir.name,
                    infer_ms=vlm.last_ms,
                    n_infer=vlm.n_infer,
                    arm_line=arm_line,
                    err=vlm.last_error,
                )
            cv2.imshow(args.window, panel)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        stop = True
        cam.stop()
        if robot is not None:
            try:
                robot.close()
            except Exception:
                logging.exception("robot close")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
