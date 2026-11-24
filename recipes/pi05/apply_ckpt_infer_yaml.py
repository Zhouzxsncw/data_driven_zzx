#!/usr/bin/env python3
"""把一份 checkpoint 写进真机推理 YAML。

相机、机械臂、脚踏板沿用模板。模型路径和训练对齐字段从 checkpoint 的
train.yaml 填入，不用手改。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_TEMPLATE = (
    _REPO
    / "infer/infer/config/lerobot/pi05_00108_cty18_bs384_ah50_rgb_d7171_step6300_ws.yaml"
)
_OUT_DIR = _REPO / "infer/infer/config/lerobot"

# 训练 image key 的最后一段 → (openpi 短名, 推理 camera key)
_CAM = {
    "cam_mid": ("front", "observation.cam_mid"),
    "front": ("front", "observation.cam_mid"),
    "base": ("front", "observation.cam_mid"),
    "cam_left": ("left", "observation.cam_left"),
    "left": ("left", "observation.cam_left"),
    "cam_right": ("right", "observation.cam_right"),
    "right": ("right", "observation.cam_right"),
}


def resolve_ckpt(raw: Path) -> Path:
    """接受 checkpoint 根目录，或其中的 pretrained_model。"""
    path = raw.expanduser().resolve()
    if (path / "model.safetensors").is_file():
        path = path.parent
    missing = []
    if not (path / "train.yaml").is_file():
        missing.append("train.yaml")
    if not (path / "assets").is_dir():
        missing.append("assets/")
    if not (path / "pretrained_model" / "model.safetensors").is_file():
        missing.append("pretrained_model/model.safetensors")
    if missing:
        raise SystemExit(f"目录不完整: {path}\n缺少: {', '.join(missing)}")
    return path


def _cam_from_image_key(key: str) -> tuple[str, str]:
    leaf = str(key).strip().split(".")[-1].lower()
    found = _CAM.get(leaf)
    if found is None:
        raise SystemExit(f"训练 image_keys 里有未识别的相机: {key}")
    return found


def apply_train(policy: dict, train: dict, ckpt: Path, prompt: str | None) -> None:
    data = train.get("data") if isinstance(train.get("data"), dict) else {}
    model = train.get("model") if isinstance(train.get("model"), dict) else {}
    config_name = train.get("config_name")
    horizon = model.get("action_horizon")
    if not config_name or horizon is None:
        raise SystemExit("train.yaml 缺少 config_name 或 model.action_horizon")
    for key in ("adapt_to_pi", "delta_action_dims", "interleave_grippers", "prompt_from_subtask"):
        if key not in data:
            raise SystemExit(f"train.yaml 的 data 缺少 {key}")
    image_keys = data.get("image_keys") or []
    if not image_keys:
        raise SystemExit("train.yaml 的 data.image_keys 为空")

    names: list[str] = []
    obs_keys: list[str] = []
    for key in image_keys:
        name, obs = _cam_from_image_key(str(key))
        names.append(name)
        obs_keys.append(obs)

    policy["ckpt_dir"] = str(ckpt / "pretrained_model")
    policy["openpi_config_name"] = str(config_name)
    policy["openpi_action_horizon"] = int(horizon)
    policy["n_action_steps_override"] = int(horizon)
    policy["openpi_adapt_to_pi"] = bool(data["adapt_to_pi"])
    policy["openpi_delta_action_dims"] = [int(x) for x in data["delta_action_dims"]]
    policy["openpi_interleave_grippers"] = bool(data["interleave_grippers"])
    policy["prompt_from_subtask"] = bool(data["prompt_from_subtask"])
    policy["task_prompt"] = prompt if prompt is not None else str(data.get("default_prompt") or "")
    policy["openpi_camera_names"] = names
    policy["camera_keys"] = obs_keys
    policy["num_inference_steps_override"] = None
    policy["rtc_enabled"] = False
    policy["rtc_config"] = None
    policy.pop("openpi_vector_norm", None)
    if data.get("vector_norm"):
        policy["openpi_vector_norm"] = str(data["vector_norm"])


def main() -> None:
    parser = argparse.ArgumentParser(description="用 checkpoint 生成真机推理 YAML")
    parser.add_argument("ckpt", type=Path, help="checkpoint 根目录，或其中的 pretrained_model")
    parser.add_argument("--template", type=Path, default=_DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=None, help="默认写到 config/lerobot/pi05_<目录名>_ws.yaml")
    parser.add_argument("--prompt", default=None, help="现场任务句；不写则用训练 default_prompt")
    parser.add_argument("--port", type=int, default=None, help="不写则沿用模板端口")
    parser.add_argument("--fps", type=int, default=None, help="控制频率；不写则沿用模板 session.fps")
    parser.add_argument("--tasks-per-group", type=int, default=None, help="一组任务段数")
    parser.add_argument(
        "--replan-steps",
        type=int,
        default=None,
        help="两次推理之间的控制步数；不写则客户端用动作块长度的五分之一",
    )
    args = parser.parse_args()
    for name, value in (
        ("port", args.port),
        ("fps", args.fps),
        ("tasks-per-group", args.tasks_per_group),
        ("replan-steps", args.replan_steps),
    ):
        if value is not None and int(value) < 1:
            raise SystemExit(f"--{name} 必须是正整数")

    ckpt = resolve_ckpt(args.ckpt)
    template = args.template.expanduser().resolve()
    if not template.is_file():
        raise SystemExit(f"找不到模板: {template}")
    out = args.output
    if out is None:
        out = _OUT_DIR / f"pi05_{ckpt.name}_ws.yaml"
    out = out.expanduser().resolve()
    if out == template:
        raise SystemExit("输出路径不能覆盖模板，请另给 --output")

    with template.open(encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
    with (ckpt / "train.yaml").open(encoding="utf-8") as fp:
        train = yaml.safe_load(fp)
    if not isinstance(cfg, dict) or not isinstance(cfg.get("policy"), dict):
        raise SystemExit(f"模板缺少 policy: {template}")
    if not isinstance(train, dict):
        raise SystemExit(f"无法读取 {ckpt / 'train.yaml'}")

    cfg["experiment_name"] = ckpt.name
    apply_train(cfg["policy"], train, ckpt, args.prompt)
    if args.port is not None:
        cfg["policy"]["lingbot_port"] = int(args.port)
    session = cfg.get("session")
    if not isinstance(session, dict):
        raise SystemExit("模板缺少 session")
    if args.fps is not None:
        session["fps"] = int(args.fps)
    if args.replan_steps is not None:
        session["replan_steps"] = int(args.replan_steps)
    elif session.get("replan_steps") is None:
        session["replan_steps"] = None
    if args.tasks_per_group is not None:
        cfg["policy"]["tasks_per_group"] = int(args.tasks_per_group)
    rec = cfg.get("recording")
    if isinstance(rec, dict):
        rec["inference_dir"] = None

    out.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
    header = (
        f"# 由 apply_ckpt_infer_yaml.py 生成。模型字段来自 {ckpt / 'train.yaml'}\n"
        f"# 真机相机、机械臂、脚踏板来自模板 {template.name}\n"
        f"# 任务句 policy.task_prompt，控制频率 session.fps，调用间隔 session.replan_steps\n"
    )
    out.write_text(header + text, encoding="utf-8")

    port = int(cfg["policy"]["lingbot_port"])
    log = _REPO / "recipes/pi05" / f"server_{ckpt.name}.log"
    py_srv = "/home/xk/VLA/openpi/cigai-agilex-train-openpi/.venv/bin/python"
    py_cli = "/home/xk/miniconda3/envs/lingbotva/bin/python"
    print(f"RUN_YAML={out}")
    print(f"PORT={port}")
    print(f"CKPT={ckpt}")
    print("PROMPT=" + str(cfg["policy"].get("task_prompt") or ""))
    print()
    print("# 端口上还没有本次服务时执行：")
    print(
        "cd /home/xk/data_Driven && "
        "OPENPI_DISABLE_TORCH_COMPILE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false "
        f"PYTHONPATH=/home/xk/data_Driven/infer nohup {py_srv} -u "
        f"/home/xk/data_Driven/recipes/pi05/serve_tong_pi05.py "
        f"--config {out} --port {port} > {log} 2>&1 &"
    )
    print()
    print("# 然后在这个 SSH 终端里执行：")
    print(
        "cd /home/xk/data_Driven/infer && "
        f"PYTHONPATH=/home/xk/data_Driven/infer {py_cli} -m infer.cli.run_async_infer "
        f"--config {out}"
    )


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
