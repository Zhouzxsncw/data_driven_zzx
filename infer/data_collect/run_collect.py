#!/usr/bin/env python3
"""
可配置数据采集入口。

新增功能：
  1. 启动 run_collect.py 后，先弹出“输入名字”窗口。
     - 点击“确认”：把名字加入本次数据集保存路径，然后继续采集。
     - 点击“取消”或关闭窗口：直接退出，不启动采集。
  2. 也支持命令行指定名字：--session-name xxx，便于远程/脚本化运行。

示例:
  cd /path/to/repo   # 含 common/ 与 data_collect/ 的仓库根
  export PYTHONPATH="$(pwd):$PYTHONPATH"

  python3 data_collect/run_collect.py \
      --config data_collect/config/dual_arm_gripper_2cam.yaml
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional


def _find_repo_root() -> Path:
    """定位含 common/ 与 data_collect/ 的仓库根。

    兼容：
      repo/data_collect/run_collect.py
      repo/scripts/data_collect/run_collect.py（旧布局）
    """
    here = Path(__file__).resolve().parent
    for cand in (here.parent, *here.parents):
        if (cand / "common").is_dir() and (cand / "data_collect").is_dir():
            return cand
    return here.parent


# 路径：repo 根目录（供 import common）+ scripts（camera_module，若存在）
_REPO_ROOT = _find_repo_root()
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
for p in (_REPO_ROOT, _SCRIPTS_DIR):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data_collect.collector import DataCollectApp
from data_collect.config_loader import CollectConfig, load_collect_config
from data_collect.logging_setup import setup_logging


def parse_args() -> argparse.Namespace:
    # default_cfg = Path(__file__).parent / "config" / "right_arm_gripper_2cam.yaml"
    #default_cfg = Path(__file__).parent / "config" / "left_arm_gripper_2cam.yaml"
    default_cfg = Path(__file__).parent / "config" / "dual_arm_gripper_2cam.yaml"
    # default_cfg = Path(__file__).parent / "config" / "default_full.yaml"

    parser = argparse.ArgumentParser(description="CQTongBot 可配置数据采集")
    parser.add_argument(
        "--config",
        type=Path,
        
        default=default_cfg,
        help=f"YAML 配置文件路径（默认: {default_cfg}）",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="覆盖配置文件中的 logging.level",
    )
    parser.add_argument(
        "--session-name",
        type=str,
        default=None,
        help="本次采集名字。指定后不弹窗，直接把名字加入数据集保存路径。",
    )
    parser.add_argument(
        "--no-name-dialog",
        action="store_true",
        help="不弹出名字输入框。如果没有 --session-name，则不修改保存路径。",
    )
    return parser.parse_args()


def _sanitize_name_for_path(name: str) -> str:
    """把用户输入的名字转换成安全的路径片段。

    保留中文、英文、数字、下划线、短横线；空格和常见非法路径字符替换为下划线。
    """
    name = str(name).strip()
    name = re.sub(r"[\\/:*?\"<>|\s]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("._-")


def _ask_session_name_by_tk() -> Optional[str]:
    """弹出名字输入框。

    Returns:
        str: 点击确认且输入有效名字。
        None: 点击取消、关闭窗口、或 GUI 不可用且命令行取消。
    """
    try:
        import tkinter as tk
        from tkinter import messagebox
    except Exception:
        raw = input("[INPUT] 请输入本次采集名字，直接回车取消退出: ").strip()
        return raw or None

    result: dict[str, Optional[str]] = {"name": None}

    root = tk.Tk()

    # Linux 上 Tk 默认字体经常不包含中文字形，会显示方框或乱码。
    # 优先选择常见中文字体；如果系统未安装中文字体，则退回英文界面，
    # 保证窗口内容始终可读。
    try:
        import tkinter.font as tkfont
        families = set(tkfont.families(root))
        candidates = (
            "Noto Sans CJK SC",
            "Noto Sans CJK TC",
            "WenQuanYi Micro Hei",
            "WenQuanYi Zen Hei",
            "Microsoft YaHei",
            "SimHei",
            "PingFang SC",
        )
        cjk_font = next((name for name in candidates if name in families), None)
    except Exception:
        cjk_font = None

    if cjk_font:
        title_text = "输入采集名字"
        label_text = "请输入本次采集名字："
        ok_text = "确认"
        cancel_text = "取消"
        empty_title = "名字不能为空"
        empty_message = "请输入一个有效名字，或点击取消退出。"
        ui_font = (cjk_font, 12)
    else:
        title_text = "Collection Name"
        label_text = "Enter collection name:"
        ok_text = "OK"
        cancel_text = "Cancel"
        empty_title = "Invalid name"
        empty_message = "Please enter a valid name or click Cancel."
        ui_font = ("DejaVu Sans", 12)

    root.title(title_text)
    root.geometry("420x170")
    root.resizable(False, False)

    # 尽量置顶，避免窗口弹出后被终端挡住。
    try:
        root.attributes("-topmost", True)
        root.after(500, lambda: root.attributes("-topmost", False))
    except Exception:
        pass

    tk.Label(root, text=label_text, font=ui_font).pack(pady=(18, 8))

    name_var = tk.StringVar()
    entry = tk.Entry(root, textvariable=name_var, width=34, font=ui_font)
    entry.pack(pady=(0, 12))
    entry.focus_set()

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=4)

    def on_ok() -> None:
        raw = name_var.get().strip()
        safe = _sanitize_name_for_path(raw)
        if not safe:
            messagebox.showwarning(empty_title, empty_message, parent=root)
            entry.focus_set()
            return
        result["name"] = safe
        root.destroy()

    def on_cancel() -> None:
        result["name"] = None
        root.destroy()

    tk.Button(btn_frame, text=ok_text, width=12, command=on_ok, font=ui_font).pack(side=tk.LEFT, padx=10)
    tk.Button(btn_frame, text=cancel_text, width=12, command=on_cancel, font=ui_font).pack(side=tk.LEFT, padx=10)

    root.bind("<Return>", lambda _e: on_ok())
    root.bind("<Escape>", lambda _e: on_cancel())
    root.protocol("WM_DELETE_WINDOW", on_cancel)

    root.mainloop()
    return result["name"]


def _apply_session_name_to_dataset_path(cfg: CollectConfig, session_name: str) -> None:
    """把名字加入保存路径。

    cfg.resolve_repo_id() 最终决定保存目录：cfg.dataset.root / repo_id。
    - repo_id=auto 时：修改 name_prefix，最后路径形如 prefix_name_h85_YYYYmmdd_HHMMSS。
    - repo_id 非 auto 时：直接在 repo_id 后追加名字，确保路径也包含名字。
    """
    safe = _sanitize_name_for_path(session_name)
    if not safe:
        return

    if cfg.dataset.repo_id == "auto":
        base = _sanitize_name_for_path(cfg.dataset.name_prefix) or "collect"
        if not base.endswith(f"_{safe}"):
            cfg.dataset.name_prefix = f"{base}_{safe}"
    else:
        base = _sanitize_name_for_path(cfg.dataset.repo_id) or "collect"
        if not base.endswith(f"_{safe}"):
            cfg.dataset.repo_id = f"{base}_{safe}"


def main() -> None:
    args = parse_args()
    cfg = load_collect_config(args.config)

    session_name: Optional[str]
    if args.session_name:
        session_name = _sanitize_name_for_path(args.session_name)
    elif args.no_name_dialog:
        session_name = None
    else:
        session_name = _ask_session_name_by_tk()
        if session_name is None:
            print("[INFO] 已取消采集。")
            return

    if session_name:
        _apply_session_name_to_dataset_path(cfg, session_name)
        print(f"[INFO] 本次采集名字: {session_name}")
        print(f"[INFO] 数据集 name_prefix/repo_id 已加入名字。")

    level = args.log_level or cfg.logging.level
    setup_logging(level=level, log_file=cfg.logging.log_file)

    app = DataCollectApp(cfg)
    app.run_interactive()


if __name__ == "__main__":
    main()
