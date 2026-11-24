#!/usr/bin/env python3
"""对 τ0-WM 上游仓库打**本地 fork 补丁**（幂等，可重复执行）。

仓库在工作区之外（``/home/xk/VLA/tau-0-wm``、h100 上 ``/data1/tau0wm/repo/tau-0-wm``），
所以不能用编辑器改，统一走这个脚本；h100 侧在 rsync 之后、启动训练之前必须跑一次。

补丁清单
--------
1. ``runner/posttrain.py`` 的 ``from utils.model_utils import forward_pass``
   上游 commit a3e4c0f 里 ``forward_pass`` **全仓库无定义**（只有这一行引用它），
   于是 ``from runner.posttrain import Trainer`` 直接 ImportError —— 训练根本起不来。
   训练循环本身是内联在 ``Trainer.train()`` 里的（noisy_actions / action_target /
   ``self.diffusion_model(...)`` / MSE），不依赖该符号，因此改成可选导入：
   上游若哪天补齐就自动生效，没补齐也不炸。

2. ``models/wan_2_2_models/transformers/model.py`` 的 ``WanModel`` 缺
   ``_supports_gradient_checkpointing = True``。
   ``forward()`` 里**本来就实现了** ``if self.training and self.gradient_checkpointing:
   torch.utils.checkpoint.checkpoint(create_custom_forward(block), ...)``，
   但 diffusers≥0.30 的 ``ModelMixin.enable_gradient_checkpointing()`` 先看这个类属性，
   没有就直接 ``ValueError: WanModel does not support gradient checkpointing``
   → ``gradient_checkpointing: true`` 的配置在 h100 冒烟阶段必崩（实测 exitcode 120）。
   补上类属性即可：diffusers 随后会把 ``self.gradient_checkpointing`` 置 True，
   真正走上游那段 checkpoint 分支（省激活显存），而不是关掉这个开关。

3. ``runner/posttrain.py`` 训练循环的 ``train_steps`` 早退只 break 内层，
   外层 epoch 循环（train_epochs=10000）继续 → 到点后永不退出。

4. ``utils/__init__.py`` 的 ``Tee.write/flush`` 在解释器 finalization 阶段
   对已析构的流抛异常 → unraisable → exitcode 120。实测冒烟 20/20 步跑完、
   epoch 跳出、存档全部成功后死在退出路上，relay 会把 rc=120 判成冒烟失败。
   ``Tee`` 改成写失败静默丢弃（此时训练早已落盘，丢的只是收尾噪声）。

5. ``runner/posttrain.py`` 的 ``%steps_to_log / %steps_to_val / %steps_to_save / %gc``
   四个周期块上游缩进在外层 epoch-stop ``break`` 之后 → **永久不可达死代码**。
   30k 正式跑实测：零中途落盘（无 latest_accelerator_state、无 step_5000..25000）、
   零 tensorboard（events 仅 88B/263B）、config 无 latest_global_step 回填。
   补丁把四个块移进内层 step 循环（``accelerator.log`` 之后、``sync_gradients`` 守住），
   并删除死区副本 —— 否则 30 万步长跑中途一个 ckpt 都不落，崩了全损。

用法::

    python patch_tau_repo.py --repo /home/xk/VLA/tau-0-wm
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

DEAD_IMPORT = "from utils.model_utils import forward_pass\n"

GUARDED = (
    "# fork patch 2026-09-14: 上游 a3e4c0f 里 forward_pass 只有 import、全仓库无定义，\n"
    "# 会让 `from runner.posttrain import Trainer` 直接 ImportError。训练循环是内联的\n"
    "# （见 Trainer.train()），不依赖该符号，故改成可选导入。\n"
    "try:\n"
    "    from utils.model_utils import forward_pass  # noqa: F401\n"
    "except ImportError:\n"
    "    forward_pass = None\n"
)

NOSPLIT = "    _no_split_modules = ['WanAttentionBlock']\n"

GC_FLAG = (
    "    _no_split_modules = ['WanAttentionBlock']\n"
    "    # fork patch 2026-09-15: forward() 里已实现 torch.utils.checkpoint 分支，但\n"
    "    # diffusers>=0.30 的 enable_gradient_checkpointing() 先查这个类属性，\n"
    "    # 缺了就 ValueError 让训练起不来。\n"
    "    _supports_gradient_checkpointing = True\n"
)

TEE_OLD = (
    "class Tee:\n"
    "    def __init__(self, *files):\n"
    "        self.files = files\n"
    "    def write(self, obj):\n"
    "        for f in self.files:\n"
    "            f.write(obj)\n"
    "            f.flush()\n"
    "    def flush(self):\n"
    "        for f in self.files:\n"
    "            f.flush()\n"
)

TEE_NEW = (
    "class Tee:\n"
    "    def __init__(self, *files):\n"
    "        self.files = files\n"
    "    def write(self, obj):\n"
    "        for f in self.files:\n"
    "            try:\n"
    "                f.write(obj)\n"
    "                f.flush()\n"
    "            except Exception:\n"
    "                # fork patch shutdown-tee: 解释器 finalization 阶段这些流可能已关闭/析构，\n"
    "                # 此时抛异常会变成 unraisable → python 以 exitcode 120 退出，torchrun 把\n"
    "                # 已经跑完并存档成功的训练报成 FAILED（实测冒烟 20/20 后 rc=120）。丢弃即可。\n"
    "                pass\n"
    "    def flush(self):\n"
    "        for f in self.files:\n"
    "            try:\n"
    "                f.flush()\n"
    "            except Exception:\n"
    "                pass\n"
)


INNER_BREAK = (
    "                if global_step >= self.state.train_steps:\n"
    '                    logger.info(">>> max train step reached")\n'
    "                    break\n"
)

EPOCH_STOP = INNER_BREAK + (
    "\n"
    "            # fork patch epoch-stop: 上面只 break 了内层 step 循环，外层 epoch 循环照样继续，\n"
    "            # 于是每个 epoch 白跑一个 optimizer step 再 break（train_epochs=10000 → 永不退出）。\n"
    "            # 实测冒烟 train_steps=20 却刷到 134 it / epoch 115，被 relay_watch 的 SMOKE_TIMEOUT\n"
    "            # 判成超时杀掉；正式跑过了 train_steps 也只是空转。这里补外层跳出，\n"
    "            # 让 train_steps 成为真正的终止条件（跳出后上游本来就有 final save_pretrained）。\n"
    "            if global_step >= self.state.train_steps:\n"
    '                logger.info(">>> outer loop: train_steps 已达，结束 epoch 循环")\n'
    "                break\n"
)

PER_STEP_MARKER = "fork patch per-step-blocks"
PER_STEP_ANCHOR = "                accelerator.log(logs, step=global_step)\n"

# 四个周期块的新家：内层 step 循环内、accelerator.log 之后。sync_gradients 守住，
# 只在真 optimizer step（global_step 已 +=1）之后触发；gradient_accumulation>1 时
# 不会在同一个 step 上重复落盘。块体逻辑与上游逐行一致（缩进加深一级）。
PER_STEP_BLOCKS = (
    "\n"
    "                # fork patch per-step-blocks: 上游的 %steps_to_log/val/save/gc 四个周期块\n"
    "                # 缩进在外层 epoch-stop break 之后 → 永久死区（30k 全程零落盘、events\n"
    "                # 仅 88B 即铁证）。这里移回内层 step 循环，sync_gradients 守住只在真优化\n"
    "                # 步后触发；save 块让 steps_to_save 真正落 ckpt 并回填 latest_global_step。\n"
    "                if accelerator.sync_gradients:\n"
    "                    if global_step % self.args.steps_to_log == 0:\n"
    "                        if accelerator.is_main_process:\n"
    "                            if self.writer is not None:\n"
    "                                self.writer.add_scalar(\"Training Loss\", loss.item(), global_step)\n"
    "                                if self.args.return_video:\n"
    "                                    self.writer.add_scalar(\"Video loss\", loss_video.item(), global_step)\n"
    "                                if self.args.return_action:\n"
    "                                    self.writer.add_scalar(\"Action loss\", loss_action.item(), global_step)\n"
    "                        accelerator.wait_for_everyone()\n"
    "\n"
    "                    if global_step % self.args.steps_to_val == 0:\n"
    "                        accelerator.wait_for_everyone()\n"
    "                        if accelerator.is_main_process:\n"
    "                            model_save_dir = os.path.join(self.save_folder,f'Validation_step_{global_step}')\n"
    "                            self.validate(accelerator, model_save_dir, global_step, n_view=n_view, n_chunk=1)\n"
    "                        accelerator.wait_for_everyone()\n"
    "\n"
    "                    if global_step % self.args.steps_to_save == 0:\n"
    "                        accelerator.save_state(os.path.join(self.save_folder, \"latest_accelerator_state\"))\n"
    "                        accelerator.wait_for_everyone()\n"
    "                        if accelerator.is_main_process:\n"
    "                            model_to_save = unwrap_model(accelerator, self.diffusion_model)\n"
    "                            dtype = (\n"
    "                                torch.float16\n"
    "                                if self.args.mixed_precision == \"fp16\"\n"
    "                                else torch.bfloat16\n"
    "                                if self.args.mixed_precision == \"bf16\"\n"
    "                                else torch.float32\n"
    "                            )\n"
    "                            model_save_dir = os.path.join(self.save_folder,f'step_{global_step}')\n"
    "                            model_to_save.save_pretrained(model_save_dir, safe_serialization=False)\n"
    "                            del  model_to_save\n"
    "\n"
    "                            tmp_cd = load(open(self.config_file, \"r\"), Loader=Loader)\n"
    "                            tmp_cd[\"latest_log_dir\"] = model_save_dir\n"
    "                            tmp_cd[\"latest_global_step\"] = global_step\n"
    "                            tmp_cd[\"latest_epoch\"] = epoch\n"
    "                            tmp_cd[\"optimizer_path\"] = os.path.join(self.save_folder, \"latest_accelerator_state\")\n"
    "                            with open(self.config_file, \"w\") as f_cfg:\n"
    "                                dump(tmp_cd, f_cfg, Dumper=Dumper)\n"
    "\n"
    "                        accelerator.wait_for_everyone()\n"
    "\n"
    "                    if global_step % 10000 == 0:\n"
    "                        accelerator.wait_for_everyone()\n"
    "                        collected = gc.collect()\n"
    "                        gc.disable()\n"
    "                        accelerator.wait_for_everyone()\n"
)

# 死区：外层 epoch-stop break 之后到 memory_statistics 之间的四个不可达周期块。
# 用正则整体移除（不问块内每一行的尾空格——上游原文可能有，正则 .*? 全包）。
DEAD_ZONE_RE = re.compile(
    r"(            if global_step >= self\.state\.train_steps:\n"
    r'                logger\.info\(">>> outer loop: train_steps 已达，结束 epoch 循环"\)\n'
    r"                break\n)"
    r"\n"
    r"                if global_step % self\.args\.steps_to_log == 0:.*?"
    r"\n            memory_statistics",
    re.S,
)

DEAD_ZONE_NOTE = (
    "\n"
    "            # fork patch dead-zone cleanup: 上面四个周期块曾是外层 break 后的永久死代码\n"
    "            # （30k 全程零落盘、events 仅 88B 即铁证），功能已由 per-step-blocks 在内层接管。\n"
    "            memory_statistics"
)

# 补丁 6（2026-09-20）：per-dim action loss。上游 loss_action 在 .mean() 时把 20 维压成
# 标量，TB 只有 Action loss 总项，违反舰队 per-dim loss 监控规范。mean 前保留 20 维分量，
# steps_to_log 同步点写 20 个 "Action loss dim/<name>" tag。配套监控：
# local_scripts/monitor_action_dim_loss.py（30min 循环 dim_monitor_loop.sh）。
PERDIM_OLD1 = """                    if self.args.return_action:
                        loss_action = (pred_all['action']-action_target)**2
                        loss_action = loss_action.mean()"""
PERDIM_NEW1 = """                    if self.args.return_action:
                        loss_action = (pred_all['action']-action_target)**2
                        # fork patch per-dim action loss: mean 前保留 20 维分量（fork patch 6, 2026-09-20）
                        _pd = loss_action.mean(dim=tuple(range(loss_action.ndim - 1)))
                        self._action_loss_per_dim = [round(float(v), 6) for v in _pd.detach().float().cpu()]
                        loss_action = loss_action.mean()"""
PERDIM_OLD2 = """                                if self.args.return_action:
                                    self.writer.add_scalar("Action loss", loss_action.item(), global_step)"""
PERDIM_NEW2 = """                                if self.args.return_action:
                                    self.writer.add_scalar("Action loss", loss_action.item(), global_step)
                                    _names = ["L_x","L_y","L_z","L_r6d0","L_r6d1","L_r6d2","L_r6d3","L_r6d4","L_r6d5","L_grip",
                                              "R_x","R_y","R_z","R_r6d0","R_r6d1","R_r6d2","R_r6d3","R_r6d4","R_r6d5","R_grip"]
                                    _pd = getattr(self, "_action_loss_per_dim", None)
                                    if _pd is not None and len(_pd) == len(_names):
                                        for _n, _v in zip(_names, _pd):
                                            self.writer.add_scalar(f"Action loss dim/{_n}", _v, global_step)"""

PATCHES: list[tuple[str, str, str, str]] = [
    ("runner/posttrain.py", DEAD_IMPORT, GUARDED, "fork patch 2026-09-14"),
    ("runner/posttrain.py", INNER_BREAK, EPOCH_STOP, "fork patch epoch-stop"),
    ("models/wan_2_2_models/transformers/model.py", NOSPLIT, GC_FLAG, "fork patch 2026-09-15"),
    ("utils/__init__.py", TEE_OLD, TEE_NEW, "fork patch shutdown-tee"),
    ("runner/posttrain.py", PERDIM_OLD1, PERDIM_NEW1, "fork patch 6 per-dim action loss"),
    ("runner/posttrain.py", PERDIM_OLD2, PERDIM_NEW2, "fork patch 6 per-dim TB tags"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=Path("/home/xk/VLA/tau-0-wm"))
    args = ap.parse_args()

    for rel, old, new, marker in PATCHES:
        f = args.repo / rel
        if not f.is_file():
            print(f"SKIP {rel}（文件不存在）")
            continue
        text = f.read_text()
        if marker in text:
            print(f"OK   {rel} 已打过补丁（{marker}）")
            continue
        n = text.count(old)
        if n != 1:
            print(f"WARN {rel} 待替换片段出现 {n} 次（需要恰好 1 次），上游可能已变更，需人工确认")
            continue
        f.write_text(text.replace(old, new, 1))
        print(f"FIX  {rel}（{marker}）")

    # 验证：trainer 模块必须能被 import（只做 import，不建模型）
    import subprocess
    import sys

    r = subprocess.run(
        [sys.executable, "-c", "import runner.posttrain as m; print('Trainer', m.Trainer.__name__)"],
        cwd=args.repo, capture_output=True, text=True,
    )
    print(r.stdout.strip() or r.stderr.strip().splitlines()[-1])

    # 验证 2：梯度检查开关的类属性必须到位（缺了的话冒烟必崩，宁可在这里先失败）
    model_py = args.repo / "models/wan_2_2_models/transformers/model.py"
    if model_py.is_file():
        if "_supports_gradient_checkpointing = True" in model_py.read_text():
            print("OK   WanModel._supports_gradient_checkpointing = True")
        else:
            print("FATAL WanModel 仍缺 _supports_gradient_checkpointing，"
                  "gradient_checkpointing: true 会在 prepare_trainable_parameters 处崩")
            raise SystemExit(1)

    # 验证 3：train_steps 必须是「外层也认」的终止条件，且改完的文件要能编译
    pt = args.repo / "runner/posttrain.py"
    if pt.is_file():
        src = pt.read_text()
        if "fork patch epoch-stop" not in src:
            print("FATAL runner/posttrain.py 仍缺外层 epoch 跳出，train_steps 到点后不会退出")
            raise SystemExit(1)
        import ast

        try:
            ast.parse(src)
            print("OK   runner/posttrain.py 语法可编译 + epoch 外层跳出已到位")
        except SyntaxError as e:
            print(f"FATAL runner/posttrain.py 补丁后语法错误：{e}")
            raise SystemExit(1)

    # 验证 4：Tee 必须关闭感知，否则训练全成功后仍会以 exitcode 120 被 relay 判死
    init_py = args.repo / "utils/__init__.py"
    if init_py.is_file():
        if "fork patch shutdown-tee" in init_py.read_text():
            print("OK   utils Tee 已关闭感知（finalization 噪声不再变成 rc=120）")
        else:
            print("FATAL utils/__init__.py 的 Tee 仍会在解释器退出时抛 unraisable → rc=120")
            raise SystemExit(1)

    # 补丁 5 + 验证 5：四个周期块必须活在内层循环里、死区必须已清空，
    # 否则长跑中途零落盘（这是 30k 全程没有 optimizer state 可续的根因）。
    if pt.is_file():
        src = pt.read_text()
        if PER_STEP_MARKER in src:
            print("OK   per-step-blocks 已移入内层循环（save/log/val/gc 逐步生效）")
            dead_alive = DEAD_ZONE_RE.search(src)
            if dead_alive:
                print("FATAL 死区周期块副本仍在（epoch-stop break 之后），上游结构异常")
                raise SystemExit(1)
        else:
            n_anchor = src.count(PER_STEP_ANCHOR)
            if n_anchor != 1 or not DEAD_ZONE_RE.search(src):
                print("WARN per-step-blocks 锚点/死区与预期不符（上游可能已变更），跳过补丁 5，需人工确认")
            else:
                src = src.replace(PER_STEP_ANCHOR, PER_STEP_ANCHOR + PER_STEP_BLOCKS, 1)
                src = DEAD_ZONE_RE.sub(lambda mm: mm.group(1) + DEAD_ZONE_NOTE, src, count=1)
                pt.write_text(src)
                print("FIX  per-step-blocks 移入内层循环 + 死区已清空")
        import ast
        try:
            ast.parse(pt.read_text())
            print("OK   runner/posttrain.py 补丁 5 后语法可编译")
        except SyntaxError as e:
            print(f"FATAL runner/posttrain.py 补丁 5 后语法错误：{e}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
