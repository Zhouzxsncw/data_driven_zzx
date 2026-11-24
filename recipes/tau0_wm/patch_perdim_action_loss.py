#!/usr/bin/env python3
"""fork patch 6: per-dim action loss TB 记录（2026-09-20）。

τ0-WM 上游 loss_action 在 .mean() 时把 20 维压成标量，TB 只有 Action loss 总项，
违反舰队 per-dim loss 监控规范。本补丁在 mean 前保留 20 维分量，
steps_to_log 同步点写 20 个 Action loss dim/* tag。
"""
import py_compile

P = "/data1/tau0wm/repo/tau-0-wm/runner/posttrain.py"
src = open(P).read()

OLD1 = """                    if self.args.return_action:
                        loss_action = (pred_all['action']-action_target)**2
                        loss_action = loss_action.mean()"""
NEW1 = """                    if self.args.return_action:
                        loss_action = (pred_all['action']-action_target)**2
                        # fork patch per-dim action loss: mean 前保留 20 维分量（fork patch 6, 2026-09-20）
                        _pd = loss_action.mean(dim=tuple(range(loss_action.ndim - 1)))
                        self._action_loss_per_dim = [round(float(v), 6) for v in _pd.detach().float().cpu()]
                        loss_action = loss_action.mean()"""

OLD2 = """                                if self.args.return_action:
                                    self.writer.add_scalar("Action loss", loss_action.item(), global_step)"""
NEW2 = """                                if self.args.return_action:
                                    self.writer.add_scalar("Action loss", loss_action.item(), global_step)
                                    _names = ["L_x","L_y","L_z","L_r6d0","L_r6d1","L_r6d2","L_r6d3","L_r6d4","L_r6d5","L_grip",
                                              "R_x","R_y","R_z","R_r6d0","R_r6d1","R_r6d2","R_r6d3","R_r6d4","R_r6d5","R_grip"]
                                    _pd = getattr(self, "_action_loss_per_dim", None)
                                    if _pd is not None and len(_pd) == len(_names):
                                        for _n, _v in zip(_names, _pd):
                                            self.writer.add_scalar(f"Action loss dim/{_n}", _v, global_step)"""

assert src.count(OLD1) == 1, f"OLD1 count={src.count(OLD1)}"
assert src.count(OLD2) == 1, f"OLD2 count={src.count(OLD2)}"
src = src.replace(OLD1, NEW1).replace(OLD2, NEW2)
open(P, "w").write(src)
py_compile.compile(P, doraise=True)
print("PATCH_OK compile_ok")
