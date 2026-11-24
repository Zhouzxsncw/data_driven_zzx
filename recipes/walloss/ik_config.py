"""WALL-OSS RM75 IK 选项。YAML ``policy.ik`` 与服务端 ``--ik-*`` / 环境变量共用这一份。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

IK_MODES = (
    "abort_rest",
    "continue_chunk",
    "psi_blend",
    "split_jump",
    "dls",
)

IK_SOLVERS = (
    "realman",
    "dls",
    "lm",
    "svd",
    "jacT",
    "ccd",
    "multiseed",
)

IK_TRAJ_OPTS = (
    "none",
    "psi_dp",
    "smooth",
    "gn",
    "retract",
)

_MODE_ALIASES = {
    "abort": "abort_rest",
    "baseline": "abort_rest",
    "continue": "continue_chunk",
    "no_abort": "continue_chunk",
    "arm_angle": "psi_blend",
    "psi": "psi_blend",
    "split": "split_jump",
    "joint_split": "split_jump",
    "numerical": "dls",
    "jacobian": "dls",
}

_TRAJ_ALIASES = {
    "off": "none",
    "greedy": "none",
    "dp": "psi_dp",
    "arm_dp": "psi_dp",
    "qp": "smooth",
    "smooth_qp": "smooth",
    "ilqr": "gn",
    "gauss_newton": "gn",
    "interp": "retract",
    "joint_interp": "retract",
}

_SOLVER_ALIASES = {
    "analytical": "realman",
    "rm": "realman",
    "sdk": "realman",
    "levenberg": "lm",
    "levenberg_marquardt": "lm",
    "pseudoinverse": "svd",
    "pinv": "svd",
    "transpose": "jacT",
    "jt": "jacT",
    "multi": "multiseed",
    "multi_seed": "multiseed",
}


@dataclass(frozen=True)
class IkSettings:
    mode: str = "abort_rest"
    solver: str = "realman"
    traj_opt: str = "none"
    max_joint_step_deg: float = 40.0
    arm_angle_step_deg: float = 15.0
    abort_on_held: bool = True
    psi_step_deg: float = 8.0
    split_max_substeps: int = 4
    dls_lambda: float = 0.08
    dls_iters: int = 12
    dls_pos_step_m: float = 0.012
    traj_w_smooth: float = 0.4
    traj_opt_iters: int = 8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def effective_solver(self) -> str:
        """兼容旧测试：``mode: dls`` 在标签上仍叫 dls。

        真机路径（``solver: realman``）已改成解析优先、再 DLS 补。
        纯数值请写 ``solver: dls``。
        """
        if self.mode == "dls" and self.solver == "realman":
            return "dls"
        return self.solver


def parse_ik_config(raw: Optional[Any] = None) -> IkSettings:
    """Accept dict / YAML ``policy.ik`` / None. Unknown keys ignored."""
    if raw is None:
        return IkSettings()
    if isinstance(raw, IkSettings):
        return raw
    if not isinstance(raw, dict):
        raise ValueError(f"policy.ik 必须是字典或 null，得到 {type(raw).__name__}")
    mode_raw = str(raw.get("mode") or "abort_rest").strip().lower()
    mode = _MODE_ALIASES.get(mode_raw, mode_raw)
    if mode not in IK_MODES:
        raise ValueError(
            f"policy.ik.mode 只能是 {list(IK_MODES)}（或别名 {list(_MODE_ALIASES)}），得到 {mode_raw!r}"
        )
    solver_raw = str(raw.get("solver") or "realman").strip()
    key = solver_raw.lower()
    if key in ("jact", "jt", "transpose"):
        solver = "jacT"
    else:
        solver = _SOLVER_ALIASES.get(key, key)
    if solver not in IK_SOLVERS:
        raise ValueError(
            f"policy.ik.solver 只能是 {list(IK_SOLVERS)}（或别名 {list(_SOLVER_ALIASES)}），得到 {solver_raw!r}"
        )
    traj_raw = str(raw.get("traj_opt") or "none").strip().lower()
    traj_opt = _TRAJ_ALIASES.get(traj_raw, traj_raw)
    if traj_opt not in IK_TRAJ_OPTS:
        raise ValueError(
            f"policy.ik.traj_opt 只能是 {list(IK_TRAJ_OPTS)}（或别名 {list(_TRAJ_ALIASES)}），得到 {traj_raw!r}"
        )

    def _f(key: str, default: float) -> float:
        val = raw.get(key, default)
        try:
            out = float(val)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"policy.ik.{key} 必须是数字，得到 {val!r}") from exc
        return out

    def _i(key: str, default: int, lo: int, hi: int) -> int:
        val = raw.get(key, default)
        try:
            out = int(val)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"policy.ik.{key} 必须是整数，得到 {val!r}") from exc
        if out < lo or out > hi:
            raise ValueError(f"policy.ik.{key} 须在 {lo}..{hi}，得到 {out}")
        return out

    step = _f("max_joint_step_deg", 40.0)
    if step < 5.0 or step > 90.0:
        raise ValueError(f"policy.ik.max_joint_step_deg 须在 5..90，得到 {step}")
    psi_step = _f("psi_step_deg", 8.0)
    if psi_step < 1.0 or psi_step > 45.0:
        raise ValueError(f"policy.ik.psi_step_deg 须在 1..45，得到 {psi_step}")
    lam = _f("dls_lambda", 0.08)
    if lam <= 0.0 or lam > 2.0:
        raise ValueError(f"policy.ik.dls_lambda 须在 (0, 2]，得到 {lam}")
    w_smooth = _f("traj_w_smooth", 0.4)
    if w_smooth < 0.0 or w_smooth > 8.0:
        raise ValueError(f"policy.ik.traj_w_smooth 须在 0..8，得到 {w_smooth}")
    return IkSettings(
        mode=mode,
        solver=solver,
        traj_opt=traj_opt,
        max_joint_step_deg=step,
        arm_angle_step_deg=max(2.0, min(45.0, _f("arm_angle_step_deg", 15.0))),
        abort_on_held=bool(raw.get("abort_on_held", True)),
        psi_step_deg=psi_step,
        split_max_substeps=_i("split_max_substeps", 4, 1, 16),
        dls_lambda=lam,
        dls_iters=_i("dls_iters", 12, 1, 40),
        dls_pos_step_m=max(0.001, min(0.08, _f("dls_pos_step_m", 0.012))),
        traj_w_smooth=w_smooth,
        traj_opt_iters=_i("traj_opt_iters", 8, 1, 30),
    )
