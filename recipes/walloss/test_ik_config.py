#!/usr/bin/env python3
"""IK YAML / mode 解析与 abort vs continue 截断语义。不连真机、不加载权重。"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from ik_config import IK_MODES, IK_SOLVERS, IK_TRAJ_OPTS, parse_ik_config  # noqa: E402
from ee_infer_pipeline import (  # noqa: E402
    wrap_add_7deg,
    wrap_delta_psi_deg,
)

CLIENT_YAMLS = [
    "infer/infer/config/lingbot/walloss_tongbot_ee_00036_s40000_local.yaml",
    "infer/infer/config/lingbot/walloss_tongbot_ee_00073_ah8_s10000_local.yaml",
    "infer/infer/config/lingbot/walloss_tongbot_ee_00086_local.yaml",
    "infer/infer/config/lingbot/walloss_tongbot_ee_00091_local.yaml",
]


def test_parse_defaults() -> dict:
    s = parse_ik_config(None)
    ok = s.mode == "abort_rest" and s.solver == "realman" and s.traj_opt == "none" and s.max_joint_step_deg == 40.0
    return {"name": "parse_defaults", "ok": ok, "mode": s.mode, "solver": s.solver, "traj_opt": s.traj_opt}


def test_aliases() -> dict:
    mapping = {
        "abort": "abort_rest",
        "continue": "continue_chunk",
        "psi": "psi_blend",
        "split": "split_jump",
        "numerical": "dls",
        "CONTINUE_CHUNK": "continue_chunk",
    }
    got = {raw: parse_ik_config({"mode": raw}).mode for raw in mapping}
    ok = got == mapping and set(IK_MODES) == {
        "abort_rest",
        "continue_chunk",
        "psi_blend",
        "split_jump",
        "dls",
    }
    return {"name": "aliases", "ok": ok, "got": got}


def test_solver_aliases() -> dict:
    mapping = {
        "realman": "realman",
        "analytical": "realman",
        "dls": "dls",
        "lm": "lm",
        "levenberg": "lm",
        "svd": "svd",
        "pinv": "svd",
        "jacT": "jacT",
        "jact": "jacT",
        "ccd": "ccd",
        "multi": "multiseed",
    }
    got = {raw: parse_ik_config({"solver": raw}).solver for raw in mapping}
    ok = got == mapping and set(IK_SOLVERS) == {
        "realman",
        "dls",
        "lm",
        "svd",
        "jacT",
        "ccd",
        "multiseed",
    }
    dls_mode = parse_ik_config({"mode": "dls"})
    ok = ok and dls_mode.solver == "realman" and dls_mode.effective_solver() == "dls"
    return {"name": "solver_aliases", "ok": ok, "got": got}


def test_traj_aliases() -> dict:
    mapping = {
        "none": "none",
        "off": "none",
        "psi_dp": "psi_dp",
        "dp": "psi_dp",
        "smooth": "smooth",
        "qp": "smooth",
        "gn": "gn",
        "ilqr": "gn",
        "retract": "retract",
        "interp": "retract",
    }
    got = {raw: parse_ik_config({"traj_opt": raw}).traj_opt for raw in mapping}
    ok = got == mapping and set(IK_TRAJ_OPTS) == {"none", "psi_dp", "smooth", "gn", "retract"}
    return {"name": "traj_aliases", "ok": ok, "got": got}


def test_reject_bad() -> dict:
    cases = []
    for raw in (
        {"mode": "magic"},
        {"mode": "abort_rest", "max_joint_step_deg": 3},
        {"mode": "abort_rest", "max_joint_step_deg": 180},
        {"solver": "magic"},
        {"traj_opt": "magic"},
        "abort_rest",
    ):
        try:
            parse_ik_config(raw)
            cases.append(False)
        except ValueError:
            cases.append(True)
    return {"name": "reject_bad", "ok": all(cases)}


def test_client_yamls() -> dict:
    missing = []
    modes = []
    for rel in CLIENT_YAMLS:
        path = REPO / rel
        if not path.is_file():
            missing.append(rel)
            continue
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        ik = (raw.get("policy") or {}).get("ik")
        if not isinstance(ik, dict):
            missing.append(rel)
            continue
        settings = parse_ik_config(ik)
        modes.append(settings.mode)
        if settings.mode != "abort_rest":
            missing.append(f"{rel}:default_not_abort_rest")
        if settings.solver != "realman":
            missing.append(f"{rel}:default_not_realman")
        if settings.traj_opt != "none":
            missing.append(f"{rel}:default_not_traj_none")
    standalone = parse_ik_config(yaml.safe_load((SCRIPT_DIR / "ik.yaml").read_text(encoding="utf-8")))
    ok = (
        not missing
        and standalone.mode == "abort_rest"
        and standalone.solver == "realman"
        and standalone.traj_opt == "none"
    )
    return {"name": "client_yamls", "ok": ok, "missing": missing, "modes": modes}


def test_solver_preset_yamls() -> dict:
    expected = {
        "ik_solver_dls.yaml": "dls",
        "ik_solver_lm.yaml": "lm",
        "ik_solver_svd.yaml": "svd",
        "ik_solver_jacT.yaml": "jacT",
        "ik_solver_ccd.yaml": "ccd",
        "ik_solver_multiseed.yaml": "multiseed",
    }
    missing = []
    got = {}
    for name, solver in expected.items():
        path = SCRIPT_DIR / name
        if not path.is_file():
            missing.append(name)
            continue
        settings = parse_ik_config(yaml.safe_load(path.read_text(encoding="utf-8")))
        got[name] = settings.solver
        if settings.solver != solver or settings.mode != "abort_rest":
            missing.append(f"{name}:{settings.solver}/{settings.mode}")
    ok = not missing and got == expected
    return {"name": "solver_preset_yamls", "ok": ok, "missing": missing, "got": got}


def test_traj_preset_yamls() -> dict:
    expected = {
        "ik_traj_psi_dp.yaml": "psi_dp",
        "ik_traj_smooth.yaml": "smooth",
        "ik_traj_gn.yaml": "gn",
        "ik_traj_retract.yaml": "retract",
    }
    missing = []
    got = {}
    for name, traj in expected.items():
        path = SCRIPT_DIR / name
        if not path.is_file():
            missing.append(name)
            continue
        settings = parse_ik_config(yaml.safe_load(path.read_text(encoding="utf-8")))
        got[name] = settings.traj_opt
        if settings.traj_opt != traj or settings.mode != "abort_rest":
            missing.append(f"{name}:{settings.traj_opt}/{settings.mode}")
    ok = not missing and got == expected
    return {"name": "traj_preset_yamls", "ok": ok, "missing": missing, "got": got}


def test_wrap_helpers() -> dict:
    q = wrap_add_7deg([179.0, 0, 0, 0, 0, 0, 350.0], [3.0, 0, 0, 0, 0, 0, 20.0])
    psi = wrap_delta_psi_deg(-170.0, 170.0)
    ok = abs(float(q[0]) + 178.0) < 1e-6 and abs(float(q[6]) + 350.0) < 1e-6 and abs(psi - 20.0) < 1e-6
    return {"name": "wrap_helpers", "ok": ok, "q0": float(q[0]), "q6": float(q[6]), "psi": psi}


@dataclass
class _FakePlan:
    cartesian_limited: bool
    held: bool


def _chunk_stop(mode: str, abort_on_held: bool, plan: _FakePlan) -> bool:
    stop_held = bool(abort_on_held and plan.held)
    stop_limit = mode == "abort_rest" and (plan.cartesian_limited or plan.held)
    return stop_held or stop_limit


def test_abort_vs_continue() -> dict:
    limited = _FakePlan(cartesian_limited=True, held=False)
    held = _FakePlan(cartesian_limited=True, held=True)
    ok = (
        _chunk_stop("abort_rest", True, limited)
        and not _chunk_stop("continue_chunk", True, limited)
        and not _chunk_stop("psi_blend", True, limited)
        and not _chunk_stop("split_jump", True, limited)
        and not _chunk_stop("dls", True, limited)
        and _chunk_stop("continue_chunk", True, held)
        and not _chunk_stop("continue_chunk", False, held)
    )
    return {"name": "abort_vs_continue", "ok": ok}


def main() -> int:
    results = [
        test_parse_defaults(),
        test_aliases(),
        test_solver_aliases(),
        test_traj_aliases(),
        test_reject_bad(),
        test_client_yamls(),
        test_solver_preset_yamls(),
        test_traj_preset_yamls(),
        test_wrap_helpers(),
        test_abort_vs_continue(),
    ]
    failed = [r for r in results if not r["ok"]]
    for r in results:
        mark = "OK" if r["ok"] else "FAIL"
        print(f"[{mark}] {r['name']} { {k: v for k, v in r.items() if k not in ('name', 'ok')} }")
    if failed:
        print(f"failed {len(failed)}/{len(results)}")
        return 1
    print(f"ok {len(results)}/{len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
