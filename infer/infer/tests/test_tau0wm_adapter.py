"""τ0-WM 独立 policy.type。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from infer.config_loader import PolicyConfig, _parse_policy_type, load_infer_config
from infer.core.types import PolicyType
from infer.policies.tau0wm_adapter import Tau0WmAdapter


def test_policy_type_tau0wm_aliases() -> None:
    assert _parse_policy_type("tau0wm") is PolicyType.TAU0WM
    assert _parse_policy_type("tau0") is PolicyType.TAU0WM
    assert _parse_policy_type("tau-0-wm") is PolicyType.TAU0WM
    assert _parse_policy_type("tau_0_wm") is PolicyType.TAU0WM


def test_tau0wm_exec_horizon_follows_action_step() -> None:
    cfg = PolicyConfig(
        type=PolicyType.TAU0WM,
        ckpt_dir=Path("/tmp"),
        task_prompt="test",
        n_action_steps_override=8,
        lingbot_port=1244,
    )
    adapter = Tau0WmAdapter(cfg)
    assert adapter.name == "tau0wm"
    assert adapter._pin_exec_horizon is None
    assert adapter._default_horizon == 33
    assert adapter._expected_model == "tau0wm"
    assert adapter._default_port == 1244
    assert adapter._resolve_exec_steps(4) == 4
    assert adapter._resolve_exec_steps(33) == 33
    assert adapter.action_space == "ee"
    assert adapter.control == "ik"
    assert adapter.action_space_source == "tau0wm_solve_ik"


def test_tau0wm_adapter_uses_client_ik_mixin() -> None:
    from infer.policies.client_ee_ik import ClientAbsEeIkMixin

    assert issubclass(Tau0WmAdapter, ClientAbsEeIkMixin)


def test_client_ee_ik_post_action_is_joint_chunk() -> None:
    """solve_ik 成功后 mixin 不得把整段 chunk 丢进 expand_action_to_15d。"""
    import numpy as np

    cfg = PolicyConfig(
        type=PolicyType.TAU0WM,
        ckpt_dir=Path("/tmp"),
        task_prompt="test",
        n_action_steps_override=8,
        lingbot_port=1244,
    )
    adapter = Tau0WmAdapter(cfg)
    joints = np.linspace(0, 1, 8 * 16, dtype=np.float32).reshape(8, 16)

    class _Rt:
        last_server_timing = {"ik_ms": 5.2}
        last_used_rtc = False

        def build_observation(self, images, state):
            return {"obs": images, "state": state}

        def infer(self, obs):
            return np.ones((8, 20), dtype=np.float32), 0.31

        def solve_ik(self, raw, state, n_exec=None, ik=None):
            info = {
                "n_planned": 8,
                "n_requested": 8,
                "aborted_ik": False,
                "cartesian_limited": False,
                "held_ik": False,
                "max_step_deg": 32.98,
                "rejected_step_deg": 0.0,
                "cartesian_t": 1.0,
                "ik_mode": "dls",
                "rpc_ms": 5.2,
            }
            return joints.copy(), info

    adapter._runtime = _Rt()
    adapter._n_action_steps = 8
    state = np.zeros(16, dtype=np.float32)
    timed = adapter.run_timed_inference(
        {"va.images": [np.zeros((64, 64, 3), dtype=np.uint8)] * 3, "va.state": state}
    )
    assert timed.post_action.shape == (8, 16)
    assert np.allclose(timed.post_action, joints)
    assert timed.raw_action.shape == (8, 20)
    assert timed.ik["n_planned"] == 8
    full = adapter.expand_action_to_15d(timed.post_action[0], state)
    assert full.shape[0] >= 16
    # 11:05 臂不动：mixin 曾把整段 chunk 丢进 expand 且漏了 state
    chunk_out = adapter.expand_action_to_15d(timed.post_action)
    assert chunk_out.shape == (8, 16)


def test_ik_abs_ee20_chunk_holds_seed_when_empty() -> None:
    import numpy as np

    from infer.policies.action_space import ik_abs_ee20_chunk

    class Dummy:
        def ik_abs_chunk(self, seed, ee, n_exec=None):
            return []

    seed = np.arange(16, dtype=np.float64)
    ee = np.zeros((8, 20), dtype=np.float64)
    out, info = ik_abs_ee20_chunk(Dummy(), ee, seed, n_exec=8)
    assert out.shape == (1, 16)
    assert np.allclose(out[0], seed)
    assert info["n_planned"] == 0
    assert info["aborted_ik"] is True


def test_ik_abs_ee20_chunk_rejects_wrong_dim() -> None:
    import numpy as np

    from infer.policies.action_space import ik_abs_ee20_chunk

    class Dummy:
        def ik_abs_chunk(self, *a, **k):
            return []

    try:
        ik_abs_ee20_chunk(Dummy(), np.zeros((8, 16)), np.zeros(16))
    except ValueError as exc:
        assert "EE20" in str(exc)
        return
    raise AssertionError("expected ValueError for 16D input")


def test_tau0wm_clamps_n_exec_to_model_chunk() -> None:
    cfg = PolicyConfig(
        type=PolicyType.TAU0WM,
        ckpt_dir=Path("/tmp"),
        task_prompt="test",
        n_action_steps_override=8,
        lingbot_port=1244,
    )
    adapter = Tau0WmAdapter(cfg)
    adapter._n_action_steps = 8
    adapter._model_chunk_size = 16
    adapter._chunk_size = 16
    adapter._runtime = SimpleNamespace(n_exec=None)
    adapter.set_n_action_steps(32)
    assert adapter._n_action_steps == 16
    assert adapter._runtime.n_exec == 16


def test_create_policy_adapter_tau0wm() -> None:
    from infer.policies import _ADAPTER_MODULES, _POLICY_REGISTRY, create_policy_adapter

    assert PolicyType.TAU0WM in _ADAPTER_MODULES
    cfg = PolicyConfig(
        type=PolicyType.TAU0WM,
        ckpt_dir=Path("/tmp"),
        task_prompt="test",
        n_action_steps_override=8,
        lingbot_port=1244,
    )
    adapter = create_policy_adapter(SimpleNamespace(policy=cfg))
    assert isinstance(adapter, Tau0WmAdapter)
    assert _POLICY_REGISTRY[PolicyType.TAU0WM] is Tau0WmAdapter


def test_load_tau0wm_yaml_policy_fields() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "lingbot"
        / "tau0wm_tongbot_ee_00036_local.yaml"
    )
    cfg = load_infer_config(path)
    assert cfg.policy.type is PolicyType.TAU0WM
    assert cfg.policy.lingbot_port == 1244
    assert cfg.policy.lingbot_expected_model == "tau0wm"
    assert cfg.policy.n_action_steps_override == 8
    assert cfg.policy.keep_dims == list(range(16))
    assert cfg.policy.va_obs_fps == 15.0
    assert cfg.policy.camera_keys == [
        "observation.images.front",
        "observation.images.left",
        "observation.images.right",
    ]


def test_load_tau0wm_00073_dls_after_continue() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "lingbot"
        / "tau0wm_tongbot_ee_00073_s240000_local.yaml"
    )
    cfg = load_infer_config(path)
    assert cfg.policy.type is PolicyType.TAU0WM
    assert cfg.policy.n_action_steps_override == 8
    assert cfg.policy.ik is not None
    assert cfg.policy.ik["mode"] == "dls"
    assert cfg.policy.ik["solver"] == "realman"
    assert cfg.policy.ik["traj_opt"] == "none"
    assert cfg.policy.ik["max_joint_step_deg"] == 40.0
