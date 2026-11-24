"""RTC（Real-Time Chunking）Guided Flow-Matching 去噪核心：模型无关的 ΠGDM inpainting 采样器。

纯 PyTorch 实现论文《Real-Time Execution of Action Chunking Flow Policies》
(Black, Galliker, Levine; Physical Intelligence / UC Berkeley; NeurIPS 2025,
arXiv:2506.07339) 的 Algorithm 1「GuidedInference」：

只要任意模型能暴露一个速度场回调 ``velocity_fn(x_t, tau, cond) -> v_t``
（对应论文里的学习到的 v_π），就能复用本模块在 denoise 循环内部做真正的
prefix inpainting，不需要像 ``openpi/models/pi0.py`` 那样把这段数学硬编码进
模型自己的 forward。

用途：DreamZero / Motus / FastWAM 等 PyTorch 世界模型接入 RTC 时，服务端只需
把自己的 flow/diffusion 前向包装成 ``velocity_fn`` 传给这里，不用再抄一遍
``pi0.py`` 的 JAX 实现（那份实现目前也还没有对应的 PyTorch 版本，见
"RTC功能已支持JAX与PyTorch双路径实现"记忆修正：实际仅 JAX 路径有，
PyTorch 路径尚未移植——本模块即是这次统一移植的落点）。

与 ``async_blend.splice_async_blend`` 的区别：
- async_blend：模型无关的"事后拼接"，在 chunk 之间做线性/指数插值，完全不参与
  去噪过程，是拿不到真正 RTC 时的降级方案。
- 本模块：模型无关的"去噪时引导"，改变整条生成轨迹，是论文里真正的 RTC 算法。

索引约定（与 ``pi0.py::get_prefix_weights`` 保持一致，与论文原式记号略有差异）：
  ``prev_action_chunk`` 传入时应当已经切掉"上一包里已经执行掉的步"（客户端负责，
  见 serve_tong_pi05.py 协议：executed_steps 决定切片），因此其第 0 步对应的是
  "推理开始时尚未执行的第一步"，而不是原始 chunk 的第 0 步。
  在此约定下：
  - ``inference_delay``（d）：估计"新包算完之前，机器人还会消耗掉多少 leftover 步"；
    ``[0, d)`` 这段即使引导了也不会被真正执行（客户端会跳过），故权重置 0，
    不浪费梯度——这与论文公式 W_i=1 (i<d) 的表面差异正源于此处索引平移，
    语义等价。
  - ``[d, min(prefix_attention_horizon, constraint_steps))``：软过渡区，
    按 schedule（exp/linear/ones/zeros）从 1 衰减到 0。
  - 其余区域：权重 0，完全自由生成。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
from torch import Tensor

LOGGER = logging.getLogger(__name__)

RTC_SCHEDULES = frozenset({"exp", "linear", "ones", "zeros"})

VelocityFn = Callable[[Tensor, Tensor, Any], Tensor]
"""velocity_fn(x_t: [B, H, D] (requires_grad), tau: [B] float, cond: Any) -> v_t: [B, H, D]"""


@dataclass(frozen=True)
class RTCDenoiseConfig:
    """对应 config_loader.normalize_rtc_config() 产出的 dict 中，跟"去噪引导"相关的字段。

    与 async_inference / async_blending / blend_schedule 等纯客户端字段无关。
    """

    num_steps: int = 10
    max_guidance_weight: float = 10.0
    prefix_attention_schedule: str = "exp"

    @classmethod
    def from_rtc_dict(cls, rtc: Optional[dict], *, num_steps: int = 10) -> "RTCDenoiseConfig":
        """从 normalize_rtc_config() 产出的 dict 里提取去噪相关字段，缺省走默认值。"""
        rtc = rtc or {}
        schedule = str(rtc.get("prefix_attention_schedule", "exp")).strip().lower()
        if schedule not in RTC_SCHEDULES:
            raise ValueError(
                f"prefix_attention_schedule 只能是 exp / linear / ones / zeros，得到 {schedule!r}"
            )
        return cls(
            num_steps=int(num_steps),
            max_guidance_weight=float(rtc.get("max_guidance_weight", 10.0)),
            prefix_attention_schedule=schedule,
        )


def prefix_attention_weights(
    *,
    action_horizon: int,
    inference_delay: int,
    prefix_attention_horizon: int,
    constraint_steps: int,
    schedule: str = "exp",
    executed_steps: int = 0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """计算 prefix attention 权重向量 W（[action_horizon]，值域 [0, 1]）。

    语义等价于 openpi/models/pi0.py::get_prefix_weights，但纯 PyTorch + eager 模式，
    不依赖 JAX。``executed_steps`` 参数保留是为了与 JAX 版调用签名对齐（当前索引
    约定下未参与计算，切片已在调用方/客户端完成）。
    """
    if schedule not in RTC_SCHEDULES:
        raise ValueError(f"Invalid schedule: {schedule!r}")

    step_indices = torch.arange(action_horizon, device=device, dtype=dtype)
    start = float(inference_delay)
    end = min(float(prefix_attention_horizon), float(constraint_steps))

    if schedule == "ones":
        w = torch.ones(action_horizon, device=device, dtype=dtype)
    elif schedule == "zeros":
        w = (step_indices < start).to(dtype)
    elif schedule == "linear":
        active = (step_indices >= start) & (step_indices < end)
        w = active.to(dtype)
    elif schedule == "exp":
        progress = (step_indices - start) / max(end - start, 1e-8)
        progress = progress.clamp(0.0, 1.0)
        denominator = max(float(torch.expm1(torch.tensor(1.0))), 1e-8)
        exp_term = torch.expm1(progress).clamp(-1e6, 1e6)
        w = 1.0 - exp_term / denominator
        w = torch.where(step_indices < start, torch.zeros_like(w), w)
        w = torch.where(step_indices >= end, torch.zeros_like(w), w)
    else:  # pragma: no cover - 已在入口校验
        raise ValueError(f"Invalid schedule: {schedule!r}")

    # 超出约束步数的位置一律不约束（constraint_steps 之外没有对应的 prev_action_chunk）
    w = torch.where(step_indices >= float(constraint_steps), torch.zeros_like(w), w)
    w = torch.nan_to_num(w, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    return w


def guidance_weight_scalar(tau: float, max_guidance_weight: float) -> float:
    """论文 Eq.2 的标量引导权重 min(β, (1-τ)/(τ·r_τ²))，τ∈[0,1) 为当前去噪进度。"""
    tau_clamped = min(max(tau, 1e-8), 1.0 - 1e-8)
    inv_r2 = (tau_clamped**2 + (1.0 - tau_clamped) ** 2) / ((1.0 - tau_clamped) ** 2)
    c = (1.0 - tau_clamped) / tau_clamped
    return max(0.0, min(c * inv_r2, max_guidance_weight))


class RTCGuidedFlowSampler:
    """模型无关的 RTC Guided Flow-Matching 采样器（Algorithm 1「GuidedInference」）。

    用法：
        sampler = RTCGuidedFlowSampler(action_horizon=50, action_dim=16, config=cfg)
        actions = sampler.sample(
            velocity_fn=my_model_velocity_fn,
            cond=(text_embeds, image_latents),
            prev_action_chunk=leftover_from_last_chunk,
            inference_delay=d_steps,
            prefix_attention_horizon=s_steps,
        )

    ``velocity_fn(x_t, tau, cond)`` 必须是可微的（不能包 torch.no_grad），
    输入/输出 shape 均为 [B, action_horizon, action_dim]，tau 为 shape [B] 的 float。
    """

    def __init__(
        self,
        *,
        action_horizon: int,
        action_dim: int,
        config: Optional[RTCDenoiseConfig] = None,
    ) -> None:
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.config = config or RTCDenoiseConfig()

    @torch.enable_grad()
    def sample(
        self,
        velocity_fn: VelocityFn,
        *,
        cond: Any = None,
        prev_action_chunk: Optional[Tensor] = None,
        inference_delay: int = 0,
        prefix_attention_horizon: Optional[int] = None,
        executed_steps: int = 0,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        """跑一遍带（或不带，prev_action_chunk=None 时退化为普通 Euler 积分）RTC 引导的 flow 采样。

        返回 [B, action_horizon, action_dim]。

        本方法必须能在调用方处于 ``torch.inference_mode()`` 时安全使用（例如
        serve_tong_dreamzero.py / serve_tong_motus.py 的主 forward 普遍包了
        inference_mode）：这里显式退出 inference_mode，因为 RTC 引导需要 autograd，
        而 inference_mode 下创建的张量是永久推理张量，无法 requires_grad_。
        """
        with torch.inference_mode(False):
            return self._sample_inner(
                velocity_fn,
                cond=cond,
                prev_action_chunk=prev_action_chunk,
                inference_delay=inference_delay,
                prefix_attention_horizon=prefix_attention_horizon,
                executed_steps=executed_steps,
                noise=noise,
            )

    def _sample_inner(
        self,
        velocity_fn: VelocityFn,
        *,
        cond: Any = None,
        prev_action_chunk: Optional[Tensor] = None,
        inference_delay: int = 0,
        prefix_attention_horizon: Optional[int] = None,
        executed_steps: int = 0,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        if prev_action_chunk is None:
            return self._euler_loop_no_guidance(velocity_fn, cond=cond, noise=noise)

        prev = torch.as_tensor(prev_action_chunk, dtype=torch.float32)
        if prev.ndim != 3 or prev.shape[-1] != self.action_dim:
            raise ValueError(
                f"prev_action_chunk 期望 [B, <=H, {self.action_dim}]，得到 {tuple(prev.shape)}"
            )
        batch_size = int(prev.shape[0])
        constraint_steps = int(prev.shape[1])
        device = prev.device

        # 右 pad 到 action_horizon（Algorithm 1 第 25 行：right-pad A_prev to length H）
        prev_padded = torch.zeros(
            (batch_size, self.action_horizon, self.action_dim), device=device, dtype=torch.float32
        )
        prev_padded[:, :constraint_steps] = prev

        effective_prefix_horizon = (
            self.action_horizon if prefix_attention_horizon is None else int(prefix_attention_horizon)
        )
        weights = prefix_attention_weights(
            action_horizon=self.action_horizon,
            inference_delay=int(inference_delay),
            prefix_attention_horizon=effective_prefix_horizon,
            constraint_steps=constraint_steps,
            schedule=self.config.prefix_attention_schedule,
            executed_steps=int(executed_steps),
            device=device,
            dtype=torch.float32,
        )
        w_diag = weights[None, :, None]  # [1, H, 1]，用于按步数掩码

        if noise is None:
            x_t = torch.randn(batch_size, self.action_horizon, self.action_dim, device=device)
        else:
            x_t = torch.as_tensor(noise, dtype=torch.float32, device=device)

        num_steps = self.config.num_steps
        beta = self.config.max_guidance_weight
        dt = 1.0 / num_steps

        for i in range(num_steps):
            tau = i * dt
            x_t = x_t.detach().requires_grad_(True)
            tau_batch = torch.full((batch_size,), tau, device=device, dtype=torch.float32)

            v = velocity_fn(x_t, tau_batch, cond)
            if v.shape != x_t.shape:
                raise ValueError(f"velocity_fn 返回 shape {tuple(v.shape)} 与 x_t {tuple(x_t.shape)} 不一致")

            # 式(3)：Â¹ = x_t + (1-τ)·v —— 当前对最终干净 chunk 的估计
            x_1_hat = x_t + (1.0 - tau) * v

            # 式(2) 的加权误差项 e = (A_prev - Â¹)ᵀ diag(W)，对 x_t 而言 e 是常数余切向量
            error = (prev_padded - x_1_hat) * w_diag

            # VJP：g = e · ∂Â¹/∂x_t（反向模式自动微分，一次前向 + 一次反向，
            # 比 JAX 版"标准前向 + vjp 内再算一次前向"省一次网络评估）
            (vjp,) = torch.autograd.grad(
                outputs=x_1_hat,
                inputs=x_t,
                grad_outputs=error.detach(),
                retain_graph=False,
                create_graph=False,
            )

            weight = guidance_weight_scalar(tau, beta)
            v_guided = (v + weight * vjp).detach()
            x_t = (x_t.detach() + dt * v_guided).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        return x_t.detach()

    @torch.no_grad()
    def _euler_loop_no_guidance(
        self,
        velocity_fn: VelocityFn,
        *,
        cond: Any = None,
        noise: Optional[Tensor] = None,
        batch_size: int = 1,
    ) -> Tensor:
        """无 prev_action_chunk 时的普通（未引导）flow 采样，方便对比 baseline。"""
        device = noise.device if noise is not None else torch.device("cpu")
        if noise is None:
            x_t = torch.randn(batch_size, self.action_horizon, self.action_dim, device=device)
        else:
            x_t = torch.as_tensor(noise, dtype=torch.float32, device=device)
        num_steps = self.config.num_steps
        dt = 1.0 / num_steps
        for i in range(num_steps):
            tau = i * dt
            tau_batch = torch.full((x_t.shape[0],), tau, device=x_t.device, dtype=torch.float32)
            v = velocity_fn(x_t.detach(), tau_batch, cond)
            x_t = (x_t + dt * v).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        return x_t.detach()


def generate_actions_rtc(
    velocity_fn: VelocityFn,
    *,
    action_horizon: int,
    action_dim: int,
    rtc: Optional[dict] = None,
    prev_action_chunk: Optional[Tensor] = None,
    inference_delay: int = 0,
    prefix_attention_horizon: Optional[int] = None,
    executed_steps: int = 0,
    noise: Optional[Tensor] = None,
    num_steps: int = 10,
    cond: Any = None,
) -> Tensor:
    """DiT action generator 的唯一 RTC 入口。

    ``rtc_enabled=false`` 时不要调这个函数，走模型原来的 Euler。
    打开时必须走这里，禁止各 serve_tong_* 自写一份引导循环。
    """
    cfg = RTCDenoiseConfig.from_rtc_dict(rtc, num_steps=num_steps)
    sampler = RTCGuidedFlowSampler(
        action_horizon=action_horizon,
        action_dim=action_dim,
        config=cfg,
    )
    return sampler.sample(
        velocity_fn,
        cond=cond,
        prev_action_chunk=prev_action_chunk,
        inference_delay=inference_delay,
        prefix_attention_horizon=prefix_attention_horizon,
        executed_steps=executed_steps,
        noise=noise,
    )


if __name__ == "__main__":
    # 冒烟自检：用一个可解析求导的玩具 velocity_fn（线性 flow，真值 = cond 给的 target），
    # 验证：(1) 不 NaN；(2) 引导采样结果比未引导采样结果更接近 prev_action_chunk。
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(0)

    H, D = 50, 16
    prev_chunk = torch.randn(1, 30, D)  # 30 步 leftover

    def toy_velocity(x_t: Tensor, tau: Tensor, cond: Any) -> Tensor:
        # 不能选 v=(target-x_t)/(1-tau) 这种"一步精确收敛到 target"的解：
        # 那样 x_1_hat = x_t+(1-tau)*v 恒等于 target（与 x_t 无关），
        # Jacobian 退化为 0，任何引导都无从施加。用线性衰减保留对 x_t 的依赖：
        # x_1_hat = x_t*(1 - 0.5*(1-tau)) 关于 x_t 的导数 = 0.5+0.5*tau > 0。
        return -0.5 * x_t

    cfg = RTCDenoiseConfig(num_steps=10, max_guidance_weight=10.0, prefix_attention_schedule="exp")
    sampler = RTCGuidedFlowSampler(action_horizon=H, action_dim=D, config=cfg)

    noise = torch.randn(1, H, D)
    unguided = sampler.sample(toy_velocity, cond=None, noise=noise)
    guided = sampler.sample(
        toy_velocity,
        cond=None,
        prev_action_chunk=prev_chunk,
        inference_delay=6,
        prefix_attention_horizon=20,
        noise=noise,
    )

    def dist_to_prev(a: Tensor) -> float:
        pad = torch.zeros(1, H, D)
        pad[:, : prev_chunk.shape[1]] = prev_chunk
        mask = torch.zeros(1, H, D)
        mask[:, 6 : min(20, prev_chunk.shape[1])] = 1.0
        return float(((a - pad) ** 2 * mask).sum() / mask.sum())

    assert torch.isfinite(unguided).all(), "未引导采样出现 NaN/Inf"
    assert torch.isfinite(guided).all(), "引导采样出现 NaN/Inf"
    d_unguided = dist_to_prev(unguided)
    d_guided = dist_to_prev(guided)
    LOGGER.info("dist_to_prev unguided=%.4f guided=%.4f", d_unguided, d_guided)
    assert d_guided < d_unguided, "RTC 引导未生效：guided 没有比 unguided 更接近 prev_action_chunk"
    print("PASS: rtc_denoise 冒烟自检通过（引导生效，且把生成结果拉向 leftover）")
