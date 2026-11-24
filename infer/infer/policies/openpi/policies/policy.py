from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        self._sample_actions_with_rtc = None
        self._has_rtc_support = False
        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
            rtc_fn = getattr(model, "sample_actions_with_rtc_guidance", None)
            if callable(rtc_fn):
                # PyTorch 路径 RTC（PI0Pytorch 直译自 JAX 实现，VJP 用 torch.autograd.grad）
                self._sample_actions_with_rtc = rtc_fn
                self._has_rtc_support = True
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)
            rtc_fn = getattr(model, "sample_actions_with_rtc_guidance", None)
            if callable(rtc_fn):
                try:
                    self._sample_actions_with_rtc = nnx_utils.module_jit(rtc_fn)
                    self._has_rtc_support = True
                except Exception:
                    logging.getLogger(__name__).warning(
                        "OpenPI RTC JIT 失败，rtc_enabled 将不可用",
                        exc_info=True,
                    )

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        if getattr(self, "_debug_attention", False):
            try:
                self._last_attention = self._model.modality_attention(observation)
            except Exception:
                logging.getLogger(__name__).exception("modality attention failed")
                self._last_attention = {"ok": False, "reason": "exception"}
            self._debug_attention = False
        start_time = time.monotonic()
        outputs = {
            "state_original": inputs["state_original"] if inputs.get("state_original") is not None else inputs["state"],
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        self._last_forward = getattr(self._model, "_last_forward", None)
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def infer_with_rtc_guidance(
        self, 
        obs: dict, 
        prev_action_chunk: np.ndarray,
        inference_delay: int,
        prefix_attention_horizon: int,
        prefix_attention_schedule: str = "exp",
        max_guidance_weight: float = 1.0,
        delta_action_mask: np.ndarray | None = None,
        executed_steps: int = 0,
        debug: bool = False,
        *,
        algorithm: str = "pi05",
        rtc_ramp_rate: float = 1.0,
        rtc_overlap_steps: int | None = None,
        rtc_frozen_steps: int | None = None,
    ) -> dict:
        """Infer with real-time chunking guidance.
        
        Note: Parameter order matches model.sample_actions_with_rtc_guidance exactly
        for consistent interface across websocket -> policy -> model layers.

        ``algorithm="pi05"`` 走前缀引导。``algorithm="gr00t"`` 走噪声前缀替换和速度掩码。
        """
        if not self._has_rtc_support:
            raise ValueError("Model does not support RTC guidance")

        algorithm = str(algorithm or "pi05").strip().lower()
        if algorithm == "gr00t" and not self._is_pytorch_model:
            logging.getLogger(__name__).warning(
                "GR00T RTC 只在 PyTorch 权重上实现，本次改走 π0.5 前缀引导"
            )
            algorithm = "pi05"
            
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)

        if self._is_pytorch_model:
            # PyTorch 权重路径：observation/prev_chunk 转 torch 后走 PyTorch 版 RTC
            return self._infer_with_rtc_guidance_pytorch(
                obs,
                inputs,
                prev_action_chunk,
                inference_delay,
                prefix_attention_horizon,
                prefix_attention_schedule,
                max_guidance_weight,
                delta_action_mask,
                executed_steps,
                debug,
                algorithm=algorithm,
                rtc_ramp_rate=rtc_ramp_rate,
                rtc_overlap_steps=rtc_overlap_steps,
                rtc_frozen_steps=rtc_frozen_steps,
            )

        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        
        # 重要：prev_action_chunk需要进行标准化处理
        # prev_action_chunk是上一次推理的输出（已经过output_transform反标准化）
        # 我们需要将其重新标准化到模型内部使用的空间
        
        # RTC数据预处理留在 numpy。JAX 模型到这里才 jnp.asarray，避免和 PyTorch 路径共用时抢 GPU。
        prev_chunk_np, delta_action_mask = self._preprocess_prev_action_for_rtc(
            prev_action_chunk, obs, executed_steps, delta_action_mask, debug=debug
        )
        prev_chunk_jax = jnp.asarray(prev_chunk_np)

        start_time = time.monotonic()
        self._rng, sample_rng = jax.random.split(self._rng)
        
        if debug:
            print(
                f"RTC Debug: inference_delay={inference_delay}, "
                f"prefix_attention_horizon={prefix_attention_horizon}"
            )
            print(
                f"RTC Debug: executed_steps={executed_steps}, "
                f"max_guidance_weight={max_guidance_weight}"
            )
            print(f"RTC Debug: prev_chunk_shape={prev_chunk_jax.shape}")
        
        try:
            # 确保参数顺序与模型函数签名完全一致
            actions = self._sample_actions_with_rtc(
                sample_rng, 
                _model.Observation.from_dict(inputs),
                prev_chunk_jax,
                inference_delay,
                prefix_attention_horizon,
                prefix_attention_schedule,
                max_guidance_weight,
                jnp.asarray(delta_action_mask) if delta_action_mask is not None else None,
                executed_steps,
                **self._sample_kwargs
            )
            
            # 检查输出是否包含NaN
            if jnp.any(jnp.isnan(actions)):
                logging.getLogger(__name__).warning(
                    "RTC 输出含 NaN，回退标准推理"
                )
                # 作为fallback，使用标准推理
                actions = self._sample_actions(sample_rng, _model.Observation.from_dict(inputs), **self._sample_kwargs)
            
            outputs = {
                "state_original": inputs["state_original"] if inputs.get("state_original") is not None else inputs["state"],
                "state": inputs["state"],
                "actions": actions,
            }
            
            if debug:
                print(
                    f"RTC Debug: actions_shape={actions.shape}, "
                    f"contains_nan={jnp.any(jnp.isnan(actions))}"
                )
            
        except Exception as e:
            logging.getLogger(__name__).warning(
                "RTC Error: %s, falling back to standard inference", e
            )
            # Fallback to standard inference
            outputs = {
                "state_original": inputs["state_original"] if inputs.get("state_original") is not None else inputs["state"],
                "state": inputs["state"],
                "actions": self._sample_actions(sample_rng, _model.Observation.from_dict(inputs), **self._sample_kwargs),
            }
        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        model_time = time.monotonic() - start_time

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def _model_action_dim(self) -> int:
        """JAX 模型为顶层属性，PyTorch(PI0Pytorch) 为 config 下属性。"""
        if hasattr(self._model, "action_dim"):
            return int(self._model.action_dim)
        return int(self._model.config.action_dim)

    def _model_action_horizon(self) -> int:
        """JAX 模型为顶层属性，PyTorch(PI0Pytorch) 为 config 下属性。"""
        if hasattr(self._model, "action_horizon"):
            return int(self._model.action_horizon)
        return int(self._model.config.action_horizon)

    def _infer_with_rtc_guidance_pytorch(
        self,
        obs: dict,
        inputs: dict,
        prev_action_chunk: np.ndarray,
        inference_delay: int,
        prefix_attention_horizon: int,
        prefix_attention_schedule: str,
        max_guidance_weight: float,
        delta_action_mask: np.ndarray | None,
        executed_steps: int,
        debug: bool,
        *,
        algorithm: str = "pi05",
        rtc_ramp_rate: float = 1.0,
        rtc_overlap_steps: int | None = None,
        rtc_frozen_steps: int | None = None,
    ) -> dict:
        """PyTorch 权重路径的 RTC 推理（与 JAX 版逐段对应）。

        inputs 为已完成 input_transform 的 numpy 域观测；此处转 torch batch。
        prev_action_chunk 预处理只走 numpy，不调用 jnp，避免 JAX 在 GPU 上预占显存。
        输出转 torch 后调模型 PyTorch 版 sample_actions_with_rtc_guidance。
        """
        device = self._pytorch_device

        def to_torch_batch(tree):
            if isinstance(tree, dict):
                return {key: to_torch_batch(value) for key, value in tree.items()}
            # numpy.bool_ 不是 ndarray。普通推理用 np.array 收成 batch，RTC 这里原来把 True 原样留下。
            if isinstance(tree, np.ndarray):
                array = np.array(tree, copy=True)
            elif isinstance(tree, (np.generic, bool, int, float)):
                array = np.asarray(tree)
            else:
                return tree
            return torch.from_numpy(array).to(device)[None, ...]

        inputs = to_torch_batch(inputs)

        if debug:
            print(
                f"RTC Debug (pytorch): algorithm={algorithm}, "
                f"inference_delay={inference_delay}, "
                f"prefix_attention_horizon={prefix_attention_horizon}, "
                f"executed_steps={executed_steps}, "
                f"max_guidance_weight={max_guidance_weight}, "
                f"rtc_ramp_rate={rtc_ramp_rate}, "
                f"rtc_overlap_steps={rtc_overlap_steps}, "
                f"rtc_frozen_steps={rtc_frozen_steps}"
            )

        prev_chunk_np, delta_mask_np = self._preprocess_prev_action_for_rtc(
            prev_action_chunk, obs, executed_steps, delta_action_mask, debug=debug
        )
        prev_chunk_torch = torch.tensor(
            np.array(prev_chunk_np, dtype=np.float32, copy=True),
            dtype=torch.float32,
            device=device,
        )
        mask_torch = (
            torch.as_tensor(np.asarray(delta_mask_np), dtype=torch.bool, device=device)
            if delta_mask_np is not None
            else None
        )

        observation = _model.Observation.from_dict(inputs)
        state_orig = (
            inputs["state_original"]
            if inputs.get("state_original") is not None
            else inputs["state"]
        )

        start_time = time.monotonic()
        try:
            if algorithm == "gr00t":
                actions = self._model.sample_actions_with_gr00t_rtc(
                    device,
                    observation,
                    prev_chunk_torch,
                    inference_delay,
                    prefix_attention_horizon,
                    ramp_rate=rtc_ramp_rate,
                    overlap_steps=rtc_overlap_steps,
                    frozen_steps=rtc_frozen_steps,
                    **self._sample_kwargs,
                )
            else:
                actions = self._sample_actions_with_rtc(
                    device,
                    observation,
                    prev_chunk_torch,
                    inference_delay,
                    prefix_attention_horizon,
                    prefix_attention_schedule,
                    max_guidance_weight,
                    mask_torch,
                    executed_steps,
                    **self._sample_kwargs,
                )
            if bool(torch.isnan(actions).any()):
                logging.getLogger(__name__).warning("RTC 输出含 NaN，回退标准推理")
                actions = self._sample_actions(
                    device, observation, **self._sample_kwargs
                )
            outputs = {
                "state_original": state_orig,
                "state": inputs["state"],
                "actions": actions,
            }
        except Exception as e:
            logging.getLogger(__name__).warning(
                "RTC Error: %s, falling back to standard inference", e
            )
            outputs = {
                "state_original": state_orig,
                "state": inputs["state"],
                "actions": self._sample_actions(
                    device, observation, **self._sample_kwargs
                ),
            }

        self._last_forward = getattr(self._model, "_last_forward", None)
        outputs = {
            key: np.asarray(value[0].detach().cpu()) if isinstance(value, torch.Tensor) else value
            for key, value in outputs.items()
        }
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": (time.monotonic() - start_time) * 1000,
        }
        return outputs

    def _pad_rtc_chunk_to_horizon(self, chunk_btd: np.ndarray) -> np.ndarray:
        """Pad leftover [1, remaining, D] on the right to [1, action_horizon, D]. Numpy only."""
        chunk_btd = np.asarray(chunk_btd)
        horizon = self._model_action_horizon()
        remaining = int(chunk_btd.shape[1])
        dim = int(chunk_btd.shape[-1])
        if remaining == horizon:
            return chunk_btd
        if remaining > horizon:
            return chunk_btd[:, :horizon, :]
        pad = np.zeros((1, horizon - remaining, dim), dtype=chunk_btd.dtype)
        return np.concatenate([chunk_btd, pad], axis=1)

    def _preprocess_prev_action_for_rtc(
        self, 
        prev_action_chunk: np.ndarray, 
        obs: dict, 
        executed_steps: int, 
        delta_action_mask: np.ndarray | None,
        debug: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """预处理 prev_action_chunk。全程 numpy，不把数组放到 JAX 设备上。

        1. 根据 executed_steps 切片取未执行部分
        2. 维度匹配和填充
        3. Delta 处理
        4. 标准化

        Returns:
            (动作数组 [1, action_horizon, action_dim], 掩码)
        """
        def _dprint(*args, **kwargs):
            if debug:
                print(*args, **kwargs)

        chunk_size = prev_action_chunk.shape[0]
        remaining_steps = chunk_size - executed_steps
        
        _dprint(f"RTC Preprocess: chunk_size={chunk_size}, executed_steps={executed_steps}, remaining_steps={remaining_steps}")
        
        # 1. 取未执行的部分：后面的 remaining_steps 对应新推理的前 remaining_steps
        if executed_steps > 0 and remaining_steps > 0:
            prev_action_unexecuted = prev_action_chunk[executed_steps:, :]  # [remaining_steps, action_dim]
            _dprint(f"RTC Preprocess: Sliced prev_action to unexecuted part, shape: {prev_action_unexecuted.shape}")
        elif remaining_steps > 0:
            prev_action_unexecuted = prev_action_chunk  # 没有执行任何步骤
        else:
            # 所有步骤都已执行，返回空数组
            _dprint("RTC Preprocess: All steps executed, returning empty constraint")
            return np.zeros((1, 0, self._model_action_dim()), dtype=np.float32), delta_action_mask
        
        # 2. 维度匹配：从14维扩展到32维
        if prev_action_unexecuted.shape[-1] < self._model_action_dim():
            pad_width = self._model_action_dim() - prev_action_unexecuted.shape[-1]
            prev_action_padded = np.pad(prev_action_unexecuted, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
            _dprint(f"RTC Preprocess: Padded action from {prev_action_unexecuted.shape[-1]} to {prev_action_padded.shape[-1]} dims")
            
            # 同时填充delta_action_mask
            if delta_action_mask is not None:
                if not isinstance(delta_action_mask, np.ndarray):
                    delta_action_mask = np.asarray(delta_action_mask)
                
                if delta_action_mask.shape[-1] < self._model_action_dim():
                    delta_mask_pad_width = self._model_action_dim() - delta_action_mask.shape[-1]
                    delta_action_mask = np.pad(delta_action_mask, (0, delta_mask_pad_width), mode='constant', constant_values=False)
                    _dprint(f"RTC Preprocess: Padded delta_mask from {delta_action_mask.shape[-1] - delta_mask_pad_width} to {delta_action_mask.shape[-1]} dims")
        else:
            prev_action_padded = prev_action_unexecuted
        
        # 3. Delta处理 + 标准化：使用原始状态进行正确的Delta计算
        try:
            # 使用原始状态，但需要确保维度匹配。LIBERO 输入是 observation/state。
            raw_state = obs.get("state", obs.get("observation/state"))
            if raw_state is None:
                raise KeyError("RTC preprocess 需要 state 或 observation/state")
            raw_state = np.asarray(raw_state)
            
            _dprint(f"RTC Delta Debug: === 开始Delta处理 ===")
            _dprint(f"RTC Delta Debug: Raw state shape: {raw_state.shape}")
            _dprint(f"RTC Delta Debug: Raw state values: {raw_state}")
            
            # 确保state也被填充到与actions相同的维度
            if raw_state.shape[-1] < self._model_action_dim():
                state_pad_width = self._model_action_dim() - raw_state.shape[-1]
                raw_state_padded = np.pad(raw_state, (0, state_pad_width), mode='constant', constant_values=0)
                _dprint(f"RTC Delta Debug: Padded state from {raw_state.shape[-1]} to {raw_state_padded.shape[-1]} dims")
            else:
                raw_state_padded = raw_state
            
            _dprint(f"RTC Delta Debug: Padded state shape: {raw_state_padded.shape}")
            _dprint(f"RTC Delta Debug: Prev action BEFORE delta shape: {prev_action_unexecuted.shape}")
            _dprint(f"RTC Delta Debug: === All timesteps BEFORE delta ===")
            for i in range(min(prev_action_unexecuted.shape[0], 20)):  # 显示前20个时间步，或全部如果少于20
                _dprint(f"  Step {i:2d}: {prev_action_unexecuted[i, :8]} ... (showing first 8 dims)")
            if prev_action_unexecuted.shape[0] > 20:
                _dprint(f"  ... (showing first 20 of {prev_action_unexecuted.shape[0]} total steps)")
            
            # 手动展示Delta计算过程（用于调试）
            if debug and delta_action_mask is not None:
                mask = np.asarray(delta_action_mask)
                _dprint(f"RTC Delta Debug: Delta action mask (14-dim): {mask[:14]}")
                _dprint(f"RTC Delta Debug: Dimensions WITH delta: {[i for i, m in enumerate(mask) if m]}")
            
            # 创建包含原始状态和原始动作的完整观测结构，用于delta+normalize处理
            temp_obs_with_actions = dict(obs)
            temp_obs_with_actions["state"] = raw_state
            temp_obs_with_actions["actions"] = prev_action_unexecuted
            temp_obs_with_actions.setdefault("prompt", "null")
            
            # 应用完整的输入变换（DeltaActions + Normalize）
            temp_transformed = self._input_transform(temp_obs_with_actions)
            processed_prev_chunk = np.asarray(temp_transformed["actions"], dtype=np.float32)[np.newaxis, ...]
            processed_prev_chunk = self._pad_rtc_chunk_to_horizon(processed_prev_chunk)
            _dprint(f"RTC Delta Debug: Final shape={processed_prev_chunk.shape}")
            return processed_prev_chunk, delta_action_mask
            
        except Exception as e:
            logging.getLogger(__name__).exception("RTC Preprocess: Transform failed")
            raise e


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
