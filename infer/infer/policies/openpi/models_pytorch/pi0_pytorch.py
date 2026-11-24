import logging
import math
import os

import numpy as np
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def _install_ppu_m1_gemm_pad() -> None:
    """Avoid a ppu0015 libacblas SIGSEGV on GEMMs whose M dimension is 1.

    ``F.linear`` on a CUDA tensor of shape ``(..., 1, K)`` (the π0.5 time MLP
    is ``(1, 1024)``) jumps to a null entry in ``libacblas_internal.so``
    (fault address 0x118, offset 0x291551). The same kernel with M>=2 returns.
    Pad one zero row and drop it. ``nn.Linear`` does not mix rows, so the
    kept row matches the unpadded product. Set ``OPENPI_ACBLAS_PAD_M1=1``.
    """
    flag = os.environ.get("OPENPI_ACBLAS_PAD_M1", "")
    if flag.lower() not in {"1", "true", "yes"}:
        return
    if getattr(F, "_openpi_m1_pad", False):
        return
    orig_linear = F.linear
    orig_matmul = torch.matmul

    def linear(input, weight, bias=None):
        if (
            torch.is_tensor(input)
            and input.is_cuda
            and input.ndim >= 2
            and int(input.shape[-2]) == 1
        ):
            padded = torch.cat((input, torch.zeros_like(input)), dim=-2)
            return orig_linear(padded, weight, bias)[..., :1, :]
        return orig_linear(input, weight, bias)

    def matmul(a, b, *args, **kwargs):
        if torch.is_tensor(a) and a.is_cuda and a.ndim >= 2 and int(a.shape[-2]) == 1:
            padded = torch.cat((a, torch.zeros_like(a)), dim=-2)
            return orig_matmul(padded, b, *args, **kwargs)[..., :1, :]
        return orig_matmul(a, b, *args, **kwargs)

    F.linear = linear
    torch.matmul = matmul
    F._openpi_m1_pad = True


_install_ppu_m1_gemm_pad()


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def _encode_modality_tag_ids(tags: dict) -> dict[str, list[int]]:
    """Paligemma ids for DEPTH/FT labels. Avoid importing tokenizer.py (pulls JAX)."""
    if not tags:
        return {}
    from pathlib import Path

    import sentencepiece

    import openpi.shared.download as download

    env_path = os.environ.get("PALIGEMMA_TOKENIZER_PATH", "")
    path = Path(env_path) if env_path else None
    if path is None or not path.is_file():
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
    sp = sentencepiece.SentencePieceProcessor(model_file=str(path))
    out: dict[str, list[int]] = {}
    for key, text in tags.items():
        cleaned = str(text).strip().replace("_", " ").replace("\n", " ")
        if cleaned:
            ids = list(sp.encode(cleaned, add_bos=False))
            if ids:
                out[str(key)] = ids
    return out


def _image_nchw(img: torch.Tensor) -> torch.Tensor:
    """SigLIP wants [B, C, H, W]. Depth copies from the loader are often HWC."""
    if img.ndim == 4 and img.shape[-1] in (1, 3) and img.shape[1] not in (1, 3):
        return img.permute(0, 3, 1, 2)
    return img


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    att_masks_int = att_masks.to(torch.int32)
    cumsum = torch.cumsum(att_masks_int, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def _modality_mass(mass, spans: list[tuple[str, int, int]]) -> dict:
    """语言/图像块之间的注意力质量，以及语言对第一路图像的粗网格。"""
    import numpy as np

    mass = np.asarray(mass, dtype=np.float64)
    images = [(n, a, b) for n, a, b in spans if n.startswith("image.")]
    langs = [(n, a, b) for n, a, b in spans if n == "language"]
    if not images or not langs or mass.ndim != 2:
        return {"ok": False, "reason": "missing_span", "spans": spans}

    def _block(qs, ks) -> float:
        if qs[1] <= qs[0] or ks[1] <= ks[0]:
            return 0.0
        sl = mass[qs[0] : qs[1], ks[0] : ks[1]]
        return float(sl.mean()) if sl.size else 0.0

    img0 = images[0]
    lang = langs[0]
    side = int(round((img0[2] - img0[1]) ** 0.5))
    grid = mass[lang[0] : lang[1], img0[1] : img0[2]].mean(axis=0)
    attn_map = None
    if side * side == grid.size and side > 0:
        attn_map = np.asarray(grid, dtype=np.float64).reshape(side, side)
        # 压到最多 8×8，只留空间形状
        step = max(1, side // 8)
        attn_map = attn_map[::step, ::step]
    relations = {
        "lang_to_image": _block(lang[1:], img0[1:]),
        "image_to_lang": _block(img0[1:], lang[1:]),
        "lang_to_lang": _block(lang[1:], lang[1:]),
        "image_to_image": _block(img0[1:], img0[1:]),
    }
    per_image = {
        name: _block(lang[1:], (a, b)) for name, a, b in images
    }
    out = {
        "ok": True,
        "image_tokens": int(sum(b - a for _, a, b in images)),
        "lang_tokens": int(lang[2] - lang[0]),
        "relations": {k: round(v, 6) for k, v in relations.items()},
        "lang_to_each_image": {k: round(v, 6) for k, v in per_image.items()},
    }
    if attn_map is not None:
        out["attention_map"] = {
            "camera": img0[0],
            "hw": [int(attn_map.shape[0]), int(attn_map.shape[1])],
            "values": [round(float(x), 5) for x in attn_map.reshape(-1)],
        }
    return out


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        self.force6d_encoder = None
        if getattr(config, "use_force6d_encoder", False):
            from openpi.models_pytorch.force6d_encoder import Force6DEncoder

            self.force6d_encoder = Force6DEncoder(
                output_dim=paligemma_config.width if self.pi05 else action_expert_config.width,
                hidden_dim=getattr(config, "force6d_hidden_dim", 32),
                num_layers=getattr(config, "force6d_num_layers", 2),
            )

        self._modality_prompt_ids: dict[str, torch.Tensor] = {}
        if getattr(config, "use_modality_prompt_tokens", False):
            tags = getattr(config, "modality_prompt_tags", None) or {}
            for key, ids in _encode_modality_tag_ids(tags).items():
                self._modality_prompt_ids[key] = torch.tensor(ids, dtype=torch.int64)

        torch.set_float32_matmul_precision("high")
        # Blackwell (sm_120) 上 torch.compile / inductor 会崩；可用 OPENPI_DISABLE_TORCH_COMPILE=1 强制 eager。
        disable_compile = os.environ.get("OPENPI_DISABLE_TORCH_COMPILE", "").lower() in ("1", "true", "yes")
        if not disable_compile and torch.cuda.is_available():
            try:
                major, _minor = torch.cuda.get_device_capability()
                if major >= 12:
                    disable_compile = True
                    logging.warning("检测到 CUDA capability sm_%s0+，跳过 torch.compile（eager 推理）", major)
            except Exception:
                pass
        if disable_compile:
            logging.info("OpenPI PyTorch sample_actions 使用 eager（未 torch.compile）")
        else:
            self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Keep extra DEPTH image keys so embed_prefix can place them after language."""
        return _preprocessing.preprocess_observation_pytorch(observation, train=train)

    def _append_modality_prompt(self, name, batch_size, device, embs, pad_masks, att_masks):
        ids = self._modality_prompt_ids.get(name)
        if ids is None or ids.numel() == 0:
            return
        ids = ids.to(device=device).unsqueeze(0).expand(batch_size, -1)

        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            return lang_emb * math.sqrt(lang_emb.shape[-1])

        tag_emb = self._apply_checkpoint(lang_embed_func, ids)
        embs.append(tag_emb)
        pad_masks.append(torch.ones(batch_size, tag_emb.shape[1], dtype=torch.bool, device=device))
        att_masks += [0] * tag_emb.shape[1]

    def _embed_named_image(self, img, img_mask, embs, pad_masks, att_masks):
        img = _image_nchw(img)

        def image_embed_func(image):
            return self.paligemma_with_expert.embed_image(image)

        img_emb = self._apply_checkpoint(image_embed_func, img)
        bsize, num_img_embs = img_emb.shape[:2]
        embs.append(img_emb)
        pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
        att_masks += [0] * num_img_embs

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, observation, spans: list | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """RGB (pretrained slots) → Task/State → DEPTH extras through the same SigLIP.

        ``spans`` 若给出，记每段模态的 ``(name, start, end)`` token 区间，供注意力汇总。
        """
        embs = []
        pad_masks = []
        att_masks = []
        cursor = 0

        def _note(name: str, n_tok: int) -> None:
            nonlocal cursor
            if spans is not None and n_tok > 0:
                spans.append((name, cursor, cursor + int(n_tok)))
            cursor += int(n_tok)
        pretrained_rgb = set(_preprocessing.IMAGE_KEYS)
        images = observation.images
        image_masks = observation.image_masks
        lang_tokens = observation.tokenized_prompt
        lang_masks = observation.tokenized_prompt_mask

        for name in _preprocessing.IMAGE_KEYS:
            if name not in images:
                continue
            n_before = sum(t.shape[1] for t in embs) if embs else 0
            self._embed_named_image(images[name], image_masks[name], embs, pad_masks, att_masks)
            _note("image." + name, (sum(t.shape[1] for t in embs) if embs else 0) - n_before)

        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            return lang_emb * math.sqrt(lang_emb.shape[-1])

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]
        _note("language", int(lang_emb.shape[1]))

        extra_names = [name for name in images if name not in pretrained_rgb]
        for name in extra_names:
            img = images[name]
            self._append_modality_prompt(name, img.shape[0], img.device, embs, pad_masks, att_masks)
            self._embed_named_image(img, image_masks[name], embs, pad_masks, att_masks)

        if self.force6d_encoder is not None and getattr(observation, "force6d", None):
            force6d = observation.force6d
            force_masks = getattr(observation, "force6d_masks", None) or {}
            for force_key in force6d:
                vec = force6d[force_key]
                self._append_modality_prompt(
                    f"force6d.{force_key}", vec.shape[0], vec.device, embs, pad_masks, att_masks
                )
                encoded = self.force6d_encoder(vec)
                token = encoded[:, None, :]
                embs.append(token)
                mask = force_masks.get(force_key)
                if mask is None:
                    mask = torch.ones(token.shape[0], dtype=torch.bool, device=token.device)
                pad_masks.append(mask[:, None].expand(token.shape[0], token.shape[1]))
                att_masks += [0] * token.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, att_masks.shape[0])
        return embs, pad_masks, att_masks

    @torch.no_grad()
    def modality_attention(self, observation) -> dict:
        """最后一层：语言/图像 token 互相看的质量，以及语言对第一路图像的粗网格。

        不返回完整注意力张量。
        """
        observation = self._preprocess_observation(observation, train=False)
        spans: list[tuple[str, int, int]] = []
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(observation, spans)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        lang_model = self.paligemma_with_expert.paligemma.language_model
        lang_model.config._attn_implementation = "eager"  # noqa: SLF001
        out = lang_model.forward(
            inputs_embeds=prefix_embs,
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            use_cache=False,
            output_attentions=True,
        )
        attns = getattr(out, "attentions", None)
        if not attns:
            return {"ok": False, "reason": "no_attentions"}
        last = attns[-1][0].float()  # [heads, S, S]
        mass = last.mean(dim=0).detach().cpu()
        del attns, last, out
        return _modality_mass(mass.numpy(), spans)

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        observation = self._preprocess_observation(observation, train=self.training)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(observation)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(observation.state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    def _act_stats(self, x: Tensor) -> dict:
        t = x.detach().float()
        finite = bool(torch.isfinite(t).all().item())
        clean = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        return {
            "min": round(float(clean.min()), 6),
            "max": round(float(clean.max()), 6),
            "mean": round(float(clean.mean()), 6),
            "finite": finite,
        }

    def _begin_forward_trace(self, observation, num_steps: int) -> dict:
        images = {}
        for name, img in observation.images.items():
            images[str(name)] = [int(s) for s in img.shape]
        prompt_tokens = None
        tokens = getattr(observation, "tokenized_prompt", None)
        if tokens is not None:
            prompt_tokens = int(tokens.shape[-1])
        self._forward_trace = {
            "num_steps": int(num_steps),
            "input": {
                "images": images,
                "state": self._act_stats(observation.state),
                "prompt_tokens": prompt_tokens,
            },
            "prefix_len": None,
            "denoise": [],
        }
        self._denoise_keep = {0, max(int(num_steps) // 2, 0), max(int(num_steps) - 1, 0)}
        self._denoise_i = 0
        return self._forward_trace

    def _note_prefix(self, prefix_pad_masks) -> None:
        trace = getattr(self, "_forward_trace", None)
        if trace is not None:
            trace["prefix_len"] = int(prefix_pad_masks.shape[-1])

    def _note_denoise(self, time: Tensor, x_t: Tensor) -> None:
        trace = getattr(self, "_forward_trace", None)
        if trace is None:
            return
        i = int(getattr(self, "_denoise_i", 0))
        if i in getattr(self, "_denoise_keep", set()):
            row = {"i": i, "time": round(float(time.detach()), 4)}
            row.update(self._act_stats(x_t))
            trace["denoise"].append(row)
        self._denoise_i = i + 1

    def _finish_forward_trace(self, x_t: Tensor) -> None:
        trace = getattr(self, "_forward_trace", None)
        if trace is None:
            return
        trace["output"] = {
            "shape": [int(s) for s in x_t.shape],
            "action_dim": int(self.config.action_dim),
            "horizon": int(self.config.action_horizon),
        }
        trace["output"].update(self._act_stats(x_t))
        self._last_forward = trace

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        observation = self._preprocess_observation(observation, train=False)
        self._begin_forward_trace(observation, num_steps)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(observation)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        self._note_prefix(prefix_pad_masks)

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        state = observation.state
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            self._note_denoise(time, x_t)
            time += dt
        self._finish_forward_trace(x_t)
        return x_t

    @torch.no_grad()
    def sample_actions_with_gr00t_rtc(
        self,
        device,
        observation,
        prev_action_chunk,
        inference_delay: int,
        prefix_attention_horizon: int,
        *,
        ramp_rate: float = 1.0,
        overlap_steps: int | None = None,
        frozen_steps: int | None = None,
        num_steps: int = 10,
        noise=None,
    ) -> Tensor:
        """GR00T N1.7 RTC，接到 π0.5 的负 dt 欧拉积分上。

        噪声仍从 N(0, I) 起步。上一包未执行的左前缀写进噪声的同一位置，
        冻结帧速度乘 0，重叠段乘指数斜坡，其余帧速度为 1。
        不做 π0.5 的 VJP 前缀引导。

        ``prev_action_chunk`` 已在 policy 层切掉已执行前缀并右侧补零，
        所以有效动作在左边，长度是 ``prefix_attention_horizon``。
        """
        bsize = observation.state.shape[0]
        horizon = int(self.config.action_horizon)
        if noise is None:
            noise = self.sample_noise(
                (bsize, horizon, self.config.action_dim), device
            )

        if not isinstance(prev_action_chunk, Tensor):
            prev = torch.as_tensor(
                np.asarray(prev_action_chunk), dtype=torch.float32, device=device
            )
        else:
            prev = prev_action_chunk.to(device=device, dtype=torch.float32)
        if prev.ndim == 2:
            prev = prev.unsqueeze(0)

        from infer.policies.gr00t_rtc import (
            gr00t_rtc_vel_strength,
            resolve_gr00t_rtc_window,
        )

        valid, overlap, frozen = resolve_gr00t_rtc_window(
            horizon,
            prefix_attention_horizon,
            int(prev.shape[1]),
            inference_delay,
            overlap_steps,
            frozen_steps,
        )
        self._last_gr00t_rtc = {
            "algorithm": "gr00t",
            "overlap_steps": overlap,
            "frozen_steps": frozen,
            "ramp_rate": float(ramp_rate),
            "prefix_horizon": valid,
        }

        observation = self._preprocess_observation(observation, train=False)
        self._begin_forward_trace(observation, num_steps)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(observation)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        self._note_prefix(prefix_pad_masks)

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise.to(dtype=torch.float32)
        if overlap > 0:
            x_t = x_t.clone()
            x_t[:, :overlap, :] = prev[:, :overlap, :].to(dtype=x_t.dtype)
        strength = torch.as_tensor(
            gr00t_rtc_vel_strength(horizon, frozen, overlap, float(ramp_rate)),
            device=device,
            dtype=x_t.dtype,
        ).view(1, horizon, 1)

        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        state = observation.state
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            x_t = x_t + dt * v_t * strength
            self._note_denoise(time, x_t)
            time += dt
        x_t = torch.nan_to_num(x_t, nan=0.0)
        self._finish_forward_trace(x_t)
        return x_t

    def _rtc_get_prefix_weights(
        self,
        inference_delay: int,
        prefix_attention_horizon: int,
        constraint_steps: int,
        schedule: str,
        executed_steps: int = 0,
    ) -> Tensor:
        """Compute prefix attention weights for real-time chunking (torch port of JAX)."""
        step_indices = torch.arange(self.config.action_horizon, device=self._rtc_device)

        start = int(inference_delay)
        end = min(int(prefix_attention_horizon), int(constraint_steps))

        if schedule == "ones":
            w = torch.ones(self.config.action_horizon, device=self._rtc_device)
        elif schedule == "zeros":
            w = (step_indices < start).to(torch.float32)
        elif schedule == "linear":
            w = torch.where(
                step_indices < start,
                torch.zeros_like(step_indices, dtype=torch.float32),
                torch.where(
                    step_indices >= end,
                    torch.zeros_like(step_indices, dtype=torch.float32),
                    torch.ones_like(step_indices, dtype=torch.float32),
                ),
            )
        elif schedule == "ramp":
            # 官方 linear：从 delay 到 prefix horizon 由 1 线性降到 0
            denom = max(float(end - start), 1.0)
            progress = (step_indices.to(torch.float32) - float(start)) / denom
            w = (1.0 - progress).clamp(0.0, 1.0)
            w = torch.where(step_indices < start, torch.zeros_like(w), w)
            w = torch.where(step_indices >= end, torch.zeros_like(w), w)
        elif schedule == "exp":
            progress = (step_indices.to(torch.float32) - start) / max(end - start, 1e-8)
            progress = progress.clamp(0.0, 1.0)
            denominator = max(math.expm1(1.0), 1e-8)
            exp_term = torch.expm1(progress).clamp(-1e6, 1e6)
            w = 1.0 - exp_term / denominator
            w = torch.where(step_indices < start, torch.zeros_like(w), w)
            w = torch.where(step_indices >= end, torch.zeros_like(w), w)
        else:
            raise ValueError(f"Invalid schedule: {schedule}")

        # 确保超出约束步数的权重为0（与 JAX 版一致）
        w = torch.where(step_indices >= constraint_steps, torch.zeros_like(w), w)

        # 数值稳定性检查（同 JAX nan_to_num 链）
        w = torch.nan_to_num(w, nan=0.0, posinf=1.0, neginf=0.0)
        w = w.clamp(0.0, 1.0)
        return w

    def sample_actions_with_rtc_guidance(
        self,
        device,
        observation,
        prev_action_chunk,
        inference_delay: int,
        prefix_attention_horizon: int,
        prefix_attention_schedule: str = "exp",
        max_guidance_weight: float = 1.0,
        delta_action_mask=None,
        executed_steps: int = 0,
        *,
        num_steps: int = 10,
    ) -> Tensor:
        """Do inference with real-time chunking guidance (torch port of JAX implementation).

        每个去噪步在普通速度场 v_t 之外，用 torch.autograd.grad 计算
        x_1 = x_t - time·v(x_t) 对 x_t 的 VJP（与 jax.vjp 等价），
        把新 chunk 前缀拉向上一 chunk 未执行部分 prev_action_chunk。

        prev_action_chunk 需已在 policy 层完成切片/Delta/标准化，
        形状 [1, action_horizon, action_dim]（右侧零填充）。
        """
        self._rtc_device = device
        bsize = observation.state.shape[0]
        actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
        noise = self.sample_noise(actions_shape, device)

        prev_delta_chunk = prev_action_chunk
        if not isinstance(prev_delta_chunk, Tensor):
            prev_delta_chunk = torch.as_tensor(
                np.asarray(prev_delta_chunk), dtype=torch.float32, device=device
            )
        else:
            prev_delta_chunk = prev_delta_chunk.to(device=device, dtype=torch.float32)

        constraint_steps = int(prev_delta_chunk.shape[1])

        observation = self._preprocess_observation(observation, train=False)
        self._begin_forward_trace(observation, num_steps)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(observation)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        self._note_prefix(prefix_pad_masks)

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        # prefix KV cache 在 no_grad 下计算一次（梯度只流经 suffix 路径，与 JAX 语义一致）
        with torch.no_grad():
            _, past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        state = observation.state
        while time >= -dt / 2 and time <= 1.1:
            expanded_time = time.expand(bsize)

            # 1) 普通速度场（无梯度）
            with torch.no_grad():
                v_t = self.denoise_step(
                    state, prefix_pad_masks, past_key_values, x_t, expanded_time
                )

            # 2) VJP 路径：denoiser_fn(x) = x + v(x)·(-time)，主输出 x_1
            x_t_grad = x_t.detach().clone().requires_grad_(True)
            v_vjp = self.denoise_step(
                state, prefix_pad_masks, past_key_values, x_t_grad, expanded_time
            )
            x_1 = x_t_grad + v_vjp * (-time)

            weights = self._rtc_get_prefix_weights(
                inference_delay,
                prefix_attention_horizon,
                constraint_steps,
                prefix_attention_schedule,
                executed_steps,
            )

            # 只对前 constraint_steps 步计算误差，其余步误差为0（权重掩码）
            with torch.no_grad():
                error_full = prev_delta_chunk - x_1.detach()
                error = error_full * weights[None, :, None]

            # VJP: ∂x_1/∂x_t_grad^T · error —— 与 jax.vjp(denoiser_fn, x_t)[1](error) 等价
            pinv_correction = torch.autograd.grad(
                outputs=x_1,
                inputs=x_t_grad,
                grad_outputs=error,
                retain_graph=False,
            )[0].detach()

            # guidance weight（RTC paper 公式，同 JAX 数值防护链）
            time_clamped = time.clamp(1e-8, 1.0 - 1e-8)
            inv_r2 = (time_clamped**2 + (1.0 - time_clamped) ** 2) / (
                (1.0 - time_clamped) ** 2
            )
            c = (1.0 - time_clamped) / time_clamped
            guidance_weight = (c * inv_r2).clamp(max=max_guidance_weight)
            guidance_weight = torch.nan_to_num(
                guidance_weight, nan=0.0, posinf=max_guidance_weight, neginf=0.0
            )

            correction_term = torch.nan_to_num(guidance_weight * pinv_correction, nan=0.0)
            v_t_guided = torch.nan_to_num(v_t + correction_term, nan=0.0)

            # Euler step（新 tensor，避免原地操作）
            step_result = x_t + dt * v_t_guided
            step_result = torch.where(
                torch.isnan(step_result), x_t, step_result
            )
            x_t = step_result
            self._note_denoise(time, x_t)
            time = time + dt

        x_t = torch.nan_to_num(x_t, nan=0.0)
        self._finish_forward_trace(x_t)
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
