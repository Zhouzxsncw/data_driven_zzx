"""Vendored ALOHA ACT / Diffusion / RDT model code used by infer.

Keep package import lightweight; CUDA models are imported lazily by
``AlohaActRuntime`` when ``act_backend: aloha`` is selected.
"""
