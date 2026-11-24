#!/usr/bin/env python3
"""Disable channels_last_3d vision patch on Ascend (NPU does not support that memory format)."""
from pathlib import Path

p = Path("/home/gr00t/src/Isaac-GR00T/gr00t/model/modules/qwen3_backbone.py")
text = p.read_text()
if "NPU_SKIP_CHANNELS_LAST_3D" in text:
    print("ALREADY_PATCHED")
    raise SystemExit(0)

old = '''    def _apply_vision_patch_embed_channels_last(self) -> None:
        """Force the Qwen3-VL vision patch-embed ``Conv3d`` to ``channels_last_3d``.

        On torch>=2.9 (cuDNN 9.10.x) the contiguous bf16 patch-embed ``Conv3d``
        dispatches to a pathologically slow kernel on sm_80; ``channels_last_3d``
        selects the fast one and is numerically equivalent (harmless elsewhere).
        """
        visual = getattr(self.model, "visual", None)
'''

new = '''    def _apply_vision_patch_embed_channels_last(self) -> None:
        """Force the Qwen3-VL vision patch-embed ``Conv3d`` to ``channels_last_3d``.

        On torch>=2.9 (cuDNN 9.10.x) the contiguous bf16 patch-embed ``Conv3d``
        dispatches to a pathologically slow kernel on sm_80; ``channels_last_3d``
        selects the fast one and is numerically equivalent (harmless elsewhere).
        """
        # NPU_SKIP_CHANNELS_LAST_3D: Ascend contiguous() only accepts contiguous
        # memory format; channels_last_3d raises ERR01007.
        if hasattr(torch, "npu") and torch.npu.is_available():
            logger.info("Skipping channels_last_3d vision patch on Ascend NPU")
            return
        visual = getattr(self.model, "visual", None)
'''

if old not in text:
    raise SystemExit("PATCH_MISS")
p.write_text(text.replace(old, new))
for c in p.parent.rglob("__pycache__"):
    import shutil
    shutil.rmtree(c, ignore_errors=True)
print("PATCHED", p)
