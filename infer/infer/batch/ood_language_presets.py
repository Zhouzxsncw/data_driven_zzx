"""OOD language 固定测试集加载与目录命名。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_DEFAULT_PRESETS_PATH = Path(__file__).parent / "config" / "ood_language_presets.yaml"

OOD_LANGUAGE_DIR = "ood-language"


@dataclass(frozen=True)
class OODLanguagePreset:
    id: int
    name: str
    category: str
    modification: str
    task_prompt: str


def load_ood_language_presets(
    path: Optional[Path] = None,
) -> tuple[str, List[OODLanguagePreset]]:
    """加载 presets YAML，返回 (baseline_prompt, presets)。"""
    preset_path = (path or _DEFAULT_PRESETS_PATH).resolve()
    with preset_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    baseline = str(raw.get("baseline_prompt", "")).strip()
    presets: List[OODLanguagePreset] = []
    for item in raw.get("presets", []):
        if not isinstance(item, dict):
            continue
        presets.append(
            OODLanguagePreset(
                id=int(item["id"]),
                name=str(item.get("name", f"test_{item['id']}")),
                category=str(item.get("category", "")),
                modification=str(item.get("modification", "")),
                task_prompt=str(item.get("task_prompt", "")).strip(),
            )
        )
    presets.sort(key=lambda p: p.id)
    seen_ids: set[int] = set()
    for p in presets:
        if p.id in seen_ids:
            raise ValueError(f"ood_language_presets.yaml 中 id 重复: {p.id}")
        seen_ids.add(p.id)
    return baseline, presets


def get_ood_preset_by_id(presets: List[OODLanguagePreset], preset_id: int) -> OODLanguagePreset:
    for p in presets:
        if p.id == preset_id:
            return p
    valid = ", ".join(str(p.id) for p in presets)
    raise KeyError(f"未知 OOD test ID: {preset_id}，可选: {valid}")


def ood_language_output_dir(output_base: Path) -> Path:
    """OOD 整组结果目录（与 action-step_20/ 同级，产物直接在此目录）。"""
    return output_base.expanduser().resolve() / OOD_LANGUAGE_DIR


def ood_language_dir_name(preset: OODLanguagePreset) -> str:
    """已废弃：旧版按 preset 分子目录 ood-language/test_1/。"""
    return f"{OOD_LANGUAGE_DIR}/{preset.name}"


def write_ood_group_meta(
    output_dir: Path,
    presets: List[OODLanguagePreset],
    baseline: str,
) -> Path:
    """在 ood-language/ 写入整组 preset 与任务段映射。"""
    meta: Dict[str, Any] = {
        "baseline_prompt": baseline,
        "tasks_per_group": len(presets),
        "presets": [
            {
                "ood_test_id": p.id,
                "ood_test_name": p.name,
                "category": p.category,
                "modification": p.modification,
                "task_prompt": p.task_prompt,
                "task_segment_index": i + 1,
            }
            for i, p in enumerate(presets)
        ],
    }
    out = output_dir / "ood_group_meta.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return out


def write_ood_preset_meta(output_dir: Path, preset: OODLanguagePreset, baseline: str) -> Path:
    """兼容旧调用；请改用 write_ood_group_meta。"""
    return write_ood_group_meta(output_dir, [preset], baseline)


def format_ood_preset_menu(presets: List[OODLanguagePreset], baseline: str) -> str:
    lines = [
        "",
        "OOD Language 固定测试集（baseline）:",
        f"  {baseline}",
        "",
        f"{'段':<4} {'ID':<4} {'类别':<16} {'修改说明':<24} task_prompt",
        "-" * 100,
    ]
    for i, p in enumerate(presets, start=1):
        prompt_preview = p.task_prompt.replace("\n", " ")
        if len(prompt_preview) > 48:
            prompt_preview = prompt_preview[:45] + "..."
        lines.append(
            f"{i:<4} {p.id:<4} {p.category:<16} {p.modification:<24} {prompt_preview}"
        )
    lines.append("")
    lines.append(
        "说明: 整组产物在 ood-language/，按上表顺序执行 "
        f"{len(presets)} 个独立任务段；每段自动切换 task_prompt（Ctrl+. 开始/结束）。"
    )
    lines.append("")
    return "\n".join(lines)


def scan_ood_progress(
    ckpt_dir: Path,
    presets: List[OODLanguagePreset],
    *,
    tasks_per_group: Optional[int] = None,
) -> dict[int, str]:
    """扫描 ood-language/ 整组进度（各 ID 显示相同组状态，便于终端展示）。"""
    from infer.batch.batch_infer_utils import inspect_ood_language_dir

    expected = tasks_per_group if tasks_per_group is not None else len(presets)
    st = inspect_ood_language_dir(ood_language_output_dir(ckpt_dir), expected_tasks=expected)
    if st.state == "complete":
        label = f"整组已完成（{expected} 个任务段 / 1 个 test）"
    elif st.state == "partial":
        label = (
            f"组内中断于 {st.test_name or '本组'} "
            f"（已完成 {st.completed_tasks}/{expected} 个任务段）"
        )
    else:
        label = "未完成"
    return {p.id: label for p in presets}
