"""批量推理人工评分表（.xlsx）：从模板复制、预填配置项、打开与保存确认。"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from infer.batch.ood_language_presets import OODLanguagePreset

_INFER_PKG = Path(__file__).resolve().parent.parent
_TEMPLATE_DIR = _INFER_PKG / "template"
ACTION_STEP_TEMPLATE = _TEMPLATE_DIR / "机械臂抓取模型反馈评分模型.xlsx"
OOD_TEMPLATE = _TEMPLATE_DIR / "机械臂抓取模型反馈评分ood.xlsx"
ACTION_STEP_FILENAME = "机械臂抓取模型反馈评分模型.xlsx"
OOD_FILENAME = "机械臂抓取模型反馈评分ood.xlsx"

# OOD 评分表 test 列起始列（0-based）；test N → col = OOD_SCORE_SHEET_FIRST_TEST_COL + (N-1)
OOD_SCORE_SHEET_FIRST_TEST_COL = 3

CellKey = Tuple[int, int]

# action-step 模板占位符扫描关键字（0-based 行列自动定位）
_ACTION_STEP_FIELD_NEEDLES: Dict[str, Tuple[str, ...]] = {
    "n_action_steps": ("N_ACTION_STEPS_OVERRIDE",),
    "num_inference_steps": ("NUM_INFERENCE_STEPS_OVERRIDE",),
    "prompt": ("提示词",),
    "paths": ("video路径", "对应video路径", "模型路径"),
}

_SCORE_SHEET_PIP = "pip install openpyxl"

# action-step 总分/任务进度公式数量（用于回归测试）
_ACTION_STEP_FORMULA_CELL_COUNT = 12


def ensure_feedback_score_sheet_deps() -> None:
    """批量推理评分表硬依赖；缺包时直接报错并提示安装命令。"""
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            f"评分表依赖未安装，请执行: {_SCORE_SHEET_PIP}"
        ) from exc


def count_workbook_formula_cells(path: Path) -> int:
    """统计 .xlsx 中的公式单元格数。"""
    import openpyxl

    wb = openpyxl.load_workbook(path, data_only=False, read_only=True)
    try:
        ws = wb.active
        total = 0
        for row in ws.iter_rows():
            for cell in row:
                if cell.data_type == "f":
                    total += 1
        return total
    finally:
        wb.close()


def _load_workbook(path: Path):
    import openpyxl

    return openpyxl.load_workbook(path, data_only=False)


def _cell_text_ws(ws, row: int, col: int) -> str:
    """读取单元格文本；合并格取合并区域左上角（row/col 为 0-based）。"""
    row, col = _merged_origin_ws(ws, row, col)
    value = ws.cell(row=row + 1, column=col + 1).value
    if value is None:
        return ""
    return str(value).strip()


def _merged_origin_ws(ws, row: int, col: int) -> CellKey:
    """0-based → 所属合并区域左上角（0-based）。"""
    for merged in ws.merged_cells.ranges:
        if (
            merged.min_row - 1 <= row < merged.max_row
            and merged.min_col - 1 <= col < merged.max_col
        ):
            return merged.min_row - 1, merged.min_col - 1
    return row, col


def _find_template_cell(
    ws,
    needles: Sequence[str],
    *,
    label: str,
) -> CellKey:
    """在模板中按关键字定位单元格（0-based）。"""
    hits: List[CellKey] = []
    for row in range(ws.max_row):
        for col in range(ws.max_column):
            text = _cell_text_ws(ws, row, col)
            if not text:
                continue
            if any(needle in text for needle in needles):
                hits.append((row, col))
    if not hits:
        raise KeyError(
            f"评分表模板中未找到 {label}（需包含: {', '.join(needles)}）"
        )
    return hits[0]


def _find_paths_cell(ws) -> CellKey:
    """定位 video/模型路径配置格；优先同格含两行，否则取含 video 的首格。"""
    video_hit: Optional[CellKey] = None
    for row in range(ws.max_row):
        for col in range(ws.max_column):
            text = _cell_text_ws(ws, row, col)
            if not text:
                continue
            has_video = "video路径" in text or "对应video路径" in text
            has_model = "模型路径" in text or "对应模型路径" in text
            if has_video and has_model:
                return (row, col)
            if has_video and video_hit is None:
                video_hit = (row, col)
    if video_hit is not None:
        return video_hit
    raise KeyError(
        "评分表模板中未找到 paths（需包含 video路径 或 对应video路径）"
    )


def _scan_action_step_template_cells(ws) -> Dict[str, CellKey]:
    cells: Dict[str, CellKey] = {}
    for field, needles in _ACTION_STEP_FIELD_NEEDLES.items():
        if field == "paths":
            cells[field] = _find_paths_cell(ws)
            continue
        cells[field] = _find_template_cell(ws, needles, label=field)
    return cells


@dataclass(frozen=True)
class FeedbackScoreSheetSpec:
    """生成评分表所需的运行配置（与当前任务保持一致）。"""

    test_dir: Path
    n_action_steps: int
    task_prompt: str
    model_path: Path
    num_inference_steps: Optional[int] = None
    phase: str = "action-step"  # action-step | ood-language
    ood_presets: Optional[Sequence["OODLanguagePreset"]] = None


def resolve_num_inference_steps(cfg, policy_adapter) -> Optional[int]:
    """与 Pi05Adapter.load() 一致：优先 override，否则读 checkpoint 训练配置。"""
    override = getattr(getattr(cfg, "policy", None), "num_inference_steps_override", None)
    if override is not None:
        return int(override)
    inner = getattr(policy_adapter, "policy", None)
    if inner is not None:
        inner_cfg = getattr(inner, "config", None)
        if inner_cfg is not None:
            trained = getattr(inner_cfg, "num_inference_steps", None)
            if trained is not None:
                return int(trained)
    return None


def _format_override_value(value: Optional[int]) -> str:
    return "None" if value is None else str(int(value))


def _score_sheet_display_paths(test_dir: Path, model_path: Path) -> tuple[str, str]:
    """评分表路径与模板一致：写入绝对路径（video 目录 + 模型 checkpoint 目录）。"""
    test_dir = test_dir.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    video_display = str((test_dir / "videos").resolve())
    model_display = str(model_path)
    return video_display, model_display


def _action_step_cell_values(spec: FeedbackScoreSheetSpec, ws) -> Dict[CellKey, str]:
    video_path, model_path = _score_sheet_display_paths(spec.test_dir, spec.model_path)
    prompt = spec.task_prompt.strip()
    cells = _scan_action_step_template_cells(ws)
    return {
        cells["n_action_steps"]: f"N_ACTION_STEPS_OVERRIDE = {int(spec.n_action_steps)}",
        cells["num_inference_steps"]: (
            f"NUM_INFERENCE_STEPS_OVERRIDE = {_format_override_value(spec.num_inference_steps)}"
        ),
        cells["prompt"]: f'提示词： "{prompt}"',
        cells["paths"]: (
            f"video路径：{video_path}\n"
            f"模型路径：{model_path}"
        ),
    }


def _ood_cell_values(spec: FeedbackScoreSheetSpec) -> Dict[CellKey, str]:
    """OOD 整组：各段 prompt 写入 test 列；列数 = len(ood_presets)。"""
    if not spec.ood_presets:
        raise ValueError("OOD 评分表需要 ood_presets（来自 ood_language_presets.yaml）")

    presets = list(spec.ood_presets)
    video_path, model_path = _score_sheet_display_paths(spec.test_dir, spec.model_path)
    baseline = spec.task_prompt.strip().rstrip(".")

    values: Dict[CellKey, str] = {
        (2, 3): baseline,
        (19, 2): f"N_ACTION_STEPS_OVERRIDE = {int(spec.n_action_steps)}",
        (22, 2): (
            f"video路径：{video_path}\n"
            f"模型路径：{model_path}"
        ),
    }

    for i, preset in enumerate(presets):
        col = OOD_SCORE_SHEET_FIRST_TEST_COL + i
        prompt = preset.task_prompt.strip()
        category = preset.category.strip()
        values[(4, col)] = f"{category}\n{prompt}"
        values[(5, col)] = preset.modification.strip()

    prompt_lines = [
        f"test {i + 1} (ID {preset.id}): {preset.task_prompt.strip()}"
        for i, preset in enumerate(presets)
    ]
    values[(21, 2)] = "TEXT PROMPT:\n" + "\n".join(prompt_lines)
    return values


def _write_xlsx_from_template(
    template_path: Path,
    dest: Path,
    cell_values: Dict[CellKey, str],
) -> None:
    """
    复制 .xlsx 模板并仅更新指定单元格的值，保留公式、合并单元格与样式。
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(template_path, dest)
    wb = _load_workbook(dest)
    ws = wb.active
    for (row, col), value in sorted(cell_values.items()):
        row, col = _merged_origin_ws(ws, row, col)
        ws.cell(row=row + 1, column=col + 1).value = value
    wb.save(dest)
    logger.info("已从 xlsx 模板写入评分表（保留公式与格式）: %s", dest)


def prepare_feedback_score_sheet(spec: FeedbackScoreSheetSpec) -> Path:
    """
    从模板复制评分表到 test_dir，并预填配置相关单元格。
    返回生成的 .xlsx 路径。
    """
    ensure_feedback_score_sheet_deps()
    spec.test_dir.mkdir(parents=True, exist_ok=True)

    if spec.phase == "ood-language":
        template_path = OOD_TEMPLATE
        filename = OOD_FILENAME
        if not template_path.is_file():
            raise FileNotFoundError(f"评分表模板不存在: {template_path}")
        cell_values = _ood_cell_values(spec)
        dest = spec.test_dir / filename
        _write_xlsx_from_template(template_path, dest, cell_values)
        logger.info("已生成 OOD 评分表（完整 prompt，无截断）: %s", dest)
        return dest

    template_path = ACTION_STEP_TEMPLATE
    filename = ACTION_STEP_FILENAME

    if not template_path.is_file():
        raise FileNotFoundError(f"评分表模板不存在: {template_path}")

    wb = _load_workbook(template_path)
    ws = wb.active
    cell_values = _action_step_cell_values(spec, ws)
    wb.close()
    dest = spec.test_dir / filename
    _write_xlsx_from_template(template_path, dest, cell_values)
    logger.info("已生成评分表: %s", dest)
    return dest


def open_feedback_score_sheet(sheet_path: Path) -> bool:
    """尝试用 WPS / 系统默认程序打开评分表。"""
    path_str = str(sheet_path.resolve())
    candidates: List[List[str]] = [
        ["wps", path_str],
        ["et", path_str],
        ["xdg-open", path_str],
    ]
    for cmd in candidates:
        try:
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            logger.info("已打开评分表 (%s): %s", " ".join(cmd[:2]), sheet_path)
            print(f"[INFO] 已尝试用 WPS 打开评分表: {sheet_path.name}")
            return True
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.warning("打开评分表失败 (%s): %s", cmd, exc)
    print(f"\n[WARN] 无法自动打开评分表，请手动打开:\n  {sheet_path}\n")
    return False


def prompt_feedback_score_sheet_save(pedal_hub, sheet_path: Path) -> None:
    """提示评分员保存并关闭表格，脚踏板/终端确认后继续。"""
    from infer.ui.pedal_input import ask_yes_no_pedal

    print(f"\n{'=' * 60}")
    print("请在 WPS 表格中保存并关闭本组评分表（记录部分由人工填写）：")
    print(f"  {sheet_path}")
    print("  操作: 文件 → 保存，然后关闭表格窗口")
    print(f"{'=' * 60}")
    while not ask_yes_no_pedal(pedal_hub, "评分表已保存并关闭？"):
        print("[INFO] 请先在 WPS 中保存并关闭评分表，再按确认键继续。")


def run_feedback_score_sheet_workflow(
    spec: FeedbackScoreSheetSpec,
    pedal_hub,
) -> Optional[Path]:
    """
    组开始前生成并打开评分表；组结束后等待人工保存关闭。
    返回评分表路径；模板缺失时记录警告并跳过。
    """
    try:
        sheet_path = prepare_feedback_score_sheet(spec)
    except FileNotFoundError as exc:
        logger.warning("评分表模板缺失，已跳过: %s", exc)
        print(f"[WARN] 评分表已跳过（模板缺失）: {exc}")
        return None
    except (KeyError, ValueError) as exc:
        logger.error("评分表预填失败: %s", exc)
        raise RuntimeError(f"评分表预填失败: {exc}") from exc

    print(f"\n[INFO] 已生成评分表: {sheet_path.name}")
    if spec.phase == "ood-language":
        n = len(spec.ood_presets or ())
        print(
            f"      各段 TEXT PROMPT 已写入第 4 行 test 1~{n} 列"
            f"（共 {n} 段，与 ood_language_presets.yaml 一致）。"
        )
    else:
        print("      配置项已自动填写；请在表格中填写测试记录。")
    open_feedback_score_sheet(sheet_path)
    return sheet_path
