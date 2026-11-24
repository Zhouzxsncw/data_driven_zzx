"""交互式提示。"""

from __future__ import annotations


def ask_yes_no(prompt: str) -> bool:
    while True:
        choice = input(f"{prompt} (y/n): ").strip().lower()
        if choice in {"y", "yes"}:
            return True
        if choice in {"n", "no"}:
            return False
        print("[INPUT] 请输入 y 或 n。")


def ask_optional_task_prompt(current: str) -> str:
    """询问是否修改 task_prompt；留空则保持不变。"""
    if not ask_yes_no("是否修改 task_prompt？"):
        return current
    print(f"当前 task_prompt: {current}")
    new = input("输入新的 task_prompt（直接回车保持不变）: ").strip()
    return new if new else current


def parse_action_step_list(text: str, *, max_value: int) -> list[int]:
    """解析逗号/空格分隔的 action-step 列表，去重排序并校验范围。"""
    if max_value < 1:
        raise ValueError(f"max_value 必须 >= 1, 得到 {max_value}")
    raw = text.replace("，", ",").replace(";", ",")
    parts = [p.strip() for p in raw.replace(" ", ",").split(",") if p.strip()]
    if not parts:
        raise ValueError("至少输入一个 action-step 整数")
    values: list[int] = []
    seen: set[int] = set()
    for part in parts:
        try:
            val = int(part)
        except ValueError as exc:
            raise ValueError(f"无效整数: {part!r}") from exc
        if val < 1 or val > max_value:
            raise ValueError(f"action-step 须在 1~{max_value} 之间，得到 {val}")
        if val not in seen:
            seen.add(val)
            values.append(val)
    return sorted(values)


def ask_choice_int(prompt: str, choices: list[int]) -> int:
    """从给定整数列表中选择一个。"""
    choice_str = ", ".join(str(c) for c in choices)
    while True:
        raw = input(f"{prompt} ({choice_str}): ").strip()
        try:
            val = int(raw)
        except ValueError:
            print(f"[INPUT] 请输入以下之一: {choice_str}")
            continue
        if val in choices:
            return val
        print(f"[INPUT] 请输入以下之一: {choice_str}")


def ask_int_with_default(prompt: str, default: int, *, min_value: int = 1) -> int:
    """输入整数；直接回车保留 default。"""
    while True:
        raw = input(f"{prompt} [默认 {default}，回车保留]: ").strip()
        if not raw:
            return int(default)
        try:
            val = int(raw)
        except ValueError:
            print("[INPUT] 请输入整数，或直接回车保留默认值。")
            continue
        if val < min_value:
            print(f"[INPUT] 须 >= {min_value}，得到 {val}")
            continue
        return val


def ask_float_with_default(prompt: str, default: float, *, min_value: float = 0.1) -> float:
    """输入浮点；直接回车保留 default。"""
    while True:
        raw = input(f"{prompt} [默认 {default}，回车保留]: ").strip()
        if not raw:
            return float(default)
        try:
            val = float(raw)
        except ValueError:
            print("[INPUT] 请输入数字，或直接回车保留默认值。")
            continue
        if val < min_value:
            print(f"[INPUT] 须 >= {min_value}，得到 {val}")
            continue
        return val


def ask_optional_int_with_default(
    prompt: str,
    default: int | None,
    *,
    min_value: int = 1,
) -> int | None:
    """输入整数或留空/null 表示 None；直接回车保留 default。"""
    default_label = "null（checkpoint 默认）" if default is None else str(default)
    while True:
        raw = input(
            f"{prompt} [默认 {default_label}；回车保留；输入 null 用 checkpoint 默认]: "
        ).strip()
        if not raw:
            return default
        if raw.lower() in {"null", "none", "-"}:
            return None
        try:
            val = int(raw)
        except ValueError:
            print("[INPUT] 请输入整数、null，或直接回车。")
            continue
        if val < min_value:
            print(f"[INPUT] 须 >= {min_value}，得到 {val}")
            continue
        return val


def ask_ood_test_id(
    valid_ids: list[int],
    *,
    status_by_id: dict[int, str] | None = None,
) -> int:
    """选择本组要运行的 OOD test ID（须手动输入，无默认）。"""
    choice_str = ", ".join(str(i) for i in valid_ids)
    if status_by_id:
        print("\nOOD 测试进度（仅供参考，无固定顺序）:")
        for i in valid_ids:
            status = status_by_id.get(i, "未完成")
            print(f"  ID {i}: {status}")
    prompt = f"请选择本组 OOD test ID（切换 task_prompt） ({choice_str}): "
    while True:
        raw = input(prompt).strip()
        try:
            val = int(raw)
        except ValueError:
            print(f"[INPUT] 请输入以下之一: {choice_str}")
            continue
        if val in valid_ids:
            return val
        print(f"[INPUT] 请输入以下之一: {choice_str}")
