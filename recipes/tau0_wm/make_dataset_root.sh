#!/usr/bin/env bash
# 为 τ0-WM 生成一棵「修好 tasks.parquet」的 00036 LeRobot v3.0 数据集目录树。
#
# 背景：/home/xk/dataset/00036_20260804_RM_01_yj 是从 v2.1 转换来的 v3.0，
# 但转换时把任务文本留在了列 `__index_level_0__` 里、索引退化成整数 RangeIndex。
# τ0-WM 的 data/example_dataset.py 用 `self.meta.tasks.iloc[task_idx].name` 取 caption，
# 于是 caption 会变成字符串 "0" 而不是真的语言指令 —— 预训练权重是靠真文本 prompt 训的，
# 喂 "0" 等于丢掉语言条件，必须修。
#
# 做法：不改动原数据集（其它 recipe 也在用），只做目录软链 + 重写 meta/tasks.parquet，
# 索引列 = 任务文本，附 task_index 列，这才是 lerobot v3.0 的规范布局。
set -euo pipefail

SRC="${SRC:-/home/xk/dataset/00036_20260804_RM_01_yj}"
DST="${DST:-/home/xk/dataset/00036_v30_for_tau0wm}"
PY="${PY:-/home/xk/miniconda3/envs/gigaworld-policy/bin/python}"

echo "[tau0_wm] SRC=${SRC}"
echo "[tau0_wm] DST=${DST}"

mkdir -p "${DST}/meta"
ln -sfn "${SRC}/data"   "${DST}/data"
ln -sfn "${SRC}/videos" "${DST}/videos"

# meta 里除 tasks.parquet 外全部软链
ln -sfn "${SRC}/meta/info.json"   "${DST}/meta/info.json"
ln -sfn "${SRC}/meta/stats.json"  "${DST}/meta/stats.json"
ln -sfn "${SRC}/meta/episodes"    "${DST}/meta/episodes"

"${PY}" - "${SRC}" "${DST}" <<'PY'
import sys
from pathlib import Path

import pandas as pd

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
df = pd.read_parquet(src / "meta" / "tasks.parquet")

TEXT_COLS = [c for c in df.columns if c in ("task", "__index_level_0__")]
if df.index.name == "task" or (not TEXT_COLS and df.index.dtype == object):
    out = df
    print("[tau0_wm] tasks.parquet 索引已是任务文本，原样落盘")
else:
    text_col = TEXT_COLS[0]
    out = df.rename(columns={text_col: "task"}).set_index("task")
    out.index.name = "task"
    print(f"[tau0_wm] 已把 '{text_col}' 列提升为索引（任务文本）")

assert "task_index" in out.columns, f"缺 task_index 列：{list(out.columns)}"
out.to_parquet(dst / "meta" / "tasks.parquet")

chk = pd.read_parquet(dst / "meta" / "tasks.parquet")
print("[tau0_wm] 行数:", len(chk), "| 索引名:", chk.index.name)
for i, name in enumerate(chk.index[:5]):
    print(f"  task_index={chk['task_index'].iloc[i]}  {name[:100]}")
PY

echo "[tau0_wm] 完成：${DST}"
ls -la "${DST}" "${DST}/meta"
