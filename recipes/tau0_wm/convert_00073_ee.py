#!/usr/bin/env python3
"""TongBot 00073：rm75_ee_v30（WALL-OSS 行约定 rot6d）→ τ0-WM 列约定 20D。

与 convert_00036_ee.py 的区别：00073_rm75_ee_v30 已是 EE 20D（RM75 FK 已做），
只需把 rot6d 从「行约定 (r00,r01,r02,r10,r11,r12)」转「列约定
(r00,r10,r20,r01,r11,r21)」，其中 r20 = r00*r11 - r01*r10（正交第三行第 0 列）。

用法（h100-103）::
    /data1/tau0wm/envs/tau0wm/bin/python convert_00073_ee.py \
        --src /data1/dataset/00073_rm75_ee_v30 \
        --dst /data1/tau0wm/datasets/00073_rm75_ee_tau0wm
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROT_LO = (3, 13)  # 左臂 rot6d [3:9]、右臂 [13:19]
EE_COLS = ("action.ee", "observation.state.ee")


def rows_to_cols(m: np.ndarray) -> np.ndarray:
    """(N,6) 行约定 → 列约定，向量化。

    输入 [r00,r01,r02, r10,r11,r12]（R 的第 0/1 行），输出
    [r00,r10,r20, r01,r11,r21]（R 的第 0/1 列）。正交阵第 2 行 =
    第 0 行 × 第 1 行：r20 = r01*r12 − r02*r11，r21 = r02*r10 − r00*r12。
    自检：重建 R 后 R @ R.T ≈ I。
    """
    r00, r01, r02 = m[:, 0], m[:, 1], m[:, 2]
    r10, r11, r12 = m[:, 3], m[:, 4], m[:, 5]
    r20 = r01 * r12 - r02 * r11
    r21 = r02 * r10 - r00 * r12
    out = np.stack([r00, r10, r20, r01, r11, r21], axis=1)
    # 重建 R 验证正交性（抽样即可，全量算力也无压力）
    R = np.stack([
        np.stack([r00, r01, r02], axis=1),
        np.stack([r10, r11, r12], axis=1),
        np.stack([r20, r21, r00 * r11 - r01 * r10], axis=1),
    ], axis=1)  # (N,3,3)
    eye = np.einsum("nij,nkj->nik", R, R)
    assert np.allclose(eye, np.eye(3), atol=1e-6), "重建 R 非正交，输入不是合法行约定 rot6d"
    return out


def convert_col(arr: np.ndarray) -> np.ndarray:
    assert arr.ndim == 2 and arr.shape[1] == 20, f"EE 20D expected, got {arr.shape}"
    out = arr.copy()
    for lo in ROT_LO:
        seg = arr[:, lo:lo + 6]
        # 输入必须是单位正交（行约定特征），否则说明源数据已坏
        c0, c1 = seg[:, 0:3], seg[:, 3:6]
        assert np.allclose(np.linalg.norm(c0, axis=1), 1, atol=1e-3), f"rot6d |r0|!=1 @lo={lo}"
        assert np.allclose(np.abs((c0 * c1).sum(1)), 0, atol=1e-3), f"rot6d not ortho @lo={lo}"
        out[:, lo:lo + 6] = rows_to_cols(seg)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    args = ap.parse_args()

    src, dst = args.src, args.dst
    assert (src / "data").is_dir() and (src / "videos").is_dir(), "src 缺少 data/videos"
    (dst / "data").mkdir(parents=True, exist_ok=True)

    # 1) meta 复制（info.json 里标注列约定，供日后审计）
    if not (dst / "meta").exists():
        shutil.copytree(src / "meta", dst / "meta")
    info_p = dst / "meta" / "info.json"
    info = json.loads(info_p.read_text())
    info["rot6d_convention"] = "tau0wm_cols (r00,r10,r20,r01,r11,r21), converted from walloss_rows"
    info_p.write_text(json.dumps(info, indent=1))

    # 2) videos 硬链（同盘 /data1，零额外空间；已存在则跳过）
    if not (dst / "videos").exists():
        shutil.copytree(src / "videos", dst / "videos", copy_function=os.link)

    # 3) parquet：rot6d 行→列，其余列原样
    n_files = n_rows = 0
    for pq_in in sorted((src / "data").rglob("*.parquet")):
        rel = pq_in.relative_to(src / "data")
        pq_out = dst / "data" / rel
        pq_out.parent.mkdir(parents=True, exist_ok=True)
        t = pq.read_table(pq_in)
        for name in EE_COLS:
            if name not in t.column_names:
                continue
            col = t.column(name).to_pylist()
            arr = np.asarray(col, dtype=np.float64)
            conv = convert_col(arr)
            idx = t.column_names.index(name)
            t = t.set_column(idx, name, pa.ListArray.from_arrays(
                pa.array(np.arange(0, len(conv) * 20 + 1, 20)), pa.array(conv.ravel())))
        pq.write_table(t, pq_out)
        n_files += 1
        n_rows += t.num_rows
        if n_files % 20 == 0:
            print(f"[{n_files}] {rel} rows={t.num_rows}", flush=True)

    # 4) 独立自检：抽首文件验证 转换可逆（列约定 matrix→6d 恒等）
    f0 = sorted((dst / "data").rglob("*.parquet"))[0]
    t0 = pq.read_table(f0)
    a = np.asarray(t0.column("action.ee").to_pylist()[:200], dtype=np.float64)
    for lo in ROT_LO:
        seg = a[:, lo:lo + 6]
        c0, c1 = seg[:, 0:3], seg[:, 3:6]
        assert np.allclose(np.linalg.norm(c0, axis=1), 1, atol=1e-3)
        assert np.allclose(np.abs((c0 * c1).sum(1)), 0, atol=1e-3)
    print(f"CONVERT_DONE files={n_files} rows={n_rows}")
    print(f"spot_check: act row0={np.round(a[0][:10],3).tolist()}")


if __name__ == "__main__":
    import os  # noqa: E402  (os.link 用于硬链)

    main()
