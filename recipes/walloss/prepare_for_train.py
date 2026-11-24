#!/usr/bin/env python3
"""Prepare TongBot datasets for WALL-OSS continuous + discrete SFT.

Same entry for data processing and prepare-to-train. It does not start training.

  python3 recipes/walloss/prepare_for_train.py
  python3 recipes/walloss/prepare_for_train.py --ids 00036 00108 --convert-workers 16

Each id runs ``ee_sft_pipeline.sh --until norm``:
joint → absolute EE (parallel FK) → LeRobot v3 → sft.yml (flow + FAST) → norm stats.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent / "ee_sft_pipeline.sh"

# src is the joint LeRobot root when EE is missing; ee is an already converted root.
DATASETS: dict[str, dict[str, str]] = {
    "00036": {"src": "/data/wall-oss/datasets/00036_rm75_ee"},
    "00073": {"src": "/data/wall-oss/datasets/00073_rm75_ee_v30"},
    "00091": {"src": "/data/wall-oss/datasets/00091_rm75_ee_v30"},
    "00108": {
        "src": "/home/xk/dataset/00108_20260922_RM_01_YJ_v21",
        "dst": "/data/wall-oss/datasets/00108_rm75_ee",
    },
}


def _run_one(dataset_id: str, args: argparse.Namespace, log_dir: Path) -> int:
    spec = DATASETS[dataset_id]
    src = spec["src"]
    log_path = log_dir / f"{dataset_id}.log"
    cmd = [
        "bash",
        str(PIPELINE),
        "--src",
        src,
        "--name",
        dataset_id,
        "--until",
        "norm",
        "--convert-workers",
        str(args.convert_workers if dataset_id == args.heavy else 1),
        "--action-tokenizer-path",
        args.action_tokenizer_path,
        "--action-tokenizer-type",
        "fast",
        "--use-fast-tokenizer",
        "true",
    ]
    if "dst" in spec:
        cmd.extend(["--dst", spec["dst"]])
    env = os.environ.copy()
    # FK and norm stats are CPU. Leave the local GPU for inference.
    env["CUDA_VISIBLE_DEVICES"] = ""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write(" ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", nargs="+", default=["00036", "00073", "00091", "00108"])
    parser.add_argument("--convert-workers", type=int, default=12)
    parser.add_argument(
        "--heavy",
        default="00108",
        help="which id receives --convert-workers (the joint→EE job)",
    )
    parser.add_argument(
        "--action-tokenizer-path",
        default=(
            "/data/wall-oss/pretrained/fast"
        ),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/data/wall-oss/datasets/prepare_logs"),
    )
    args = parser.parse_args()
    unknown = [i for i in args.ids if i not in DATASETS]
    if unknown:
        raise SystemExit(f"未知数据集: {unknown}；可选 {sorted(DATASETS)}")
    if not Path(args.action_tokenizer_path).is_dir():
        raise SystemExit(f"找不到 FAST tokenizer: {args.action_tokenizer_path}")

    print(f"parallel datasets={args.ids} convert_workers={args.convert_workers} heavy={args.heavy}", flush=True)
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=len(args.ids)) as pool:
        futures = {pool.submit(_run_one, dataset_id, args, args.log_dir): dataset_id for dataset_id in args.ids}
        for fut in as_completed(futures):
            dataset_id = futures[fut]
            code = fut.result()
            status = "ok" if code == 0 else f"exit {code}"
            print(f"[{dataset_id}] {status}  log={args.log_dir / (dataset_id + '.log')}", flush=True)
            if code != 0:
                failures.append(dataset_id)
    if failures:
        raise SystemExit(f"未完成: {failures}")


if __name__ == "__main__":
    main()
