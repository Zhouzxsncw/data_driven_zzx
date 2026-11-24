#!/usr/bin/env python3
"""τ0-WM ACTION per-dim loss 健康监控（fork patch 6 配套）。

读训练 run 目录下最新 TensorBoard events，对 20 个 Action loss dim/* tag 做：
  * 首值/最新值/变化率/线性斜率
  * 健康判定：DOWN（健康）/ STUCK / EXPLODE / OUTLIER
  * step < warmup_min_step（默认 2000）时只观察不下结论（warmup 期波动大）

用法（h100-103）::
    /data1/tau0wm/envs/tau0wm/bin/python monitor_action_dim_loss.py \
        --run-dir /data1/tau0wm/outputs/tongbot_00073_ee/tau0wm_vam5b_ee_rela_192x256_bs1_300k

退出码：0=全部健康或观察期，1=有异常维度。
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np


def latest_events(run_dir: str) -> str:
    es = glob.glob(str(Path(run_dir) / "events.out.tfevents.*"))
    if not es:
        sys.exit(f"NO_EVENTS: {run_dir} 下没有 events 文件")
    return max(es, key=lambda p: Path(p).stat().st_mtime)


def slope(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    xm, ym = sum(xs) / n, sum(ys) / n
    den = sum((x - xm) ** 2 for x in xs)
    return sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / den if den > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--warmup-min-step", type=int, default=2000)
    ap.add_argument("--out", default=None, help="JSON 健康报告落盘路径（追加）")
    args = ap.parse_args()

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ev_path = latest_events(args.run_dir)
    ea = EventAccumulator(ev_path)
    ea.Reload()
    dims = sorted(t for t in ea.Tags()["scalars"] if t.startswith("Action loss dim/"))
    if len(dims) != 20:
        print(f"WARN: dim tags={len(dims)} != 20（训练未起或 patch 未生效）")

    rows, alerts = [], []
    last_step = 0
    vals = {}
    for t in dims:
        ev = ea.Scalars(t)
        name = t.split("/")[-1]
        xs = [e.step for e in ev]
        ys = [e.value for e in ev]
        last_step = max(last_step, xs[-1] if xs else 0)
        vals[name] = ys[-1] if ys else float("nan")

    med = float(np.median(list(vals.values()))) if vals else 1.0

    for t in dims:
        ev = ea.Scalars(t)
        name = t.split("/")[-1]
        if not ev:
            rows.append((name, None, None, None, None, "NO_DATA"))
            alerts.append(f"{name}: NO_DATA")
            continue
        xs = [e.step for e in ev]
        ys = [e.value for e in ev]
        f, l = ys[0], ys[-1]
        chg = (l - f) / f * 100 if f else 0.0
        sl = slope([float(x) for x in xs], ys)

        if last_step < args.warmup_min_step:
            verdict = "OBSERVE"  # warmup 期不下结论
        elif len(ys) >= 5 and np.std(ys[-5:]) / max(np.mean(ys[-5:]), 1e-9) < 0.01:
            verdict = "STUCK"
            alerts.append(f"{name}: STUCK last5={ys[-5:]}")
        elif l > 3 * max(f, 1e-6) and l > 1.0:
            verdict = "EXPLODE"
            alerts.append(f"{name}: EXPLODE first={f:.4f} last={l:.4f}")
        elif med > 0 and l > 20 * med:
            verdict = "OUTLIER"
            alerts.append(f"{name}: OUTLIER last={l:.4f} med={med:.4f}")
        else:
            verdict = "DOWN" if sl < 0 else "FLAT"
        rows.append((name, f, l, chg, sl, verdict))

    ts = time.strftime("%F %T")
    print(f"[{ts}] run={Path(args.run_dir).name} events={Path(ev_path).name} step={last_step}")
    print(f"{'dim':<8} {'first':>9} {'last':>9} {'chg%':>8} {'slope':>12}  verdict")
    for name, f, l, chg, sl, v in rows:
        if f is None:
            print(f"{name:<8} {'-':>9} {'-':>9} {'-':>8} {'-':>12}  {v}")
        else:
            print(f"{name:<8} {f:>9.4f} {l:>9.4f} {chg:>7.1f}% {sl:>+12.2e}  {v}")

    verdicts = {r[-1] for r in rows}
    summary = {
        "ts": ts, "step": last_step, "events": Path(ev_path).name,
        "n_dims": len(dims), "verdicts": sorted(verdicts),
        "alerts": alerts,
        "last": vals,
    }
    print(f"SUMMARY: dims={len(dims)} verdicts={sorted(verdicts)} alerts={len(alerts)}")
    if args.out:
        with open(args.out, "a") as fp:
            fp.write(json.dumps(summary, ensure_ascii=False) + "\n")
    sys.exit(1 if alerts else 0)


if __name__ == "__main__":
    main()
