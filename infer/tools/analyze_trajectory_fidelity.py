#!/usr/bin/env python3
"""Offline D1 first release: raw-time ZOH comparison, never a physical accuracy claim."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def stats(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not a.size:
        return {"count": 0}
    return {"count": int(a.size), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
            "max": float(a.max())}


def match_reference(command_times, targets, feedback_times, *, max_gap=0.3):
    """At feedback query END, ZOH from last attempted command; failures break support.

    No extrapolation beyond max_gap, no future commands, no time-shift fitting.
    Query interval uncertainty is additionally filtered by analyze().
    """
    times = np.asarray(command_times)
    feedback_times = np.asarray(feedback_times)
    index = np.searchsorted(times, feedback_times, side="right") - 1
    valid = index >= 0
    index = np.maximum(index, 0)
    age = feedback_times - times[index]
    ref = np.asarray(targets)[index]
    valid &= (age <= max_gap) & np.isfinite(ref).all(axis=1)
    return ref, valid, age


def write_signals(path, times, values, prefix):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_sec", *[f"{prefix}_j{i+1}_rad" for i in range(7)]])
        w.writerows([t, *q] for t, q in zip(times, values))


def analyze(directory, *, max_gap=0.3, plots=True):
    root = Path(directory)
    meta = json.loads((root / "run_meta.json").read_text())
    if meta.get("schema_version") != 1:
        raise ValueError("Unsupported schema version")
    commands = read_rows(root / "commands.csv")
    feedback = read_rows(root / "feedback.csv")
    events = read_rows(root / "events.csv")
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    issues = []
    if not meta.get("closed") or not meta.get("recording_complete"):
        issues.append("Recorder incomplete, dropped rows, or writer errors: metrics are exploratory only")
    if not meta.get("run_returned"):
        issues.append("Inference did not return normally")
    if not commands or not feedback:
        issues.append("Missing commands or feedback")
    report = {
        "schema_version": 1, "comparison": "host query end vs last command start; ZOH",
        "synthetic_data": bool(meta.get("source_metadata", {}).get("synthetic", False)),
        "limitations": ["Not controller receipt time or hardware sampling time",
                        "Query intervals containing command transitions are excluded",
                        "No fitted physical delay, FK, or task success assessment in this release"],
        "max_reference_age_sec": max_gap, "quality_issues": issues, "arms": {},
    }
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    boundary_times = [float(r["time_sec"]) for r in events if r["event"] == "chunk_install"]
    barriers = np.array(sorted(float(r["time_sec"]) for r in events
                               if r["event"] in {"task_boundary", "group_boundary", "run_end"}
                               or r["event"].startswith(("exo_", "device_"))))
    for arm in ("left", "right"):
        c = sorted([r for r in commands if r["arm"] == arm], key=lambda r: float(r["send_start_sec"]))
        f = sorted([r for r in feedback if r["arm"] == arm and r["return_code"] == "0"
                    and not r["error"] and all(r[f"q{i}_rad"] != "" for i in range(7))],
                   key=lambda r: float(r["query_end_sec"]))
        result = {"command_attempts": len(c), "feedback_valid_rows": len(f),
                  "failed_commands": sum(r["return_code"] != "0" or bool(r["error"]) for r in c),
                  "failed_feedback": sum(r["arm"] == arm and (r["return_code"] != "0" or bool(r["error"]))
                                         for r in feedback)}
        report["arms"][arm] = result
        if not c or not f:
            continue
        ct = np.array([float(r["send_start_sec"]) for r in c])
        cq = np.array([[float(r[f"q{i}_rad"]) for i in range(7)] for r in c])
        for k, row in enumerate(c):
            if row["return_code"] != "0" or row["error"] or not row["chunk_id"]:
                cq[k] = np.nan
        ft = np.array([float(r["query_end_sec"]) for r in f])
        fs = np.array([float(r["query_start_sec"]) for r in f])
        fq = np.array([[float(r[f"q{i}_rad"]) for i in range(7)] for r in f])
        ref, valid, age = match_reference(ct, cq, ft, max_gap=max_gap)
        # A command sent during the query makes the association ambiguous.
        start_index = np.searchsorted(ct, fs, side="right") - 1
        end_index = np.searchsorted(ct, ft, side="right") - 1
        valid &= (start_index == end_index) & (ft >= fs)
        if barriers.size:
            barrier_index = np.searchsorted(barriers, ft, side="right") - 1
            last_barrier = np.where(barrier_index >= 0, barriers[np.maximum(barrier_index, 0)], -np.inf)
            command_index = np.maximum(end_index, 0)
            valid &= ct[command_index] >= last_barrier
        result.update(send_intervals_sec=stats(np.diff(ct)), feedback_intervals_sec=stats(np.diff(ft)),
                      query_durations_sec=stats(ft - fs), reference_age_sec=stats(age[valid]),
                      matched_samples=int(valid.sum()), excluded_samples=int((~valid).sum()))
        delta = fq[valid] - ref[valid]
        if delta.size:
            result["joint_error_rad"] = [
                {"joint": i + 1, "rmse": float(np.sqrt(np.mean(delta[:, i] ** 2))),
                 "abs_p95": float(np.percentile(np.abs(delta[:, i]), 95)),
                 "abs_max": float(np.abs(delta[:, i]).max())} for i in range(7)]
        write_signals(out / f"commands_{arm}.csv", ct, cq, f"target_{arm}")
        write_signals(out / f"feedback_{arm}.csv", ft, fq, f"measured_{arm}")
        with (out / f"matched_{arm}.csv").open("w", newline="", encoding="utf-8") as handle:
            w = csv.writer(handle)
            w.writerow(["time_sec", "reference_age_sec", *[f"error_j{i+1}_rad" for i in range(7)]])
            w.writerows([t, a, *e] for t, a, e in zip(ft[valid], age[valid], delta))
        if plots:
            fig, axes = plt.subplots(7, 2, figsize=(15, 16), sharex=True)
            for i in range(7):
                axes[i, 0].step(ct, cq[:, i], where="post", label="sent target")
                axes[i, 0].plot(ft, fq[:, i], ".-", markersize=2, label="feedback (query end)")
                axes[i, 0].set_ylabel(f"J{i+1} rad")
                axes[i, 1].plot(ft[valid], delta[:, i] if delta.size else [], ".", markersize=2)
                axes[i, 1].set_ylabel("error rad")
                for b in boundary_times:
                    for ax in axes[i]:
                        ax.axvline(b, color="gray", alpha=0.2)
            axes[0, 0].legend()
            axes[0, 1].set_title("Matched samples only; unshifted time axis")
            axes[-1, 0].set_xlabel("monotonic elapsed seconds")
            axes[-1, 1].set_xlabel("monotonic elapsed seconds")
            label = "SYNTHETIC DEMO — " if report["synthetic_data"] else ""
            fig.suptitle(f"{label}{arm}: gray lines = chunk installations")
            fig.tight_layout()
            fig.savefig(out / f"tracking_{arm}.png", dpi=130)
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(ct[1:], np.diff(ct) * 1000, ".-", label="send interval")
            ax.plot(ft[1:], np.diff(ft) * 1000, ".-", label="feedback interval")
            ax.set(xlabel="seconds", ylabel="milliseconds", title=arm)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / f"timing_{arm}.png", dpi=130)
            plt.close(fig)
    pairs = {}
    for r in commands:
        pairs.setdefault(r["command_id"], {})[r["arm"]] = float(r["send_start_sec"])
    skew = [v["left"] - v["right"] for v in pairs.values() if "left" in v and "right" in v]
    report["left_minus_right_send_start_sec"] = stats(skew)
    report["dual_arm_skew_semantics"] = "SDK submission gap, not physical synchronization error"
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="run/trajectory_fidelity directory")
    parser.add_argument("--max-gap", type=float, default=0.3)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.max_gap <= 0:
        parser.error("--max-gap must be positive")
    report = analyze(args.directory, max_gap=args.max_gap, plots=not args.no_plots)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
