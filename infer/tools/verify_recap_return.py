#!/usr/bin/env python3
"""Hard-gate validation for RECAP inference return data."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def require_columns(table, name: str, columns: set[str]) -> None:
    missing = sorted(columns - set(table.column_names))
    if missing:
        raise RuntimeError(f"{name}: missing columns: {missing}")


def finite_vector_column(table, name: str, dim: int) -> None:
    values = table[name].to_pylist()
    for i, value in enumerate(values):
        if value is None or len(value) != dim or not np.isfinite(value).all():
            raise RuntimeError(f"{name}: invalid row {i}, expected finite dim={dim}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run = args.run_dir
    steps = pq.read_table(run / "record/steps.parquet")
    ticks = pq.read_table(run / "record/ticks.parquet")

    require_columns(
        steps,
        "steps",
        {"tick_id", "task_id", "step_index", "t_send", "t_state_pre", "t_state_post",
         "state_pre", "state_post", "action_cmd", "t_model_in", "t_model_out"},
    )
    require_columns(
        ticks,
        "ticks",
        {"tick_id", "task_id", "t_wall", "t_state", "t_image", "state", "raw_action", "post_action"},
    )
    if steps.num_rows == 0 or ticks.num_rows == 0:
        raise RuntimeError("empty steps or ticks table")
    # One model tick expands to multiple robot execution steps, so steps.tick_id
    # is intentionally repeatable. The tick table itself must remain unique.
    if len(set(ticks["tick_id"].to_pylist())) != ticks.num_rows:
        raise RuntimeError("ticks.tick_id is not unique")
    finite_vector_column(steps, "state_pre", 16)
    # state_post can be absent on the final command because there is no
    # subsequent feedback sample; state_pre is the RECAP input state.
    finite_vector_column(steps, "action_cmd", 16)
    finite_vector_column(ticks, "state", 16)
    for name in ("raw_action", "post_action"):
        for i, value in enumerate(ticks[name].to_pylist()):
            if value is None or len(value) == 0 or any(len(row) != 16 or not np.isfinite(row).all() for row in value):
                raise RuntimeError(f"{name}: invalid row {i}, expected finite [horizon,16]")
    for name in ("t_send", "t_state_pre", "t_model_in", "t_model_out"):
        if not np.isfinite(np.asarray(steps[name].to_numpy())).all():
            raise RuntimeError(f"steps.{name}: non-finite timestamp")
    for name in ("t_wall", "t_state", "t_image"):
        if not np.isfinite(np.asarray(ticks[name].to_numpy())).all():
            raise RuntimeError(f"ticks.{name}: non-finite timestamp")
    tick_ids = set(ticks["tick_id"].to_pylist())
    missing_tick = sorted(set(steps["tick_id"].to_pylist()) - tick_ids)
    if missing_tick:
        raise RuntimeError(f"steps have no matching tick rows: {missing_tick[:10]}")
    missing_post = sum(v is None for v in steps["state_post"].to_pylist())
    print(f"RECAP_RETURN_OK run={run} ticks={ticks.num_rows} steps={steps.num_rows} "
          f"action_dim=16 missing_state_post={missing_post}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
