#!/usr/bin/env python3
import argparse
import os
import signal
import subprocess
import sys
import time

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import _bootstrap


def main() -> int:
    parser = argparse.ArgumentParser(description="下位机统一启动")
    parser.add_argument("--gripper-only", action="store_true", help="仅夹爪")
    parser.add_argument("--no-exo", action="store_true", help="不启动外骨骼")
    args = parser.parse_args()

    required = (
        "only_gripper.py",
        "gripper_udp_packet.py",
        "gripper_motor_read.py",
        "exo_kinematics.py",
        "exo_read_safety.py",
        "_bootstrap.py",
        os.path.join("TG_exo", "__init__.py"),
        os.path.join("TG_exo", "tg_exo_2.py"),
    )
    for name in required:
        path = os.path.join(_DIR, name)
        if not os.path.isfile(path):
            print(f"[LAUNCH] 缺少文件: {path}", flush=True)
            return 1

    env = os.environ.copy()
    env["PYTHONPATH"] = _bootstrap.pythonpath_env(env.get("PYTHONPATH"))
    procs: list[subprocess.Popen] = []
    exo_proc: subprocess.Popen | None = None

    gripper = os.path.join(_DIR, "only_gripper.py")
    print("[LAUNCH] only_gripper.py", flush=True)
    procs.append(
        subprocess.Popen(
            [
                sys.executable,
                gripper,
                "--skip-arm-init",
            ],
            env=env,
            cwd=_DIR,
        )
    )

    if not args.gripper_only and not args.no_exo:
        exo = os.path.join(_DIR, "exo_read_safety.py")
        print("[LAUNCH] exo_read_safety.py", flush=True)
        exo_proc = subprocess.Popen(
            [sys.executable, exo],
            env=env,
            cwd=_DIR,
        )
        procs.append(exo_proc)

    shutting_down = False

    def shutdown(signum=None, frame=None):
        nonlocal shutting_down
        if shutting_down:
            return
        shutting_down = True
        print("\n[LAUNCH] 退出...", flush=True)
        for p in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
        deadline = time.time() + 5.0
        for p in procs:
            try:
                p.wait(timeout=max(0.0, deadline - time.time()))
            except subprocess.TimeoutExpired:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    print("[LAUNCH] 已启动，Ctrl+C 退出", flush=True)

    try:
        while True:
            for p in list(procs):
                if p.poll() is None:
                    continue
                code = p.returncode
                if p is exo_proc:
                    print(
                        f"[LAUNCH] exo 进程退出 code={code}，夹爪继续运行",
                        flush=True,
                    )
                    procs.remove(p)
                    exo_proc = None
                    continue
                print(f"[LAUNCH] 夹爪进程退出 code={code}", flush=True)
                shutdown()
            time.sleep(0.5)
    except KeyboardInterrupt:
        shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
