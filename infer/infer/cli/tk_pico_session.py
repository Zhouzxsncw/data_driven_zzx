"""Prepare TK's PICO stream, launch one RM75 client, and clean up owned processes."""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import yaml

from infer.io.pico_input import PicoSubscriber


def active_inference_clients(proc_root: Path = Path("/proc")) -> list[int]:
    """Identify the Python module a process is executing from its argv."""
    modules = {b"infer.cli.run_async_infer", b"infer.cli.run_infer"}
    matches = []
    for entry in proc_root.iterdir():
        if not entry.name.isdecimal() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        executable = os.path.basename(argv[0])
        if not re.fullmatch(rb"python(?:[0-9]+(?:\.[0-9]+)*)?", executable):
            continue
        flags = iter(argv[1:])
        for arg in flags:
            if arg == b"-m":
                if next(flags, None) in modules:
                    matches.append(int(entry.name))
                break
            if arg in (b"-c", b"--") or not arg.startswith(b"-"):
                break
            if arg in (b"-W", b"-X", b"--check-hash-based-pycs"):
                next(flags, None)
    return sorted(matches)


def listening(port: int) -> bool:
    result = subprocess.run(
        ["ss", "-H", "-ltn", f"sport = :{port}"],
        check=True, capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


def stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        proc.wait()
        return
    try:
        proc.wait(timeout=12.)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def start(stack: ExitStack, argv: list[str], **kwargs) -> subprocess.Popen:
    process = subprocess.Popen(argv, start_new_session=True, **kwargs)
    stack.callback(stop_process, process)
    return process


def check_child(proc: subprocess.Popen | None, name: str) -> None:
    if proc is not None and proc.poll() is not None:
        raise RuntimeError(f"{name}已退出，检查 pico/state 中的日志")


def prepare_pico(stack: ExitStack, root: Path, endpoint: str, timeout: float) -> None:
    host, port_text = endpoint.removeprefix("tcp://").rsplit(":", 1)
    if not endpoint.startswith("tcp://") or host != "127.0.0.1":
        raise ValueError("自动启动只支持本机 PICO TCP 端口")
    port = int(port_text)
    state = root / "pico" / "state"
    state.mkdir(parents=True, exist_ok=True)
    service = publisher = None
    deadline = time.monotonic() + timeout
    if not listening(port):
        if not (listening(60061) or listening(63901)):
            output = stack.enter_context((state / "pc_service.log").open("a"))
            service = start(stack, ["bash", str(root / "run_pico_service_zzx.sh")],
                            stdout=output, stderr=subprocess.STDOUT)
            while not (listening(60061) or listening(63901)):
                check_child(service, "PC Service")
                if time.monotonic() >= deadline:
                    raise RuntimeError("PC Service 启动超时，检查 pico/state/pc_service.log")
                time.sleep(0.2)

        output = stack.enter_context((state / "publisher.log").open("a"))
        publisher = start(stack, ["bash", str(root / "run_pico_zzx.sh"), "--port", str(port)],
                          stdout=output, stderr=subprocess.STDOUT)

    subscriber = PicoSubscriber(endpoint)
    try:
        print("[PICO] 等待头显连接 TK、应用启用头显和双手柄追踪。", flush=True)
        while time.monotonic() < deadline:
            check_child(service, "PC Service")
            check_child(publisher, "PICO 发布器")
            if subscriber.poll().frame is not None:
                return
            time.sleep(0.02)
        raise RuntimeError("未收到有效 PICO 帧，检查头显连接和 pico/state/publisher.log")
    finally:
        subscriber.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=90.)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not argv or not 0 < args.timeout < float("inf"):
        parser.error("需要有效的等待秒数和待运行命令")

    def stop(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        with ExitStack() as stack:
            lock = stack.enter_context((args.root / ".rm75-control.lock").open("a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("已有独立遥操或推理客户端控制机械臂") from exc
            clients = active_inference_clients()
            if clients:
                pids = ", ".join(map(str, clients))
                raise RuntimeError(f"已有推理客户端运行（PID {pids}），请先结束当前任务")
            cfg = yaml.safe_load(args.config.read_text())
            endpoint = cfg.get("pico_recovery", {}).get("endpoint", "tcp://127.0.0.1:5555")
            prepare_pico(stack, args.root, endpoint, args.timeout)
            client = start(stack, argv)
            return client.wait()
    except KeyboardInterrupt:
        return 130
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"[PICO] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
