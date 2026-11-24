#!/usr/bin/env python3
"""在 cigai20 (192.168.1.12) 上把下位机夹爪 30016 转发到 gpu-master 192.168.1.14。"""
from __future__ import annotations

import socket

LISTEN = ("192.168.1.12", 30016)
DEST = ("192.168.1.14", 30016)


def main() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(LISTEN)
    print(f"relay {LISTEN[0]}:{LISTEN[1]} -> {DEST[0]}:{DEST[1]}", flush=True)
    while True:
        data, _addr = sock.recvfrom(4096)
        sock.sendto(data, DEST)


if __name__ == "__main__":
    raise SystemExit(main())
