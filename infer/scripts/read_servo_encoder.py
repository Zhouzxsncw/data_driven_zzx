#!/usr/bin/env python3
"""
读取伺服电机当前位置编码值。

默认使用 Feetech SCS/SMS 协议，也支持 Dynamixel Protocol 2.0。

用法：
  python3 read_servo_encoder.py                      # 默认 Feetech, ID=21, /dev/ttyUSB2, 1M
  python3 read_servo_encoder.py --proto dxl          # 使用 Dynamixel 协议
  python3 read_servo_encoder.py --id 5 --port /dev/ttyUSB0 --baud 115200 --proto dxl
  python3 read_servo_encoder.py --loop               # 循环读取（Ctrl+C 退出）

依赖：
  pip install pyserial
"""

import argparse
import struct
import sys
import time


# ---------------------------------------------------------------------------
# 协议定义
# ---------------------------------------------------------------------------

class FeetechProto:
    """Feetech SCS/SMS 系列伺服协议"""

    BROADCAST_ID = 0xFE
    INSTRUCTION_READ = 0x02
    INSTRUCTION_WRITE = 0x03

    # 常用寄存器地址
    ADDR_PRESENT_POSITION = 0x38  # 当前位置（2 字节，只读）

    @staticmethod
    def checksum(data: bytes) -> int:
        return (~sum(data) & 0xFF)

    @classmethod
    def build_read_packet(cls, servo_id: int, register_addr: int, read_len: int) -> bytes:
        """构建读寄存器指令包"""
        params = bytes([register_addr, read_len])
        pkt = struct.pack(
            ">BBBB",
            0xFF,
            0xFF,
            servo_id,
            4,  # length = ID + Length + Instruction + 2 params
        )
        pkt += bytes([cls.INSTRUCTION_READ]) + params
        chk = cls.checksum(pkt[2:])  # 校验从 ID 开始
        return pkt + bytes([chk])

    @classmethod
    def parse_read_response(cls, raw: bytes) -> int | None:
        """
        解析读回应包，返回解码后的值（小端序 2 字节）。
        返回 None 表示解析失败。
        """
        if len(raw) < 6:
            return None
        header, sid, length, status = struct.unpack_from(">BBBB", raw, 0)
        if header != 0xFF:
            return None
        # 第二个 header 是 raw[1]，已在 struct 中作为 sid 之前读入
        # 实际上格式是: 0xFF 0xFF ID Len Status Data... Checksum
        # 上面 struct 读的是 0xFF(header) 0xFF(sid位置) — 需要修正
        return None  # placeholder，下面重写


def read_feetech_position(ser, servo_id: int) -> int | None:
    """
    通过 Feetech 协议读取单个舵机的当前位置。

    回应包格式:
      0xFF 0xFF ID Len Status Param... Checksum
    位置为 Param 的前 2 字节（小端序），范围 0~1023（对应 0~300°）。
    """
    pkt = FeetechProto.build_read_packet(servo_id, FeetechProto.ADDR_PRESENT_POSITION, 2)
    ser.reset_input_buffer()
    ser.write(pkt)
    time.sleep(0.002)  # 2ms 等待回应

    # 预期回应长度: 2(header) + 1(id) + 1(len) + 1(status) + 2(data) + 1(checksum) = 8
    raw = ser.read(8)
    if len(raw) < 7:
        print(f"  [WARN] 回应过短 ({len(raw)} bytes): {raw.hex()}")
        return None

    # 校验包头
    if raw[0] != 0xFF or raw[1] != 0xFF:
        print(f"  [WARN] 包头错误: {raw[:2].hex()}")
        return None

    resp_id = raw[2]
    if resp_id != servo_id:
        print(f"  [WARN] 回应 ID 不匹配: 期望 {servo_id}, 实际 {resp_id}")

    err = raw[4]
    if err != 0:
        print(f"  [WARN] 舵机报错: status=0x{err:02X}")

    # 小端序解析位置值
    pos = raw[5] | (raw[6] << 8)
    return pos


# ---------------------------------------------------------------------------
# Dynamixel Protocol 2.0
# ---------------------------------------------------------------------------

def _dxl_update_crc(crc_accum: int, data_blk: bytes) -> int:
    """DXL 2.0 CRC-16/XMODEM"""
    crc_table = [
        0x0000, 0x8005, 0x800F, 0x000A, 0x801B, 0x001E, 0x0014, 0x8011,
        0x8033, 0x0036, 0x003C, 0x8039, 0x0028, 0x802D, 0x8027, 0x0022,
        0x8063, 0x0066, 0x006C, 0x8069, 0x0078, 0x807D, 0x8077, 0x0072,
        0x0050, 0x8055, 0x805F, 0x005A, 0x804B, 0x004E, 0x0044, 0x8041,
        0x80C3, 0x00C6, 0x00CC, 0x80C9, 0x00D8, 0x80DD, 0x80D7, 0x00D2,
        0x00F0, 0x80F5, 0x80FF, 0x00FA, 0x80EB, 0x00EE, 0x00E4, 0x80E1,
        0x00A0, 0x80A5, 0x80AF, 0x00AA, 0x80BB, 0x00BE, 0x00B4, 0x80B1,
        0x8093, 0x0096, 0x009C, 0x8099, 0x0088, 0x808D, 0x8087, 0x0082,
        0x8183, 0x0186, 0x018C, 0x8189, 0x0198, 0x819D, 0x8197, 0x0192,
        0x01B0, 0x81B5, 0x81BF, 0x01BA, 0x81AB, 0x01AE, 0x01A4, 0x81A1,
        0x01E0, 0x81E5, 0x81EF, 0x01EA, 0x81FB, 0x01FE, 0x01F4, 0x81F1,
        0x81D3, 0x01D6, 0x01DC, 0x81D9, 0x01C8, 0x81CD, 0x81C7, 0x01C2,
        0x0140, 0x8145, 0x814F, 0x014A, 0x815B, 0x015E, 0x0154, 0x8151,
        0x8173, 0x0176, 0x017C, 0x8179, 0x0168, 0x816D, 0x8167, 0x0162,
        0x8123, 0x0126, 0x012C, 0x8129, 0x0138, 0x813D, 0x8137, 0x0132,
        0x0110, 0x8115, 0x811F, 0x011A, 0x810B, 0x010E, 0x0104, 0x8101,
        0x8303, 0x0306, 0x030C, 0x8309, 0x0318, 0x831D, 0x8317, 0x0312,
        0x0330, 0x8335, 0x833F, 0x033A, 0x832B, 0x032E, 0x0324, 0x8321,
        0x0360, 0x8365, 0x836F, 0x036A, 0x837B, 0x037E, 0x0374, 0x8371,
        0x8353, 0x0356, 0x035C, 0x8359, 0x0348, 0x834D, 0x8347, 0x0342,
        0x03C0, 0x83C5, 0x83CF, 0x03CA, 0x83DB, 0x03DE, 0x03D4, 0x83D1,
        0x83F3, 0x03F6, 0x03FC, 0x83F9, 0x03E8, 0x83ED, 0x83E7, 0x03E2,
        0x83A3, 0x03A6, 0x03AC, 0x83A9, 0x03B8, 0x83BD, 0x83B7, 0x03B2,
        0x0390, 0x8395, 0x839F, 0x039A, 0x838B, 0x038E, 0x0384, 0x8381,
        0x0280, 0x8285, 0x828F, 0x028A, 0x829B, 0x029E, 0x0294, 0x8291,
        0x82B3, 0x02B6, 0x02BC, 0x82B9, 0x02A8, 0x82AD, 0x82A7, 0x02A2,
        0x82E3, 0x02E6, 0x02EC, 0x82E9, 0x02F8, 0x82FD, 0x82F7, 0x02F2,
        0x02D0, 0x82D5, 0x82DF, 0x02DA, 0x82CB, 0x02CE, 0x02C4, 0x82C1,
        0x8243, 0x0246, 0x024C, 0x8249, 0x0258, 0x825D, 0x8257, 0x0252,
        0x0270, 0x8275, 0x827F, 0x027A, 0x826B, 0x026E, 0x0264, 0x8261,
        0x0220, 0x8225, 0x822F, 0x022A, 0x823B, 0x023E, 0x0234, 0x8231,
        0x8213, 0x0216, 0x021C, 0x8219, 0x0208, 0x820D, 0x8207, 0x0202,
    ]
    for b in data_blk:
        idx = ((crc_accum >> 8) ^ b) & 0xFF
        crc_accum = ((crc_accum << 8) ^ crc_table[idx]) & 0xFFFF
    return crc_accum


def read_dxl_position(ser, servo_id: int) -> int | None:
    """
    通过 Dynamixel Protocol 2.0 读取 Present Position (addr 132, 4 bytes)。
    """
    # 构建 DXL 2.0 Read 指令包
    # [0xFF,0xFF,0xFD,0x00, ID, Len_L,Len_H, 0x02, Addr_L,Addr_H, DataLen_L,DataLen_H, CRC_L,CRC_H]
    addr = 132  # Present Position
    data_len = 4

    pkt = bytes(
        [
            0xFF,
            0xFF,
            0xFD,
            0x00,  # reserved
            servo_id,
            0x07,  # Len_L  (Instruction + Addr(2) + DataLen(2) + CRC(2) = 1+2+2+2 = 7)
            0x00,  # Len_H
            0x02,  # READ
            addr & 0xFF,
            (addr >> 8) & 0xFF,
            data_len & 0xFF,
            (data_len >> 8) & 0xFF,
        ]
    )
    crc = _dxl_update_crc(0, pkt)
    pkt += struct.pack("<H", crc)

    ser.reset_input_buffer()
    ser.write(pkt)
    time.sleep(0.005)

    # 读取回应 (最大 14 bytes)
    raw = ser.read(14)
    if len(raw) < 11:
        print(f"  [WARN] 回应过短 ({len(raw)} bytes)")
        return None

    if raw[0] != 0xFF or raw[1] != 0xFF or raw[2] != 0xFD:
        print(f"  [WARN] 包头错误: {raw[:4].hex()}")
        return None

    resp_id = raw[4]
    if resp_id != servo_id:
        print(f"  [WARN] 回应 ID 不匹配: 期望 {servo_id}, 实际 {resp_id}")

    err = raw[8]
    if err != 0:
        print(f"  [WARN] 舵机报错: status=0x{err:02X}")

    # 小端序 4 字节
    pos = struct.unpack_from("<i", raw, 9)[0]
    return pos


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="读取舵机编码器当前位置值",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                                       # Feetech, ID=21
  %(prog)s --proto dxl                           # Dynamixel, ID=21
  %(prog)s --id 5 --port /dev/ttyUSB0 --baud 115200
  %(prog)s --loop --interval 0.5                 # 每 0.5s 循环读取
        """,
    )
    parser.add_argument("--port", default="/dev/ttyUSB2", help="串口设备路径 (默认 /dev/ttyUSB2)")
    parser.add_argument("--baud", type=int, default=1000000, help="波特率 (默认 1000000)")
    parser.add_argument("--id", type=int, nargs="+", default=[21], help="舵机 ID，支持多个 (默认 21)")
    parser.add_argument(
        "--proto",
        choices=["feetech", "dxl"],
        default="feetech",
        help="伺服协议: feetech (默认) / dxl (Dynamixel 2.0)",
    )
    parser.add_argument("--loop", action="store_true", help="循环读取模式")
    parser.add_argument("--interval", type=float, default=1.0, help="循环读取间隔秒数 (默认 1.0)")

    args = parser.parse_args()

    # 串口参数校验
    try:
        import serial
    except ImportError:
        print("[ERROR] 缺少 pyserial，请先安装: pip install pyserial", file=sys.stderr)
        sys.exit(1)

    print(f"串口: {args.port}")
    print(f"波特率: {args.baud}")
    print(f"舵机 ID: {args.id}")
    print(f"协议: {args.proto.upper()}")
    print()

    try:
        ser = serial.Serial(
            port=args.port,
            baudrate=args.baud,
            timeout=0.05,
        )
    except Exception as e:
        print(f"[ERROR] 无法打开串口 {args.port}: {e}", file=sys.stderr)
        sys.exit(1)

    read_fn = read_feetech_position if args.proto == "feetech" else read_dxl_position

    try:
        if args.loop:
            print(f"循环读取中（间隔 {args.interval}s），按 Ctrl+C 停止...\n")
            count = 0
            while True:
                count += 1
                ts = time.strftime("%H:%M:%S")
                for sid in args.id:
                    pos = read_fn(ser, sid)
                    if pos is not None:
                        print(f"[{ts}] #{count:04d}  ID={sid}  编码值={pos:6d}  (0x{pos:04X})")
                    else:
                        print(f"[{ts}] #{count:04d}  ID={sid}  读取失败")
                time.sleep(args.interval)
        else:
            for sid in args.id:
                pos = read_fn(ser, sid)
                if pos is not None:
                    print(f"ID={sid}  编码值 = {pos}  (0x{pos:04X})")
                else:
                    print(f"ID={sid}  读取失败", file=sys.stderr)
    except KeyboardInterrupt:
        print("\n中断退出")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
