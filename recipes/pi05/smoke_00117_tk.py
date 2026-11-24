#!/usr/bin/env python3
"""Read-only tk smoke test. Never sends arm/gripper motion commands."""
import argparse
import json
import math
import socket
import subprocess
import sys
import time
from pathlib import Path
import yaml

p = argparse.ArgumentParser()
p.add_argument('--root', default='/home/tk/data_Driven')
p.add_argument('--idle-hardware', action='store_true', help='Open cameras, read SDK state, receive UDP only when collection/control is stopped')
p.add_argument('--output', default=None)
a = p.parse_args()
root = Path(a.root)
config = root/'infer/infer/config/lerobot/pi05_00117_compare_step001302_tk_ws.yaml'
cfg = yaml.safe_load(config.read_text())
report = {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'config': str(config), 'checks': {}}
checks = report['checks']
def result(name, status, **details):
    checks[name] = dict(status=status, **details)
    print(name, json.dumps(checks[name], ensure_ascii=False), flush=True)
def run(argv, timeout=15):
    q = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    return q.returncode, q.stdout + q.stderr
procs = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True)
busy = [x for x in procs.splitlines() if any(k in x for k in ('run_collect.py', 'run_batch_infer.py', 'exo_read_mode_lock', 'new_exo_control')) and 'smoke_00117' not in x]
result('active_hardware_processes', 'BUSY' if busy else 'IDLE', processes=busy)
for host in ('192.168.1.17', '192.168.1.18', '192.168.1.13'):
    code, text = run(['ip', 'route', 'get', host])
    result('route_'+host, 'PASS' if code == 0 and 'src 192.168.1.12' in text else 'FAIL', evidence=text.strip())
for name, host, port in [('model_tcp', '127.0.0.1', 1297), ('right_arm_tcp', '192.168.1.17', 8080), ('left_arm_tcp', '192.168.1.18', 8080)]:
    try:
        with socket.create_connection((host, port), timeout=3): pass
        result(name, 'PASS', host=host, port=port, limitation='TCP connection only')
    except Exception as e: result(name, 'FAIL', error=str(e))
try:
    import pyrealsense2 as rs
    serials = [d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()]
    expected = [d['serial'] for d in cfg['camera_sender']['devices']]
    result('camera_enumeration', 'PASS' if set(expected) <= set(serials) else 'FAIL', expected=expected, detected=serials)
except Exception as e: result('camera_enumeration', 'FAIL', error=str(e))
if any('run_batch_infer.py' in x for x in busy):
    result('model_websocket', 'SKIP', reason='An actual inference client is running')
else:
    try:
        code, text = run([sys.executable, str(root/'recipes/pi05/test_pi05_client.py'), '--port', '1297'], timeout=120)
        result('model_websocket', 'PASS' if code == 0 and 'PASS:' in text else 'FAIL', evidence=text)
    except Exception as e: result('model_websocket', 'FAIL', error=str(e))
if not a.idle_hardware or busy:
    result('idle_hardware_tests', 'SKIP', reason='Requires --idle-hardware and no collection/control process; no ports or cameras were opened')
else:
    try:
        sys.path.insert(0, cfg['direct_robot']['sdk_path'])
        from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
        for i, side in enumerate(('right', 'left')):
            arm = RoboticArm(rm_thread_mode_e(2)) if i == 0 else RoboticArm()
            host = cfg['direct_robot']['robot_r_host' if side == 'right' else 'robot_l_host']
            handle = arm.rm_create_robot_arm(host, 8080)
            if handle.id <= 0:
                result(side+'_arm_sdk', 'FAIL', handle=handle.id)
                continue
            try:
                samples = []
                for _ in range(3):
                    code, state = arm.rm_get_current_arm_state()
                    joints = state.get('joint', [])
                    if code != 0 or len(joints) != 7 or not all(math.isfinite(float(x)) for x in joints):
                        raise RuntimeError(str((code, state)))
                    samples.append(state)
                    time.sleep(.1)
                result(side+'_arm_sdk', 'PASS', samples=samples, limitation='Read-only; motion untested')
            finally: arm.rm_delete_robot_arm()
    except Exception as e: result('arm_sdk', 'FAIL', error=str(e))
    pipelines = []
    try:
        for d in cfg['camera_sender']['devices']:
            pipeline, settings = rs.pipeline(), rs.config()
            settings.enable_device(d['serial'])
            settings.enable_stream(rs.stream.color, d['width'], d['height'], rs.format.rgb8, d['fps'])
            pipeline.start(settings)
            pipelines.append((d, pipeline))
        for d, pipeline in pipelines:
            samples = []
            for _ in range(3):
                frame = pipeline.wait_for_frames(10000).get_color_frame()
                if not frame: raise RuntimeError('Missing color frame')
                samples.append(dict(number=frame.get_frame_number(), timestamp_ms=frame.get_timestamp(), width=frame.get_width(), height=frame.get_height()))
            if len(set(x['number'] for x in samples)) != 3: raise RuntimeError('Frame numbers did not advance')
            result('camera_frames_'+d['key'], 'PASS', serial=d['serial'], frames=samples)
    except Exception as e: result('camera_frames', 'FAIL', error=str(e))
    finally:
        for _, pipeline in pipelines: pipeline.stop()
    for name, port in [('gripper_feedback', 30016), ('exo_feedback', 18140)]:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.bind(('192.168.1.12', port))  # Deliberately no SO_REUSEADDR.
            s.settimeout(4)
            packets = []
            for _ in range(3):
                data, sender = s.recvfrom(65535)
                packets.append(dict(sender=sender, bytes=len(data), hex=data[:128].hex()))
            result(name, 'RECEIVED', packets=packets, limitation='Transport verified; payload and physical calibration require separate verification')
        except Exception as e: result(name, 'UNVERIFIED', error=str(e))
        finally: s.close()
result('actuation', 'UNTESTED', reason='No robot, gripper, or exoskeleton command is sent by this script')
out = Path(a.output) if a.output else root/'recipes/pi05/00117_tk_smoke_report.json'
out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
print('Report:', out)
failed = any(v['status'] == 'FAIL' for v in checks.values())
incomplete = a.idle_hardware and any(v['status'] in ('SKIP', 'UNVERIFIED') for v in checks.values())
sys.exit(1 if failed else 2 if incomplete else 0)
