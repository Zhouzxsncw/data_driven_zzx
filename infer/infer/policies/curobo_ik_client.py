"""Local cuRobo subprocess; no model-server RPC or robot control IO."""
import atexit
import json
import os
from pathlib import Path
import selectors
import subprocess
import time
import numpy as np


class CuroboIKClient:
    def __init__(self):
        python=os.environ.get('CUROBO_PYTHON', '/home/xk/miniconda3/envs/RoboTwin/bin/python')
        script=Path(__file__).resolve().parents[3]/'recipes/gr00t/curobo/rm75_ik.py'
        self.process=subprocess.Popen([python,'-u',str(script)],stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE,bufsize=0)
        self.buffer=b''
        atexit.register(self.close)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()

    def solve(self,ee,seed,n_exec,cap):
        request={'ee':np.asarray(ee).tolist(),'seed':np.asarray(seed).tolist(),
                 'n_exec':int(n_exec),'cap':float(cap)}
        try:
            self.process.stdin.write((json.dumps(request,allow_nan=False)+'\n').encode())
            deadline=time.monotonic()+180
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdout,selectors.EVENT_READ)
                while time.monotonic()<deadline:
                    while b'\n' in self.buffer:
                        line,self.buffer=self.buffer.split(b'\n',1)
                        if not line.startswith(b'CUROBO_RPC '):continue
                        response=json.loads(line[len(b'CUROBO_RPC '):])
                        if 'error' in response:raise RuntimeError(response['error'])
                        return np.asarray(response['actions'],dtype=np.float32),response['info']
                    if selector.select(timeout=min(1,max(0,deadline-time.monotonic()))):
                        chunk=os.read(self.process.stdout.fileno(),65536)
                        if not chunk:raise RuntimeError('cuRobo worker exited')
                        self.buffer+=chunk
            raise TimeoutError('Local cuRobo solve timed out')
        except Exception:
            self.close()
            raise
