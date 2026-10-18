#!/usr/bin/env python3
"""Real TCP/WebSocket e2e test with a fake model, no GPU."""
import asyncio
import io
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image
import websockets

def main():
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
    source_root = str(Path(__file__).resolve().parent.parent)
    server_code = f"""
import sys
sys.path.insert(0, {source_root!r})
from sam3_service_v4.api import create_app
from sam3_service_v4.core import Settings
from sam3_service_v4.test_v4 import FakeBackend
import uvicorn
cfg=Settings(engine='hf',hf_model='test-fixture',device='cpu',api_key='testsecret')
app=create_app(cfg,backend_factory=lambda _:FakeBackend(streaming=True))
uvicorn.run(app,host='127.0.0.1',port={port},log_level='error')
"""
    proc = subprocess.Popen([sys.executable, "-c", server_code],
                            env=dict(os.environ, PYTHONPATH=source_root),
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)
    try:
        for attempt in range(100):
            if proc.poll() is not None:
                raise RuntimeError(f"Server exited: {proc.stderr.read()[-600:]}")
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health/ready", timeout=.3
                ) as response:
                    if response.status == 200:
                        break
            except Exception:
                time.sleep(.05)
        else:
            raise RuntimeError("Readiness did not complete")

        async def check():
            frame = io.BytesIO()
            Image.new("RGB", (64, 48), (100, 150, 200)).save(frame, format="PNG")
            async with websockets.connect(
                f"ws://127.0.0.1:{port}/video/stream",
                additional_headers={"Authorization": "Bearer testsecret"},
                proxy=None, open_timeout=4, max_size=4*1024*1024
            ) as ws:
                await ws.send(json.dumps(
                    {"type":"start","prompt":"brush","max_frames":3}))
                ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                assert ready["type"] == "ready", ready
                for i in range(3):
                    await ws.send(frame.getvalue())
                    reply = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                    assert reply["type"] == "frame_result" and reply["frame_index"] == i
                done = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                assert done == {"type":"done","frames_processed":3,"reason":"frame_limit"}
        asyncio.run(check())
        print("REAL_NETWORK_E2E_PASS: 3 JPEG/PNG frames via TCP/WebSocket with ACK")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

if __name__ == "__main__":
    main()
