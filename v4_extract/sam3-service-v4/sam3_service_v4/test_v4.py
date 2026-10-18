"""Integration tests without SAM3 weights or GPU.

Uses real FastAPI ASGI, real FFmpeg for truncation, and fake backends
that explicitly model model-load/session/open/close/inference lifecycles.
"""
from __future__ import annotations

import base64
import io
import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from starlette.websockets import WebSocketDisconnect

from sam3_service_v4.api import create_app, decode_image, ffprobe_count
from sam3_service_v4.core import FrameMasks, Settings, normalize_frame_response


class FakeBackend:
    name = "fake-sam3"
    def __init__(self, streaming=True, fail_at=None, loading_event=None):
        self.supports_streaming = streaming
        self.load_calls = 0
        self.new_stream_calls = 0
        self.close_calls = 0
        self.batch_calls = 0
        self.shutdown_calls = 0
        self.frames_processed = 0
        self.fail_at = fail_at
        self.loading_event = loading_event

    def load(self):
        self.load_calls += 1
        if self.loading_event is not None:
            self.loading_event.wait(timeout=10)

    @staticmethod
    def _result(i, h, w):
        mask = np.zeros((h, w), dtype=bool)
        mask[5:15, 10:25] = True
        return FrameMasks(i, [3], [.81], [mask])

    def segment_video(self, path, prompt, threshold):
        self.batch_calls += 1
        cap = cv2.VideoCapture(path)
        assert cap.isOpened()
        try:
            index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if self.fail_at is not None and index == self.fail_at:
                    raise RuntimeError("intentional inference failure")
                yield self._result(index, frame.shape[0], frame.shape[1])
                index += 1
        finally:
            cap.release()
            self.close_calls += 1

    def new_stream(self, prompt):
        self.new_stream_calls += 1
        return {"count": 0}

    def process_frame(self, session, image, threshold):
        i = session["count"]
        session["count"] = i + 1
        self.frames_processed += 1
        return self._result(i, image.height, image.width)

    def close_stream(self, session):
        self.close_calls += 1

    def shutdown(self):
        self.shutdown_calls += 1


@pytest.fixture(scope="session")
def sample_mp4():
    import tempfile
    with tempfile.TemporaryDirectory(prefix="sam3_v4_fixture_") as directory:
        path = str(Path(directory) / "fixture.mp4")
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (64, 48))
        assert out.isOpened()
        for i in range(24):
            frame = np.full((48, 64, 3), 20 + i * 4, dtype=np.uint8)
            cv2.rectangle(frame, (10+i%8, 5), (24+i%8, 17), (200, 200, 200), -1)
            out.write(frame)
        out.release()
        assert ffprobe_count(path) == 24
        yield Path(path).read_bytes()


@pytest.fixture(scope="session")
def sample_png():
    img = Image.new("RGB", (64, 48), (23, 44, 77))
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()


def make_app(fake, *, key="testsecret", allow_unauth=False, max_upload=None, max_frames=400):
    settings = Settings(
        engine="hf" if fake.supports_streaming else "meta",
        hf_model="mock-HF-weight",
        device="cpu",
        api_key=key,
        allow_unauth_local=allow_unauth,
        max_frames=max_frames,
        max_upload_bytes=max_upload or 1024 * 1024,
        idle_timeout_s=2.0,
        max_session_s=20.0,
        max_frame_pixels=64 * 48,
    )
    return create_app(settings, backend_factory=lambda _: fake)


def ready(client):
    for _ in range(300):
        r = client.get("/health/ready")
        if r.status_code == 200:
            return
        time.sleep(.01)
    pytest.fail("backend failed to become ready")


def test_health_and_one_load():
    fake = FakeBackend()
    app = make_app(fake)
    with TestClient(app) as client:
        ready(client)
        assert client.get("/health/live").status_code == 200
        assert client.get("/v1/capabilities").json()["true_incremental_streaming"]
        assert fake.load_calls == 1
    assert fake.shutdown_calls == 1


def test_not_ready_until_startup_load_finishes():
    gate = threading.Event()
    fake = FakeBackend(loading_event=gate)
    app = make_app(fake)
    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        first = client.get("/health/ready")
        assert first.status_code == 503
        assert first.json()["status"] in ("starting", "loading")
        gate.set()
        ready(client)
    assert fake.load_calls == 1


@pytest.mark.parametrize("length", [1, 5, 10, 24])
def test_exact_batch_frame_limit(length, sample_mp4):
    fake = FakeBackend(streaming=False)
    app = make_app(fake)
    with TestClient(app) as client:
        ready(client)
        response = client.post(
            "/video/segment",
            data={"text_prompt": "brush", "max_frames": str(length),
                  "frame_stride": "1", "return_masks_png": "false"},
            files={"file": ("original.mp4", sample_mp4, "video/mp4")},
            headers={"Authorization": "Bearer testsecret"},
        )
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["num_frames"] == length
        assert result["frame_indices"] == list(range(length))
        assert result["frames_with_masks"] == length
        assert result["num_objects"] == 1
        assert result["object_scores"] == [.81] * length
        assert fake.load_calls == 1
        assert fake.close_calls == 1


def test_frame_stride_and_original_indices(sample_mp4):
    fake = FakeBackend(streaming=False)
    with TestClient(make_app(fake)) as client:
        ready(client)
        r = client.post("/video/segment",
            data={"text_prompt": "brush", "max_frames": "6", "frame_stride": "3"},
            files={"file": ("video.mp4", sample_mp4, "video/mp4")},
            headers={"Authorization": "Bearer testsecret"})
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["frame_indices"] == [0, 3, 6, 9, 12, 15]
        assert d["processed_frame_indices"] == list(range(6))


def test_zip_masks_and_metadata(sample_mp4):
    fake = FakeBackend(streaming=False)
    with TestClient(make_app(fake)) as client:
        ready(client)
        response = client.post(
            "/video/segment",
            data={"text_prompt": "brush", "max_frames": "10",
                  "return_masks_png": "true"},
            files={"file": ("original.mp4", sample_mp4, "video/mp4")},
            headers={"Authorization": "Bearer testsecret"},
        )
        assert response.status_code == 200, response.text
        import zipfile
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            files = archive.namelist()
            assert len(files) == 11
            assert archive.testzip() is None
            metadata = json.loads(archive.read("meta.json"))
            assert metadata["num_frames"] == 10
            assert metadata["num_objects"] == 1
            assert metadata["frame_indices"] == list(range(10))
            mask_file = next(x for x in files if x.startswith("frame_0000_"))
            with Image.open(io.BytesIO(archive.read(mask_file))) as image:
                assert image.size == (64, 48)


def test_http_bearer_auth(sample_mp4):
    fake = FakeBackend(streaming=False)
    with TestClient(make_app(fake)) as client:
        ready(client)
        common = {"data": {"text_prompt": "brush"},
                  "files": {"file": ("video.mp4", sample_mp4, "video/mp4")}}
        assert client.post("/video/segment", **common).status_code == 401
        assert client.post("/video/segment", headers={"Authorization": "Bearer bad"},
                           **common).status_code == 401
        assert fake.batch_calls == 0


def test_unauth_local_rejects_remote_clients(sample_mp4):
    fake = FakeBackend(streaming=False)
    app = make_app(fake, key="", allow_unauth=True)
    with TestClient(app, client=("203.0.113.7", 54321)) as client:
        ready(client)
        r = client.post("/video/segment",
            data={"text_prompt": "brush"},
            files={"file": ("v.mp4", sample_mp4, "video/mp4")})
        assert r.status_code == 403
    with TestClient(app, client=("127.0.0.1", 4321)) as client:
        ready(client)
        r = client.post("/video/segment",
            data={"text_prompt": "brush", "max_frames": "1"},
            files={"file": ("v.mp4", sample_mp4, "video/mp4")})
        assert r.status_code == 200


def test_batch_bad_video_and_oversize_are_rejected():
    fake = FakeBackend(streaming=False)
    with TestClient(make_app(fake, max_upload=1000)) as client:
        ready(client)
        hdr={"Authorization": "Bearer testsecret"}
        r = client.post("/video/segment", data={"text_prompt": "brush"},
            files={"file": ("huge.mp4", b"X"*2000, "video/mp4")}, headers=hdr)
        assert r.status_code == 413
        r = client.post("/video/segment", data={"text_prompt": "brush"},
            files={"file": ("bad.mp4", b"not a video", "video/mp4")}, headers=hdr)
        assert r.status_code == 422


def test_batch_inference_failure_releases_resources(sample_mp4):
    fake = FakeBackend(streaming=False, fail_at=2)
    with TestClient(make_app(fake)) as client:
        ready(client)
        r = client.post("/video/segment",
            data={"text_prompt": "brush", "max_frames": "5"},
            files={"file": ("video.mp4", sample_mp4, "video/mp4")},
            headers={"Authorization": "Bearer testsecret"})
        assert r.status_code == 500
        assert fake.close_calls == 1
        assert client.get("/health/ready").status_code == 200


def test_true_incremental_websocket(sample_png):
    fake=FakeBackend(streaming=True)
    with TestClient(make_app(fake)) as client:
        ready(client)
        with client.websocket_connect("/v1/video/stream",
             headers={"Authorization": "Bearer testsecret"}) as ws:
            ws.send_json({"type": "start", "prompt": "brush", "max_frames": 3,
                          "return_masks_png": True})
            greeting=ws.receive_json()
            assert greeting["type"] == "ready"
            assert greeting["ack_required"]
            for i in range(3):
                ws.send_json({"type": "frame_meta", "frame_index": i,
                              "timestamp_ms": i*50})
                ws.send_bytes(sample_png)
                res=ws.receive_json()
                assert res["type"] == "frame_result"
                assert res["frame_index"] == i
                assert res["timestamp_ms"] == i*50
                assert res["next_frame_index"] == i+1
                obj=res["objects"][0]
                assert obj["mask_pixels"] == 150
                with Image.open(io.BytesIO(base64.b64decode(obj["mask_png_base64"]))) as img:
                    assert img.size == (64,48)
            done=ws.receive_json()
            assert done == {"type": "done", "frames_processed": 3,
                            "reason": "frame_limit"}
        assert fake.frames_processed == 3
        assert fake.new_stream_calls == 1
        assert fake.close_calls == 1


def test_ws_cancel_disposes_state(sample_png):
    fake=FakeBackend(streaming=True)
    with TestClient(make_app(fake)) as client:
        ready(client)
        with client.websocket_connect("/video/stream",
             headers={"Authorization": "Bearer testsecret"}) as ws:
            ws.send_json({"type":"start","prompt":"brush"})
            assert ws.receive_json()["type"]=="ready"
            ws.send_bytes(sample_png)
            assert ws.receive_json()["type"]=="frame_result"
            ws.send_json({"type":"cancel"})
            assert ws.receive_json()["reason"]=="cancel"
        assert fake.close_calls == 1


def test_ws_bad_order_and_bad_image_rejected(sample_png):
    fake=FakeBackend(streaming=True)
    with TestClient(make_app(fake)) as client:
        ready(client)
        with client.websocket_connect("/video/stream",
             headers={"Authorization": "Bearer testsecret"}) as ws:
            ws.send_json({"type":"start","prompt":"brush"})
            ws.receive_json()
            ws.send_json({"type":"frame_meta","frame_index":9})
            err=ws.receive_json()
            assert err["code"]=="BAD_FRAME_OR_SESSION"
        assert fake.close_calls == 1
        with client.websocket_connect("/video/stream",
             headers={"Authorization": "Bearer testsecret"}) as ws:
            ws.send_json({"type":"start","prompt":"brush"})
            ws.receive_json()
            ws.send_bytes(b"invalid")
            assert ws.receive_json()["code"]=="BAD_FRAME_OR_SESSION"
        assert fake.close_calls == 2


def test_meta_batch_cannot_fake_streaming():
    fake=FakeBackend(streaming=False)
    with TestClient(make_app(fake)) as client:
        ready(client)
        with client.websocket_connect("/video/stream",
             headers={"Authorization": "Bearer testsecret"}) as ws:
            result=ws.receive_json()
            assert result["code"]=="UNSUPPORTED_BACKEND"
        assert fake.new_stream_calls == 0


def test_ws_auth_failure():
    fake=FakeBackend(streaming=True)
    with TestClient(make_app(fake)) as client:
        ready(client)
        with client.websocket_connect("/video/stream") as ws:
            assert ws.receive_json()["code"]=="UNAUTHORIZED"
        assert fake.new_stream_calls == 0


def test_parse_nested_meta_schema():
    m=np.zeros((1,48,64),dtype=bool)
    m[0,5:15,10:25]=True
    raw={"frame_index":0,"outputs":{
        "out_binary_masks":m,"out_obj_ids":np.array([3]),
        "out_probs":np.array([.81],dtype=np.float32)}}
    frame=normalize_frame_response(raw,16)
    assert frame.object_ids==[3]
    assert frame.as_json()["objects"][0]["mask_pixels"]==150
    with pytest.raises(ValueError, match="outputs"):
        normalize_frame_response({"frame_index":0,"out_binary_masks":m},16)


def test_image_dimension_cap(sample_png):
    cfg=Settings(engine="meta",api_key="x",max_frame_pixels=10)
    with pytest.raises(ValueError,match="max_frame_pixels"):
        decode_image(sample_png,cfg)


def test_missing_hf_weights_refuses_success():
    cfg = Settings(engine="hf",api_key="x",device="cpu",hf_model="")
    with pytest.raises(ValueError, match="HF_SAM3_MODEL"):
        cfg.validate()


def test_physical_frame_cap_applies_even_without_requested_limit(sample_mp4):
    fake = FakeBackend(streaming=False)
    with TestClient(make_app(fake, max_frames=10)) as client:
        ready(client)
        result = client.post(
            "/video/segment", data={"text_prompt":"brush"},
            files={"file":("input.mp4",sample_mp4,"video/mp4")},
            headers={"Authorization":"Bearer testsecret"},
        )
        assert result.status_code == 422
        assert "max_frames safety cap" in result.text
        assert fake.batch_calls == 0


def test_origin_protection_blocks_cross_site_browser(sample_mp4):
    fake = FakeBackend(streaming=False)
    app = make_app(fake, key="", allow_unauth=True)
    with TestClient(app, client=("127.0.0.1", 4444)) as client:
        ready(client)
        r = client.post("/video/segment",
            data={"text_prompt": "brush", "max_frames": "1"},
            files={"file": ("v.mp4", sample_mp4, "video/mp4")},
            headers={"Origin": "https://attacker.example"})
        assert r.status_code == 403
    with TestClient(make_app(FakeBackend(streaming=True), key="", allow_unauth=True),
                    client=("127.0.0.1", 4444)) as client:
        ready(client)
        with client.websocket_connect("/video/stream",
             headers={"Origin": "https://attacker.example"}) as ws:
            assert ws.receive_json()["code"] == "UNAUTHORIZED"


def test_old_meta_predictor_session_adapter():
    from sam3_service_v4.backends import MetaBatchBackend

    class DummyModel:
        def init_state(self, resource_path, offload_video_to_cpu=False):
            return {"path": resource_path}

    class OldPredictor:
        model = DummyModel()
        def __init__(self):
            self._ALL_INFERENCE_STATES = {}
        def start_session(self, resource_path, offload_state_to_cpu=False):
            raise AssertionError("Legacy method should be patched")

    old = OldPredictor()
    MetaBatchBackend._patch_session_if_needed(old)
    session = old.start_session("synthetic.mp4")
    assert session["session_id"] in old._ALL_INFERENCE_STATES
    assert old._ALL_INFERENCE_STATES[session["session_id"]]["state"]["path"] == "synthetic.mp4"
    with pytest.raises(ValueError, match="unsupported"):
        old.start_session("synthetic.mp4", offload_state_to_cpu=True)


def test_hf_adapter_streams_image_without_preloading_video():
    import torch
    from types import SimpleNamespace
    from sam3_service_v4.backends import HFIncrementalBackend

    class MockProcessor:
        def init_video_session(self, **kwargs):
            assert "video" not in kwargs
            assert kwargs["video_storage_device"] == "cpu"
            return SimpleNamespace(counter=0)
        def add_text_prompt(self, inference_session, text):
            assert text == "brush"
        def __call__(self, images, return_tensors):
            return SimpleNamespace(
                pixel_values=torch.ones(1,3,images.height,images.width),
                original_sizes=[[images.height, images.width]]
            )
        def postprocess_outputs(self, session, outputs, original_sizes):
            h,w = original_sizes[0]
            return {
                "object_ids":torch.tensor([17]),
                "scores":torch.tensor([.92]),
                "masks":torch.ones(1,h,w,dtype=torch.bool)
            }
    class MockModel:
        def __call__(self, inference_session, frame, reverse):
            inference_session.counter+=1
            return {"frame_processed":True}

    backend=HFIncrementalBackend(Settings(engine="hf", hf_model="dummy",
                                          device="cpu", api_key="secret"))
    backend.processor=MockProcessor()
    backend.model=MockModel()
    backend.torch=torch
    backend.dtype=torch.float32
    session=backend.new_stream("brush")
    image=Image.new("RGB",(64,48))
    for i in range(3):
        frame=backend.process_frame(session,image,0.1)
        assert frame.object_ids == [17]
        assert frame.as_json()["objects"][0]["mask_pixels"] == 64*48
    assert session.counter == 3
    backend.close_stream(session)


def test_ws_rejects_frame_too_large(sample_png):
    fake=FakeBackend(streaming=True)
    cfg=Settings(engine="hf",hf_model="dummy",device="cpu",api_key="testsecret",
                 max_frame_bytes=32,max_frame_pixels=64*48)
    with TestClient(create_app(cfg,backend_factory=lambda _:fake)) as client:
        ready(client)
        with client.websocket_connect("/video/stream",
             headers={"Authorization":"Bearer testsecret"}) as ws:
            ws.send_json({"type":"start","prompt":"brush"})
            assert ws.receive_json()["type"] == "ready"
            ws.send_bytes(sample_png)
            assert ws.receive_json()["code"] == "BAD_FRAME_OR_SESSION"
        assert fake.frames_processed == 0
        assert fake.close_calls == 1
