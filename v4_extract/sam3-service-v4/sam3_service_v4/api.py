"""Production-minded SAM3 API: lifespan warm-up, bounded HTTP, true WS streaming."""
from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import os
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile
from contextlib import asynccontextmanager, closing
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import JSONResponse, Response
from starlette.websockets import WebSocketDisconnect

from .backends import HFIncrementalBackend, MetaBatchBackend
from .core import FrameMasks, Settings, is_local_client
from .model_pool import ModelPool, get_global_pool, shutdown_global_pool
from .workers import BatchWorker, StreamWorker, SingleWorker

log = logging.getLogger("sam3.service")
CHUNK = 1024 * 1024


class NotEnoughFrames(ValueError):
    pass


def ffprobe_count(path: str, timeout_s: float = 30.0) -> int:
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-count_frames", "-show_entries", "stream=nb_read_frames",
               "-of", "default=noprint_wrappers=1:nokey=1", path]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=True,
            timeout=timeout_s
        )
        count = int(result.stdout.strip())
        if count < 1:
            raise ValueError("Video has no decodable frames")
        return count
    except (ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        raise ValueError(f"Cannot verify video frame count via FFprobe: {exc}") from exc


def probe_video(path: str, timeout_s: float = 30.0) -> tuple[int, int, int]:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-count_frames", "-show_entries",
           "stream=nb_read_frames,width,height", "-of", "json", path]
    try:
        response = subprocess.run(cmd, capture_output=True, text=True,
                                  check=True, timeout=timeout_s)
        streams = json.loads(response.stdout)["streams"]
        if len(streams) != 1:
            raise ValueError("Expected one selected video stream")
        row = streams[0]
        count = int(row["nb_read_frames"])
        width = int(row["width"])
        height = int(row["height"])
        if min(count, width, height) <= 0:
            raise ValueError("Invalid stream frame count or resolution")
        return count, width, height
    except (KeyError, IndexError, ValueError, OSError,
            subprocess.CalledProcessError, subprocess.TimeoutExpired,
            json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid or unsupported input video: {exc}") from exc


def prepare_video(source: str, output: str, original_frames: int,
                  max_frames: int | None, stride: int, cfg: Settings) -> tuple[str, int]:
    available = (original_frames + stride - 1) // stride
    requested = max_frames if max_frames is not None else available
    if requested > available:
        raise NotEnoughFrames(f"Requested {requested} frames, but input has {available} sampled frames")
    if stride == 1 and requested == original_frames:
        return source, original_frames

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
           "-y", "-i", source]
    if stride > 1:
        cmd.extend(["-vf", f"select=not(mod(n\\,{stride}))", "-fps_mode", "vfr"])
    cmd.extend(["-frames:v", str(requested), "-an", "-c:v", "libx264",
                "-preset", "ultrafast", "-crf", "16", "-pix_fmt", "yuv420p", output])
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True,
                       timeout=cfg.ffmpeg_timeout_s)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        detail = getattr(exc, "stderr", b"")
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        raise ValueError(f"FFmpeg frame extraction failed: {str(detail)[-500:]}") from exc
    actual = ffprobe_count(output)
    if actual != requested:
        raise ValueError(f"Extracted video has {actual} frames, expected {requested}")
    return output, requested


def decode_image(raw: bytes, cfg: Settings) -> Image.Image:
    if not raw or len(raw) > cfg.max_frame_bytes:
        raise ValueError("Frame exceeds max_frame_bytes or is empty")
    try:
        with Image.open(io.BytesIO(raw)) as file_img:
            if file_img.format not in ("JPEG", "PNG"):
                raise ValueError("Only JPEG/PNG input frames are allowed")
            if file_img.width * file_img.height > cfg.max_frame_pixels:
                raise ValueError("Image exceeds max_frame_pixels")
            image = ImageOps.exif_transpose(file_img).convert("RGB")
            image.load()
            return image
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Invalid encoded image: {exc}") from exc


def _safe_browser_origin(origin: str | None) -> bool:
    if origin is None:
        return True
    try:
        parsed = urlsplit(origin)
        return parsed.scheme in ("http", "https") and is_local_client(parsed.hostname) or (
            parsed.scheme in ("http", "https") and parsed.hostname == "localhost"
        )
    except (TypeError, ValueError):
        return False


def _verify_http_auth(request: Request, cfg: Settings, auth: str | None):
    if cfg.api_key:
        if auth is None or not secrets.compare_digest(auth, f"Bearer {cfg.api_key}"):
            raise HTTPException(401, "Unauthorized")
    elif not cfg.allow_unauth_local or not is_local_client(request.client.host if request.client else None):
        raise HTTPException(403, "Local-only API; client must use loopback")
    elif not _safe_browser_origin(request.headers.get("origin")) or (
        request.headers.get("sec-fetch-site", "").lower() == "cross-site"
    ):
        raise HTTPException(403, "Cross-site browser requests are blocked in local-only mode")


def _verify_ws_auth(ws: WebSocket, cfg: Settings) -> bool:
    if cfg.api_key:
        auth = ws.headers.get("authorization")
        return auth is not None and secrets.compare_digest(auth, f"Bearer {cfg.api_key}")
    return (
        cfg.allow_unauth_local
        and is_local_client(ws.client.host if ws.client else None)
        and _safe_browser_origin(ws.headers.get("origin"))
    )


class Runtime:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.status = "starting"
        self.loading_error: str | None = None
        self.loading_task: asyncio.Task | None = None
        self.device_lock = threading.Lock()
        self.gate = asyncio.Lock()
        self.active_sessions = 0

        # Model pool (shared across workers)
        self.pool: ModelPool | None = None
        self.batch_worker: BatchWorker | None = None
        self.stream_worker: StreamWorker | None = None
        self.single_worker: SingleWorker | None = None

    def gpu_call(self, fn, *args):
        with self.device_lock:
            return fn(*args)

    async def load_at_startup(self):
        self.status = "loading"
        started = time.perf_counter()
        try:
            # Initialize shared model pool
            self.pool = get_global_pool(
                model_path=self.cfg.model_path,
                dtype=self.cfg.hf_dtype,
                device=self.cfg.device,
                pool_size=2,
                hf_model=self.cfg.hf_model,
                bpe_path=self.cfg.bpe_path,
            )

            # Initialize workers
            self.batch_worker = BatchWorker(self.cfg, self.pool)
            self.stream_worker = StreamWorker(self.cfg, self.pool)
            self.single_worker = SingleWorker(self.cfg, self.pool)

            # Load all workers
            await asyncio.to_thread(self.batch_worker.load)
            await asyncio.to_thread(self.stream_worker.load)
            await asyncio.to_thread(self.single_worker.load)

            self.status = "ready"
            log.info("model_load_done elapsed_s=%.2f", time.perf_counter() - started)
        except Exception as exc:
            self.status = "error"
            self.loading_error = f"{type(exc).__name__}: {exc}"
            log.exception("SAM3 model startup load failed")

    async def shutdown(self):
        if self.loading_task:
            await self.loading_task
        # Shutdown workers
        if self.batch_worker:
            await asyncio.to_thread(self.batch_worker.shutdown)
        if self.stream_worker:
            await asyncio.to_thread(self.stream_worker.shutdown)
        if self.single_worker:
            await asyncio.to_thread(self.single_worker.shutdown)
        # Shutdown global pool
        shutdown_global_pool()
        self.status = "stopped"

    def mark_runtime_error(self, exc: BaseException):
        if "out of memory" in str(exc).lower() or "cuda oom" in str(exc).lower():
            self.status = "degraded"
            self.loading_error = "GPUOutOfMemory: operator restart required"
            log.critical("GPU OOM detected; refusing new work until controlled restart")

    async def acquire(self):
        try:
            await asyncio.wait_for(self.gate.acquire(), self.cfg.gate_wait_s)
            self.active_sessions += 1
            return True
        except asyncio.TimeoutError:
            return False

    def release(self):
        self.active_sessions -= 1
        self.gate.release()


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or Settings.from_env()
    cfg.validate()
    rt = Runtime(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        rt.loading_task = asyncio.create_task(rt.load_at_startup())
        try:
            yield
        finally:
            await rt.shutdown()

    app = FastAPI(title="SAM3 Video Service v4", lifespan=lifespan)
    app.state.runtime = rt

    @app.get("/health/live")
    def health_live():
        return {"status": "alive"}

    @app.get("/health/ready")
    def health_ready():
        if rt.status != "ready":
            return JSONResponse({"status": rt.status,
                                  "error_type": rt.loading_error.split(":", 1)[0] if rt.loading_error else None,
                                 "backend": cfg.engine}, status_code=503)
        return {"status": "ready", "backend": "hf-pool",
                "streaming_input": rt.stream_worker is not None}

    @app.get("/health")
    def legacy_health():
        return {"status": "ok" if rt.status == "ready" else rt.status,
                "model_loaded": rt.status == "ready", "backend": cfg.engine}

    @app.get("/v1/capabilities")
    def capabilities():
        return {"backend": "hf-pool", "ready": rt.status == "ready",
                "modes": ["batch", "stream", "single"],
                "max_frames": cfg.max_frames}

    # Unified /v1/segment endpoint with mode routing
    @app.post("/v1/segment")
    async def unified_segment(
        request: Request,
        file: UploadFile = File(...),
        text_prompt: str = Form(...),
        mode: str = Form("batch"),  # batch | stream | single
        frame_stride: int = Form(1),
        max_frames: int | None = Form(None),
        return_masks_png: bool = Form(False),
        output_prob_thresh: float | None = Form(None),
        authorization: str | None = Header(None),
    ):
        _verify_http_auth(request, cfg, authorization)
        if rt.status != "ready":
            raise HTTPException(503, f"Model not ready ({rt.status})")
        if not text_prompt.strip() or len(text_prompt) > 256:
            raise HTTPException(422, "Invalid text prompt")
        if mode not in ("batch", "stream", "single"):
            raise HTTPException(422, "mode must be batch|stream|single")
        if frame_stride < 1 or frame_stride > 100:
            raise HTTPException(422, "frame_stride must be 1..100")
        if max_frames is not None and not (1 <= max_frames <= cfg.max_frames):
            raise HTTPException(422, "max_frames outside allowed range")
        threshold = cfg.output_threshold if output_prob_thresh is None else output_prob_thresh
        if not 0 <= threshold <= 1:
            raise HTTPException(422, "output_prob_thresh must be 0..1")

        # Route to appropriate worker
        if mode == "batch":
            return await _handle_batch_mode(rt, cfg, file, text_prompt, threshold,
                                            frame_stride, max_frames, return_masks_png)
        elif mode == "stream":
            return await _handle_stream_mode(rt, cfg, file, text_prompt, threshold,
                                             max_frames, return_masks_png)
        else:  # single
            return await _handle_single_mode(rt, cfg, file, text_prompt, threshold)

    async def _handle_batch_mode(rt: Runtime, cfg: Settings, file: UploadFile,
                                 prompt: str, threshold: float, stride: int,
                                 max_frames: int | None, include_masks: bool):
        if not await rt.acquire():
            raise HTTPException(429, "GPU busy; retry later")
        try:
            import hashlib
            with tempfile.TemporaryDirectory(prefix="sam3_v4_") as folder:
                path = os.path.join(folder, "uploaded.mp4")
                sha = hashlib.sha256()
                length = 0
                with open(path, "wb") as target:
                    while True:
                        data = await file.read(CHUNK)
                        if not data:
                            break
                        length += len(data)
                        if length > cfg.max_upload_bytes:
                            raise HTTPException(413, "Video upload exceeds size cap")
                        sha.update(data)
                        target.write(data)
                if not length:
                    raise HTTPException(422, "Empty video upload")
                original_count, width, height = await asyncio.to_thread(probe_video, path)
                if width * height > cfg.max_frame_pixels:
                    raise ValueError("Video resolution exceeds max_frame_pixels")
                sampled_available = (original_count + stride - 1) // stride
                if max_frames is None and sampled_available > cfg.max_frames:
                    raise ValueError(
                        "Video exceeds max_frames safety cap; supply max_frames explicitly"
                    )
                source, count = await asyncio.to_thread(
                    prepare_video, path, os.path.join(folder, "sampled.mp4"),
                    original_count, max_frames, stride, cfg
                )
                started = time.perf_counter()
                result, content_type = await rt.batch_worker.segment_video(
                    video_path=source,
                    prompt=prompt.strip(),
                    threshold=threshold,
                    actual_frames=count,
                    stride=stride,
                    include_masks=include_masks,
                    source_sha256=sha.hexdigest(),
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if content_type == "application/zip":
                    return Response(result, media_type=content_type,
                                    headers={"Content-Disposition":
                                             'attachment; filename="sam3_masks.zip"',
                                             "X-Inference-Ms": f"{elapsed_ms:.1f}"})
                result["inference_ms"] = elapsed_ms
                return JSONResponse(result)
        except HTTPException:
            raise
        except Exception as exc:
            rt.mark_runtime_error(exc)
            log.exception("SAM3 batch inference failed")
            raise HTTPException(500, "SAM3 batch inference failed; see server logs")
        finally:
            rt.release()

    async def _handle_stream_mode(rt: Runtime, cfg: Settings, file: UploadFile,
                                  prompt: str, threshold: float,
                                  max_frames: int | None, include_masks: bool):
        # Stream mode expects WebSocket connection, not file upload
        # This endpoint is for backward compatibility - redirect to WS
        raise HTTPException(400, "Stream mode requires WebSocket connection at /v1/video/stream")

    async def _handle_single_mode(rt: Runtime, cfg: Settings, file: UploadFile,
                                  prompt: str, threshold: float):
        if not await rt.acquire():
            raise HTTPException(429, "GPU busy; retry later")
        try:
            image = decode_image(await file.read(), cfg)
            started = time.perf_counter()
            result = await rt.single_worker.segment_single(image, prompt.strip(), threshold)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return JSONResponse({
                "success": True,
                "backend": "hf-pool-single",
                "inference_ms": elapsed_ms,
                "frame_result": result.as_json(include_masks=True)
            })
        except HTTPException:
            raise
        except Exception as exc:
            rt.mark_runtime_error(exc)
            log.exception("SAM3 single inference failed")
            raise HTTPException(500, "SAM3 single inference failed; see server logs")
        finally:
            rt.release()

    # Legacy /video/segment endpoint (batch only, for backward compatibility)
    @app.post("/video/segment")
    @app.post("/v1/video/segment")
    async def video_segment_legacy(
        request: Request,
        file: UploadFile = File(...),
        text_prompt: str = Form(...),
        frame_stride: int = Form(1),
        max_frames: int | None = Form(None),
        return_masks_png: bool = Form(False),
        output_prob_thresh: float | None = Form(None),
        authorization: str | None = Header(None),
    ):
        return await _handle_batch_mode(rt, cfg, file, text_prompt,
                                        cfg.output_threshold if output_prob_thresh is None else output_prob_thresh,
                                        frame_stride, max_frames, return_masks_png)

    # WebSocket streaming endpoint
    @app.websocket("/video/stream")
    @app.websocket("/v1/video/stream")
    async def video_stream(ws: WebSocket):
        await ws.accept()
        if not _verify_ws_auth(ws, cfg):
            await ws.send_json({"type": "error", "code": "UNAUTHORIZED"})
            await ws.close(code=1008)
            return
        if rt.status != "ready":
            await ws.send_json({"type": "error", "code": "NOT_READY", "state": rt.status})
            await ws.close(code=1013)
            return
        if rt.stream_worker is None:
            await ws.send_json({"type": "error", "code": "UNSUPPORTED",
                                "message": "Stream worker not available"})
            await ws.close(code=1003)
            return
        if not await rt.acquire():
            await ws.send_json({"type": "error", "code": "GPU_BUSY"})
            await ws.close(code=1013)
            return

        session_id = None
        expected_idx = 0
        last_source_timestamp = None
        pending_metadata = None
        try:
            init = await asyncio.wait_for(ws.receive_json(), timeout=cfg.idle_timeout_s)
            if not isinstance(init, dict) or init.get("type") != "start":
                raise ValueError("First message must be JSON: {type:'start',prompt:'...'}")
            prompt = init.get("prompt", "")
            if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 256:
                raise ValueError("Invalid stream prompt")
            threshold = float(init.get("output_prob_thresh", cfg.output_threshold))
            if not np.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError("Invalid output_prob_thresh")
            max_frames = int(init.get("max_frames", cfg.max_frames))
            if not 1 <= max_frames <= cfg.max_frames:
                raise ValueError("Streaming frame cap exceeded")
            include_masks = init.get("return_masks_png", True)
            if not isinstance(include_masks, bool):
                raise ValueError("return_masks_png must be boolean")

            # Create session via StreamWorker
            session_id = await rt.stream_worker.create_session(prompt.strip(), threshold, max_frames)
            await ws.send_json({"type": "ready", "session_id": session_id,
                                "backend": "hf-pool-stream",
                                "frame_format": ["JPEG", "PNG"],
                                "next_frame_index": 0,
                                "ack_required": True,
                                "max_frames": max_frames})

            dimensions = None
            while expected_idx < max_frames:
                if time.monotonic() - start_time > cfg.max_session_s:
                    raise TimeoutError("Session lifetime exceeded")
                received = await asyncio.wait_for(ws.receive(), timeout=cfg.idle_timeout_s)
                if received.get("type") == "websocket.disconnect":
                    return
                binary = received.get("bytes")
                if binary is None:
                    text = received.get("text")
                    if text is None:
                        raise ValueError("Expected JPEG/PNG binary frame or control JSON")
                    control = json.loads(text)
                    if not isinstance(control, dict):
                        raise ValueError("Control message must be a JSON object")
                    command = control.get("type")
                    if command in ("end", "cancel"):
                        await ws.send_json({"type": "done", "frames_processed": expected_idx,
                                            "reason": command})
                        return
                    if command == "ping":
                        await ws.send_json({"type": "pong"})
                        continue
                    if command != "frame_meta" or pending_metadata is not None:
                        raise ValueError("Unsupported or duplicate frame metadata control")
                    idx = control.get("frame_index")
                    if type(idx) is not int or idx != expected_idx:
                        raise ValueError(f"Expected frame_index {expected_idx}")
                    timestamp = control.get("timestamp_ms")
                    if timestamp is not None:
                        if type(timestamp) not in (int, float) or not math.isfinite(timestamp):
                            raise ValueError("Invalid timestamp_ms")
                        if timestamp < 0 or (
                            last_source_timestamp is not None and timestamp <= last_source_timestamp
                        ):
                            raise ValueError("Source timestamps must increase monotonically")
                    pending_metadata = {"timestamp_ms": timestamp}
                    continue

                image = decode_image(binary, cfg)
                if dimensions is None:
                    dimensions = image.size
                elif dimensions != image.size:
                    raise ValueError("Frame resolution changed during session")
                inference_start = time.perf_counter()
                result = await rt.stream_worker.process_frame(session_id, image)
                inference_ms = (time.perf_counter() - inference_start) * 1000.0
                output = result.as_json(
                    include_masks=include_masks,
                    max_mask_bytes=min(cfg.max_frame_bytes, cfg.max_output_bytes),
                )
                if len(json.dumps(output)) > cfg.max_output_bytes:
                    raise ValueError("Frame response exceeds max_output_bytes")
                output["type"] = "frame_result"
                output["inference_ms"] = round(inference_ms, 3)
                output["session_id"] = session_id
                output["timestamp_ms"] = (
                    pending_metadata.get("timestamp_ms") if pending_metadata else
                    (time.monotonic() - start_time) * 1000.0
                )
                if pending_metadata and pending_metadata["timestamp_ms"] is not None:
                    last_source_timestamp = pending_metadata["timestamp_ms"]
                output["next_frame_index"] = expected_idx + 1
                pending_metadata = None
                await ws.send_json(output)
                expected_idx += 1
            await ws.send_json({"type": "done", "frames_processed": expected_idx,
                                "reason": "frame_limit"})
        except WebSocketDisconnect:
            pass
        except asyncio.TimeoutError:
            await ws.send_json({"type": "error", "code": "IDLE_TIMEOUT"})
            await ws.close(code=1001)
        except (ValueError, TimeoutError, json.JSONDecodeError) as exc:
            try:
                await ws.send_json({"type": "error", "code": "BAD_FRAME_OR_SESSION",
                                    "message": str(exc)})
                await ws.close(code=1008)
            except (RuntimeError, WebSocketDisconnect):
                pass
        except Exception as exc:
            rt.mark_runtime_error(exc)
            log.exception("Incremental SAM3 inference failure")
            try:
                await ws.send_json({"type": "error", "code": "INFERENCE_ERROR"})
                await ws.close(code=1011)
            except (RuntimeError, WebSocketDisconnect):
                pass
        finally:
            if session_id:
                try:
                    await rt.stream_worker.end_session(session_id)
                except Exception:
                    log.exception("Failed to free streaming session state")
            rt.release()

    return app