"""SAM3 HTTP /video/segment client for e-Brush Studio integration.

Adapts SAM3 v1.2 API (/video/segment endpoint) for real-time brush video segmentation.
Replaces deprecated /image/segment endpoint.
"""
from __future__ import annotations

import os, sys, json, time, tempfile, zipfile, io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import threading
from collections import deque
import re

import cv2
import numpy as np
import requests


@dataclass
class SAM3Config:
    """SAM3 client configuration"""
    base_url: str = "https://desktop-kc6mgc1.tailb2b872.ts.net/sam3"
    api_key: str = ""
    prompt: str = "calligraphy brush"
    jpeg_quality: int = 88
    timeout_s: float = 30.0
    clip_frames: int = 30
    clip_fps: int = 30
    min_interval_s: float = 0.5

    @classmethod
    def from_env(cls) -> "SAM3Config":
        """Load from environment variables and config file"""
        config_path = Path(__file__).parent.parent / "config" / "vision_validation.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f).get("sam3", {})
            return cls(
                base_url=cfg.get("base_url", cls.base_url),
                api_key=os.environ.get(cfg.get("api_key_env", "SAM3_API_KEY"), ""),
                prompt=cfg.get("prompt", cls.prompt),
                jpeg_quality=cfg.get("jpeg_quality", cls.jpeg_quality),
                timeout_s=cfg.get("timeout_s", cls.timeout_s),
                clip_frames=cfg.get("clip_frames", cls.clip_frames),
                clip_fps=cfg.get("clip_fps", cls.clip_fps),
                min_interval_s=cfg.get("min_interval_s", cls.min_interval_s)
            )
        return cls(
            api_key=os.environ.get("SAM3_API_KEY", "")
        )


@dataclass
class SamResult:
    frame_id: int
    host_ns: int
    mask: Optional[np.ndarray]
    score: float = 0.0
    bbox: Optional[tuple] = None
    latency_ms: float = 0.0
    error: str = ""
    prompt: str = ""
    camera_context: object = None


class SAM3Client:
    """
    Async SAM3 video segmentation client

    Design points:
    - Buffer frames to assemble short video clips
    - Use /video/segment endpoint (SAM3 v1.2+)
    - Support real-time streaming: prioritize latest frames, drop old
    - Return best mask for latest frame
    """

    def __init__(self, config: SAM3Config):
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self.api_key = config.api_key
        self.prompt = config.prompt
        self.jpeg_quality = config.jpeg_quality
        self.timeout = config.timeout_s
        self.clip_frames = config.clip_frames
        self.clip_fps = config.clip_fps
        self.min_interval = config.min_interval_s

        self._frame_buffer = deque(maxlen=max(1, self.clip_frames))
        self._lock = threading.Lock()
        self._result: Optional[SamResult] = None
        self._stop = False
        self._wake = threading.Event()
        self._last_call = 0.0

        self.thread = threading.Thread(target=self._run, name="SAM3Worker", daemon=True)
        self.thread.start()

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def health(self) -> dict:
        """Health check"""
        r = requests.get(self.base_url + "/health", headers=self._headers(), timeout=min(5, self.timeout))
        r.raise_for_status()
        return r.json()

    def submit(self, rgb: np.ndarray, frame_id: int, host_ns: int, camera_context: object = None):
        """Submit frame for segmentation"""
        with self._lock:
            self._frame_buffer.append((np.asarray(rgb).copy(), int(frame_id), int(host_ns), str(self.prompt), camera_context))
        self._wake.set()

    def latest(self) -> Optional[SamResult]:
        with self._lock:
            return self._result

    @staticmethod
    def _decode_zip_masks(content: bytes, shape: tuple[int, int],
                          frame_index: int, expected_frames: int | None = None):
        """Defensively select only the exact predicted camera clip frame."""
        if len(content) > 100 * 1024 * 1024:
            raise ValueError("SAM3 prediction archive too large")
        masks = []
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            names = z.namelist()
            if len(names) != len(set(names)) or len(names) > 12000:
                raise ValueError("Duplicate or excessive ZIP members")
            if sum(item.file_size for item in z.infolist()) > 200 * 1024 * 1024:
                raise ValueError("SAM3 uncompressed prediction exceeds limit")
            if "meta.json" not in names:
                raise ValueError("SAM3 ZIP missing meta.json")
            meta = json.loads(z.read("meta.json"))
            count = meta.get("num_frames")
            if type(count) is not int or count <= frame_index or frame_index < 0:
                raise ValueError("SAM3 ZIP frame count mismatch")
            if expected_frames is not None and count != expected_frames:
                raise ValueError("SAM3 ZIP does not match requested clip length")
            if "frame_indices" in meta and meta["frame_indices"] != list(range(count)):
                raise ValueError("SAM3 ZIP frame index mapping mismatch")
            confidence = {}
            rows = meta.get("frame_results")
            if isinstance(rows, list):
                if len(rows) != count:
                    raise ValueError("SAM3 frame_results count mismatch")
                for i, row in enumerate(rows):
                    if row.get("frame_index") != i:
                        raise ValueError("SAM3 frame_results index mismatch")
                    for obj in row.get("objects", []):
                        confidence[(i, int(obj["object_id"]))] = float(obj["score"])
            elif isinstance(meta.get("object_scores"), list):
                vals = meta["object_scores"]
                if frame_index < len(vals) and vals[frame_index] is not None:
                    confidence[(frame_index, 0)] = float(vals[frame_index])
            seen = set()
            for n in names:
                m = re.fullmatch(r"frame_(\d+)_obj_(\d+)(?:_score([0-9.]+))?\.png", n)
                if m is None or int(m.group(1)) != frame_index:
                    continue
                obj = int(m.group(2))
                if obj in seen:
                    raise ValueError("Duplicate SAM3 object for requested frame")
                seen.add(obj)
                data = z.read(n)
                decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
                if decoded is None or decoded.shape != shape:
                    raise ValueError("SAM3 mask decoding or size mismatch")
                score = confidence.get((frame_index, obj))
                if score is None and m.group(3) is not None:
                    score = float(m.group(3))
                if score is None or not np.isfinite(score):
                    raise ValueError("SAM3 mask has no usable confidence value")
                masks.append((decoded > 127, float(score), n))
        return masks

    @staticmethod
    def _pick(masks: list, prev_mask: Optional[np.ndarray] = None):
        """Pick best mask: combine SAM score, shape prior, temporal consistency"""
        if not masks:
            return None, 0.0, None
        best = None
        bestv = -1e9
        for m, score, n in masks:
            ys, xs = np.nonzero(m)
            if len(xs) < 20:
                continue
            w = max(1, xs.max() - xs.min() + 1)
            h = max(1, ys.max() - ys.min() + 1)
            elong = max(w, h) / max(1, min(w, h))
            area = float(m.sum())
            iou = 0.0
            if prev_mask is not None and prev_mask.shape == m.shape:
                inter = np.logical_and(m, prev_mask).sum()
                union = np.logical_or(m, prev_mask).sum()
                iou = float(inter / max(1, union))
            v = float(score) + 0.05 * min(elong, 8) / 8 + 0.20 * iou
            if v > bestv:
                bestv = v
                best = (m, score, (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)))
        return best if best else (None, 0.0, None)

    def _write_clip_to_temp(self, frames: list[np.ndarray]) -> str:
        """Write frame sequence to temporary MP4 file"""
        if not frames:
            return ""
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        tmp_path = tmp.name
        tmp.close()
        writer = cv2.VideoWriter(tmp_path, fourcc, self.clip_fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError("Failed to create SAM3 clip MP4")
        try:
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        return tmp_path

    def _run(self):
        """Worker thread: periodically take frame buffer, assemble clip, request segmentation"""
        prev_mask = None
        while not self._stop:
            self._wake.wait(0.2)
            self._wake.clear()

            # Take most recent clip_frames frames
            with self._lock:
                frames_to_process = self._frame_buffer[-self.clip_frames:] if self._frame_buffer else []
                self._frame_buffer.clear()

            if not frames_to_process:
                continue

            wait = self.min_interval - (time.perf_counter() - self._last_call)
            if wait > 0:
                time.sleep(wait)

            t0 = time.perf_counter()
            clip_path = ""
            try:
                frame_arrays = [f[0] for f in frames_to_process]
                clip_path = self._write_clip_to_temp(frame_arrays)

                with open(clip_path, 'rb') as f:
                    files = {'file': (os.path.basename(clip_path), f, 'video/mp4')}
                    data = {
                        'text_prompt': frames_to_process[-1][3],
                        'max_frames': str(len(frames_to_process)),
                        'frame_stride': '1',
                        'return_masks_png': 'true'
                    }
                    r = requests.post(
                        self.base_url + "/video/segment",
                        headers=self._headers(),
                        files=files,
                        data=data,
                        timeout=self.timeout
                    )
                    r.raise_for_status()

                masks = self._decode_zip_masks(r.content, frame_arrays[0].shape[:2],
                                               len(frame_arrays) - 1,
                                               expected_frames=len(frame_arrays))

                if masks:
                    mask, score, bbox = self._pick(masks, None)
                    prev_mask = mask if mask is not None else prev_mask
                    _, fid, hns, prompt, camera_context = frames_to_process[-1]
                    res = SamResult(fid, hns, mask, score, bbox, (time.perf_counter() - t0) * 1000, "", prompt, camera_context)
                else:
                    _, fid, hns, prompt, camera_context = frames_to_process[-1]
                    res = SamResult(fid, hns, None, 0, None, (time.perf_counter() - t0) * 1000, "No valid masks", prompt, camera_context)

            except Exception as e:
                if frames_to_process:
                    _, fid, hns, prompt, camera_context = frames_to_process[-1]
                else:
                    fid, hns, prompt, camera_context = -1, int(time.perf_counter_ns()), self.prompt, None
                res = SamResult(fid, hns, None, 0, None, (time.perf_counter() - t0) * 1000, f"{type(e).__name__}: {e}", prompt, camera_context)
            finally:
                if clip_path and os.path.exists(clip_path):
                    try:
                        os.unlink(clip_path)
                    except:
                        pass

            self._last_call = time.perf_counter()
            with self._lock:
                self._result = res

    def close(self):
        self._stop = True
        self._wake.set()
        if self.thread.is_alive():
            self.thread.join(timeout=min(self.timeout + 2, 20))


# Convenience function
def create_sam3_client(config: Optional[SAM3Config] = None) -> SAM3Client:
    """Create SAM3 client instance"""
    if config is None:
        config = SAM3Config.from_env()
    return SAM3Client(config)


# Backward compatibility wrapper for Studio
class StudioSAM3Client(SAM3Client):
    """Backward compatible SAM3 client for Studio (wraps SAM3Client)."""
    def __init__(self, base_url: str, api_key: str = "",
                 prompt: str = "calligraphy brush", jpeg_quality: int = 88,
                 timeout_s: float = 30.0, clip_frames: int = 30,
                 clip_fps: int = 30, min_interval_s: float = 0.0):
        from dataclasses import replace
        config = SAM3Config(
            base_url=base_url, api_key=api_key, prompt=prompt,
            jpeg_quality=jpeg_quality, timeout_s=timeout_s,
            clip_frames=clip_frames, clip_fps=clip_fps,
            min_interval_s=min_interval_s
        )
        super().__init__(config)