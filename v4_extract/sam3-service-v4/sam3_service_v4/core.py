"""SAM3 service shared configuration, results, and defensive parsing."""
from __future__ import annotations

import base64
import io
import ipaddress
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Settings:
    engine: str = "meta"  # meta: proven offline pipeline; hf: true incremental input
    model_path: str = "/mnt/e/models/sam3/models/facebook--sam3/snapshots/master/sam3.pt"
    bpe_path: str = "/mnt/e/models/sam3/models/facebook--sam3/snapshots/master/bpe_simple_vocab_16e6.txt.gz"
    hf_model: str = ""  # explicit HF-compatible Transformers checkpoint/config path or repo
    device: str = "cuda:0"  # logical CUDA device; CUDA_VISIBLE_DEVICES selects physical GPU
    hf_dtype: str = "float32"
    api_key: str = ""
    allow_unauth_local: bool = False
    detection_threshold: float = 0.1
    output_threshold: float = 0.1
    max_upload_bytes: int = 256 * 1024 * 1024
    max_frame_bytes: int = 8 * 1024 * 1024
    max_frame_pixels: int = 640 * 480
    max_frames: int = 400
    max_output_bytes: int = 128 * 1024 * 1024
    max_objects: int = 16
    idle_timeout_s: float = 30.0
    max_session_s: float = 600.0
    ffmpeg_timeout_s: float = 180.0
    gate_wait_s: float = 0.05

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ
        b = lambda key: env.get(key, "").strip().lower() in ("1", "true", "yes", "on")
        obj = cls(
            engine=env.get("SAM3_ENGINE", "meta").strip().lower(),
            model_path=env.get("SAM3_MODEL_PATH", cls.model_path),
            bpe_path=env.get("SAM3_BPE_PATH", cls.bpe_path),
            hf_model=env.get("HF_SAM3_MODEL", ""),
            device=env.get("SAM3_DEVICE", "cuda:0"),
            hf_dtype=env.get("HF_SAM3_DTYPE", "float32"),
            api_key=env.get("API_KEY", ""),
            allow_unauth_local=b("ALLOW_UNAUTH_LOCAL"),
            detection_threshold=float(env.get("SAM3_DETECTION_THRESHOLD", "0.1")),
            output_threshold=float(env.get("SAM3_OUTPUT_PROB_THRESH", "0.1")),
            max_upload_bytes=int(env.get("SAM3_MAX_UPLOAD_BYTES", str(256*1024*1024))),
            max_frame_bytes=int(env.get("SAM3_MAX_FRAME_BYTES", str(8*1024*1024))),
            max_frame_pixels=int(env.get("SAM3_MAX_FRAME_PIXELS", str(640*480))),
            max_frames=int(env.get("SAM3_MAX_FRAMES", "400")),
            max_output_bytes=int(env.get("SAM3_MAX_OUTPUT_BYTES", str(128*1024*1024))),
            max_objects=int(env.get("SAM3_MAX_OBJECTS", "16")),
            idle_timeout_s=float(env.get("SAM3_IDLE_TIMEOUT_S", "30")),
            max_session_s=float(env.get("SAM3_MAX_SESSION_S", "600")),
            ffmpeg_timeout_s=float(env.get("SAM3_FFMPEG_TIMEOUT_S", "180")),
            gate_wait_s=float(env.get("SAM3_GATE_WAIT_S", "0.05")),
        )
        obj.validate()
        return obj

    def validate(self) -> None:
        if self.engine not in ("meta", "hf"):
            raise ValueError("SAM3_ENGINE must be 'meta' or 'hf'")
        if not self.api_key and not self.allow_unauth_local:
            raise ValueError("Set API_KEY or ALLOW_UNAUTH_LOCAL=1 for local-only development")
        if self.engine == "hf" and not self.hf_model:
            raise ValueError(
                "HF_SAM3_MODEL must name an HF Transformers-compatible SAM3 weight directory or repo; "
                "the original SAM3.1 .pt checkpoint is not interchangeable"
            )
        if not (0 <= self.detection_threshold <= 1 and 0 <= self.output_threshold <= 1):
            raise ValueError("Thresholds must be in [0,1]")
        if self.hf_dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("HF_SAM3_DTYPE must be float32, float16, or bfloat16")
        if min(self.max_upload_bytes, self.max_frame_bytes, self.max_frame_pixels,
               self.max_frames, self.max_output_bytes, self.max_objects) < 1:
            raise ValueError("All caps must be positive")
        if min(self.idle_timeout_s, self.max_session_s, self.ffmpeg_timeout_s,
               self.gate_wait_s) <= 0:
            raise ValueError("Timeouts must be positive")


def is_local_client(host: str | None) -> bool:
    if not host:
        return False
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass
class FrameMasks:
    frame_index: int
    object_ids: list[int] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.object_ids) != len(self.scores) or len(self.scores) != len(self.masks):
            raise ValueError("Mismatch between mask, object ID and score counts")
        if len(set(self.object_ids)) != len(self.object_ids):
            raise ValueError("Duplicate object IDs within a frame")
        for score in self.scores:
            if not np.isfinite(score):
                raise ValueError("Non-finite detection score")
        for mask in self.masks:
            if mask.ndim != 2:
                raise ValueError(f"Expected HxW binary mask, got {mask.shape}")
            if mask.dtype != np.bool_:
                raise ValueError("Normalized masks must be binary numpy arrays")

    def as_json(self, include_masks: bool = False, max_mask_bytes: int = 4*1024*1024) -> dict:
        objs = []
        for obj_id, score, mask in zip(self.object_ids, self.scores, self.masks):
            row = {"object_id": int(obj_id), "score": float(score),
                   "mask_pixels": int(np.count_nonzero(mask))}
            if include_masks:
                buf = io.BytesIO()
                Image.fromarray(mask.astype(np.uint8) * 255).save(buf, "PNG")
                if buf.tell() > max_mask_bytes:
                    raise ValueError("One encoded mask exceeds response size cap")
                row["mask_png_base64"] = base64.b64encode(buf.getvalue()).decode("ascii")
            objs.append(row)
        return {"frame_index": int(self.frame_index), "num_objects": len(objs), "objects": objs}


def normalize_frame_response(raw: dict, max_objects: int) -> FrameMasks:
    """Parse the real Meta API's response['outputs'] without guessing missing keys."""
    if not isinstance(raw, dict) or "frame_index" not in raw:
        raise ValueError("Missing frame_index in SAM3 response")
    outputs = raw.get("outputs")
    if not isinstance(outputs, dict) or "out_binary_masks" not in outputs:
        raise ValueError("SAM3 response missing outputs.out_binary_masks")
    value = outputs["out_binary_masks"]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    masks = np.asarray(value)
    if masks.ndim == 2:
        masks = masks[None]
    if masks.size == 0 and masks.ndim <= 1:
        masks = np.empty((0, 0, 0), dtype=bool)
    if masks.ndim != 3:
        raise ValueError(f"Invalid out_binary_masks shape {masks.shape}")
    count = masks.shape[0]
    if count > max_objects:
        raise ValueError(f"Too many objects ({count} > {max_objects})")
    ids = outputs.get("out_obj_ids")
    probs = outputs.get("out_probs")
    if count and (ids is None or probs is None):
        raise ValueError("Masks returned without object IDs or probabilities")
    ids_arr = np.asarray(ids if ids is not None else []).reshape(-1)
    probs_arr = np.asarray(probs if probs is not None else []).reshape(-1)
    if count != len(ids_arr) or count != len(probs_arr):
        raise ValueError("Mask/ID/probability count mismatch")
    return FrameMasks(
        int(raw["frame_index"]),
        [int(v) for v in ids_arr],
        [float(v) for v in probs_arr],
        [np.asarray(m, dtype=bool) for m in masks],
    )
