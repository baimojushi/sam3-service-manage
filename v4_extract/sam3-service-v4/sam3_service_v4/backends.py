"""Backends kept separate: legacy Meta whole-video vs HF true incremental SAM3.

Meta's handle_request dispatcher does not expose add_frame. Only HF mode advertises
genuine incremental, persistent-state video inference.
"""
from __future__ import annotations

import inspect
import logging
import os
import time
import uuid
from typing import Iterator

import numpy as np
from PIL import Image

from .core import FrameMasks, Settings, normalize_frame_response

log = logging.getLogger("sam3.service.backend")


class MetaBatchBackend:
    supports_streaming = False
    name = "meta-batch"

    def __init__(self, settings: Settings):
        self.cfg = settings
        self.predictor = None

    @staticmethod
    def _patch_session_if_needed(predictor) -> None:
        """Adapt older SAM3.1 multiplex init_state signature without overwriting it blindly."""
        sig = inspect.signature(predictor.model.init_state)
        params = sig.parameters
        if "offload_state_to_cpu" in params or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        ):
            return

        def compat_start_session(resource_path, session_id=None,
                                 offload_video_to_cpu=False, offload_state_to_cpu=False):
            if offload_state_to_cpu:
                raise ValueError(
                    "offload_state_to_cpu is unsupported by this SAM3.1 model; "
                    "use the supported video offload setting"
                )
            kwargs = {
                "resource_path": resource_path,
                "offload_video_to_cpu": offload_video_to_cpu,
                "async_loading_frames": getattr(predictor, "async_loading_frames", False),
                "video_loader_type": getattr(predictor, "video_loader_type", "cv2"),
            }
            kwargs = {k: v for k, v in kwargs.items() if k in params}
            state = predictor.model.init_state(**kwargs)
            key = session_id or str(uuid.uuid4())
            registry = getattr(predictor, "_all_inference_states", None)
            if registry is None:
                registry = getattr(predictor, "_ALL_INFERENCE_STATES", None)
            if registry is None:
                raise RuntimeError("SAM3 session registry not found")
            registry[key] = {"state": state, "session_id": key,
                             "start_time": time.time(), "last_use_time": time.time()}
            return {"session_id": key}

        predictor.start_session = compat_start_session
        log.warning("Applied legacy SAM3.1 init_state parameter compatibility adapter")

    @staticmethod
    def _repair_projection_if_needed(predictor):
        """Optional numerical safeguard, keeps all downstream token weights untouched."""
        import torch
        model = predictor.model
        candidates = [
            getattr(getattr(getattr(model, "detector", None), "backbone", None),
                    "language_backbone", None),
            getattr(getattr(model, "backbone", None), "language_backbone", None),
            getattr(model, "language_backbone", None),
        ]
        for backbone in candidates:
            encoder = getattr(backbone, "encoder", None)
            proj = getattr(encoder, "text_projection", None)
            if proj is None:
                continue
            tensor = proj if isinstance(proj, torch.nn.Parameter) else getattr(proj, "weight", None)
            if tensor is None:
                continue
            with torch.no_grad():
                maximum = float(tensor.detach().abs().max().item())
                if not np.isfinite(maximum) or maximum > 1e10:
                    generator = torch.Generator(device="cpu").manual_seed(423)
                    std = tensor.shape[0] ** -0.5 if isinstance(proj, torch.nn.Parameter) \
                        else tensor.shape[1] ** -0.5
                    replacement = torch.randn(tensor.shape, generator=generator, dtype=torch.float32)
                    tensor.copy_((replacement * std).to(tensor.device, dtype=tensor.dtype))
                    log.warning("Reinitialized out-of-range unused pooled text projection")
            break

    def load(self):
        import torch
        if self.cfg.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable for Meta backend")
        import sys
        sam3_root = os.getenv("SAM3_SOURCE_ROOT", "/opt/sam3-src")
        if sam3_root not in sys.path:
            sys.path.insert(0, sam3_root)
        from sam3.model_builder import build_sam3_video_predictor

        device_idx = int(self.cfg.device.split(":")[-1]) if ":" in self.cfg.device else 0
        self.predictor = build_sam3_video_predictor(
            checkpoint_path=self.cfg.model_path,
            async_loading_frames=False,
            apply_temporal_disambiguation=True,
            has_presence_token=True,
            geo_encoder_use_img_cross_attn=True,
            strict_state_dict_loading=True,
            video_loader_type="cv2",
            bpe_path=self.cfg.bpe_path,
            gpus_to_use=[device_idx],
        )
        self._patch_session_if_needed(self.predictor)
        if os.getenv("SAM3_PATCH_TEXT_PROJECTION", "1").lower() in ("1", "true", "yes"):
            self._repair_projection_if_needed(self.predictor)
        if hasattr(self.predictor.model, "score_threshold_detection"):
            self.predictor.model.score_threshold_detection = self.cfg.detection_threshold

    def segment_video(self, path: str, prompt: str, threshold: float) -> Iterator[FrameMasks]:
        if self.predictor is None:
            raise RuntimeError("Meta backend not loaded")
        session_id = None
        stream = None
        try:
            start = self.predictor.handle_request({
                "type": "start_session", "resource_path": path
            })
            session_id = start["session_id"]
            prompt_out = self.predictor.handle_request({
                "type": "add_prompt", "session_id": session_id, "frame_index": 0,
                "text": prompt, "output_prob_thresh": threshold
            })
            normalize_frame_response(prompt_out, self.cfg.max_objects)
            stream = self.predictor.handle_stream_request({
                "type": "propagate_in_video", "session_id": session_id,
                "propagation_direction": "forward", "start_frame_index": 0,
                "output_prob_thresh": threshold,
            })
            for raw in stream:
                yield normalize_frame_response(raw, self.cfg.max_objects)
        finally:
            if stream is not None:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
            if session_id is not None:
                self.predictor.handle_request(
                    {"type": "close_session", "session_id": session_id}
                )

    def shutdown(self):
        if self.predictor is not None:
            shutdown = getattr(self.predictor, "shutdown", None)
            if callable(shutdown):
                shutdown()
            self.predictor = None


class HFIncrementalBackend:
    """HF Sam3VideoModel streaming, using a different checkpoint layout from native SAM3.1."""
    supports_streaming = True
    name = "hf-incremental"

    def __init__(self, settings: Settings):
        self.cfg = settings
        self.model = None
        self.processor = None
        self.torch = None
        self.dtype = None

    def load(self):
        import torch
        from transformers import Sam3VideoModel, Sam3VideoProcessor
        if self.cfg.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable for HF SAM3")
        self.torch = torch
        self.dtype = getattr(torch, self.cfg.hf_dtype)
        if self.cfg.device == "cpu" and self.dtype != torch.float32:
            raise ValueError("Float16/bfloat16 CPU streaming is unsupported in this deployment")
        # Requires a transformers-compatible config and weights. Native Meta .pt alone won't work.
        try:
            model = Sam3VideoModel.from_pretrained(self.cfg.hf_model, dtype=self.dtype)
        except TypeError as exc:
            if "dtype" not in str(exc):
                raise
            model = Sam3VideoModel.from_pretrained(self.cfg.hf_model, torch_dtype=self.dtype)
        self.model = model.to(self.cfg.device).eval()
        self.processor = Sam3VideoProcessor.from_pretrained(self.cfg.hf_model)

    def new_stream(self, prompt: str):
        session = self.processor.init_video_session(
            inference_device=self.cfg.device,
            inference_state_device="cpu",
            processing_device="cpu",
            video_storage_device="cpu",
            max_vision_features_cache_size=1,
            dtype=self.dtype,
        )
        self.processor.add_text_prompt(inference_session=session, text=prompt)
        return session

    def process_frame(self, session, image: Image.Image, threshold: float) -> FrameMasks:
        torch = self.torch
        inputs = self.processor(images=image, return_tensors="pt")
        pixels = inputs.pixel_values[0].to(device=self.cfg.device, dtype=self.dtype)
        with torch.inference_mode():
            output = self.model(inference_session=session, frame=pixels, reverse=False)
            processed = self.processor.postprocess_outputs(
                session, output, original_sizes=inputs.original_sizes
            )
        ids = processed["object_ids"]
        scores = processed["scores"]
        masks = processed["masks"]
        if hasattr(ids, "detach"):
            ids = ids.detach().cpu().numpy()
        if hasattr(scores, "detach"):
            scores = scores.detach().float().cpu().numpy()
        if hasattr(masks, "detach"):
            masks = masks.detach().cpu().numpy()
        ids = np.asarray(ids).reshape(-1)
        scores = np.asarray(scores).reshape(-1)
        masks = np.asarray(masks)
        if masks.ndim == 2:
            masks = masks[None]
        if masks.size == 0:
            masks = np.empty((0, image.height, image.width), dtype=bool)
        if masks.ndim != 3 or len(ids) != len(masks) or len(scores) != len(masks):
            raise ValueError("Unexpected HF SAM3 streaming postprocessing shape")
        if len(ids) > self.cfg.max_objects:
            raise ValueError("Excessive tracked objects; session limit reached")
        out_ids, out_scores, out_masks = [], [], []
        for oid, score, mask in zip(ids, scores, masks):
            score = float(score)
            if not np.isfinite(score):
                raise ValueError("Non-finite confidence")
            if score < threshold:
                continue
            binary = np.asarray(mask, dtype=bool)
            if binary.shape != (image.height, image.width):
                raise ValueError(f"Mask dimensions mismatch: {binary.shape}")
            if not np.any(binary):
                continue
            out_ids.append(int(oid))
            out_scores.append(score)
            out_masks.append(binary)
        # The model keeps a sequential frame counter inside its inference session.
        return FrameMasks(-1, out_ids, out_scores, out_masks)

    def close_stream(self, session):
        reset = getattr(session, "reset_inference_session", None)
        if callable(reset):
            reset()

    def segment_video(self, path: str, prompt: str, threshold: float) -> Iterator[FrameMasks]:
        import cv2
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot decode video: {path}")
        session = self.new_stream(prompt)
        i = 0
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                frame = self.process_frame(session, image, threshold)
                frame.frame_index = i
                yield frame
                i += 1
        finally:
            cap.release()
            self.close_stream(session)

    def shutdown(self):
        self.model = None
        self.processor = None
