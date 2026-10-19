"""Three worker classes for Batch/Stream/Single modes sharing ModelPool."""
from __future__ import annotations

import asyncio
import io
import json
import os
import time
import uuid
from contextlib import closing
from typing import AsyncGenerator, Optional, Generator

import numpy as np
from PIL import Image, ImageOps

from .model_pool import ModelPool, get_global_pool, shutdown_global_pool
from .core import FrameMasks, Settings, is_local_client
from .backends import HFIncrementalBackend, MetaBatchBackend
from .sam3_client import create_sam3_client, SAM3Config


class BaseWorker:
    """Base class with common functionality."""
    
    def __init__(self, cfg: Settings, pool: ModelPool):
        self.cfg = cfg
        self.pool = pool
        self._lock = asyncio.Lock()
    
    async def acquire_gpu(self) -> bool:
        """Acquire GPU access with timeout."""
        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=self.cfg.gate_wait_s)
            return True
        except asyncio.TimeoutError:
            return False
    
    def release_gpu(self) -> None:
        self._lock.release()


class BatchWorker(BaseWorker):
    """
    Batch Worker: Whole-video offline processing.
    Reuses MetaBatchBackend for proven production path.
    """
    
    def __init__(self, cfg: Settings, pool: ModelPool):
        super().__init__(cfg, pool)
        self._meta_backend: Optional[MetaBatchBackend] = None
        self._hf_backend: Optional[HFIncrementalBackend] = None
    
    def _get_backend(self):
        """Get appropriate backend based on engine."""
        if self.cfg.engine == "meta":
            if self._meta_backend is None:
                self._meta_backend = MetaBatchBackend(self.cfg)
            return self._meta_backend
        else:
            if self._hf_backend is None:
                self._hf_backend = HFIncrementalBackend(self.cfg)
            return self._hf_backend
    
    def load(self):
        """Load backend model."""
        backend = self._get_backend()
        backend.load()
    
    async def segment_video(
        self,
        video_path: str,
        prompt: str,
        threshold: float,
        actual_frames: int,
        stride: int,
        include_masks: bool,
        source_sha256: str,
    ) -> tuple[bytes | dict, str]:
        """Process whole video file."""
        backend = self._get_backend()
        
        # Run in thread pool to avoid blocking event loop
        def _batch_infer():
            metadata = []
            object_ids: set[int] = set()
            frames_with_masks = 0
            buffer = io.BytesIO() if include_masks else None
            z = zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) if buffer is not None else None
            
            try:
                with closing(backend.segment_video(video_path, prompt, threshold)) as stream:
                    for frame in stream:
                        expected_idx = len(metadata)
                        if frame.frame_index != expected_idx:
                            raise ValueError(
                                f"SAM3 frame index gap: expected {expected_idx}, got {frame.frame_index}"
                            )
                        if expected_idx >= actual_frames:
                            raise ValueError("SAM3 returned more frames than preprocessed video")
                        if len(frame.object_ids) > self.cfg.max_objects:
                            raise ValueError("Object count exceeds safety limit")
                        
                        row = frame.as_json(include_masks=False)
                        row["source_frame_index"] = expected_idx * stride
                        metadata.append(row)
                        
                        any_mask = False
                        if z is not None:
                            for oid, score, mask in zip(frame.object_ids, frame.scores, frame.masks):
                                pixels = int(np.count_nonzero(mask))
                                if not pixels:
                                    continue
                                any_mask = True
                                object_ids.add(oid)
                                img = Image.fromarray(mask.astype(np.uint8) * 255)
                                img_data = io.BytesIO()
                                img.save(img_data, format="PNG")
                                z.writestr(
                                    f"frame_{expected_idx:04d}_obj_{oid:02d}_score{score:.4f}.png",
                                    img_data.getvalue(),
                                )
                        else:
                            for oid, mask in zip(frame.object_ids, frame.masks):
                                if np.any(mask):
                                    any_mask = True
                                    object_ids.add(oid)
                        
                        if any_mask:
                            frames_with_masks += 1
                        
                        if buffer is not None and buffer.tell() > self.cfg.max_output_bytes:
                            raise ValueError("Predicted masks ZIP exceeds max_output_bytes")
                
                if len(metadata) != actual_frames:
                    raise ValueError(
                        f"Model returned {len(metadata)} frames but input has {actual_frames}"
                    )
                
                seed = metadata[0]
                result = {
                    "success": True,
                    "backend": backend.name,
                    "num_frames": len(metadata),
                    "num_objects": len(object_ids),
                    "object_ids": sorted(object_ids),
                    "frames_with_masks": frames_with_masks,
                    "text_prompt": prompt,
                    "source_sha256": source_sha256,
                    "frame_indices": [r["source_frame_index"] for r in metadata],
                    "processed_frame_indices": [r["frame_index"] for r in metadata],
                    "object_scores": [
                        (r["objects"][0]["score"] if r["objects"] else None) for r in metadata
                    ],
                    "prompt_frame": {
                        "obj_ids": [o["object_id"] for o in seed["objects"]],
                        "scores": [o["score"] for o in seed["objects"]],
                    },
                    "frame_results": metadata,
                    "output_prob_thresh": threshold,
                }
                
                if z is not None:
                    z.writestr("meta.json", json.dumps(result, ensure_ascii=False, indent=2))
                    z.close()
                    if buffer.tell() > self.cfg.max_output_bytes:
                        raise ValueError("Predicted masks ZIP exceeds output cap")
                    return buffer.getvalue(), "application/zip"
                return result, "application/json"
            finally:
                if z is not None:
                    z.close()
        
        return await asyncio.to_thread(_batch_infer)
    
    def shutdown(self):
        if self._meta_backend:
            self._meta_backend.shutdown()
        if self._hf_backend:
            self._hf_backend.shutdown()


class StreamWorker(BaseWorker):
    """
    Stream Worker: True incremental frame-by-frame processing.
    Uses HFIncrementalBackend with persistent session.
    """
    
    def __init__(self, cfg: Settings, pool: ModelPool):
        super().__init__(cfg, pool)
        self._hf_backend: Optional[HFIncrementalBackend] = None
        self._active_sessions: dict[str, tuple] = {}  # session_id -> (session, prompt)
    
    def _get_backend(self) -> HFIncrementalBackend:
        if self._hf_backend is None:
            self._hf_backend = HFIncrementalBackend(self.cfg)
        return self._hf_backend
    
    def load(self):
        backend = self._get_backend()
        backend.load()
    
    async def create_session(self, prompt: str, threshold: float, max_frames: int) -> str:
        """Create new streaming session."""
        backend = self._get_backend()
        session = await asyncio.to_thread(backend.new_stream, prompt.strip())
        session_id = str(uuid.uuid4())
        self._active_sessions[session_id] = (session, prompt.strip(), threshold, max_frames, 0)
        return session_id
    
    async def process_frame(
        self,
        session_id: str,
        image: Image.Image,
    ) -> FrameMasks:
        """Process single frame in streaming session."""
        if session_id not in self._active_sessions:
            raise ValueError("Invalid or expired session")
        
        session, prompt, threshold, max_frames, frame_idx = self._active_sessions[session_id]
        if frame_idx >= max_frames:
            raise ValueError("Frame limit exceeded")
        
        backend = self._get_backend()
        
        def _process():
            return backend.process_frame(session, image, threshold)
        
        result = await asyncio.to_thread(_process)
        result.frame_index = frame_idx
        
        # Update frame index
        self._active_sessions[session_id] = (session, prompt, threshold, max_frames, frame_idx + 1)
        return result
    
    async def end_session(self, session_id: str) -> dict:
        """End streaming session and return summary."""
        if session_id not in self._active_sessions:
            return {"frames_processed": 0, "reason": "not_found"}
        
        session, prompt, threshold, max_frames, frame_idx = self._active_sessions.pop(session_id)
        backend = self._get_backend()
        await asyncio.to_thread(backend.close_stream, session)
        
        return {
            "frames_processed": frame_idx,
            "reason": "completed" if frame_idx >= max_frames else "cancelled",
        }
    
    def shutdown(self):
        for session_id in list(self._active_sessions.keys()):
            session, _, _, _, _ = self._active_sessions.pop(session_id)
            if self._hf_backend:
                self._hf_backend.close_stream(session)
        if self._hf_backend:
            self._hf_backend.shutdown()


class SingleWorker(BaseWorker):
    """
    Single Worker: Lightweight single-image inference without tracking.
    Uses HF model pool directly, bypassing video session overhead.
    """
    
    def __init__(self, cfg: Settings, pool: ModelPool):
        super().__init__(cfg, pool)
        self._processor: Optional[Sam3VideoProcessor] = None
        self._model_idx = 0  # Round-robin model index
    
    def load(self):
        """Ensure pool is initialized."""
        self.pool.initialize()
    
    def _get_model_processor(self):
        """Get model+processor from pool (round-robin)."""
        model, processor = self.pool.get_model_processor(self._model_idx)
        self._model_idx = (self._model_idx + 1) % self.pool.pool_size
        return model, processor
    
    async def segment_single(
        self,
        image: Image.Image,
        prompt: str,
        threshold: float,
    ) -> FrameMasks:
        """Process single image without tracking state."""
        model, processor = self._get_model_processor()
        
        def _infer():
            import torch
            inputs = processor(images=image, return_tensors="pt")
            pixels = inputs.pixel_values[0].to(device=self.cfg.device, dtype=model.dtype)
            
            with torch.inference_mode():
                # For single image, we use the model directly without video session
                # This requires accessing internal components
                vision_feats = model.vision_encoder(pixels.unsqueeze(0))
                
                # Get prompt embeddings
                text_inputs = processor.tokenizer(
                    prompt, return_tensors="pt", padding=True, truncation=True
                ).to(self.cfg.device)
                prompt_embeds = model.detector_model.get_text_features(**text_inputs)
                
                # Decode masks
                masks = model.mask_decoder(
                    vision_feats=vision_feats,
                    prompt_embeds=prompt_embeds,
                )
                
                # Post-process (simplified - no tracking)
                masks = masks.detach().cpu().numpy()
                if masks.ndim == 3:
                    masks = masks[0]  # Take first frame
                if masks.ndim == 2:
                    masks = masks[None]
                
                # Dummy object IDs and scores for compatibility
                object_ids = list(range(masks.shape[0]))
                scores = [0.9] * masks.shape[0]
                
                return FrameMasks(
                    frame_index=0,
                    object_ids=object_ids,
                    scores=scores,
                    masks=[m.astype(bool) for m in masks],
                )
        
        try:
            return await asyncio.to_thread(_infer)
        except AttributeError:
            # Fallback: use video session but close immediately
            backend = HFIncrementalBackend(self.cfg)
            backend.load()
            session = backend.new_stream(prompt)
            try:
                result = backend.process_frame(session, image, threshold)
                backend.close_stream(session)
                return result
            finally:
                backend.shutdown()
    
    def shutdown(self):
        pass  # Pool manages model lifecycle


# Import zipfile at module level for BatchWorker
import zipfile