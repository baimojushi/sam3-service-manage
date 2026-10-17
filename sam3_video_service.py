#!/usr/bin/env python3
"""
SAM3 Video Segmentation Service
FastAPI service for SAM3 video segmentation and tracking
"""

import io
import os
import json
import time
import threading
import zipfile
import uuid
from typing import Any, Optional, List, Dict
from dataclasses import dataclass, asdict

import numpy as np
import torch
from PIL import Image, ImageOps
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel

# SAM3 video via model_builder
import sys
sys.path.insert(0, '/opt/sam3-src')
from sam3.model_builder import build_sam3_video_predictor

MODEL_PATH = os.getenv("SAM3_MODEL_PATH", "/mnt/e/models/sam3/models/facebook--sam3/snapshots/master/sam3.pt")
API_KEY = os.getenv("API_KEY", "")
ALLOW_UNAUTH_LOCAL = os.getenv("ALLOW_UNAUTH_LOCAL", "").strip().lower() in {"1", "true", "yes", "on"}
MAX_SIDE = int(os.getenv("MAX_SIDE", "1024"))
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="SAM3 Video Segmentation API")
_PREDICTOR = None
_LOCK = threading.Lock()


def require_auth(authorization: Optional[str]):
    if not API_KEY:
        if ALLOW_UNAUTH_LOCAL:
            return
        raise HTTPException(500, "API_KEY is not configured; set ALLOW_UNAUTH_LOCAL=1 for local development")
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(401, "Unauthorized")


def _patch_text_projection(model_or_predictor):
    """Patch the corrupted text_projection in SAM3's TextTransformer.

    The text_projection parameter gets corrupted during checkpoint loading
    (values like -9.58e18). Re-initialize with proper CLIP-style initialization.
    """
    import torch
    text_transformer = None
    if hasattr(model_or_predictor, 'model'):
        model = model_or_predictor.model
    else:
        model = model_or_predictor
    if hasattr(model, 'detector') and hasattr(model.detector, 'backbone'):
        backbone = model.detector.backbone
        if hasattr(backbone, 'language_backbone') and hasattr(backbone.language_backbone, 'encoder'):
            text_transformer = backbone.language_backbone.encoder
    if text_transformer is None and hasattr(model, 'backbone'):
        backbone = model.backbone
        if hasattr(backbone, 'language_backbone') and hasattr(backbone.language_backbone, 'encoder'):
            text_transformer = backbone.language_backbone.encoder
    if text_transformer is None and hasattr(model, 'language_backbone'):
        lang_bb = model.language_backbone
        if hasattr(lang_bb, 'encoder'):
            text_transformer = lang_bb.encoder
    if text_transformer is None and model.__class__.__name__ == 'TextTransformer':
        text_transformer = model
    if text_transformer is None:
        print(f"[_patch_text_projection] Could not find TextTransformer in {type(model_or_predictor).__name__}")
        return model_or_predictor
    if hasattr(text_transformer, 'text_projection'):
        text_proj = text_transformer.text_projection
        min_val = text_proj.min().item()
        max_val = text_proj.max().item()
        if min_val < -1e10 or max_val > 1e10:
            print(f"[_patch_text_projection] Detected corrupted text_projection: min={min_val:.4e}, max={max_val:.4e}")
            with torch.no_grad():
                if isinstance(text_proj, torch.nn.Parameter):
                    std = text_proj.shape[0] ** -0.5
                    torch.nn.init.normal_(text_proj, std=std)
                else:
                    if hasattr(text_proj, 'weight'):
                        std = text_proj.weight.shape[1] ** -0.5
                        torch.nn.init.normal_(text_proj.weight, std=std)
                    if hasattr(text_proj, 'bias') and text_proj.bias is not None:
                        torch.nn.init.zeros_(text_proj.bias)
            new_min = text_proj.min().item()
            new_max = text_proj.max().item()
            print(f"[_patch_text_projection] Fixed text_projection: min={new_min:.4f}, max={new_max:.4f}")
        else:
            print(f"[_patch_text_projection] text_projection OK: min={min_val:.4f}, max={max_val:.4f}")
    else:
        print(f"[_patch_text_projection] No text_projection found in TextTransformer")
    return model_or_predictor


def _patch_predictor(predictor):
    """Monkey-patch SAM3.1 multiplex predictor to filter offload_state_to_cpu.

    SAM3.1 multiplex model's init_state() doesn't support offload_state_to_cpu parameter,
    but Sam3BasePredictor.start_session() passes it by default.
    """
    if hasattr(predictor, 'start_session'):
        original_start_session = predictor.start_session

        def patched_start_session(self, resource_path, session_id=None,
                                 offload_video_to_cpu=False, offload_state_to_cpu=False):
            # SAM3.1 multiplex doesn't support offload_state_to_cpu, filter it out
            init_kwargs = dict(
                resource_path=resource_path,
                offload_video_to_cpu=offload_video_to_cpu,
                # offload_state_to_cpu is NOT supported by multiplex model
            )
            if hasattr(self, "async_loading_frames"):
                init_kwargs["async_loading_frames"] = self.async_loading_frames
            if hasattr(self, "video_loader_type"):
                init_kwargs["video_loader_type"] = self.video_loader_type

            inference_state = self.model.init_state(**init_kwargs)

            if not session_id:
                session_id = str(uuid.uuid4())
            self._ALL_INFERENCE_STATES[session_id] = {
                "state": inference_state,
                "session_id": session_id,
                "start_time": time.time(),
                "last_use_time": time.time(),
            }
            logger.info(f"started new session {session_id}")
            return {"session_id": session_id}

        from sam3.logger import get_logger
        logger = get_logger(__name__)

        # Bind the patched method
        predictor.start_session = patched_start_session.__get__(predictor, predictor.__class__)

    return predictor


def get_predictor():
    global _PREDICTOR
    print(f"DEBUG: get_predictor() CALLED, _PREDICTOR={_PREDICTOR}", flush=True)
    with open("/tmp/threshold_debug.log", "a") as f:
        f.write(f"DEBUG: get_predictor() CALLED, _PREDICTOR={_PREDICTOR}\n")
        f.flush()
    # Test write immediately
    with open("/tmp/threshold_debug.log", "a") as f:
        f.write("DEBUG: Test write at top of get_predictor\n")
        f.flush()
    if _PREDICTOR is None:
        with _LOCK:
            if _PREDICTOR is None:
                with open("/tmp/threshold_debug.log", "a") as f:
                    f.write("DEBUG: Creating new predictor\n")
                print(f"DEBUG: Creating new predictor", flush=True)
                print(f"Loading SAM3 video predictor from {MODEL_PATH}...")
                _PREDICTOR = build_sam3_video_predictor(
                    checkpoint_path=MODEL_PATH,
                    async_loading_frames=False,
                    apply_temporal_disambiguation=True,
                    has_presence_token=True,
                    geo_encoder_use_img_cross_attn=True,
                    strict_state_dict_loading=True,
                    video_loader_type="cv2",
                    bpe_path="/mnt/e/models/sam3/models/facebook--sam3/snapshots/master/bpe_simple_vocab_16e6.txt.gz",
                    gpus_to_use=[0],
                )
                print("SAM3 video predictor loaded!")
                # Apply monkey-patch to fix offload_state_to_cpu issue
                _PREDICTOR = _patch_predictor(_PREDICTOR)
                # Apply text_projection corruption fix
                _PREDICTOR = _patch_text_projection(_PREDICTOR)
                # Lower detection threshold: default 0.5 filters out valid detections (brush prob ~0.24)
                with open("/tmp/threshold_debug.log", "a") as f:
                    f.write(f"DEBUG: Before threshold set: {_PREDICTOR.model.score_threshold_detection}\n")
                _PREDICTOR.model.score_threshold_detection = 0.1
                with open("/tmp/threshold_debug.log", "a") as f:
                    f.write(f"DEBUG: After threshold set: {_PREDICTOR.model.score_threshold_detection}\n")
                print(f"Lowered score_threshold_detection to {_PREDICTOR.model.score_threshold_detection}", flush=True)
    else:
        with open("/tmp/threshold_debug.log", "a") as f:
            f.write(f"DEBUG: Using cached predictor, threshold={_PREDICTOR.model.score_threshold_detection}\n")
    return _PREDICTOR


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "sam3.1",
        "device": DEVICE,
        "api": "sam3-video-segmentation-api",
        "model_path": MODEL_PATH
    }


class VideoSegmentRequest(BaseModel):
    """Video segmentation request parameters"""
    text_prompt: str
    frame_stride: Optional[int] = 1
    max_frames: Optional[int] = None


class VideoSegmentResponse(BaseModel):
    """Video segmentation response"""
    success: bool
    session_id: Optional[str] = None
    num_frames: Optional[int] = None
    num_objects: Optional[int] = None
    inference_time_ms: float = 0.0
    error: Optional[str] = None


@app.post("/video/start_session", response_model=VideoSegmentResponse)
async def video_start_session(
    file: UploadFile = File(...),
    text_prompt: str = Form(...),
    frame_stride: int = Form(1),
    max_frames: Optional[int] = Form(None),
    authorization: Optional[str] = Header(None)
):
    """
    Start a video segmentation session with a text prompt on the first frame.
    Returns a session_id for subsequent frame propagation.
    """
    require_auth(authorization)

    try:
        predictor = get_predictor()

        # Read video frames
        import cv2
        raw = await file.read()
        temp_path = f"/tmp/video_{int(time.time())}.mp4"
        with open(temp_path, 'wb') as f:
            f.write(raw)

        cap = cv2.VideoCapture(temp_path)
        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Resize if needed
            if max(pil_image.size) > MAX_SIDE:
                pil_image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)

            frames.append(np.array(pil_image))
            frame_count += 1

            if max_frames and frame_count >= max_frames:
                break

            # Skip frames based on stride
            for _ in range(frame_stride - 1):
                cap.read()

        cap.release()

        if len(frames) == 0:
            os.unlink(temp_path)
            raise HTTPException(400, "No frames extracted from video")

        print(f"Extracted {len(frames)} frames")

        # Start session with first frame
        start = time.perf_counter()
        resource_path = temp_path  # We'll use the video path as resource

        # Use the predictor's handle_request API
        # First frame with text prompt
        response = predictor.handle_request({
            "type": "start_session",
            "resource_path": resource_path
        })

        session_id = response.get("session_id")
        if not session_id:
            os.unlink(temp_path)
            raise RuntimeError(f"Failed to start session: {response}")

        # Add text prompt on first frame
        response = predictor.handle_request({
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": text_prompt
        })

        inference_time_ms = (time.perf_counter() - start) * 1000

        # Masks are nested under response["outputs"]["out_binary_masks"]
        outputs = response.get("outputs", {})
        num_objects = len(outputs.get("out_binary_masks", []))

        # Clean up temp file
        os.unlink(temp_path)

        return VideoSegmentResponse(
            success=True,
            session_id=session_id,
            num_frames=len(frames),
            num_objects=num_objects,
            inference_time_ms=inference_time_ms
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return VideoSegmentResponse(
            success=False,
            error=str(e),
            inference_time_ms=0.0
        )


@app.post("/video/propagate")
async def video_propagate(
    session_id: str = Form(...),
    authorization: Optional[str] = Header(None)
):
    """
    Propagate segmentation through the video.
    Returns masklets (spatio-temporal masks) for all objects.
    """
    require_auth(authorization)

    try:
        predictor = get_predictor()

        start = time.perf_counter()

        # Stream results
        results = []
        for out in predictor.handle_stream_request({
            "type": "propagate_in_video",
            "session_id": session_id
        }):
            outputs = out.get("outputs", {})
            masks = outputs.get("out_binary_masks", [])
            if masks:
                results.append({
                    "frame_index": out.get("frame_index"),
                    "num_objects": len(masks),
                    "mask_shapes": [m.shape for m in masks]
                })

        inference_time_ms = (time.perf_counter() - start) * 1000

        return JSONResponse({
            "success": True,
            "session_id": session_id,
            "propagation_results": results,
            "inference_time_ms": inference_time_ms
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/video/segment")
async def video_segment(
    file: UploadFile = File(...),
    text_prompt: str = Form(...),
    frame_stride: int = Form(1),
    max_frames: Optional[int] = Form(None),
    return_masks_png: bool = Form(False),
    authorization: Optional[str] = Header(None)
):
    """
    Segment a video with a text prompt and return all masklets.
    This is a convenience endpoint that does start_session + propagate in one call.
    """
    require_auth(authorization)

    try:
        predictor = get_predictor()

        # Read video frames
        import cv2
        raw = await file.read()
        temp_path = f"/tmp/video_{int(time.time())}.mp4"
        with open(temp_path, 'wb') as f:
            f.write(raw)

        # Start session
        response = predictor.handle_request({
            "type": "start_session",
            "resource_path": temp_path
        })

        session_id = response.get("session_id")
        if not session_id:
            raise RuntimeError(f"Failed to start session: {response}")

        # Add text prompt on first frame
        response = predictor.handle_request({
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": text_prompt
        })

        # DEBUG: Log the raw response
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"add_prompt raw response keys: {list(response.keys()) if isinstance(response, dict) else type(response)}")
        if isinstance(response, dict) and "outputs" in response:
            outputs = response["outputs"]
            logger.info(f"add_prompt outputs keys: {list(outputs.keys())}")
            logger.info(f"add_prompt out_binary_masks: {len(outputs.get('out_binary_masks', []))}")
            logger.info(f"add_prompt out_probs: {outputs.get('out_probs', [])}")
            logger.info(f"add_prompt out_obj_ids: {outputs.get('out_obj_ids', [])}")

        # Propagate
        all_masks = []
        all_scores = []
        frame_indices = []

        for out in predictor.handle_stream_request({
            "type": "propagate_in_video",
            "session_id": session_id
        }):
            frame_idx = out.get("frame_index", 0)
            outputs = out.get("outputs", {})
            masks = outputs.get("out_binary_masks", [])
            scores = outputs.get("out_probs", [1.0] * len(masks))

            frame_indices.append(frame_idx)
            all_masks.append(masks)
            all_scores.append(scores)

        # Clean up session
        predictor.handle_request({
            "type": "close_session",
            "session_id": session_id
        })

        # Clean up temp file
        os.unlink(temp_path)

        # Prepare response
        response_data = {
            "success": True,
            "session_id": session_id,
            "num_frames": len(frame_indices),
            "num_objects": len(all_masks[0]) if all_masks else 0,
            "frame_indices": frame_indices,
            "object_scores": all_scores[0] if all_scores else []
        }

        if return_masks_png:
            # Create ZIP with PNG masks for each frame
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr("meta.json", json.dumps({
                    "text_prompt": text_prompt,
                    "num_frames": len(frame_indices),
                    "num_objects": len(all_masks[0]) if all_masks else 0,
                    "frame_indices": frame_indices,
                    "object_scores": all_scores[0] if all_scores else []
                }, ensure_ascii=False, indent=2))

                for frame_idx, (masks, scores) in enumerate(zip(all_masks, all_scores)):
                    for obj_idx, (mask, score) in enumerate(zip(masks, scores)):
                        mask_np = np.array(mask, dtype=np.uint8) * 255
                        mask_img = Image.fromarray(mask_np)
                        mask_buf = io.BytesIO()
                        mask_img.save(mask_buf, format="PNG")
                        z.writestr(f"frame_{frame_idx:04d}_obj_{obj_idx:02d}_score{score:.4f}.png", mask_buf.getvalue())

            return Response(
                buf.getvalue(),
                media_type="application/zip",
                headers={"Content-Disposition": f'attachment; filename="sam3_video_masks_{text_prompt.replace(" ", "_")}.zip"'}
            )

        return JSONResponse(response_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("SAM3_VIDEO_PORT", "6014"))
    uvicorn.run(app, host="0.0.0.0", port=port)