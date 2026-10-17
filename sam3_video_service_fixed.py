#!/usr/bin/env python3
import io, os, json, time, threading, zipfile, uuid, subprocess, sys, glob
from typing import Optional
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel

sys.path.insert(0, '/opt/sam3-src')
from sam3.model_builder import build_sam3_video_predictor

MODEL_PATH = os.getenv("SAM3_MODEL_PATH", "/mnt/e/models/sam3/models/facebook--sam3/snapshots/master/sam3.pt")
API_KEY = os.getenv("API_KEY", "")
ALLOW_UNAUTH_LOCAL = os.getenv("ALLOW_UNAUTH_LOCAL", "").strip().lower() in {"1", "true", "yes", "on"}
MAX_SIDE = int(os.getenv("MAX_SIDE", "1024"))
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

STAGES_LOG = "/tmp/sam3-segment-stages.jsonl"
FATAL_LOG = "/tmp/sam3-fatal.log"

def log_stage(stage, **kwargs):
    rec = {"stage": stage, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **kwargs}
    with open(STAGES_LOG, "a") as f:
        f.write(json.dumps(rec, default=str) + "\\n")
        f.flush()

def log_fatal(msg):
    with open(FATAL_LOG, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}\\n")
        f.flush()

def get_frame_count(video_path):
    """Get the number of frames in a video file using FFprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=nokey=1:noprint_wrappers=1",
        video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return int(result.stdout.strip())
    except Exception as e:
        log_fatal(f"Failed to get frame count for {video_path}: {e}")
        # Fallback: try to estimate by reading the video with OpenCV? 
        # For now, we return 0 to avoid breaking the service, but this will cause issues.
        return 0

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
    if hasattr(predictor, 'start_session'):
        def patched_start_session(self, resource_path, session_id=None,
                                 offload_video_to_cpu=False, offload_state_to_cpu=False):
            init_kwargs = dict(
                resource_path=resource_path,
                offload_video_to_cpu=offload_video_to_cpu,
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
            from sam3.logger import get_logger
            logger = get_logger(__name__)
            logger.info(f"started new session {session_id}")
            return {"session_id": session_id}

        from sam3.logger import get_logger
        logger = get_logger(__name__)
        predictor.start_session = patched_start_session.__get__(predictor, predictor.__class__)

    return predictor


def get_predictor():
    global _PREDICTOR
    if _PREDICTOR is None:
        with _LOCK:
            if _PREDICTOR is None:
                log_stage("predictor.load.begin", model_path=MODEL_PATH)
                print(f"Loading SAM3 video predictor from {MODEL_PATH}...", flush=True)
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
                print("SAM3 video predictor loaded!", flush=True)
                _PREDICTOR = _patch_predictor(_PREDICTOR)
                _PREDICTOR = _patch_text_projection(_PREDICTOR)
                _PREDICTOR.model.score_threshold_detection = 0.1
                print(f"Lowered score_threshold_detection to {_PREDICTOR.model.score_threshold_detection}", flush=True)
                log_stage("predictor.load.done", threshold=_PREDICTOR.model.score_threshold_detection)
    return _PREDICTOR


def _ffmpeg_extract_frames_to_video(input_path, output_path, max_frames=None, frame_stride=1):
    """Extract frames from input video to output video using FFmpeg, honoring max_frames and frame_stride.
    Returns the number of frames in the output video.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", input_path]
    if max_frames is not None:
        cmd += ["-frames:v", str(max_frames)]
    if frame_stride and frame_stride > 1:
        cmd += ["-vf", f"select='not(mod(n\\,{frame_stride}))'"]
    cmd += [output_path]
    subprocess.run(cmd, check=True)
    # Get the number of frames in the output video
    return get_frame_count(output_path)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "sam3.1",
        "device": DEVICE,
        "api": "sam3-video-segmentation-api",
        "model_path": MODEL_PATH
    }


@app.post("/video/segment")
async def video_segment(
    file: UploadFile = File(...),
    text_prompt: str = Form(...),
    frame_stride: int = Form(1),
    max_frames: Optional[int] = Form(None),
    return_masks_png: bool = Form(False),
    authorization: Optional[str] = Header(None)
):
    require_auth(authorization)
    seg_id = str(uuid.uuid4())[:8]
    try:
        predictor = get_predictor()

        # Read uploaded video
        raw = await file.read()
        temp_path = f"/tmp/video_{int(time.time())}.mp4"
        with open(temp_path, 'wb') as f:
            f.write(raw)

        # Get the original frame count of the uploaded video
        original_frame_count = get_frame_count(temp_path)
        log_stage("ffmpeg.extract.begin", seg_id=seg_id, video_path=temp_path,
                  max_frames=max_frames, frame_stride=frame_stride,
                  original_frame_count=original_frame_count)

        # Extract frames to a new video file
        extracted_path = f"/tmp/extracted_{int(time.time())}.mp4"
        num_frames_extracted = _ffmpeg_extract_frames_to_video(temp_path, extracted_path, max_frames=max_frames, frame_stride=frame_stride)
        log_stage("ffmpeg.extract.done", seg_id=seg_id, num_frames=num_frames_extracted,
                  extracted_video_path=extracted_path)

        # Start session with the extracted video
        log_stage("session.start.begin", seg_id=seg_id)
        response = predictor.handle_request({
            "type": "start_session",
            "resource_path": extracted_path
        })
        session_id = response.get("session_id")
        if not session_id:
            raise RuntimeError(f"Failed to start session: {response}")
        log_stage("session.start.done", seg_id=seg_id, session_id=session_id)

        # Add text prompt
        log_stage("prompt.add.begin", seg_id=seg_id, text_prompt=text_prompt)
        response = predictor.handle_request({
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": text_prompt
        })
        outputs = response.get("outputs", {})
        num_objects = len(outputs.get("out_binary_masks", []))
        log_stage("prompt.add.done", seg_id=seg_id, num_objects=num_objects,
                  out_probs=outputs.get("out_probs", []))

        # Propagate
        log_stage("propagation.begin", seg_id=seg_id)
        all_masks, all_scores, frame_indices_extracted = [], [], []
        for out in predictor.handle_stream_request({
            "type": "propagate_in_video",
            "session_id": session_id
        }):
            frame_idx = out.get("frame_index", 0)  # frame index in the extracted video
            outs = out.get("outputs", {})
            masks = outs.get("out_binary_masks", [])
            scores = outs.get("out_probs", [1.0] * len(masks))
            frame_indices_extracted.append(frame_idx)
            all_masks.append(masks)
            all_scores.append(scores)
            # Convert frame index in extracted video to original video frame index
            original_frame_index = frame_idx * frame_stride
            log_stage("propagation.frame", seg_id=seg_id, frame_index=original_frame_index,
                      num_masks=len(masks), scores=scores)
        log_stage("propagation.done", seg_id=seg_id, num_frames=len(frame_indices_extracted))

        # Close session
        predictor.handle_request({
            "type": "close_session",
            "session_id": session_id
        })

        # Clean up temp files
        os.unlink(temp_path)
        os.unlink(extracted_path)

        # Prepare response
        # Convert frame indices in extracted video to original video frame indices
        frame_indices_original = [int(idx * frame_stride) for idx in frame_indices_extracted]
        # Object scores: we return the scores for the first object across all frames
        # all_scores is a list of lists: [frame0_scores, frame1_scores, ...] where each frame_scores is a list of scores for all objects in that frame.
        # We want: [frame0_score_obj0, frame1_score_obj0, ...]
        object_scores = [float(frame_scores[0]) for frame_scores in all_scores] if all_scores and all_scores[0] else []

        response_data = {
            "success": True,
            "session_id": session_id,
            "num_frames": len(frame_indices_original),
            "num_objects": len(all_masks[0]) if all_masks else 0,
            "frame_indices": frame_indices_original,
            "object_scores": object_scores
        }

        if return_masks_png:
            log_stage("zip.ready", seg_id=seg_id, num_frames=len(frame_indices_original))
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr("meta.json", json.dumps({
                    "text_prompt": text_prompt,
                    "num_frames": len(frame_indices_original),
                    "num_objects": len(all_masks[0]) if all_masks else 0,
                    "frame_indices": frame_indices_original,
                    "object_scores": object_scores
                }, ensure_ascii=False, indent=2))
                for frame_idx, (masks, scores) in enumerate(zip(all_masks, all_scores)):
                    for obj_idx, (mask, score) in enumerate(zip(masks, scores)):
                        mask_np = np.array(mask, dtype=np.uint8) * 255
                        mask_img = Image.fromarray(mask_np)
                        mask_buf = io.BytesIO()
                        mask_img.save(mask_buf, format="PNG")
                        z.writestr(f"frame_{frame_idx:04d}_obj_{obj_idx:02d}_score{score:.4f}.png", mask_buf.getvalue())
            log_stage("zip.ready", seg_id=seg_id, zip_size_bytes=len(buf.getvalue()))
            return Response(
                buf.getvalue(),
                media_type="application/zip",
                headers={"Content-Disposition": f'attachment; filename="sam3_video_masks_{text_prompt.replace(" ", "_")}.zip"'}
            )

        log_stage("segment.done", seg_id=seg_id, num_frames=len(frame_indices_original))
        return JSONResponse(response_data)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log_fatal(f"seg_id={seg_id} exception: {e}\\n{tb}")
        traceback.print_exc()
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("SAM3_VIDEO_PORT", "6014"))
    host = "127.0.0.1" if not API_KEY else os.getenv("SAM3_VIDEO_HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)