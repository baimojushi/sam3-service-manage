#!/usr/bin/env python3
"""Diagnose frame 11 OOM: per-frame GPU/CPU memory + tracker state audit."""
import os, sys, time, gc, psutil, traceback
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
VENV = "/opt/molink/app/.venv_hf"
sys.path.insert(0, VENV + "/bin")

import torch
import numpy as np
from PIL import Image
import cv2
from transformers import Sam3VideoModel, Sam3VideoProcessor

MODEL = "/mnt/e/models/sam3"
VIDEO = "/mnt/c/Users/root/Downloads/sam3_calligraphy_brush_640x480_20fps_20s.mp4"
PROMPT = "brush"
THRESH = 0.1

def gpu_mem():
    if torch.cuda.is_available():
        return {
            "alloc_mb": torch.cuda.memory_allocated() / 1e6,
            "reserved_mb": torch.cuda.memory_reserved() / 1e6,
            "max_alloc_mb": torch.cuda.max_memory_allocated() / 1e6,
            "max_reserved_mb": torch.cuda.max_memory_reserved() / 1e6,
        }
    return {"alloc_mb": 0, "reserved_mb": 0, "max_alloc_mb": 0, "max_reserved_mb": 0}

def cpu_mem():
    return psutil.Process(os.getpid()).memory_info().rss / 1e6

def audit_session(session, frame_idx):
    """Audit tracker state sizes."""
    state = {
        "frame_idx": frame_idx,
        "num_obj_ids": len(session.obj_ids),
        "obj_ids": list(session.obj_ids),
        "processed_frames": len(session.processed_frames) if session.processed_frames else 0,
        "mask_inputs_per_obj": {k: len(v) for k, v in session.mask_inputs_per_obj.items()},
        "point_inputs_per_obj": {k: len(v) for k, v in session.point_inputs_per_obj.items()},
        "output_dict_per_obj": {},
        "frames_tracked_per_obj": {},
        "obj_id_to_score": dict(session.obj_id_to_score),
        "obj_id_to_tracker_score_frame_wise": {k: dict(v) for k, v in session.obj_id_to_tracker_score_frame_wise.items()},
        "obj_id_to_last_occluded": dict(session.obj_id_to_last_occluded),
        "max_obj_id": session.max_obj_id,
        "obj_first_frame_idx": dict(session.obj_first_frame_idx),
        "unmatched_frame_inds": {k: list(v) for k, v in session.unmatched_frame_inds.items()},
        "overlap_pair_to_frame_inds": {str(k): list(v) for k, v in session.overlap_pair_to_frame_inds.items()},
        "trk_keep_alive": dict(session.trk_keep_alive),
        "removed_obj_ids": list(session.removed_obj_ids),
        "suppressed_obj_ids": {k: list(v) for k, v in session.suppressed_obj_ids.items()},
    }
    # Detailed per-object output dict sizes
    for obj_idx, out_dict in session.output_dict_per_obj.items():
        state["output_dict_per_obj"][obj_idx] = {
            "cond": {k: len(v) for k, v in out_dict.get("cond_frame_outputs", {}).items()},
            "non_cond": {k: len(v) for k, v in out_dict.get("non_cond_frame_outputs", {}).items()},
        }
    for obj_idx, frames_dict in session.frames_tracked_per_obj.items():
        state["frames_tracked_per_obj"][obj_idx] = {k: len(v) if isinstance(v, dict) else v for k, v in frames_dict.items()}
    return state

print("=== load model ===")
t0 = time.time()
model = Sam3VideoModel.from_pretrained(MODEL, dtype=torch.float32).cuda().eval()
proc = Sam3VideoProcessor.from_pretrained(MODEL)
print(f"load_s={time.time()-t0:.1f} {gpu_mem()} cpu={cpu_mem():.0f}MB")

# Adapter fix
_orig_get_text = model.detector_model.get_text_features
def _patched_get_text(input_ids, attention_mask=None, **kwargs):
    out = _orig_get_text(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    return out.pooler_output if hasattr(out, "pooler_output") else out
model.detector_model.get_text_features = _patched_get_text

cap = cv2.VideoCapture(VIDEO)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"video frames: {total}")

session = proc.init_video_session(
    inference_device=torch.device("cuda:0"),
    inference_state_device=torch.device("cuda:0"),
    processing_device=torch.device("cpu"),
    video_storage_device=torch.device("cpu"),
    max_vision_features_cache_size=1,
    dtype=torch.float32,
)
proc.add_text_prompt(inference_session=session, text=PROMPT)

print("=== initial session config ===")
print(f"inference_device={session.inference_device}")
print(f"inference_state_device={session.inference_state_device}")
print(f"video_storage_device={session.video_storage_device}")

frame_audits = []
try:
    for i in range(min(15, total)):  # Run to frame 14 to catch frame 11
        ok, bgr = cap.read()
        if not ok:
            break
        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        
        mem_before = gpu_mem()
        cpu_before = cpu_mem()
        
        inputs = proc(images=image, return_tensors="pt")
        pixels = inputs.pixel_values[0].to(device="cuda:0", dtype=torch.float32)
        
        try:
            with torch.inference_mode():
                out = model(inference_session=session, frame=pixels, reverse=False)
                processed = proc.postprocess_outputs(
                    session, out, original_sizes=inputs.original_sizes)
        except Exception as e:
            mem_after = gpu_mem()
            cpu_after = cpu_mem()
            print(f"=== FRAME {i} EXCEPTION ===")
            print(f"mem_before: {mem_before}")
            print(f"mem_after: {mem_after}")
            print(f"cpu_before: {cpu_before:.0f}MB cpu_after: {cpu_after:.0f}MB")
            traceback.print_exc()
            break
        
        mem_after = gpu_mem()
        cpu_after = cpu_mem()
        
        audit = audit_session(session, i)
        frame_audits.append(audit)
        
        print(f"=== FRAME {i} ===")
        print(f"  gpu_alloc={mem_after['alloc_mb']:.0f}MB reserved={mem_after['reserved_mb']:.0f}MB peak_alloc={mem_after['max_alloc_mb']:.0f}MB")
        print(f"  cpu={cpu_after:.0f}MB")
        print(f"  obj_ids={audit['obj_ids']} num_obj={audit['num_obj_ids']}")
        print(f"  processed_frames={audit['processed_frames']}")
        print(f"  mask_inputs={audit['mask_inputs_per_obj']}")
        print(f"  output_dict_cond={audit['output_dict_per_obj']}")
        print(f"  frames_tracked={audit['frames_tracked_per_obj']}")
        print(f"  scores={audit['obj_id_to_score']}")
        print(f"  tracker_scores={audit['obj_id_to_tracker_score_frame_wise']}")
        print(f"  trk_keep_alive={audit['trk_keep_alive']}")
        print(f"  removed={audit['removed_obj_ids']} suppressed={audit['suppressed_obj_ids']}")
        
        del out, processed, pixels, inputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

finally:
    cap.release()
    reset = getattr(session, "reset_inference_session", None)
    if callable(reset): reset()

print("=== SUMMARY ===")
import json
with open("/tmp/frame_audits.json", "w") as f:
    json.dump(frame_audits, f, indent=2)
print("Saved /tmp/frame_audits.json")