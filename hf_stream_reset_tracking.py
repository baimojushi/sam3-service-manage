#!/usr/bin/env python3
"""HF SAM3 streaming: bfloat16 + reset_tracking_data every 10 frames (lightweight)."""
import os, sys, time, json, gc, psutil
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
RESET_EVERY = 10

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

print("=== load model (bfloat16) ===")
t0 = time.time()
model = Sam3VideoModel.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
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

# Single session for entire run
session = proc.init_video_session(
    inference_device=torch.device("cuda:0"),
    inference_state_device=torch.device("cuda:0"),
    processing_device=torch.device("cpu"),
    video_storage_device=torch.device("cpu"),
    max_vision_features_cache_size=1,
    dtype=torch.bfloat16,
)
proc.add_text_prompt(inference_session=session, text=PROMPT)

frame_times, gpu_peak_alloc, gpu_peak_reserved = [], 0, 0
cpu_peak = 0
obj_ids_history = []
masks_per_frame = []

i = 0
t_start = time.time()
try:
    while i < total:
        ok, bgr = cap.read()
        if not ok:
            break
        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        
        mem_before = gpu_mem()
        cpu_before = cpu_mem()
        
        inputs = proc(images=image, return_tensors="pt")
        pixels = inputs.pixel_values[0].to(device="cuda:0", dtype=torch.bfloat16)
        
        with torch.inference_mode():
            out = model(inference_session=session, frame=pixels, reverse=False)
            processed = proc.postprocess_outputs(
                session, out, original_sizes=inputs.original_sizes)
        
        mem_after = gpu_mem()
        cpu_after = cpu_mem()
        
        dt = time.time() - t_start if i == 0 else time.time() - (t_start + sum(frame_times))
        frame_times.append(dt)
        
        ids = processed["object_ids"]
        scores = processed["scores"]
        masks = processed["masks"]
        
        if hasattr(ids, "detach"): ids = ids.detach().cpu().numpy()
        if hasattr(scores, "detach"): scores = scores.detach().float().cpu().numpy()
        if hasattr(masks, "detach"): masks = masks.detach().cpu().numpy()
        
        ids = np.asarray(ids).reshape(-1)
        scores = np.asarray(scores).reshape(-1)
        masks = np.asarray(masks)
        if masks.ndim == 2: masks = masks[None]
        
        kept_ids = [int(ids[j]) for j, s in enumerate(scores) if float(s) >= THRESH]
        kept_masks = masks[[j for j, s in enumerate(scores) if float(s) >= THRESH]]
        obj_ids_history.append(kept_ids)
        masks_per_frame.append(len(kept_masks))
        
        gpu_peak_alloc = max(gpu_peak_alloc, mem_after["alloc_mb"])
        gpu_peak_reserved = max(gpu_peak_reserved, mem_after["reserved_mb"])
        cpu_peak = max(cpu_peak, cpu_after)
        
        if i % 10 == 0:
            print(f"  frame {i}: time={dt:.3f}s gpu_alloc={mem_after['alloc_mb']:.0f}MB gpu_reserved={mem_after['reserved_mb']:.0f}MB cpu={cpu_after:.0f}MB objs={kept_ids} masks={len(kept_masks)}")
        
        # Lightweight reset: clear tracker dicts only, KEEP vision cache
        if (i + 1) % RESET_EVERY == 0:
            print(f"  >>> reset_tracking_data at frame {i+1}")
            session.reset_tracking_data()
            gc.collect()
            torch.cuda.empty_cache()
            mem_after_reset = gpu_mem()
            print(f"  after_reset: alloc={mem_after_reset['alloc_mb']:.0f}MB reserved={mem_after_reset['reserved_mb']:.0f}MB")
        
        i += 1
        del out, processed, pixels, inputs
        
finally:
    cap.release()
    reset = getattr(session, "reset_inference_session", None)
    if callable(reset): reset()

elapsed = time.time() - t_start
fps = i / elapsed if elapsed > 0 else 0

# Object ID continuity check
continuity_ok = True
for f in range(1, len(obj_ids_history)):
    if set(obj_ids_history[f]) != set(obj_ids_history[f-1]):
        continuity_ok = False
        break

print(f"\n=== RESULT ===")
print(f"frames={i}, elapsed={elapsed:.1f}s, fps={fps:.2f}")
print(f"peak_alloc={gpu_peak_alloc:.0f}MB peak_reserved={gpu_peak_reserved:.0f}MB cpu_peak={cpu_peak:.0f}MB")
print(f"continuity={continuity_ok} avg_masks={np.mean(masks_per_frame):.2f}")
print(f"avg_frame_s={np.mean(frame_times):.4f} max_frame_s={max(frame_times):.4f}")