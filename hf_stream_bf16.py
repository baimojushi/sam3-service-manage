#!/usr/bin/env python3
"""HF SAM3 streaming test with bfloat16 + aggressive memory management."""
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
RESET_EVERY = 50

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

session = proc.init_video_session(
    inference_device=torch.device("cuda:0"),
    inference_state_device=torch.device("cuda:0"),
    processing_device=torch.device("cpu"),
    video_storage_device=torch.device("cpu"),
    max_vision_features_cache_size=1,
    dtype=torch.bfloat16,
)
proc.add_text_prompt(inference_session=session, text=PROMPT)

frame_times = []
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
        if hasattr(ids, "detach"): ids = ids.detach().cpu().numpy()
        if hasattr(scores, "detach"): scores = scores.detach().float().cpu().numpy()
        ids = np.asarray(ids).reshape(-1); scores = np.asarray(scores).reshape(-1)
        kept = sum(1 for s in scores if float(s) >= THRESH)
        
        if i % 10 == 0 or i >= 45:
            print(f"=== FRAME {i} ===")
            print(f"  before: alloc={mem_before['alloc_mb']:.0f}MB reserved={mem_before['reserved_mb']:.0f}MB")
            print(f"  after: alloc={mem_after['alloc_mb']:.0f}MB reserved={mem_after['reserved_mb']:.0f}MB")
            print(f"  cpu: {cpu_before:.0f} -> {cpu_after:.0f}MB")
            print(f"  delta_alloc: {mem_after['alloc_mb'] - mem_before['alloc_mb']:.1f}MB")
            print(f"  delta_reserved: {mem_after['reserved_mb'] - mem_before['reserved_mb']:.1f}MB")
            print(f"  obj_ids={session.obj_ids} num_obj={len(session.obj_ids)}")
            print(f"  non_cond_frames={len(session.output_dict_per_obj.get(0, {}).get('non_cond_frame_outputs', {}))}")
        
        if (i + 1) % RESET_EVERY == 0:
            print(f"  >>> RESET at frame {i+1}")
            session.reset_inference_session()
            proc.add_text_prompt(inference_session=session, text=PROMPT)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            mem_after_reset = gpu_mem()
            print(f"  after_reset: alloc={mem_after_reset['alloc_mb']:.0f}MB reserved={mem_after_reset['reserved_mb']:.0f}MB")
        
        i += 1
        del out, processed, pixels, inputs
        
        if i % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

finally:
    cap.release()
    reset = getattr(session, "reset_inference_session", None)
    if callable(reset): reset()

elapsed = time.time() - t_start
fps = i / elapsed if elapsed > 0 else 0
print(f"\n=== FINAL: frames={i}, elapsed={elapsed:.1f}s, fps={fps:.2f} ===")
print(f"final gpu: {gpu_mem()} cpu={cpu_mem():.0f}MB")