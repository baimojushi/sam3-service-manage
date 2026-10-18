#!/usr/bin/env python3
"""Long streaming validation: 50/100/400 frames with stateless per-frame approach."""
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

# Track per-frame results for IoU computation later
frame_results = []
obj_id_continuity = []  # Track object ID continuity

def process_frame(image, frame_idx):
    """Process single frame with fresh session (stateless)."""
    session = proc.init_video_session(
        inference_device=torch.device("cuda:0"),
        inference_state_device=torch.device("cuda:0"),
        processing_device=torch.device("cpu"),
        video_storage_device=torch.device("cpu"),
        max_vision_features_cache_size=1,
        dtype=torch.bfloat16,
    )
    proc.add_text_prompt(inference_session=session, text=PROMPT)
    
    inputs = proc(images=image, return_tensors="pt")
    pixels = inputs.pixel_values[0].to(device="cuda:0", dtype=torch.bfloat16)
    
    with torch.inference_mode():
        out = model(inference_session=session, frame=pixels, reverse=False)
        processed = proc.postprocess_outputs(
            session, out, original_sizes=inputs.original_sizes)
    
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
    
    # Cleanup
    del session, out, processed, pixels, inputs
    gc.collect()
    torch.cuda.empty_cache()
    
    return ids, scores, masks

# Target frame counts to validate
targets = [50, 100, 400]
results_by_target = {}

for target_frames in targets:
    print(f"\n=== STREAMING {target_frames} FRAMES ===")
    n = min(target_frames, total)
    
    # Reset video to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    frame_times, gpu_peak_alloc, gpu_peak_reserved = [], 0, 0
    cpu_peak = 0
    obj_ids_history = []
    masks_per_frame = []
    
    i = 0
    t_start = time.time()
    try:
        while i < n:
            ok, bgr = cap.read()
            if not ok:
                break
            image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            
            mem_before = gpu_mem()
            cpu_before = cpu_mem()
            
            t1 = time.time()
            ids, scores, masks = process_frame(image, i)
            dt = time.time() - t1
            
            mem_after = gpu_mem()
            cpu_after = cpu_mem()
            
            frame_times.append(dt)
            gpu_peak_alloc = max(gpu_peak_alloc, mem_after["alloc_mb"])
            gpu_peak_reserved = max(gpu_peak_reserved, mem_after["reserved_mb"])
            cpu_peak = max(cpu_peak, cpu_after)
            
            kept_ids = [int(ids[j]) for j, s in enumerate(scores) if float(s) >= THRESH]
            kept_masks = masks[[j for j, s in enumerate(scores) if float(s) >= THRESH]]
            obj_ids_history.append(kept_ids)
            masks_per_frame.append(len(kept_masks))
            
            if i % 10 == 0:
                print(f"  frame {i}: time={dt:.3f}s gpu_alloc={mem_after['alloc_mb']:.0f}MB gpu_reserved={mem_after['reserved_mb']:.0f}MB cpu={cpu_after:.0f}MB objs={kept_ids} masks={len(kept_masks)}")
            
            i += 1
            
    finally:
        pass  # cap released at end
    
    elapsed = time.time() - t_start
    fps = i / elapsed if elapsed > 0 else 0
    
    # Object ID continuity check
    continuity_ok = True
    for f in range(1, len(obj_ids_history)):
        if set(obj_ids_history[f]) != set(obj_ids_history[f-1]):
            continuity_ok = False
            break
    
    stats = {
        "target_frames": target_frames,
        "frames_completed": i,
        "elapsed_s": round(elapsed, 2),
        "fps": round(fps, 2),
        "avg_frame_s": round(np.mean(frame_times), 4) if frame_times else 0,
        "max_frame_s": round(max(frame_times), 4) if frame_times else 0,
        "gpu_alloc_peak_mb": round(gpu_peak_alloc, 1),
        "gpu_reserved_peak_mb": round(gpu_peak_reserved, 1),
        "cpu_peak_mb": round(cpu_peak, 1),
        "obj_id_continuity": continuity_ok,
        "obj_ids_per_frame": obj_ids_history,
        "masks_per_frame": masks_per_frame,
        "avg_masks_per_frame": round(np.mean(masks_per_frame), 2) if masks_per_frame else 0,
    }
    
    results_by_target[f"STREAMING_{target_frames}"] = stats
    print(f"  RESULT: frames={i}, fps={fps:.2f}, peak_alloc={gpu_peak_alloc:.0f}MB, peak_reserved={gpu_peak_reserved:.0f}MB, continuity={continuity_ok}, avg_masks={stats['avg_masks_per_frame']}")

cap.release()

print("\n=== FINAL SUMMARY ===")
for k, v in results_by_target.items():
    print(f"{k}: {json.dumps(v, indent=2)}")

with open("/tmp/streaming_validation_results.json", "w") as f:
    json.dump(results_by_target, f, indent=2)
print("\nSaved /tmp/streaming_validation_results.json")