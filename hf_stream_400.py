#!/usr/bin/env python3
"""Lightweight 400-frame streaming test with periodic memory sampling."""
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

results = {}

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

frame_times, mem_snapshots = [], []
i = 0
t_start = time.time()
try:
    while i < total:
        ok, bgr = cap.read()
        if not ok:
            break
        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        t1 = time.time()
        inputs = proc(images=image, return_tensors="pt")
        pixels = inputs.pixel_values[0].to(device="cuda:0", dtype=torch.float32)
        with torch.inference_mode():
            out = model(inference_session=session, frame=pixels, reverse=False)
            processed = proc.postprocess_outputs(
                session, out, original_sizes=inputs.original_sizes)
        dt = time.time() - t1
        ids = processed["object_ids"]
        scores = processed["scores"]
        if hasattr(ids, "detach"): ids = ids.detach().cpu().numpy()
        if hasattr(scores, "detach"): scores = scores.detach().float().cpu().numpy()
        ids = np.asarray(ids).reshape(-1); scores = np.asarray(scores).reshape(-1)
        kept = sum(1 for s in scores if float(s) >= THRESH)
        frame_times.append(dt)
        
        # Periodic memory snapshot (every 25 frames)
        if i % 25 == 0 or i < 15:
            mem_snapshots.append({
                "frame": i,
                "gpu": gpu_mem(),
                "cpu_mb": cpu_mem(),
                "obj_ids": list(session.obj_ids),
                "num_obj": len(session.obj_ids),
            })
        
        i += 1
        del out, processed, pixels, inputs
        if i % 25 == 0:
            gc.collect()
            torch.cuda.empty_cache()
finally:
    cap.release()
    reset = getattr(session, "reset_inference_session", None)
    if callable(reset): reset()

elapsed = time.time() - t_start
fps = i / elapsed if elapsed > 0 else 0
stats = {
    "frames": i, "elapsed_s": round(elapsed, 2), "fps": round(fps, 2),
    "avg_frame_s": round(np.mean(frame_times), 4) if frame_times else 0,
    "max_frame_s": round(max(frame_times), 4) if frame_times else 0,
    "gpu_alloc_peak_mb": round(max(s["gpu"]["alloc_mb"] for s in mem_snapshots), 1),
    "gpu_reserved_peak_mb": round(max(s["gpu"]["reserved_mb"] for s in mem_snapshots), 1),
    "cpu_peak_mb": round(max(s["cpu_mb"] for s in mem_snapshots), 1),
    "objects_per_frame_mean": round(float(np.mean([kept])), 2),
}
results["HF_STREAMING_400"] = stats

print("=== HF_STREAMING_400 ===")
print(json.dumps(stats, indent=2))
print("\n=== MEMORY SNAPSHOTS ===")
for s in mem_snapshots:
    print(f"frame={s['frame']:3d} alloc={s['gpu']['alloc_mb']:.0f}MB reserved={s['gpu']['reserved_mb']:.0f}MB peak={s['gpu']['max_alloc_mb']:.0f}MB cpu={s['cpu_mb']:.0f}MB objs={s['num_obj']}")

print("=== session release ===")
del session
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print(f"after_release gpu_alloc={gpu_mem()['alloc_mb']:.0f}MB gpu_reserved={gpu_mem()['reserved_mb']:.0f}MB cpu={cpu_mem():.0f}MB")

print("=== FINAL ===")
print(json.dumps(results, indent=2))
with open("/tmp/hf_stream_400_results.json", "w") as f:
    json.dump({"stats": stats, "mem_snapshots": mem_snapshots}, f, indent=2)
print("HF_STREAM_TEST_DONE")