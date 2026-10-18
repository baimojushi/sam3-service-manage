#!/usr/bin/env python3
"""HF SAM3 streaming test v2: inference_state_device=cuda:0 to avoid GPU accumulation."""
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

def gpu_mem_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6, torch.cuda.memory_reserved() / 1e6
    return 0.0, 0.0

def cpu_mem_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e6

print("=== load model ===")
t0 = time.time()
model = Sam3VideoModel.from_pretrained(MODEL, dtype=torch.float32).cuda().eval()
proc = Sam3VideoProcessor.from_pretrained(MODEL)
print("load_s=%.1f gpu_alloc=%.0fMB gpu_reserved=%.0fMB cpu=%.0fMB" % (
    time.time()-t0, *gpu_mem_mb(), cpu_mem_mb()))

# --- Adapter fix: transformers 5.0.0 CLIP get_text_features returns
#     CLIPTextModelOutput; SAM3 video run_detection stores it verbatim in
#     prompt_embeddings and later passes it as encoder_hidden_states to
#     create_bidirectional_mask, which expects a tensor. Extract pooler_output.
_orig_get_text = model.detector_model.get_text_features
def _patched_get_text(input_ids, attention_mask=None, **kwargs):
    out = _orig_get_text(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    return out.pooler_output if hasattr(out, "pooler_output") else out
model.detector_model.get_text_features = _patched_get_text

def run_streaming(max_frames, tag):
    cap = cv2.VideoCapture(VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(max_frames, total)
    session = proc.init_video_session(
        inference_device="cuda:0",
        inference_state_device="cuda:0",   # keep cached vision features on GPU, no CPU<->GPU roundtrip accumulation
        processing_device="cpu",
        video_storage_device="cpu",
        max_vision_features_cache_size=1,
        dtype=torch.float32,
    )
    proc.add_text_prompt(inference_session=session, text=PROMPT)
    frame_times, gpu_al, gpu_res, cpu = [], [], [], []
    i = 0
    t_start = time.time()
    try:
        while i < n:
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
            masks = processed["masks"]
            if hasattr(ids, "detach"): ids = ids.detach().cpu().numpy()
            if hasattr(scores, "detach"): scores = scores.detach().float().cpu().numpy()
            if hasattr(masks, "detach"): masks = masks.detach().cpu().numpy()
            ids = np.asarray(ids).reshape(-1); scores = np.asarray(scores).reshape(-1)
            masks = np.asarray(masks)
            if masks.ndim == 2: masks = masks[None]
            kept = sum(1 for s in scores if float(s) >= THRESH)
            frame_times.append(dt)
            gpu_al.append(gpu_mem_mb()[0]); gpu_res.append(gpu_mem_mb()[1])
            cpu.append(cpu_mem_mb())
            i += 1
            del out, processed, pixels, inputs
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
        "gpu_alloc_peak_mb": round(max(gpu_al), 1) if gpu_al else 0,
        "gpu_reserved_peak_mb": round(max(gpu_res), 1) if gpu_res else 0,
        "cpu_peak_mb": round(max(cpu), 1) if cpu else 0,
        "objects_per_frame_mean": round(float(np.mean([kept])), 2),
    }
    results[tag] = stats
    print("=== %s ===" % tag)
    print(json.dumps(stats, indent=2))
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return stats

run_streaming(1, "HF_STREAMING_1")
run_streaming(10, "HF_STREAMING_10")
run_streaming(400, "HF_STREAMING_400")

print("=== session release ===")
del session
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("after_release gpu_alloc=%.0fMB gpu_reserved=%.0fMB cpu=%.0fMB" % (
    *gpu_mem_mb(), cpu_mem_mb()))

print("=== FINAL ===")
print(json.dumps(results, indent=2))
with open("/tmp/hf_stream_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("HF_STREAM_TEST_DONE")