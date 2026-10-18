#!/usr/bin/env python3
"""Memory A/B/C test: three independent runs with different cleanup strategies."""
import os, sys, time, json, gc, psutil, traceback
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
MAX_FRAMES = 20

def gpu_mem_detailed():
    if torch.cuda.is_available():
        stats = torch.cuda.memory_stats()
        return {
            "alloc_mb": torch.cuda.memory_allocated() / 1e6,
            "reserved_mb": torch.cuda.memory_reserved() / 1e6,
            "max_alloc_mb": torch.cuda.max_memory_allocated() / 1e6,
            "max_reserved_mb": torch.cuda.max_memory_reserved() / 1e6,
            "inactive_split_bytes": stats.get("inactive_split_bytes", 0) / 1e6,
            "allocation_retries": stats.get("allocation_retries", 0),
        }
    return {"alloc_mb": 0, "reserved_mb": 0, "max_alloc_mb": 0, "max_reserved_mb": 0, "inactive_split_bytes": 0, "allocation_retries": 0}

def cpu_mem():
    return psutil.Process(os.getpid()).memory_info().rss / 1e6

def run_test(tag, cleanup_level):
    """
    cleanup_level:
    0 = no cleanup (A)
    1 = del + gc.collect (B)
    2 = del + gc.collect + empty_cache (C)
    """
    print(f"\n=== TEST {tag} (cleanup_level={cleanup_level}) ===")
    
    # Fresh model load per test (independent process simulation)
    model = Sam3VideoModel.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    proc = Sam3VideoProcessor.from_pretrained(MODEL)
    
    _orig_get_text = model.detector_model.get_text_features
    def _patched_get_text(input_ids, attention_mask=None, **kwargs):
        out = _orig_get_text(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        return out.pooler_output if hasattr(out, "pooler_output") else out
    model.detector_model.get_text_features = _patched_get_text
    
    cap = cv2.VideoCapture(VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(MAX_FRAMES, total)
    
    session = proc.init_video_session(
        inference_device=torch.device("cuda:0"),
        inference_state_device=torch.device("cuda:0"),
        processing_device=torch.device("cpu"),
        video_storage_device=torch.device("cpu"),
        max_vision_features_cache_size=1,
        dtype=torch.bfloat16,
    )
    proc.add_text_prompt(inference_session=session, text=PROMPT)
    
    frame_times, gpu_snapshots = [], []
    i = 0
    oom_info = None
    t_start = time.time()
    
    try:
        while i < n:
            ok, bgr = cap.read()
            if not ok:
                break
            image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            
            mem_before = gpu_mem_detailed()
            cpu_before = cpu_mem()
            
            inputs = proc(images=image, return_tensors="pt")
            pixels = inputs.pixel_values[0].to(device="cuda:0", dtype=torch.bfloat16)
            
            try:
                with torch.inference_mode():
                    out = model(inference_session=session, frame=pixels, reverse=False)
                    processed = proc.postprocess_outputs(
                        session, out, original_sizes=inputs.original_sizes)
            except RuntimeError as e:
                mem_after = gpu_mem_detailed()
                cpu_after = cpu_mem()
                oom_info = {
                    "frame": i,
                    "error": str(e),
                    "mem_before": mem_before,
                    "mem_after": mem_after,
                    "cpu_before_mb": cpu_before,
                    "cpu_after_mb": cpu_after,
                    "traceback": traceback.format_exc(),
                }
                print(f"  FRAME {i} OOM: {e}")
                break
            
            mem_after = gpu_mem_detailed()
            cpu_after = cpu_mem()
            
            dt = time.time() - t_start if i == 0 else time.time() - (t_start + sum(frame_times))
            frame_times.append(dt)
            
            gpu_snapshots.append({
                "frame": i,
                "before": mem_before,
                "after": mem_after,
                "cpu_before_mb": cpu_before,
                "cpu_after_mb": cpu_after,
                "delta_alloc_mb": mem_after["alloc_mb"] - mem_before["alloc_mb"],
                "delta_reserved_mb": mem_after["reserved_mb"] - mem_before["reserved_mb"],
            })
            
            # Cleanup per level
            if cleanup_level >= 1:
                del out, processed
                gc.collect()
            if cleanup_level >= 2:
                torch.cuda.empty_cache()
            
            if cleanup_level == 0:
                del out, processed
            
            i += 1
            del pixels, inputs
            
    finally:
        cap.release()
        reset = getattr(session, "reset_inference_session", None)
        if callable(reset): reset()
        del model, proc, session
        gc.collect()
        torch.cuda.empty_cache()
    
    elapsed = time.time() - t_start
    fps = i / elapsed if elapsed > 0 else 0
    
    result = {
        "tag": tag,
        "cleanup_level": cleanup_level,
        "frames_completed": i,
        "oom": oom_info is not None,
        "oom_info": oom_info,
        "elapsed_s": round(elapsed, 2),
        "fps": round(fps, 2),
        "avg_frame_s": round(np.mean(frame_times), 4) if frame_times else 0,
        "max_frame_s": round(max(frame_times), 4) if frame_times else 0,
        "gpu_snapshots": gpu_snapshots,
        "final_gpu": gpu_mem_detailed(),
        "final_cpu_mb": cpu_mem(),
    }
    
    print(f"  frames={i}, oom={oom_info is not None}, fps={fps:.2f}")
    if oom_info:
        print(f"  OOM at frame {oom_info['frame']}: {oom_info['error'][:100]}")
    else:
        print(f"  peak_alloc={max(s['after']['alloc_mb'] for s in gpu_snapshots):.0f}MB")
        print(f"  peak_reserved={max(s['after']['reserved_mb'] for s in gpu_snapshots):.0f}MB")
    
    return result

# Run A, B, C sequentially (independent processes would be better but sequential works)
results = []
for tag, level in [("A_NO_CLEANUP", 0), ("B_DEL_GC", 1), ("C_DEL_GC_EMPTYCACHE", 2)]:
    results.append(run_test(tag, level))
    # Full cleanup between tests
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    time.sleep(2)

print("\n=== SUMMARY ===")
for r in results:
    print(f"{r['tag']}: frames={r['frames_completed']}, oom={r['oom']}, fps={r['fps']:.2f}")

with open("/tmp/mem_ab_test_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved /tmp/mem_ab_test_results.json")