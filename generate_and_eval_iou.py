#!/usr/bin/env python3
"""Generate SAM3 prediction masks ZIP and evaluate IoU against ground truth."""
import os, sys, time, json, gc, zipfile, io, shutil
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
EXPECTED_FRAMES = 400

# Output paths
PRED_ZIP = "/tmp/sam3_hf_predictions.zip"
GT_DIR = "/mnt/wslg/distro/opt/molink/app/gt/brush"
GT_ZIP = "/tmp/brush_gt.zip"
OUTDIR = "/tmp/iou_eval"

def gpu_mem():
    if torch.cuda.is_available():
        return {
            "alloc_mb": torch.cuda.memory_allocated() / 1e6,
            "reserved_mb": torch.cuda.memory_reserved() / 1e6,
        }
    return {"alloc_mb": 0, "reserved_mb": 0}

def cpu_mem():
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss / 1e6

# First, create ground truth ZIP in expected format
print("=== Creating ground truth ZIP ===")
if not os.path.exists(GT_ZIP):
    with zipfile.ZipFile(GT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i in range(EXPECTED_FRAMES):
            src = os.path.join(GT_DIR, f"{i:06d}.png")
            if os.path.exists(src):
                zf.write(src, f"brush/{i:06d}.png")
            else:
                print(f"Warning: missing {src}")
    print(f"Created {GT_ZIP}")
else:
    print(f"Using existing {GT_ZIP}")

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

# Generate prediction masks
print(f"\n=== Generating {EXPECTED_FRAMES} prediction masks ===")
frame_data = []  # List of (frame_idx, obj_id, score, mask_png_bytes)
prompt_frame_info = {"obj_ids": [], "scores": []}

i = 0
t_start = time.time()
try:
    while i < EXPECTED_FRAMES:
        ok, bgr = cap.read()
        if not ok:
            break
        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        
        # Fresh session per frame (stateless)
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
        
        # Track first frame object info for meta.json
        if i == 0:
            for j, (oid, sc) in enumerate(zip(ids, scores)):
                if float(sc) >= THRESH:
                    prompt_frame_info["obj_ids"].append(int(oid))
                    prompt_frame_info["scores"].append(float(sc))
        
        # Save masks above threshold
        for j, (oid, sc, mask) in enumerate(zip(ids, scores, masks)):
            if float(sc) >= THRESH:
                # Convert mask to PNG bytes
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                buf = io.BytesIO()
                mask_img.save(buf, format="PNG")
                png_bytes = buf.getvalue()
                frame_data.append({
                    "frame_idx": i,
                    "obj_id": int(oid),
                    "score": float(sc),
                    "png_bytes": png_bytes,
                })
        
        if i % 50 == 0:
            print(f"  frame {i}: {len([d for d in frame_data if d['frame_idx']==i])} masks, gpu={gpu_mem()['alloc_mb']:.0f}MB")
        
        # Cleanup
        del session, out, processed, pixels, inputs, ids, scores, masks
        gc.collect()
        torch.cuda.empty_cache()
        
        i += 1
        
finally:
    cap.release()

elapsed = time.time() - t_start
print(f"Generated {len(frame_data)} masks in {elapsed:.1f}s ({i/elapsed:.2f} fps)")

# Write prediction ZIP
print(f"\n=== Writing prediction ZIP to {PRED_ZIP} ===")
os.makedirs(OUTDIR, exist_ok=True)

with zipfile.ZipFile(PRED_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    # Write meta.json
    meta = {
        "frame_indices": list(range(i)),
        "num_frames": i,
        "prompt_frame": prompt_frame_info,
    }
    zf.writestr("meta.json", json.dumps(meta, indent=2))
    
    # Write mask PNGs
    for fd in frame_data:
        name = f"frame_{fd['frame_idx']:04d}_obj_{fd['obj_id']:02d}_score{fd['score']:.4f}.png"
        zf.writestr(name, fd["png_bytes"])

print(f"ZIP written: {len(frame_data)} masks + meta.json")

# Run evaluation
print(f"\n=== Running IoU evaluation ===")
sys.path.insert(0, "/mnt/e/项目/深度相机/v4_extract/sam3-service-v4/scripts")
from evaluate_sam3_iou import evaluate

report = evaluate(
    pred_zip=PRED_ZIP,
    gt_zip=GT_ZIP,
    target="brush",
    outdir=OUTDIR,
    expected_frames=EXPECTED_FRAMES,
)

print("\n=== IoU REPORT ===")
print(json.dumps(report, indent=2, ensure_ascii=False))

with open("/tmp/hf_iou_report.json", "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print("\nSaved /tmp/hf_iou_report.json")