#!/usr/bin/env python3
import sys
import os
import cv2
import torch
import numpy as np
import time
import json
import csv
from pathlib import Path
from PIL import Image

sys.path.insert(0, '/opt/sam3-src')
from sam3.model_builder import build_sam3_video_predictor

MODEL_PATH = "/mnt/e/models/sam3/models/facebook--sam3/snapshots/master/sam3.pt"
BPE_PATH = "/mnt/e/models/sam3/models/facebook--sam3/snapshots/master/bpe_simple_vocab_16e6.txt.gz"
VIDEO_PATH = "/mnt/e/项目/art hanging/brush_cup_realistic.mp4"
GT_DIR = "/tmp/ground_truth/brush"
GT_CUP = "/tmp/ground_truth/cup_COCO_static.png"
OUTPUT_DIR = "/tmp/sam3_eval"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "pred_masks"), exist_ok=True)

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def load_gt():
    gt_brush = []
    for i in range(400):
        path = os.path.join(GT_DIR, f"{i:06d}.png")
        if os.path.exists(path):
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            gt_brush.append(mask > 128)
        else:
            gt_brush.append(np.zeros((480, 640), dtype=bool))
    gt_cup = cv2.imread(GT_CUP, cv2.IMREAD_GRAYSCALE) > 128 if os.path.exists(GT_CUP) else np.zeros((480, 640), dtype=bool)
    return gt_brush, gt_cup

print("=" * 60)
print("SAM3 Evaluation Pipeline")
print("=" * 60)

# Stage A
print("\n[Stage A] Loading model...")
start = time.time()
predictor = build_sam3_video_predictor(
    checkpoint_path=MODEL_PATH,
    async_loading_frames=False,
    apply_temporal_disambiguation=True,
    has_presence_token=True,
    geo_encoder_use_img_cross_attn=True,
    strict_state_dict_loading=True,
    video_loader_type='cv2',
    bpe_path=BPE_PATH,
    gpus_to_use=[1],
)
load_time = time.time() - start
print(f"Loaded in {load_time:.1f}s on {predictor.device}")

print("\n[Stage A] Starting session...")
start = time.time()
resp = predictor.handle_request({"type": "start_session", "resource_path": VIDEO_PATH})
session_time = time.time() - start
session_id = resp.get("session_id")
print(f"Session {session_id} in {session_time:.2f}s")

print("\n[Stage A] Adding prompt...")
start = time.time()
predictor.handle_request({"type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": "Chinese calligraphy brush"})
prompt_time = time.time() - start
print(f"Prompt in {prompt_time:.2f}s")

print("\n[Stage A/C] Propagating...")
start = time.time()
all_masks = []
all_scores = []
for out in predictor.handle_stream_request({"type": "propagate_in_video", "session_id": session_id}):
    frame_idx = out.get("frame_index", 0)
    masks = out.get("outputs", {}).get("out_binary_masks", [])
    scores = out.get("outputs", {}).get("out_probs", [1.0] * len(masks))
    all_masks.append(masks)
    all_scores.append(scores)
    if frame_idx % 50 == 0:
        print(f"  Frame {frame_idx}: {len(masks)} objects")
propagate_time = time.time() - start
print(f"Propagation: {propagate_time:.2f}s, frames: {len(all_masks)}")

predictor.handle_request({"type": "close_session", "session_id": session_id})

# Stage D
print("\n[Stage D] Computing IoU...")
gt_brush, gt_cup = load_gt()

iou_brush = []
iou_cup = []
for frame_idx, (masks, scores) in enumerate(zip(all_masks, all_scores)):
    for obj_idx, (mask, score) in enumerate(zip(masks, scores)):
        mask_np = np.array(mask, dtype=np.uint8) * 255
        Image.fromarray(mask_np).save(os.path.join(OUTPUT_DIR, "pred_masks", f"frame_{frame_idx:04d}_obj_{obj_idx:02d}_score{score:.4f}.png"))

for frame_idx, masks in enumerate(all_masks):
    if len(masks) > 0:
        pred = masks[0] > 0.5
        gt = gt_brush[frame_idx] if frame_idx < len(gt_brush) else np.zeros_like(pred)
        iou_brush.append(compute_iou(pred, gt))
    else:
        iou_brush.append(0.0)
    if len(masks) > 1:
        pred = masks[1] > 0.5
        iou_cup.append(compute_iou(pred, gt_cup))
    else:
        iou_cup.append(0.0)

mean_brush = np.mean(iou_brush)
mean_cup = np.mean(iou_cup)
min_brush = np.min(iou_brush)
max_brush = np.max(iou_brush)
brush_ge_90 = sum(1 for x in iou_brush if x >= 0.90)
cup_ge_90 = sum(1 for x in iou_cup if x >= 0.90)

print(f"\n=== RESULTS ===")
print(f"Brush IoU: mean={mean_brush:.4f}, min={min_brush:.4f}, max={max_brush:.4f}")
print(f"Cup IoU: mean={mean_cup:.4f}")
print(f"Brush >= 0.90: {brush_ge_90}/400")
print(f"Cup >= 0.90: {cup_ge_90}/400")

# Save outputs
with open(os.path.join(OUTPUT_DIR, "metrics.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "iou_brush", "iou_cup", "num_objects", "scores"])
    for i in range(400):
        writer.writerow([i, iou_brush[i] if i < len(iou_brush) else 0, iou_cup[i] if i < len(iou_cup) else 0, len(all_masks[i]) if i < len(all_masks) else 0, str(all_scores[i]) if i < len(all_scores) else "[]"])

report = {
    "model": "SAM3", "video": "brush_cup_realistic.mp4", "prompt": "Chinese calligraphy brush",
    "load_time_s": load_time, "session_time_s": session_time, "prompt_time_s": prompt_time,
    "propagate_time_s": propagate_time, "mean_iou_brush": float(mean_brush),
    "min_iou_brush": float(min_brush), "max_iou_brush": float(max_brush),
    "mean_iou_cup": float(mean_cup), "brush_iou_ge_090": brush_ge_90,
    "cup_iou_ge_090": cup_ge_90, "total_frames": 400,
    "passed": mean_brush >= 0.90 and mean_cup >= 0.90
}
with open(os.path.join(OUTPUT_DIR, "report.json"), "w") as f:
    json.dump(report, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "report.md"), "w") as f:
    f.write(f"# SAM3 Evaluation Report\n\n")
    f.write(f"**Video:** brush_cup_realistic.mp4\n\n")
    f.write(f"**Prompt:** Chinese calligraphy brush\n\n")
    f.write(f"## Metrics\n\n")
    f.write(f"| Object | Mean IoU | Min IoU | Max IoU | Frames >= 0.90 |\n")
    f.write(f"|--------|----------|---------|---------|----------------|\n")
    f.write(f"| Brush | {mean_brush:.4f} | {min_brush:.4f} | {max_brush:.4f} | {brush_ge_90}/400 |\n")
    f.write(f"| Cup | {mean_cup:.4f} | - | - | {cup_ge_90}/400 |\n\n")
    f.write(f"## Pass\n")
    f.write(f"- Brush IoU >= 0.90: {'PASS' if mean_brush >= 0.90 else 'FAIL'}\n")
    f.write(f"- Cup IoU >= 0.90: {'PASS' if mean_cup >= 0.90 else 'FAIL'}\n")
    f.write(f"- Overall: {'PASS' if report['passed'] else 'FAIL'}\n\n")
    f.write(f"## Timing\n")
    f.write(f"- Load: {load_time:.1f}s\n")
    f.write(f"- Session: {session_time:.2f}s\n")
    f.write(f"- Prompt: {prompt_time:.2f}s\n")
    f.write(f"- Propagate: {propagate_time:.2f}s\n")
    f.write(f"- Total: {load_time + session_time + prompt_time + propagate_time:.1f}s\n")

print(f"\nOutputs in {OUTPUT_DIR}/")
print(f"  report.md, report.json, metrics.csv, pred_masks/")

sys.exit(0 if report["passed"] else 1)