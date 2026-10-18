#!/usr/bin/env python3
"""Evaluate *actual* SAM3 exported PNG masks against the supplied ground truth."""
import argparse
import csv
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


def read_binary(z, name):
    with Image.open(io.BytesIO(z.read(name))) as image:
        return np.asarray(image.convert("L")) > 127


def evaluate(pred_zip, gt_zip, target, outdir, expected_frames=400):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(pred_zip) as pred, zipfile.ZipFile(gt_zip) as gt:
        if "meta.json" not in pred.namelist():
            raise ValueError("Prediction ZIP is missing meta.json")
        meta = json.loads(pred.read("meta.json"))
        indices = meta.get("frame_indices")
        if indices != list(range(expected_frames)):
            raise ValueError(
                f"Prediction frame indices not contiguous 0..{expected_frames-1}: "
                f"length={len(indices) if isinstance(indices, list) else 'missing'}"
            )
        if meta.get("num_frames") != expected_frames:
            raise ValueError("Prediction frame count does not match expected_frames")

        pattern = re.compile(r"^frame_(\d+)_obj_(\d+)(?:_score[0-9.]+)?\.png$")
        files = {}
        for name in pred.namelist():
            m = pattern.fullmatch(name)
            if m:
                key = (int(m.group(1)), int(m.group(2)))
                if key in files:
                    raise ValueError(f"Duplicate prediction mask: {key}")
                files[key] = name

        # Choose a single stable object ID, using prediction outputs only.
        # Prefer the prompt-frame object ID; never use GT to choose per-frame winners.
        seed = meta.get("prompt_frame") or {}
        seed_ids = seed.get("obj_ids") or []
        seed_scores = seed.get("scores") or []
        if seed_ids:
            if len(seed_scores) == len(seed_ids):
                chosen_id = int(seed_ids[int(np.argmax(seed_scores))])
            else:
                chosen_id = int(seed_ids[0])
        elif files:
            earliest_frame = min(f for f, _ in files)
            chosen_id = min(obj for f, obj in files if f == earliest_frame)
        else:
            chosen_id = None

        cup_gt = None
        if target == "cup":
            cup_gt = read_binary(gt, "cup_COCO_static.png")

        rows = []
        for i in range(expected_frames):
            truth = cup_gt if target == "cup" else read_binary(gt, f"brush/{i:06d}.png")
            name = files.get((i, chosen_id)) if chosen_id is not None else None
            estimate = read_binary(pred, name) if name is not None else np.zeros_like(truth)

            if estimate.shape != truth.shape:
                raise ValueError(f"Frame {i}: mask shape mismatch {estimate.shape} vs {truth.shape}")

            intersection = int(np.logical_and(estimate, truth).sum())
            union = int(np.logical_or(estimate, truth).sum())
            iou = intersection / union if union else 1.0
            rows.append({
                "frame_index": i, "object_id": chosen_id if chosen_id is not None else "",
                "iou": iou, "gt_pixels": int(truth.sum()),
                "pred_pixels": int(estimate.sum()),
                "intersection_pixels": intersection, "union_pixels": union,
            })

    mean_iou = sum(r["iou"] for r in rows) / len(rows)
    passing_frames = sum(r["iou"] >= 0.90 for r in rows)
    report = {
        "target": target, "expected_frames": expected_frames,
        "selected_object_id": chosen_id,
        "object_selection": "first prompt object with highest initial score; fixed across frames",
        "mean_iou": mean_iou, "frames_iou_ge_0_90": passing_frames,
        "fraction_iou_ge_0_90": passing_frames / expected_frames,
        "pass": mean_iou >= 0.90 and passing_frames >= 0.90 * expected_frames,
        "prediction_zip": str(pred_zip), "ground_truth_zip": str(gt_zip),
    }
    with (outdir / "metrics.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (outdir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="SAM3 mask ZIP from return_masks_png=true")
    parser.add_argument("--gt", required=True)
    parser.add_argument("--target", required=True, choices=["cup", "brush"])
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--expected-frames", type=int, default=400)
    args = parser.parse_args()
    print(json.dumps(
        evaluate(args.pred, args.gt, args.target, args.outdir, args.expected_frames),
        ensure_ascii=False, indent=2
    ))
