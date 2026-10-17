#!/usr/bin/env python3
import sys
sys.path.insert(0, "/opt/sam3-src")
from sam3.model_builder import build_sam3_video_predictor
import os

MODEL_PATH = "/mnt/e/models/sam3/models/facebook--sam3/snapshots/master/sam3.pt"
print("Loading model...")
predictor = build_sam3_video_predictor(
    checkpoint_path=MODEL_PATH,
    async_loading_frames=False,
    apply_temporal_disambiguation=True,
    has_presence_token=True,
    geo_encoder_use_img_cross_attn=True,
    strict_state_dict_loading=True,
    video_loader_type="cv2",
    bpe_path="/mnt/e/models/sam3/models/facebook--sam3/snapshots/master/bpe_simple_vocab_16e6.txt.gz",
    gpus_to_use=[0],
)
print("Model loaded successfully")
predictor.model.score_threshold_detection = 0.1
print("Threshold:", predictor.model.score_threshold_detection)

video_path = "/mnt/c/Users/root/Downloads/sam3_calligraphy_brush_640x480_20fps_20s.mp4"
print("Video exists:", os.path.exists(video_path))

# Service start_session
resp = predictor.handle_request({"type": "start_session", "resource_path": video_path})
print("start_session:", resp)
session_id = resp["session_id"]

# Service add_prompt
resp = predictor.handle_request({"type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": "Chinese-calligraphy-brush"})
print("add_prompt keys:", list(resp.keys()) if isinstance(resp, dict) else type(resp))
if isinstance(resp, dict) and "outputs" in resp:
    outputs = resp["outputs"]
    print("  outputs keys:", list(outputs.keys()))
    masks = outputs.get("out_binary_masks", [])
    print("  masks:", len(masks))

# Service propagate
frame_count = 0
for out in predictor.handle_stream_request({"type": "propagate_in_video", "session_id": session_id}):
    frame_idx = out.get("frame_index", 0)
    outputs = out.get("outputs", {})
    masks = outputs.get("out_binary_masks", [])
    frame_count += 1
    if frame_count <= 3:
        print(f"Frame {frame_idx}: {len(masks)} masks")
    if frame_count >= 10:
        break
print(f"Total frames: {frame_count}")

# Close
predictor.handle_request({"type": "close_session", "session_id": session_id})
print("DONE!")