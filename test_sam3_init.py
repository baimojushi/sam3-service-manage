#!/usr/bin/env python3
import sys
sys.path.insert(0, "/opt/sam3-src")
from sam3.model_builder import build_sam3_video_predictor
import os

MODEL_PATH = os.getenv("SAM3_MODEL_PATH", "/mnt/e/models/sam3/models/facebook--sam3/snapshots/master/sam3.pt")
print("Loading model from:", MODEL_PATH)
print("Model exists:", os.path.exists(MODEL_PATH))

try:
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
    print("Predictor loaded successfully!")
    print("Model:", predictor.model)
    print("Score threshold:", predictor.model.score_threshold_detection)
except Exception as e:
    import traceback
    traceback.print_exc()
    print("ERROR:", e)