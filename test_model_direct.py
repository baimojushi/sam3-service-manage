import sys
sys.path.insert(0, '/opt/sam3-src')
from sam3.model_builder import build_sam3_video_predictor
import time

MODEL_PATH = "/mnt/e/models/sam3/models/facebook--sam3/snapshots/master/sam3.pt"
BPE_PATH = "/mnt/e/models/sam3/models/facebook--sam3/snapshots/master/bpe_simple_vocab_16e6.txt.gz"

print("Loading SAM3...")
start = time.time()
predictor = build_sam3_video_predictor(
    checkpoint_path=MODEL_PATH,
    async_loading_frames=False,
    apply_temporal_disambiguation=True,
    has_presence_token=True,
    geo_encoder_use_img_cross_attn=True,
    strict_state_dict_loading=True,
    video_loader_type="cv2",
    bpe_path=BPE_PATH,
    gpus_to_use=[1],
)
print(f"Loaded in {time.time() - start:.1f}s")

# Test handle_request
print("Testing handle_request...")
response = predictor.handle_request({
    "type": "start_session",
    "resource_path": "/tmp/test_video.mp4"
})
print(f"Session: {response}")