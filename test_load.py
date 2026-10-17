import torch
import sys
sys.path.insert(0, '/opt/sam3-src')
from sam3.model_builder import build_sam3_video_predictor

CKPT = '/mnt/e/models/sam3/models/facebook--sam3/snapshots/master/sam3.pt'
raw = torch.load(CKPT, map_location='cpu', weights_only=False)
sd = raw['model'] if isinstance(raw, dict) and 'model' in raw else raw

predictor = build_sam3_video_predictor(
    checkpoint_path=None,
    load_from_HF=False,
    async_loading_frames=False,
    apply_temporal_disambiguation=True,
    has_presence_token=True,
    geo_encoder_use_img_cross_attn=True,
    strict_state_dict_loading=True,
    video_loader_type='cv2',
    bpe_path='/mnt/e/models/sam3/models/facebook--sam3/snapshots/master/bpe_simple_vocab_16e6.txt.gz',
    gpus_to_use=[1],
)

model = predictor.model
tt = model.detector.backbone.language_backbone.encoder
print('Before load:', type(tt.text_projection))
tp = tt.text_projection
print('min={:.6e}, max={:.6e}'.format(tp.min().item(), tp.max().item()))
missing, unexpected = model.load_state_dict(sd, strict=False)
print('Missing:', len(missing), 'Unexpected:', len(unexpected))
for k in missing[:10]:
    print('  MISSING:', k)
tp = tt.text_projection
print('After load min={:.6e}, max={:.6e}'.format(tp.min().item(), tp.max().item()))