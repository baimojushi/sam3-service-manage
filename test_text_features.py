import torch
import sys
sys.path.insert(0, '/opt/sam3-src')

# Load raw checkpoint
CKPT = '/mnt/e/models/sam3/models/facebook--sam3/snapshots/master/sam3.pt'
raw = torch.load(CKPT, map_location='cpu', weights_only=False)
sd = raw['model'] if isinstance(raw, dict) and 'model' in raw else raw

# Build model WITHOUT loading checkpoint
from sam3.model_builder import build_sam3_video_predictor
predictor = build_sam3_video_predictor(
    checkpoint_path=None,
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

# Check text_projection type and value BEFORE loading checkpoint
print('Before load:')
print('  text_projection type:', type(tt.text_projection))
tp = tt.text_projection
print('  text_projection min={:.6e}, max={:.6e}'.format(tp.min().item(), tp.max().item()))

# Now manually load checkpoint (skip text_projection)
missing, unexpected = model.load_state_dict(sd, strict=False)
print('Missing keys:', len(missing))
print('Unexpected keys:', len(unexpected))
for k in missing[:20]:
    print('  MISSING:', k)

# Check if text_projection was loaded
print('After load (text_projection still random):')
tp = tt.text_projection
print('  text_projection min={:.6e}, max={:.6e}'.format(tp.min().item(), tp.max().item()))

# Now test text encoding
print('\nTesting text encoding with cup...')
ve_encoder = model.detector.backbone.language_backbone.encoder

# Tokenize manually using BPE
from sam3.model.tokenizer_ve import SimpleTokenizer as get_tokenizer
tokenizer_path = '/mnt/e/models/sam3/models/facebook--sam3/snapshots/master/bpe_simple_vocab_16e6.txt.gz'
tok = get_tokenizer(tokenizer_path)
tokenized = tok(['cup'], context_length=32).to('cpu')
print('Tokenized shape:', tokenized.shape)

# Run through VETextEncoder
with torch.no_grad():
    text_attention_mask, text_memory_resized, inputs_embeds = ve_encoder(['cup'], device='cpu')
    
    print('text_memory_resized: min={:.6e}, max={:.6e}, NaN={}, Inf={}'.format(
        text_memory_resized.min().item(), text_memory_resized.max().item(),
        torch.isnan(text_memory_resized).sum().item(), torch.isinf(text_memory_resized).sum().item()))
    print('inputs_embeds: min={:.6e}, max={:.6e}, NaN={}, Inf={}'.format(
        inputs_embeds.min().item(), inputs_embeds.max().item(),
        torch.isnan(inputs_embeds).sum().item(), torch.isinf(inputs_embeds).sum().item()))
    
    # Also check pooled from TextTransformer
    pooled, tokens = ve_encoder.encoder(tokenized)
    print('pooled: min={:.6e}, max={:.6e}, NaN={}, Inf={}'.format(
        pooled.min().item(), pooled.max().item(),
        torch.isnan(pooled).sum().item(), torch.isinf(pooled).sum().item()))
    print('tokens: min={:.6e}, max={:.6e}, NaN={}, Inf={}'.format(
        tokens.min().item(), tokens.max().item(),
        torch.isnan(tokens).sum().item(), torch.isinf(tokens).sum().item()))

# Now test with Chinese calligraphy brush
print('\nTesting with Chinese calligraphy brush...')
text2 = ['Chinese calligraphy brush']
tokenized2 = tok(text2, context_length=32).to('cpu')
with torch.no_grad():
    text_attention_mask2, text_memory_resized2, inputs_embeds2 = ve_encoder(['Chinese calligraphy brush'], device='cpu')
    print('text_memory_resized: min={:.6e}, max={:.6e}, NaN={}'.format(
        text_memory_resized2.min().item(), text_memory_resized2.max().item(),
        torch.isnan(text_memory_resized2).sum().item()))
    pooled2, tokens2 = ve_encoder.encoder(tokenized2)
    print('pooled: min={:.6e}, max={:.6e}'.format(
        pooled2.min().item(), pooled2.max().item()))
    print('tokens: min={:.6e}, max={:.6e}'.format(
        tokens2.min().item(), tokens2.max().item()))

# Test with patch applied
print('\n\n=== NOW TESTING WITH PATCH APPLIED ===')
# Re-apply the patch
import torch.nn as nn
with torch.no_grad():
    if isinstance(tt.text_projection, nn.Parameter):
        std = tt.text_projection.shape[0] ** -0.5
        torch.nn.init.normal_(tt.text_projection, std=std)
    else:
        if hasattr(tt.text_projection, 'weight'):
            std = tt.text_projection.weight.shape[1] ** -0.5
            torch.nn.init.normal_(tt.text_projection.weight, std=std)
            if tt.text_projection.bias is not None:
                torch.nn.init.zeros_(tt.text_projection.bias)

print('After patch:')
tp = tt.text_projection
print('  text_projection min={:.6e}, max={:.6e}'.format(tp.min().item(), tp.max().item()))

# Test text encoding again
print('\nTesting text encoding with cup (patched)...')
with torch.no_grad():
    text_attention_mask, text_memory_resized, inputs_embeds = ve_encoder(['cup'], device='cpu')
    print('text_memory_resized: min={:.6e}, max={:.6e}, NaN={}, Inf={}'.format(
        text_memory_resized.min().item(), text_memory_resized.max().item(),
        torch.isnan(text_memory_resized).sum().item(), torch.isinf(text_memory_resized).sum().item()))
    pooled, tokens = ve_encoder.encoder(tokenized)
    print('pooled: min={:.6e}, max={:.6e}, NaN={}, Inf={}'.format(
        pooled.min().item(), pooled.max().item(),
        torch.isnan(pooled).sum().item(), torch.isinf(pooled).sum().item()))
    print('tokens: min={:.6e}, max={:.6e}, NaN={}, Inf={}'.format(
        tokens.min().item(), tokens.max().item(),
        torch.isnan(tokens).sum().item(), torch.isinf(tokens).sum().item()))

print('\nTesting with Chinese calligraphy brush (patched)...')
with torch.no_grad():
    text_attention_mask2, text_memory_resized2, inputs_embeds2 = ve_encoder(['Chinese calligraphy brush'], device='cpu')
    print('text_memory_resized: min={:.6e}, max={:.6e}, NaN={}'.format(
        text_memory_resized2.min().item(), text_memory_resized2.max().item(),
        torch.isnan(text_memory_resized2).sum().item()))
    pooled2, tokens2 = ve_encoder.encoder(tokenized2)
    print('pooled: min={:.6e}, max={:.6e}'.format(
        pooled2.min().item(), pooled2.max().item()))
    print('tokens: min={:.6e}, max={:.6e}'.format(
        tokens2.min().item(), tokens2.max().item()))