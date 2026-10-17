content = open('/opt/molink/app/sam3_video_service.py').read()
if 'def _patch_text_projection' not in content:
    idx = content.find('def require_auth')
    if idx == -1:
        print('require_auth not found')
    else:
        lines = content[idx:].split('\n')
        end_idx = idx
        for i, line in enumerate(lines):
            if line.strip() == '':
                end_idx = idx + len('\n'.join(lines[:i+1]))
        patch_func = '''\ndef _patch_text_projection(model_or_predictor):
    import torch
    text_transformer = None
    if hasattr(model_or_predictor, 'model'):
        model = model_or_predictor.model
    else:
        model = model_or_predictor
    if hasattr(model, 'detector') and hasattr(model.detector, 'backbone'):
        backbone = model.detector.backbone
        if hasattr(backbone, 'language_backbone') and hasattr(backbone.language_backbone, 'encoder'):
            text_transformer = backbone.language_backbone.encoder
    if text_transformer is None and hasattr(model, 'backbone'):
        backbone = model.backbone
        if hasattr(backbone, 'language_backbone') and hasattr(backbone.language_backbone, 'encoder'):
            text_transformer = backbone.language_backbone.encoder
    if text_transformer is None and hasattr(model, 'language_backbone'):
        lang_bb = model.language_backbone
        if hasattr(lang_bb, 'encoder'):
            text_transformer = lang_bb.encoder
    if text_transformer is None and model.__class__.__name__ == 'TextTransformer':
        text_transformer = model
    if text_transformer is None:
        print(f"[_patch_text_projection] Could not find TextTransformer in {type(model_or_predictor).__name__}")
        return model_or_predictor
    if hasattr(text_transformer, 'text_projection'):
        text_proj = text_transformer.text_projection
        min_val = text_proj.min().item()
        max_val = text_proj.max().item()
        if min_val < -1e10 or max_val > 1e10:
            print(f"[_patch_text_projection] Detected corrupted text_projection: min={min_val:.4e}, max={max_val:.4e}")
            with torch.no_grad():
                if isinstance(text_proj, torch.nn.Parameter):
                    std = text_proj.shape[0] ** -0.5
                    torch.nn.init.normal_(text_proj, std=std)
                else:
                    if hasattr(text_proj, 'weight'):
                        std = text_proj.weight.shape[1] ** -0.5
                        torch.nn.init.normal_(text_proj.weight, std=std)
                    if hasattr(text_proj, 'bias') and text_proj.bias is not None:
                        torch.nn.init.zeros_(text_proj.bias)
            new_min = text_proj.min().item()
            new_max = text_proj.max().item()
            print(f"[_patch_text_projection] Fixed text_projection: min={new_min:.4f}, max={new_max:.4f}")
        else:
            print(f"[_patch_text_projection] text_projection OK: min={min_val:.4f}, max={max_val:.4f}")
    else:
        print(f"[_patch_text_projection] No text_projection found in TextTransformer")
    return model_or_predictor
'''
        new_content = content[:end_idx] + patch_func + '\n' + content[end_idx:]
        with open('/opt/molink/app/sam3_video_service.py', 'w') as f:
            f.write(new_content)
        print('Patched file')
else:
    print('Function already exists')