#!/usr/bin/env python3
"""Checkpoint audit: inspect raw checkpoint data for text_projection."""
import torch
import sys
import os

CKPT = "/mnt/e/models/sam3/models/facebook--sam3/snapshots/master/sam3.pt"

print("=" * 70)
print("CHECKPOINT AUDIT - raw checkpoint inspection")
print("=" * 70)

# Load raw checkpoint
print("\n[1] Loading raw checkpoint...")
raw = torch.load(CKPT, map_location="cpu", weights_only=False)
print(f"    Type: {type(raw)}")
if isinstance(raw, dict):
    print(f"    Top-level keys: {list(raw.keys())[:20]}")
    if "model" in raw:
        sd = raw["model"]
        print(f"    Has 'model' key, state_dict keys: {len(sd)}")
    else:
        sd = raw
else:
    sd = raw

# Find text_projection in checkpoint
print("\n[2] Searching for text_projection in checkpoint...")
text_proj_keys = [k for k in sd.keys() if "text_projection" in k.lower()]
print(f"    Found {len(text_proj_keys)} keys:")
for k in text_proj_keys:
    t = sd[k]
    print(f"    {k}: dtype={t.dtype}, shape={tuple(t.shape)}, min={t.min().item():.6e}, max={t.max().item():.6e}, NaN={torch.isnan(t).sum().item()}, Inf={torch.isinf(t).sum().item()}")

# Find positional_embedding
print("\n[3] Searching for positional_embedding...")
pe_keys = [k for k in sd.keys() if "positional_embedding" in k.lower()]
print(f"    Found {len(pe_keys)} keys:")
for k in pe_keys:
    t = sd[k]
    print(f"    {k}: dtype={t.dtype}, shape={tuple(t.shape)}, min={t.min().item():.6e}, max={t.max().item():.6e}")

# Find all text encoder keys
print("\n[4] All text encoder related keys:")
text_keys = [k for k in sd.keys() if any(x in k.lower() for x in ["text", "language", "token", "bert", "clip", "encoder"])]
print(f"    Found {len(text_keys)} keys")
for k in sorted(text_keys)[:30]:
    t = sd[k]
    if hasattr(t, 'shape'):
        print(f"    {k}: dtype={t.dtype}, shape={tuple(t.shape)}, min={t.min().item():.6e}, max={t.max().item():.6e}")
    else:
        print(f"    {k}: {type(t)}")

# Check checkpoint version
print("\n[5] Checkpoint metadata:")
if isinstance(raw, dict):
    for k in ["version", "epoch", "global_step", "checkpoint_version"]:
        if k in raw:
            print(f"    {k}: {raw[k]}")

print("\nAudit complete.")