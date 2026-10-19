"""Shared HF SAM3 model pool for multi-worker architecture."""
from __future__ import annotations

import threading
import os
import sys
from contextlib import contextmanager
from typing import Optional, Generator

import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor

# Adapter fix for CLIP get_text_features
def _patch_get_text_features(model: Sam3VideoModel) -> None:
    _orig_get_text = model.detector_model.get_text_features
    def _patched_get_text(input_ids, attention_mask=None, **kwargs):
        out = _orig_get_text(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        return out.pooler_output if hasattr(out, "pooler_output") else out
    model.detector_model.get_text_features = _patched_get_text


class ModelPool:
    """
    Thread-safe pool of HF SAM3 models sharing weights.
    
    All instances share the same model parameters (frozen, requires_grad=False).
    Each acquire() returns a model instance; release() returns it to pool.
    """
    
    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        device: str = "cuda:0",
        pool_size: int = 2,
        hf_model: str = "",
        bpe_path: str = "",
    ):
        self.model_path = model_path
        self.hf_model = hf_model or model_path
        self.dtype = getattr(torch, dtype)
        self.device = torch.device(device)
        self.pool_size = pool_size
        self.bpe_path = bpe_path
        
        self._models: list[Sam3VideoModel] = []
        self._processors: list[Sam3VideoProcessor] = []
        self._lock = threading.Lock()
        self._idx = 0
        self._initialized = False
    
    def initialize(self) -> None:
        """Load all model instances. Call once at startup."""
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            for i in range(self.pool_size):
                print(f"[ModelPool] Loading model instance {i+1}/{self.pool_size}...")
                model = Sam3VideoModel.from_pretrained(
                    self.hf_model, dtype=self.dtype
                ).to(self.device).eval()
                
                # Freeze weights for sharing
                for param in model.parameters():
                    param.requires_grad = False
                
                _patch_get_text_features(model)
                
                processor = Sam3VideoProcessor.from_pretrained(self.hf_model)
                
                self._models.append(model)
                self._processors.append(processor)
            
            self._initialized = True
            print(f"[ModelPool] Initialized {self.pool_size} model instances on {self.device}")
    
    def shutdown(self) -> None:
        """Release all models."""
        with self._lock:
            self._models.clear()
            self._processors.clear()
            self._initialized = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    @contextmanager
    def acquire(self) -> Generator[tuple[Sam3VideoModel, Sam3VideoProcessor], None, None]:
        """
        Acquire a model+processor pair from pool.
        
        Usage:
            with pool.acquire() as (model, processor):
                # use model, processor
                pass
        """
        if not self._initialized:
            raise RuntimeError("ModelPool not initialized. Call initialize() first.")
        
        with self._lock:
            model = self._models[self._idx]
            processor = self._processors[self._idx]
            self._idx = (self._idx + 1) % self.pool_size
        
        try:
            yield model, processor
        finally:
            pass  # Model returned to pool automatically via round-robin
    
    def get_model_processor(self, index: int = 0) -> tuple[Sam3VideoModel, Sam3VideoProcessor]:
        """Get specific model instance (for workers that need persistent reference)."""
        if not self._initialized:
            raise RuntimeError("ModelPool not initialized")
        return self._models[index % self.pool_size], self._processors[index % self.pool_size]


# Global pool instance (lazy initialization)
_global_pool: Optional[ModelPool] = None
_pool_lock = threading.Lock()


def get_global_pool(
    model_path: str,
    dtype: str = "bfloat16",
    device: str = "cuda:0",
    pool_size: int = 2,
    hf_model: str = "",
    bpe_path: str = "",
) -> ModelPool:
    """Get or create global model pool singleton."""
    global _global_pool
    with _pool_lock:
        if _global_pool is None:
            _global_pool = ModelPool(
                model_path=model_path,
                dtype=dtype,
                device=device,
                pool_size=pool_size,
                hf_model=hf_model,
                bpe_path=bpe_path,
            )
            _global_pool.initialize()
        return _global_pool


def shutdown_global_pool() -> None:
    """Shutdown global pool."""
    global _global_pool
    with _pool_lock:
        if _global_pool is not None:
            _global_pool.shutdown()
            _global_pool = None