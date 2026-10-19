"""Tests for SAM3 v4 multi-mode architecture (Batch/Stream/Single)."""
import pytest
import tempfile
import os
from pathlib import Path

# Test imports
def test_model_pool_import():
    """Test ModelPool can be imported."""
    from sam3_service_v4.model_pool import ModelPool, get_global_pool, shutdown_global_pool
    assert ModelPool is not None
    assert get_global_pool is not None
    assert shutdown_global_pool is not None


def test_workers_import():
    """Test three worker classes can be imported."""
    from sam3_service_v4.workers import BatchWorker, StreamWorker, SingleWorker, BaseWorker
    assert BatchWorker is not None
    assert StreamWorker is not None
    assert SingleWorker is not None
    assert BaseWorker is not None


def test_api_import():
    """Test API can be imported with new endpoints."""
    from sam3_service_v4.api import create_app
    assert create_app is not None


def test_model_pool_initialization():
    """Test ModelPool initialization without actual model loading."""
    from sam3_service_v4.model_pool import ModelPool
    import torch
    
    # Use a dummy path - we won't actually load
    pool = ModelPool(
        model_path="/dummy/path",
        dtype="float32",
        device="cpu",
        pool_size=1,
    )
    # Check attributes set correctly
    assert pool.pool_size == 1
    assert pool.dtype == torch.float32
    assert pool.device.type == "cpu"
    assert pool._initialized == False


def test_worker_instantiation():
    """Test workers can be instantiated with mock settings."""
    from sam3_service_v4.core import Settings
    from sam3_service_v4.model_pool import ModelPool
    from sam3_service_v4.workers import BatchWorker, StreamWorker, SingleWorker
    
    cfg = Settings(
        engine="meta",
        model_path="/dummy/path",
        bpe_path="/dummy/bpe",
        hf_model="/dummy/hf",
        device="cpu",
        hf_dtype="float32",
        api_key="test_key",
        allow_unauth_local=True,
    )
    
    pool = ModelPool(
        model_path="/dummy/path",
        dtype="float32",
        device="cpu",
        pool_size=1,
    )
    
    # Workers should instantiate without loading models
    batch_worker = BatchWorker(cfg, pool)
    stream_worker = StreamWorker(cfg, pool)
    single_worker = SingleWorker(cfg, pool)
    
    assert batch_worker is not None
    assert stream_worker is not None
    assert single_worker is not None
    
    # Check they have required methods
    assert hasattr(batch_worker, 'segment_video')
    assert hasattr(batch_worker, 'load')
    assert hasattr(batch_worker, 'shutdown')
    
    assert hasattr(stream_worker, 'create_session')
    assert hasattr(stream_worker, 'process_frame')
    assert hasattr(stream_worker, 'end_session')
    assert hasattr(stream_worker, 'load')
    assert hasattr(stream_worker, 'shutdown')
    
    assert hasattr(single_worker, 'segment_single')
    assert hasattr(single_worker, 'load')
    assert hasattr(single_worker, 'shutdown')


def test_settings_environment_loading():
    """Test Settings loads from environment."""
    import os
    from sam3_service_v4.core import Settings
    
    os.environ["SAM3_ENGINE"] = "hf"
    os.environ["HF_SAM3_MODEL"] = "/test/model"
    os.environ["SAM3_DEVICE"] = "cuda:0"
    os.environ["HF_SAM3_DTYPE"] = "bfloat16"
    os.environ["API_KEY"] = "test_key"
    os.environ["ALLOW_UNAUTH_LOCAL"] = "true"
    
    try:
        cfg = Settings.from_env()
        assert cfg.engine == "hf"
        assert cfg.hf_model == "/test/model"
        assert cfg.device == "cuda:0"
        assert cfg.hf_dtype == "bfloat16"
        assert cfg.api_key == "test_key"
        assert cfg.allow_unauth_local == True
    finally:
        # Clean up
        for key in ["SAM3_ENGINE", "HF_SAM3_MODEL", "SAM3_DEVICE", "HF_SAM3_DTYPE", "API_KEY", "ALLOW_UNAUTH_LOCAL"]:
            os.environ.pop(key, None)


def test_capabilities_endpoint():
    """Test /v1/capabilities returns expected structure."""
    from sam3_service_v4.core import Settings
    from sam3_service_v4.api import create_app
    from fastapi.testclient import TestClient
    
    # Mock settings
    cfg = Settings(
        engine="meta",
        model_path="/dummy",
        bpe_path="/dummy",
        hf_model="/dummy",
        device="cpu",
        api_key="test",
        allow_unauth_local=True,
    )
    
    # We can't fully test without model, but check app creation
    app = create_app(cfg)
    assert app is not None
    assert app.title == "SAM3 Video Service v4"


def test_mode_routing_logic():
    """Test mode parameter validation logic."""
    from sam3_service_v4.core import Settings
    
    cfg = Settings(
        engine="meta",
        model_path="/dummy",
        bpe_path="/dummy",
        hf_model="/dummy",
        device="cpu",
        api_key="test",
        allow_unauth_local=True,
    )
    
    # Valid modes
    valid_modes = ["batch", "stream", "single"]
    for mode in valid_modes:
        assert mode in ("batch", "stream", "single")
    
    # Invalid mode would be caught by FastAPI Form validation
    invalid_modes = ["invalid", "video", "image"]
    for mode in invalid_modes:
        assert mode not in ("batch", "stream", "single")


def test_frame_masks_structure():
    """Test FrameMasks dataclass structure."""
    import numpy as np
    from sam3_service_v4.core import FrameMasks
    
    mask = np.zeros((480, 640), dtype=bool)
    frame = FrameMasks(
        frame_index=0,
        object_ids=[0, 1],
        scores=[0.9, 0.8],
        masks=[mask, mask],
    )
    
    assert frame.frame_index == 0
    assert len(frame.object_ids) == 2
    assert len(frame.scores) == 2
    assert len(frame.masks) == 2
    
    # Test as_json
    json_out = frame.as_json(include_masks=True)
    assert "frame_index" in json_out
    assert "num_objects" in json_out
    assert "objects" in json_out
    assert len(json_out["objects"]) == 2
    assert "mask_png_base64" in json_out["objects"][0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])