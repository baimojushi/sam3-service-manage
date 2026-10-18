"""Entry point with secure listening and exactly one model process."""
from __future__ import annotations
import logging
import os

from .api import create_app
from .core import Settings


def main():
    import uvicorn
    logging.basicConfig(
        level=os.getenv("SAM3_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    cfg = Settings.from_env()
    # Do not trust an external Uvicorn invocation to enforce bind security;
    # the ASGI request handler additionally checks loopback clients.
    requested_host = os.getenv("SAM3_BIND_HOST", "127.0.0.1")
    host = requested_host if cfg.api_key else "127.0.0.1"
    port = int(os.getenv("SAM3_VIDEO_PORT", "6014"))
    if not (1 <= port <= 65535):
        raise ValueError("Invalid SAM3_VIDEO_PORT")
    uvicorn.run(
        create_app(cfg),
        host=host,
        port=port,
        workers=1,
        ws_max_size=cfg.max_frame_bytes + 1024,
        ws_max_queue=2,
        timeout_keep_alive=5,
        log_level=os.getenv("SAM3_LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()
