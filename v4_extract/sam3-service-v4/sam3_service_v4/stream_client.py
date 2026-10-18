#!/usr/bin/env python3
"""True frame-by-frame SAM3 v4 WebSocket smoke/evaluation client.

Each JPEG/PNG frame is sent only after the previous inference ACK is received.
The produced ZIP can be evaluated independently against the original GT.
Requires: websockets, opencv-python, numpy.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import inspect
import io
import json
import os
import zipfile

import cv2
import websockets


async def run(args):
    headers = {}
    token = os.getenv("API_KEY") or args.api_key
    if token:
        headers["Authorization"] = f"Bearer {token}"
    kw = {"max_size": 32 * 1024 * 1024, "max_queue": 2,
          "open_timeout": 30, "ping_timeout": 60}
    param = "additional_headers" if "additional_headers" in inspect.signature(websockets.connect).parameters \
        else "extra_headers"
    kw[param] = headers

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot decode input video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    frames = []
    index = 0
    try:
        async with websockets.connect(args.url, **kw) as ws:
            await ws.send(json.dumps({
                "type": "start", "prompt": args.prompt,
                "max_frames": args.frames, "output_prob_thresh": args.threshold,
                "return_masks_png": not args.metadata_only,
            }))
            greeting = json.loads(await ws.recv())
            if greeting.get("type") != "ready":
                raise RuntimeError(f"Stream did not start: {greeting}")
            print("STREAM_READY", json.dumps(greeting, ensure_ascii=False))
            with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
                while index < args.frames:
                    ok, bgr = cap.read()
                    if not ok:
                        raise RuntimeError(f"Input ends after {index}/{args.frames} frames")
                    ok, png = cv2.imencode(".png", bgr)
                    if not ok:
                        raise RuntimeError("PNG encode failed")
                    await ws.send(json.dumps({
                        "type": "frame_meta", "frame_index": index,
                        "timestamp_ms": round(index * 1000.0 / fps, 3),
                    }))
                    await ws.send(png.tobytes())
                    result = json.loads(await ws.recv())
                    if result.get("type") != "frame_result" or result.get("frame_index") != index:
                        raise RuntimeError(f"Frame ACK mismatch at {index}: {result}")
                    objects = result.get("objects", [])
                    for item in objects:
                        encoded = item.get("mask_png_base64")
                        if encoded:
                            blob = base64.b64decode(encoded, validate=True)
                            archive.writestr(
                                f"frame_{index:04d}_obj_{int(item['object_id']):02d}_"
                                f"score{float(item['score']):.4f}.png", blob
                            )
                    frames.append({
                        "frame_index": index, "timestamp_ms": result["timestamp_ms"],
                        "inference_ms": result.get("inference_ms"),
                        "objects": [{k: v for k, v in item.items() if k != "mask_png_base64"}
                                    for item in objects],
                    })
                    index += 1
                    if index == 1 or index % 10 == 0:
                        print(f"frame={index}/{args.frames} objects={len(objects)} "
                              f"inference_ms={result.get('inference_ms')}")
                done = json.loads(await ws.recv())
                if done.get("type") != "done" or done.get("frames_processed") != args.frames:
                    raise RuntimeError(f"Invalid stream completion: {done}")
                archive.writestr("meta.json", json.dumps({
                    "engine": greeting["backend"], "text_prompt": args.prompt,
                    "num_frames": len(frames), "frame_indices": list(range(len(frames))),
                    "frame_results": frames,
                    "prompt_frame": {
                        "obj_ids": [o["object_id"] for o in frames[0]["objects"]],
                        "scores": [o["score"] for o in frames[0]["objects"]],
                    },
                }, ensure_ascii=False, indent=2))
    finally:
        cap.release()
    print(f"STREAM_PASS frames={index} zip={args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:6014/video/stream")
    parser.add_argument("--video", required=True)
    parser.add_argument("--prompt", default="brush")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--output", default="sam3_stream_predictions.zip")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
