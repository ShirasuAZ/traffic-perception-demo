"""HTTP + WebSocket service for the frontend.

  uvicorn perception.service:app --host 0.0.0.0 --port 8000

  POST /api/upload            multipart file -> {path}
  POST /api/start             {source, calib?, band_bottom?, light_model?, rtdetr_size?, scene_period?, loop?}
  POST /api/stop
  GET  /api/status
  GET  /api/files             uploaded / known videos
  WS   /ws                    text = WorldState JSON, binary = PCL1 point cloud (see server.py)
"""
import asyncio
import json
import os
import re
import time
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .paths import DATA
from .runner import PipelineRunner, calib_path_for
from .server import pack_cloud
from .webrtc import WebRTCHub

UPLOADS = DATA / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="traffic perception")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
runner = PipelineRunner()
hub = WebRTCHub(lambda: runner.video_slot)


class OfferReq(BaseModel):
    sdp: str
    type: str
    codec: str = "H264"


@app.post("/api/webrtc/offer")
async def webrtc_offer(req: OfferReq):
    return await hub.offer(req.sdp, req.type, prefer=req.codec)


@app.on_event("shutdown")
async def _shutdown():
    await hub.close_all()


class StartReq(BaseModel):
    source: str
    calib: str | None = None
    band_bottom: float = 0.85
    light_model: str = "rtdetr"
    rtdetr_size: str = "r18"
    scene_period: float = 0.25
    loop: bool = True
    lane_width: float = 3.75
    send_frame: bool = False   # embed JPEG frames in the state (fallback when WebRTC is unavailable)


@app.get("/api/status")
def status():
    return runner.status()


@app.get("/api/files")
def files():
    out = []
    for p in sorted(list(UPLOADS.glob("*")) + list(DATA.glob("*.mp4"))):
        if p.suffix.lower() in (".mp4", ".mov", ".mkv", ".webm", ".avi"):
            out.append({"path": str(p), "name": p.name, "size_mb": round(p.stat().st_size / 1e6, 1),
                        "calib": os.path.isfile(calib_path_for(str(p)))})
    return out


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    name = re.sub(r"[^\w.\-]+", "_", file.filename or "video.mp4")
    dst = UPLOADS / f"{int(time.time())}_{name}"
    with open(dst, "wb") as f:
        while True:
            chunk = await file.read(4 << 20)
            if not chunk:
                break
            f.write(chunk)
    return {"path": str(dst), "size_mb": round(dst.stat().st_size / 1e6, 1)}


@app.post("/api/start")
def start(req: StartReq):
    runner.start(req.source, calib_path=req.calib, band_bottom=req.band_bottom, light_model=req.light_model,
                 rtdetr_size=req.rtdetr_size, scene_period=req.scene_period, loop=req.loop, lane_width=req.lane_width,
                 send_frame=req.send_frame)
    return {"ok": True}


@app.post("/api/stop")
def stop():
    runner.stop()
    return {"ok": True, "state": runner.state}


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    last_version, last_scene_ts = -1, None
    try:
        while True:
            state, cloud, version = runner.world.get()
            if state is not None and version != last_version and runner.state == "running":
                last_version = version
                await websocket.send_text(json.dumps(state))
                if cloud is not None and cloud[0] != last_scene_ts:
                    last_scene_ts = cloud[0]
                    await websocket.send_bytes(pack_cloud(cloud[1], cloud[2]))
            await asyncio.sleep(1 / 30)
    except WebSocketDisconnect:
        pass
