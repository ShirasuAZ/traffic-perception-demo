"""Connect to the pipeline WebSocket and report message rates/sizes (contract check for the frontend).

  python -m scripts.ws_probe [ws://localhost:8765] [seconds]
"""
import asyncio
import json
import struct
import sys
import time
import numpy as np
import websockets


async def main(url, seconds):
    n_txt = n_bin = 0
    bytes_txt = bytes_bin = 0
    last = None
    t0 = time.time()
    cloud_stats = None
    async with websockets.connect(url, max_size=None) as ws:
        while time.time() - t0 < seconds:
            try:
                msg = await ws.recv()
            except websockets.ConnectionClosed as e:
                print(f"connection closed after {n_txt} text / {n_bin} binary messages: {e!r}")
                break
            if isinstance(msg, bytes):
                assert msg[:4] == b"PCL1"
                n = struct.unpack("<I", msg[4:8])[0]
                xyz = np.frombuffer(msg[8:8 + n * 12], np.float32).reshape(n, 3)
                n_bin += 1
                bytes_bin += len(msg)
                cloud_stats = (n, xyz.min(0).round(1).tolist(), xyz.max(0).round(1).tolist())
            else:
                last = json.loads(msg)
                n_txt += 1
                bytes_txt += len(msg)
    dt = time.time() - t0
    print(f"state msgs: {n_txt / dt:.1f} Hz, {bytes_txt / max(n_txt, 1) / 1024:.0f} KB avg")
    print(f"cloud msgs: {n_bin / dt:.1f} Hz, {bytes_bin / max(n_bin, 1) / 1024:.0f} KB avg")
    if n_bin:
        print("last cloud: N=%d min=%s max=%s (ego frame, m)" % cloud_stats)
    if last:
        keys = {k: (type(v).__name__ if not isinstance(v, (list, dict)) else f"{type(v).__name__}[{len(v)}]") for k, v in last.items()}
        print("state keys:", keys)
        print("metrics:", last["metrics"])
        print("association:", last["association"])
        if last["traffic_lights"]:
            print("light[0]:", {k: v for k, v in last["traffic_lights"][0].items()})
        if last["lanes"]:
            lane = last["lanes"][0]
            print("lane[0]:", {k: (v if k not in ("points_3d", "points_2d") else f"{len(v)} pts") for k, v in lane.items()})


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8765"
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 5
    asyncio.run(main(url, secs))
