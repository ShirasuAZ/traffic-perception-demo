"""WebSocket broadcaster. Text frames carry the WorldState JSON; binary frames carry the point cloud:
  magic b"PCL1" | uint32 N | float32[N,3] xyz (ego frame) | uint8[N,3] rgb
The cloud is only re-sent when the scene has been updated."""
import asyncio
import json
import struct
import threading
import numpy as np
import websockets

MAGIC = b"PCL1"


def pack_cloud(xyz, rgb):
    n = len(xyz)
    return MAGIC + struct.pack("<I", n) + np.ascontiguousarray(xyz, np.float32).tobytes() + np.ascontiguousarray(rgb, np.uint8).tobytes()


class Broadcaster(threading.Thread):
    def __init__(self, world, host="0.0.0.0", port=8765, rate_hz=30.0):
        super().__init__(daemon=True, name="ws")
        self.world, self.host, self.port, self.period = world, host, port, 1.0 / rate_hz
        self.clients = 0

    async def _client(self, ws):
        self.clients += 1
        last_version, last_scene_ts = -1, None
        print(f"[ws] client connected ({self.clients})", flush=True)
        try:
            while True:
                state, cloud, version = self.world.get()
                if state is not None and version != last_version:
                    last_version = version
                    await ws.send(json.dumps(state))
                    if cloud is not None and cloud[0] != last_scene_ts:
                        last_scene_ts = cloud[0]
                        await ws.send(pack_cloud(cloud[1], cloud[2]))
                await asyncio.sleep(self.period)
        except websockets.ConnectionClosed as e:
            print(f"[ws] client closed: {e!r}", flush=True)
        except Exception:
            import traceback
            traceback.print_exc()
        finally:
            self.clients -= 1

    async def _main(self):
        async with websockets.serve(self._client, self.host, self.port, max_size=None, ping_interval=None):
            await asyncio.Future()

    def run(self):
        asyncio.run(self._main())
