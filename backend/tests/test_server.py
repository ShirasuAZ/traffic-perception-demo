import asyncio
import json
import time
import numpy as np
import websockets
from perception.pipeline import WorldState
from perception.server import Broadcaster

world = WorldState()
world.set({"hello": 1, "big": "x" * 300000, "metrics": {}, "association": {}, "traffic_lights": [], "lanes": []},
          (0.0, np.zeros((10, 3), np.float32), np.zeros((10, 3), np.uint8)))
b = Broadcaster(world, port=8799)
b.start()
time.sleep(1.0)


async def main():
    async with websockets.connect("ws://localhost:8799", max_size=None) as ws:
        for i in range(3):
            msg = await ws.recv()
            print(i, type(msg).__name__, len(msg), msg[:40] if isinstance(msg, str) else msg[:8])
            world.set({"hello": i + 2, "metrics": {}, "association": {}, "traffic_lights": [], "lanes": []})

asyncio.run(main())
print("server test ok")
