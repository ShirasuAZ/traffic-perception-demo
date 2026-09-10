"""Fusion loop + WorldState. Independent of how model results are produced (live workers or replay)."""
import base64
import threading
import time
import cv2
import numpy as np

from .calib import Calib
from .fusion.lanes import LaneFusion
from .fusion.lights import LightFusion
from .fusion.scene import SceneFusion
from .fusion.assoc import associate
from .workers import RateMeter


class WorldState:
    def __init__(self):
        self._lock = threading.Lock()
        self.state = None
        self.cloud = None          # (scene_ts, xyz float32 (N,3), rgb uint8 (N,3))
        self.version = 0

    def set(self, state, cloud=None):
        with self._lock:
            self.state = state
            if cloud is not None:
                self.cloud = cloud
            self.version += 1

    def get(self):
        with self._lock:
            return self.state, self.cloud, self.version


class Fusion(threading.Thread):
    def __init__(self, calib: Calib, lane_worker, light_worker, scene_worker, world: WorldState,
                 decoder=None, tick_hz=60.0, send_frame=True, frame_width=960, debug_writer=None):
        super().__init__(daemon=True, name="fusion")
        self.calib = calib
        self.lane_w, self.light_w, self.scene_w = lane_worker, light_worker, scene_worker
        self.world, self.decoder = world, decoder
        self.tick = 1.0 / tick_hz
        self.send_frame, self.frame_width = send_frame, frame_width
        self.debug_writer = debug_writer
        self.lanes_f, self.lights_f, self.scene_f = LaneFusion(calib), LightFusion(calib), SceneFusion(calib)
        self.lanes, self.lights, self.scene, self.assoc = [], [], None, {}
        self.seen = {"lane": -1, "light": -1, "scene": -1}
        self.t0_ns = time.monotonic_ns()
        self.meter = RateMeter()
        self.latest_frame = None
        self.latest_frame_ts = 0
        self._stop_evt = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def _ts_s(self, ts_ns):
        return (ts_ns - self.t0_ns) / 1e9

    def run(self):
        while not self._stop_evt.is_set():
            t_start = time.perf_counter()
            updated = False
            if self.scene_w is not None:
                r = self.scene_w.poll(self.seen["scene"])
                if r is not None:
                    seq, ts_ns, views, frame = r
                    self.seen["scene"] = seq
                    self.scene = self.scene_f.update(views, self._ts_s(ts_ns), frame)
                    updated = True
            if self.lane_w is not None:
                r = self.lane_w.poll(self.seen["lane"])
                if r is not None:
                    seq, ts_ns, lanes_2d, frame = r
                    self.seen["lane"] = seq
                    depth = self.scene["depth"] if self.scene else None
                    ratio = self.scene["depth_ratio"] if self.scene else None
                    self.lanes = self.lanes_f.update(lanes_2d, self._ts_s(ts_ns), depth, ratio)
                    self.raw_lanes = lanes_2d
                    self.latest_frame, self.latest_frame_ts = frame, ts_ns
                    updated = True
            if self.light_w is not None:
                r = self.light_w.poll(self.seen["light"])
                if r is not None:
                    seq, ts_ns, housings, frame = r
                    self.seen["light"] = seq
                    self.lights = self.lights_f.update(housings, self._ts_s(ts_ns))
                    if ts_ns > self.latest_frame_ts:
                        self.latest_frame, self.latest_frame_ts = frame, ts_ns
                    updated = True
            if updated:
                self.assoc = associate(self.lanes, self.lights, self.calib)
                self.publish()
                self.meter.tick()
            rest = self.tick - (time.perf_counter() - t_start)
            if rest > 0:
                time.sleep(rest)

    # ---- output ---------------------------------------------------------
    def metrics(self):
        m = {"fusion_hz": round(self.meter.hz, 1), "source": getattr(self, "source", "live")}
        if self.decoder is not None:
            m["decode_hz"] = round(self.decoder.decode_hz, 1)
        for name, w in (("lane", self.lane_w), ("light", self.light_w), ("scene", self.scene_w)):
            if w is None:
                continue
            m[f"{name}_hz"] = round(w.meter.hz, 1)
            m[f"{name}_latency_ms"] = round(w.latency_ms, 1)
            m[f"{name}_model_ms"] = round(w.model_ms, 1)
            if w.error:
                m[f"{name}_error"] = w.error
        return m

    def publish(self):
        now_ns = time.monotonic_ns()
        sc = self.scene
        state = {
            "frame_ts_ns": int(self.latest_frame_ts),
            "state_ts_ns": int(now_ns),
            "e2e_latency_ms": round((now_ns - self.latest_frame_ts) / 1e6, 1) if self.latest_frame_ts else None,
            "coords": {"ego": "x forward, y left, z up, origin camera ground point, metres",
                       "world": "ego frame at start; only recent history is consistent"},
            "calib": {"fx": self.calib.fx, "fy": self.calib.fy, "cx": self.calib.cx, "cy": self.calib.cy,
                      "width": self.calib.width, "height": self.calib.height,
                      "cam_height_m": self.calib.cam_height_m, "pitch_deg": self.calib.pitch_deg},
            "ego": {
                "T_world_ego": sc["T_world_ego"] if sc else np.eye(4).tolist(),
                "speed_mps": sc["speed_mps"] if sc else 0.0,
                "yaw_rate_rps": sc["yaw_rate_rps"] if sc else 0.0,
                "odom_confidence": sc["odom_confidence"] if sc else 0.0,
            },
            "lanes": self.lanes,
            "ego_lane_id": LaneFusion.ego_lane_id(self.lanes),
            "ego_lane_turn": self.assoc.get("ego_lane_turn"),
            "traffic_lights": self.lights,
            "association": {k: self.assoc.get(k) for k in ("ego_lane_light_track_id", "confidence", "score_breakdown")},
            "scene": {
                "scene_ts_s": sc["ts"] if sc else None,
                "n_points": int(len(sc["points_xyz"])) if sc else 0,
                "voxel_size_m": 0.15,
                "scale_factor": sc["scale_factor"] if sc else None,
                "scale_confidence": sc["scale_confidence"] if sc else 0.0,
            },
            "metrics": self.metrics(),
        }
        if self.send_frame and self.latest_frame is not None:
            f = self.latest_frame
            if f.shape[1] > self.frame_width:
                s = self.frame_width / f.shape[1]
                f = cv2.resize(f, (self.frame_width, int(f.shape[0] * s)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                state["frame"] = {"jpeg_b64": base64.b64encode(buf.tobytes()).decode(), "width": f.shape[1],
                                  "height": f.shape[0], "scale": f.shape[1] / self.calib.width}
        cloud = (sc["ts"], sc["points_xyz"], sc["points_rgb"]) if sc else None
        self.world.set(state, cloud)
        if self.debug_writer is not None and self.latest_frame is not None:
            self.debug_writer.write(draw_overlay(self.latest_frame.copy(), state, self.calib, getattr(self, "raw_lanes", [])))


def draw_overlay(img, state, calib, raw_lanes=()):
    for uv, conf in raw_lanes:                       # thin blue = raw 2D detections
        pts = np.asarray(uv)
        pts = pts[np.isfinite(pts).all(1)].astype(np.int32)
        if len(pts) > 1:
            cv2.polylines(img, [pts], False, (255, 120, 0), 1)
    for lane in state["lanes"]:
        pts = np.array(lane["points_2d"])
        pts = pts[np.isfinite(pts).all(1)].astype(np.int32)
        col = (0, 255, 0) if lane["role"].startswith("ego") else (255, 200, 0)
        if lane["coasting"]:
            col = (128, 128, 128)
        if len(pts) > 1:
            cv2.polylines(img, [pts], False, col, 3)
            cv2.putText(img, f"{lane['id']}:{lane['role']}", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    a_id = state["association"]["ego_lane_light_track_id"]
    for lt in state["traffic_lights"]:
        x1, y1, x2, y2 = map(int, lt["bbox_2d"])
        col = (0, 0, 255) if lt["track_id"] == a_id else (0, 200, 255)
        if lt["coasting"]:
            col = (140, 140, 140)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2 if not lt["coasting"] else 1)
        lamps = ",".join(f"{l['shape'][:1]}{l['shape'][6:7]}:{l['color']}{'*' if l['forced'] else ''}" for l in lt["lamps"])
        cv2.putText(img, f"#{lt['track_id']} {lamps} {lt['distance_m']}m", (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    m = state["metrics"]
    lines = [f"lane {m.get('lane_hz', 0)}Hz {m.get('lane_latency_ms', 0)}ms | light {m.get('light_hz', 0)}Hz "
             f"{m.get('light_latency_ms', 0)}ms | scene {m.get('scene_hz', 0)}Hz {m.get('scene_latency_ms', 0)}ms",
             f"decode {m.get('decode_hz', 0)}Hz fusion {m.get('fusion_hz', 0)}Hz e2e {state['e2e_latency_ms']}ms "
             f"scale {state['scene']['scale_factor']} ({state['scene']['scale_confidence']}) "
             f"speed {state['ego']['speed_mps']} m/s",
             f"ego lane {state['ego_lane_id']} turn {state['ego_lane_turn']['allowed'] if state['ego_lane_turn'] else None} "
             f"-> light {a_id} conf {state['association']['confidence']}"]
    for i, t in enumerate(lines):
        cv2.putText(img, t, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img
