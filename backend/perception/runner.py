"""Pipeline lifecycle for the service: cached models, auto-calibration, start/stop on a file or stream."""
import os
import threading
import time
import traceback
import cv2

from .calib import Calib
from .frames import LatestFrame, Decoder
from .pipeline import Fusion, WorldState
from .workers import Worker
from .autocalib import calibrate_frames, sample_frames_from_video, sample_frames_from_stream


def calib_path_for(video_path):
    """<stem>.calib.yaml next to the video (e.g. data/x.calib.yaml for data/x.mp4); legacy <file>.calib.yaml also accepted."""
    stem = os.path.splitext(video_path)[0] + ".calib.yaml"
    legacy = f"{video_path}.calib.yaml"
    return stem if os.path.isfile(stem) or not os.path.isfile(legacy) else legacy


class PipelineRunner:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = "idle"            # idle | loading | calibrating | running | error
        self.error = None
        self.source = None
        self.calib = None
        self.calib_info = None
        self.world = WorldState()
        self._models = {}
        self.decoder = self.workers = self.fusion = None
        self.video_slot = None         # LatestFrame fed by the decoder, consumed by the WebRTC track
        self.log_lines = []

    # ---- models (cached across runs) ----------------------------------
    def _log(self, s):
        self.log_lines.append(f"{time.strftime('%H:%M:%S')} {s}")
        self.log_lines = self.log_lines[-200:]
        print(s, flush=True)

    def models(self, light_model="rtdetr", rtdetr_size="r18", band_bottom=0.85):
        from .models.lanes_clrernet import CLRerNetDetector
        from .models.scene_mapanything import MapAnythingScene
        if "lane" not in self._models:
            self._log("loading CLRerNet")
            self._models["lane"] = CLRerNetDetector(band_bottom_frac=band_bottom)
        self._models["lane"].band_bottom_frac = band_bottom
        if "scene" not in self._models:
            self._log("loading MapAnything")
            self._models["scene"] = MapAnythingScene()
        key = f"light:{light_model}:{rtdetr_size}"
        if key not in self._models:
            self._log(f"loading light model {light_model} {rtdetr_size}")
            if light_model == "rtdetr":
                from .models.lights_rtdetr import RTDetrLightDetector
                self._models[key] = RTDetrLightDetector(size=rtdetr_size, conf=0.4, colour="apollo")
            else:
                from .models.lights_openlenda import OpenLendaDetector
                self._models[key] = OpenLendaDetector(conf=0.4, tsize=1280)
        return self._models["lane"], self._models[key], self._models["scene"]

    # ---- lifecycle -------------------------------------------------------
    def start(self, source, calib_path=None, band_bottom=0.85, light_model="rtdetr", rtdetr_size="r18",
              scene_period=0.25, scene_window=0.15, loop=True, lane_width=3.75, send_frame=False):
        self.stop()
        t = threading.Thread(target=self._start_impl, daemon=True, name="runner-start",
                             args=(source, calib_path, band_bottom, light_model, rtdetr_size, scene_period, scene_window, loop, lane_width, send_frame))
        t.start()

    def _start_impl(self, source, calib_path, band_bottom, light_model, rtdetr_size, scene_period, scene_window, loop, lane_width, send_frame):
        try:
            self.state, self.error, self.source = "loading", None, source
            lane, light, scene = self.models(light_model, rtdetr_size, band_bottom)
            is_file = os.path.isfile(str(source))
            src = int(source) if str(source).isdigit() else source
            # calibration
            if calib_path and os.path.isfile(calib_path):
                self.calib = Calib.load(calib_path)
                self.calib_info = {"from": calib_path}
                self._log(f"calib loaded {calib_path}")
            else:
                self.state = "calibrating"
                auto_path = calib_path_for(source) if is_file else None
                if auto_path and os.path.isfile(auto_path):
                    self.calib = Calib.load(auto_path)
                    self.calib_info = {"from": auto_path}
                    self._log(f"calib loaded {auto_path}")
                else:
                    if is_file:
                        frames = sample_frames_from_video(source, 16)
                    else:
                        cap = cv2.VideoCapture(src)
                        if not cap.isOpened():
                            raise RuntimeError(f"cannot open stream {source}")
                        frames = sample_frames_from_stream(cap, 10, 0.4)
                        cap.release()
                    if not frames:
                        raise RuntimeError("no frames for calibration")
                    self._log(f"auto-calibrating on {len(frames)} frames")
                    self.calib, self.calib_info = calibrate_frames(frames, scene, lane, lane_width_m=lane_width, log=self._log)
                    if auto_path:
                        self.calib.save(auto_path)
                        self.calib_info["from"] = auto_path
            # pipeline
            slots = [LatestFrame(), LatestFrame(), LatestFrame()]
            video_slot = LatestFrame()
            dec = Decoder(src, slots + [video_slot], realtime=is_file, loop=loop and is_file)
            if (dec.width, dec.height) != (self.calib.width, self.calib.height):
                self._log(f"WARNING calib {self.calib.width}x{self.calib.height} vs video {dec.width}x{dec.height}")
            workers = (Worker("lane", slots[0], lane.detect),
                       Worker("light", slots[1], light.detect),
                       Worker("scene", slots[2], scene.infer, window_s=scene_window, min_period_s=scene_period))
            fusion = Fusion(self.calib, *workers, self.world, decoder=dec, send_frame=send_frame)
            fusion.source = "live"
            with self.lock:
                self.decoder, self.workers, self.fusion = dec, workers, fusion
                self.video_slot = video_slot
            for w in workers:
                w.start()
            fusion.start()
            dec.start()
            self.state = "running"
            self._log(f"running on {source}")
        except Exception as e:
            self.state, self.error = "error", f"{e!r}"
            self._log("start failed: " + traceback.format_exc())

    def stop(self):
        with self.lock:
            dec, workers, fusion = self.decoder, self.workers, self.fusion
            self.decoder = self.workers = self.fusion = None
        if dec is not None:
            dec.stop()
        if workers:
            for w in workers:
                w.stop()
            for w in workers:
                w.join(timeout=2.0)
        if fusion is not None:
            fusion.stop()
            fusion.join(timeout=2.0)
        if dec is not None:
            dec.join(timeout=2.0)
        if self.state in ("running",):
            self.state = "idle"

    def status(self):
        st, _, _ = self.world.get()
        d = {"state": self.state, "error": self.error, "source": self.source,
             "calib": self.calib.__dict__ if self.calib else None, "calib_info": self.calib_info,
             "metrics": st["metrics"] if st and self.state == "running" else None,
             "models_loaded": sorted(self._models.keys()), "log": self.log_lines[-30:]}
        if self.decoder is not None:
            d["video"] = {"width": self.decoder.width, "height": self.decoder.height, "fps": self.decoder.fps}
        return d
