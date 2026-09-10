"""Run the full perception pipeline on a video (live inference) or on dumped model outputs (replay).

  python -m scripts.run_pipeline --video data/x.mp4 --calib data/x.calib.yaml [--ws-port 8765] [--debug-out out.mp4]
  python -m scripts.run_pipeline --video data/x.mp4 --calib ... --replay data/x.replay.pkl
"""
import argparse
import pickle
import sys
import time
import cv2

from perception.calib import Calib
from perception.frames import LatestFrame, Decoder
from perception.pipeline import Fusion, WorldState
from perception.workers import Worker, ReplayWorker


def build_light_model(args):
    """--light-model: rtdetr (COCO/Objects365 detector + Apollo colour CNN; Chinese lights) or openlenda (Japanese lights, arrows)."""
    if args.light_model == "rtdetr":
        from perception.models.lights_rtdetr import RTDetrLightDetector
        return RTDetrLightDetector(size=args.rtdetr_size, conf=args.light_conf, colour="apollo")
    from perception.models.lights_openlenda import OpenLendaDetector
    return OpenLendaDetector(conf=args.light_conf, tsize=1280)


def add_model_args(ap):
    ap.add_argument("--light-model", choices=["rtdetr", "openlenda"], default="rtdetr")
    ap.add_argument("--rtdetr-size", choices=["r18", "r50"], default="r50")
    ap.add_argument("--light-conf", type=float, default=0.4)
    ap.add_argument("--band-bottom", type=float, default=0.9, help="fraction of frame height where the road band ends (exclude hood)")


def build_workers(args, slots):
    if args.replay:
        with open(args.replay, "rb") as f:
            tables = pickle.load(f)
        return (ReplayWorker("lane", slots[0], tables["lane"], 0.02),
                ReplayWorker("light", slots[1], tables["light"], 0.01),
                ReplayWorker("scene", slots[2], tables["scene"], 0.08))
    from perception.models.lanes_clrernet import CLRerNetDetector
    from perception.models.scene_mapanything import MapAnythingScene
    lane = CLRerNetDetector(band_bottom_frac=args.band_bottom)
    light = build_light_model(args)
    scene = MapAnythingScene()
    return (Worker("lane", slots[0], lane.detect),
            Worker("light", slots[1], light.detect),
            Worker("scene", slots[2], scene.infer, window_s=args.scene_window, min_period_s=args.scene_period))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--replay", default=None)
    ap.add_argument("--ws-port", type=int, default=8765)
    ap.add_argument("--no-ws", action="store_true")
    ap.add_argument("--debug-out", default=None)
    ap.add_argument("--duration", type=float, default=0, help="stop after N seconds (0 = run until video ends / Ctrl-C)")
    ap.add_argument("--no-loop", action="store_true")
    add_model_args(ap)
    ap.add_argument("--scene-window", type=float, default=0.15)
    ap.add_argument("--scene-period", type=float, default=0.0)
    ap.add_argument("--no-frame", action="store_true", help="do not embed JPEG frames in the state")
    args = ap.parse_args()

    calib = Calib.load(args.calib)
    slots = [LatestFrame(), LatestFrame(), LatestFrame()]
    dec = Decoder(args.video, slots, realtime=True, loop=not args.no_loop)
    if (dec.width, dec.height) != (calib.width, calib.height):
        print(f"WARNING: calib is {calib.width}x{calib.height} but video is {dec.width}x{dec.height}", file=sys.stderr)
    workers = build_workers(args, slots)
    world = WorldState()
    writer = None
    if args.debug_out:
        writer = cv2.VideoWriter(args.debug_out, cv2.VideoWriter_fourcc(*"mp4v"), dec.fps, (dec.width, dec.height))
    fusion = Fusion(calib, *workers, world, decoder=dec, send_frame=not args.no_frame, debug_writer=writer)
    fusion.source = "replay" if args.replay else "live"
    if not args.no_ws:
        from perception.server import Broadcaster
        Broadcaster(world, port=args.ws_port).start()
        print(f"websocket on ws://localhost:{args.ws_port}")
    for w in workers:
        w.start()
    fusion.start()
    dec.start()
    t0 = time.time()
    try:
        while dec.is_alive():
            time.sleep(1.0)
            st, _, _ = world.get()
            if st:
                m = st["metrics"]
                print(f"[{time.time() - t0:5.1f}s] " + " ".join(f"{k}={v}" for k, v in m.items() if not k.endswith("model_ms"))
                      + f" lanes={len(st['lanes'])} lights={len(st['traffic_lights'])} assoc={st['association']['ego_lane_light_track_id']}")
            if args.duration and time.time() - t0 > args.duration:
                break
    except KeyboardInterrupt:
        pass
    dec.stop()
    for w in workers:
        w.stop()
    for w in workers:
        w.join(timeout=3.0)
    fusion.stop()
    fusion.join(timeout=2.0)
    dec.join(timeout=2.0)
    if writer is not None:
        writer.release()
        print("debug video written:", args.debug_out)
    st, _, _ = world.get()
    if st:
        import json
        import os
        os.makedirs("logs", exist_ok=True)
        with open("logs/last_state.json", "w") as f:
            f.write(json.dumps({k: v for k, v in st.items() if k != "frame"}, indent=1))
        print("last state written to logs/last_state.json")


if __name__ == "__main__":
    main()
