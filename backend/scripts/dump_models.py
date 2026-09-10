"""Run the three models over a whole video (not realtime) and dump per-frame results for replay.

  python -m scripts.dump_models --video data/x.mp4 --out data/x.replay.pkl [--scene-every 3] [--max-frames N]
"""
import argparse
import pickle
import time
import cv2

from perception.models.lanes_clrernet import CLRerNetDetector
from perception.models.scene_mapanything import MapAnythingScene
from scripts.run_pipeline import build_light_model, add_model_args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene-every", type=int, default=3, help="run MapAnything every N frames")
    ap.add_argument("--scene-gap", type=int, default=4, help="frames between the two views of a window")
    ap.add_argument("--max-frames", type=int, default=0)
    add_model_args(ap)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    lane, light, scene = CLRerNetDetector(band_bottom_frac=args.band_bottom), build_light_model(args), MapAnythingScene()
    tables = {"lane": {}, "light": {}, "scene": {}, "meta": {"video": args.video, "fps": cap.get(cv2.CAP_PROP_FPS)}}
    hist, seq, t0 = {}, 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and seq >= args.max_frames):
            break
        tables["lane"][seq] = lane.detect(frame)
        tables["light"][seq] = light.detect(frame)
        hist[seq] = frame
        if seq % args.scene_every == 0:
            prev = hist.get(seq - args.scene_gap)
            views = scene.infer([prev, frame] if prev is not None else [frame])
            for v in views:          # drop bulky things we do not need for replay
                v.pop("conf", None)
            tables["scene"][seq] = views
        for k in [k for k in hist if k < seq - args.scene_gap]:
            del hist[k]
        seq += 1
        if seq % 30 == 0:
            print(f"{seq} frames, {seq / (time.time() - t0):.1f} fps")
    with open(args.out, "wb") as f:
        pickle.dump(tables, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("wrote", args.out, "frames", seq)


if __name__ == "__main__":
    main()
