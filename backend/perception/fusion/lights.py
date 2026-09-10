"""Traffic-light fusion: IoU tracking, per-lamp temporal voting with a legal-transition state machine
(with a forced-accept fallback), housing-size ranging and 3D placement in the ego frame."""
from collections import deque, Counter
import numpy as np

COLORS = ("red", "green", "yellow", "empty")
SHAPES = ("straight", "left", "right", "other")
# "off" = housing seen but this lamp slot not lit (arrows). Any lamp may switch to/from off.
LEGAL_NEXT = {"green": {"yellow", "empty", "off"}, "yellow": {"red", "empty", "off"}, "red": {"green", "empty", "off"},
              "empty": {"red", "green", "yellow", "off"}, "off": {"red", "green", "yellow", "empty"},
              None: set(COLORS) | {"off"}}
VOTE_WINDOW = 5
FORCE_AFTER = 10            # frames a rejected state must persist before it is accepted anyway
MAX_MISS_S = 0.7
MIN_HITS = 3
IOU_GATE = 0.3
# physical size priors (m): 3-lamp housing width (horizontal) / height (vertical)
HOUSING_W_H = 1.15
HOUSING_H_V = 1.15


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class LampState:
    """Temporal state of one lamp slot (the circle colour, or one arrow direction)."""
    def __init__(self):
        self.state = None
        self.votes = deque(maxlen=VOTE_WINDOW)
        self.stable_frames = 0
        self.pending = None
        self.pending_frames = 0
        self.forced = False
        self.transition_ts = None
        self.conf = 0.0

    def push(self, observed, conf, ts):
        """observed: colour string or None (lamp not seen / off)."""
        self.votes.append(observed)
        self.conf = 0.7 * self.conf + 0.3 * (conf if observed else 0.0)
        winner, n = Counter(self.votes).most_common(1)[0]
        if n < (len(self.votes) // 2 + 1):
            return
        if winner == self.state:
            self.stable_frames += 1
            self.pending, self.pending_frames = None, 0
            return
        if winner is None:
            return                               # keep last known state while unseen
        if winner in LEGAL_NEXT.get(self.state, set()) or self.state is None:
            self._switch(winner, ts, forced=False)
        else:
            if self.pending == winner:
                self.pending_frames += 1
            else:
                self.pending, self.pending_frames = winner, 1
            if self.pending_frames >= FORCE_AFTER:
                self._switch(winner, ts, forced=True)

    def _switch(self, s, ts, forced):
        self.state, self.forced = s, forced
        self.stable_frames = 0
        self.pending, self.pending_frames = None, 0
        self.transition_ts = ts


class LightTrack:
    _next_id = 1

    def __init__(self, housing, ts):
        self.id = LightTrack._next_id
        LightTrack._next_id += 1
        self.bbox = np.array(housing["bbox"], float)
        self.vel = np.zeros(4)
        self.orientation = housing["orientation"]
        self.circle = LampState()
        self.arrows = {s: LampState() for s in ("straight", "left", "right")}
        self.hits, self.miss, self.last_seen = 0, 0, ts
        self.conf = 0.0
        self.observe(housing, ts)

    def predict(self):
        self.bbox = self.bbox + self.vel

    def observe(self, housing, ts):
        nb = np.array(housing["bbox"], float)
        if self.hits > 0:
            self.vel = 0.5 * self.vel + 0.5 * (nb - self.bbox)
        self.bbox = 0.4 * self.bbox + 0.6 * nb
        self.orientation = housing["orientation"]
        self.hits += 1
        self.miss = 0
        self.last_seen = ts
        self.conf = 0.7 * self.conf + 0.3 * housing["conf"]
        colours = {l["label"]: l["conf"] for l in housing["lamps"] if l["label"] in COLORS}
        shapes = {l["label"]: l["conf"] for l in housing["lamps"] if l["label"] in SHAPES}
        if colours:
            c = max(colours, key=colours.get)
            self.circle.push(c, colours[c], ts)
        else:
            self.circle.push(None, 0.0, ts)
        for s, st in self.arrows.items():
            # OpenLenda arrows are lit (green) arrows; housing seen without the arrow label => arrow off
            st.push("green" if s in shapes else "off", shapes.get(s, 0.0), ts)

    def coast(self, ts):
        self.miss += 1
        self.circle.push(None, 0.0, ts)
        for st in self.arrows.values():
            st.push(None, 0.0, ts)


class LightFusion:
    def __init__(self, calib):
        self.calib = calib
        self.tracks = []

    def update(self, housings, ts):
        for t in self.tracks:
            t.predict()
        unmatched = set(range(len(housings)))
        if self.tracks and housings:
            # affinity = IoU, or a centre-distance fallback for small/fast-moving boxes
            iou = np.zeros((len(self.tracks), len(housings)))
            for i, t in enumerate(self.tracks):
                tc = np.array([(t.bbox[0] + t.bbox[2]) / 2, (t.bbox[1] + t.bbox[3]) / 2])
                diag = max(np.hypot(t.bbox[2] - t.bbox[0], t.bbox[3] - t.bbox[1]), 8.0)
                for j, h in enumerate(housings):
                    b = h["bbox"]
                    hc = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
                    d = np.linalg.norm(tc - hc) / diag
                    iou[i, j] = max(_iou(t.bbox, b), IOU_GATE + 0.3 * max(0.0, 1.0 - d / 1.5))
                    if d > 1.5 and _iou(t.bbox, b) < IOU_GATE:
                        iou[i, j] = 0.0
            while True:
                r, c = np.unravel_index(iou.argmax(), iou.shape)
                if iou[r, c] < IOU_GATE:
                    break
                self.tracks[r].observe(housings[c], ts)
                unmatched.discard(c)
                iou[r, :] = -1
                iou[:, c] = -1
        for c in unmatched:
            self.tracks.append(LightTrack(housings[c], ts))
        for t in self.tracks:
            if t.last_seen != ts:
                t.coast(ts)
        self.tracks = [t for t in self.tracks if ts - t.last_seen <= MAX_MISS_S]
        return self.publish()

    # ---- geometry --------------------------------------------------------
    def _range(self, t):
        w = t.bbox[2] - t.bbox[0]
        h = t.bbox[3] - t.bbox[1]
        if t.orientation == "horizontal":
            px, real, f = w, HOUSING_W_H, self.calib.fx
        else:
            px, real, f = h, HOUSING_H_V, self.calib.fy
        px = max(px, 1.0)
        dist = f * real / px
        conf = float(np.clip((px - 6.0) / 24.0, 0.05, 1.0))  # 6 px -> 0.05, 30 px -> 1.0
        return dist, conf

    def _position(self, t, dist):
        cx, cy = (t.bbox[0] + t.bbox[2]) / 2, (t.bbox[1] + t.bbox[3]) / 2
        ray = self.calib.pixel_rays_ego([[cx, cy]])[0]
        # dist is along the optical axis (pinhole ranging); convert to along-ray
        cam_fwd = self.calib.R_cam2ego[:, 2]
        s = dist / max(float(ray @ cam_fwd), 1e-3)
        return self.calib.t_cam2ego + s * ray

    # ---- output ----------------------------------------------------------
    def publish(self):
        out = []
        for t in self.tracks:
            if t.hits < MIN_HITS:
                continue
            dist, dconf = self._range(t)
            pos = self._position(t, dist)
            lamps = []
            if t.circle.state:
                lamps.append({"shape": "circle", "color": t.circle.state, "confidence": round(t.circle.conf, 3),
                              "stable_frames": t.circle.stable_frames, "forced": t.circle.forced})
            for s, st in t.arrows.items():
                if st.state == "green":
                    lamps.append({"shape": f"arrow_{s}", "color": "green", "confidence": round(st.conf, 3),
                                  "stable_frames": st.stable_frames, "forced": st.forced})
            trans = [x.transition_ts for x in [t.circle, *t.arrows.values()] if x.transition_ts is not None]
            out.append({
                "track_id": t.id,
                "bbox_2d": t.bbox.round(1).tolist(),
                "orientation": t.orientation,
                "confidence": round(float(t.conf), 3),
                "coasting": t.miss > 0,
                "distance_m": round(float(dist), 1),
                "distance_confidence": round(dconf, 2),
                "position_ego": pos.round(2).tolist(),
                "lamps": lamps,
                "transition_ts": max(trans) if trans else None,
            })
        return out
