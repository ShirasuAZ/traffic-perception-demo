"""Ego lane <-> traffic light association (rule based, with an explicit confidence)."""
import numpy as np

from .lanes import eval_poly

X_FAR = 30.0


def ego_lane_turns(lanes):
    """Allowed turns for the ego lane from lane layout heuristics. Returns (set, confidence)."""
    offs = {l["id"]: eval_poly(l["poly_bev"], 5.0) for l in lanes}
    ego_left = next((l for l in lanes if l["role"] == "ego_left"), None)
    ego_right = next((l for l in lanes if l["role"] == "ego_right"), None)
    if ego_left is None or ego_right is None:
        return {"straight"}, 0.3
    n_left = sum(1 for l in lanes if offs[l["id"]] > offs[ego_left["id"]] + 0.5)
    n_right = sum(1 for l in lanes if offs[l["id"]] < offs[ego_right["id"]] - 0.5)
    n_lanes = n_left + n_right + 1
    allowed = {"straight"}
    if n_lanes >= 3 and n_left == 0:
        allowed.add("left")
    if n_lanes >= 3 and n_right == 0:
        allowed.add("right")
    return allowed, 0.5


def ego_lane_heading(lanes):
    """Bearing (rad, + left) of the ego lane centre line at X_FAR, from the two ego boundaries."""
    ego = [l for l in lanes if l["role"] in ("ego_left", "ego_right")]
    if not ego:
        return 0.0
    y = np.mean([eval_poly(l["poly_bev"], X_FAR) for l in ego])
    return float(np.arctan2(y, X_FAR))


def associate(lanes, lights, calib):
    """Returns dict(ego_lane_light_track_id, confidence, score_breakdown, ego_lane_turn)."""
    allowed, turn_conf = ego_lane_turns(lanes)
    heading = ego_lane_heading(lanes)
    best, scores = None, []
    for lt in lights:
        pos = np.array(lt["position_ego"])
        bearing = float(np.arctan2(pos[1], max(pos[0], 1e-3)))
        d_ang = abs(np.degrees(bearing - heading))
        s_bearing = 0.5 * float(np.clip((25.0 - d_ang) / 17.0, 0.0, 1.0))      # <8 deg full, 25 deg zero
        lamps = lt["lamps"]
        arrows = {l["shape"].replace("arrow_", "") for l in lamps if l["shape"].startswith("arrow_")}
        if arrows & allowed:
            s_sem = 1.0
        elif arrows:
            s_sem = -1.0                                                        # arrows for a different movement
        elif any(l["shape"] == "circle" for l in lamps):
            s_sem = 0.6
        else:
            s_sem = 0.0
        s_dist = 0.3 if 15.0 < lt["distance_m"] < 120.0 else 0.0
        s_h = 0.2 if 3.0 < pos[2] < 8.0 else 0.0
        total = s_sem + s_bearing + s_dist + s_h
        total *= 0.5 + 0.5 * lt["distance_confidence"]
        if lt.get("coasting"):
            total *= 0.5
        scores.append((total, lt["track_id"], {"semantic": s_sem, "bearing": round(s_bearing, 2),
                                                "distance": s_dist, "height": s_h}))
    scores.sort(key=lambda s: -s[0])
    result = {"ego_lane_light_track_id": None, "confidence": 0.0, "score_breakdown": None,
              "ego_lane_turn": {"allowed": sorted(allowed), "confidence": turn_conf}}
    if not scores:
        return result
    top, tid, br = scores[0]
    margin = top - scores[1][0] if len(scores) > 1 else 1.0
    conf = float(np.clip(top / 2.0, 0.0, 1.0))
    if top < 1.0 or margin < 0.3:
        conf = min(conf, 0.4)
    result.update({"ego_lane_light_track_id": tid, "confidence": round(conf, 2), "score_breakdown": br})
    return result
