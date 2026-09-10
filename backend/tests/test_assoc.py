from perception.calib import Calib
from perception.fusion.assoc import associate

calib = Calib(fx=1000, fy=1000, cx=960, cy=540, width=1920, height=1080, cam_height_m=1.5)
lanes = [
    {"id": 1, "role": "right", "poly_bev": [0, 0, -5.4]},
    {"id": 2, "role": "ego_right", "poly_bev": [0, 0, -1.8]},
    {"id": 3, "role": "ego_left", "poly_bev": [0, 0, 1.8]},
    {"id": 4, "role": "left", "poly_bev": [0, 0, 5.4]},
]


def light(tid, pos, lamps, dist_conf=1.0):
    return {"track_id": tid, "position_ego": pos, "distance_m": pos[0], "distance_confidence": dist_conf, "lamps": lamps}


circle_red = [{"shape": "circle", "color": "red"}]
left_arrow = [{"shape": "circle", "color": "red"}, {"shape": "arrow_left", "color": "green"}]
# middle lane: straight only. Light A ahead (circle), light B left-arrow, light C cross street far right
lights = [light(10, [50, 0.5, 5.5], circle_red), light(11, [50, 6.0, 5.5], left_arrow), light(12, [20, -30, 5.0], circle_red)]
print("middle lane:", associate(lanes, lights, calib))
# leftmost lane -> left allowed -> arrow light wins
lanes_left = [{"id": 2, "role": "ego_right", "poly_bev": [0, 0, -1.8]}, {"id": 3, "role": "ego_left", "poly_bev": [0, 0, 1.8]},
              {"id": 1, "role": "right", "poly_bev": [0, 0, -5.4]}, {"id": 0, "role": "right", "poly_bev": [0, 0, -9.0]}]
print("left lane:  ", associate(lanes_left, lights, calib))
print("no lights:  ", associate(lanes, [], calib))
