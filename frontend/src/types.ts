export interface Lamp {
  shape: string; // circle | arrow_left | arrow_straight | arrow_right
  color: string; // red | yellow | green | empty
  confidence: number;
  stable_frames: number;
  forced: boolean;
}

export interface TrafficLight {
  track_id: number;
  bbox_2d: [number, number, number, number];
  orientation: string;
  confidence: number;
  coasting: boolean;
  distance_m: number;
  distance_confidence: number;
  position_ego: [number, number, number];
  lamps: Lamp[];
  transition_ts: number | null;
}

export interface Lane {
  id: number;
  role: string; // ego_left | ego_right | left | right
  confidence: number;
  age_frames: number;
  coasting: boolean;
  poly_bev: [number, number, number];
  points_3d: [number, number, number][];
  points_2d: [number, number][];
}

export interface WorldState {
  frame_ts_ns: number;
  state_ts_ns: number;
  e2e_latency_ms: number | null;
  calib: { fx: number; fy: number; cx: number; cy: number; width: number; height: number; cam_height_m: number; pitch_deg: number };
  ego: { T_world_ego: number[][]; speed_mps: number; yaw_rate_rps: number; odom_confidence: number };
  lanes: Lane[];
  ego_lane_id: number | null;
  ego_lane_turn: { allowed: string[]; confidence: number } | null;
  traffic_lights: TrafficLight[];
  association: { ego_lane_light_track_id: number | null; confidence: number; score_breakdown: Record<string, number> | null };
  scene: { scene_ts_s: number | null; n_points: number; voxel_size_m: number; scale_factor: number | null; scale_confidence: number };
  metrics: Record<string, number | string>;
  frame?: { jpeg_b64: string; width: number; height: number; scale: number };
}

export interface Cloud {
  ts: number;
  xyz: Float32Array; // N*3, ego frame (x fwd, y left, z up)
  rgb: Uint8Array; // N*3
  n: number;
}

export interface Status {
  state: string;
  error: string | null;
  source: string | null;
  calib: Record<string, number> | null;
  calib_info: Record<string, unknown> | null;
  metrics: Record<string, number | string> | null;
  models_loaded: string[];
  log: string[];
  video?: { width: number; height: number; fps: number };
}
