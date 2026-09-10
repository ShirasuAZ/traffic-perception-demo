# traffic-perception-demo

单目行车视频的实时感知演示：车道线、交通灯（颜色）、米制 3D 场景与自车运动，以及车道与交通灯的关联。零训练，全部使用开源预训练权重，单卡 GPU 实时推理。后端 Python，前端 React + Three.js。

## 架构

```
video file / stream
      │  解码线程，每个消费者一个 LatestFrame 槽位（覆盖写，永不积压）
      ├──► LaneWorker   CLRerNet            2D 车道点
      ├──► LightWorker  RT-DETR + Apollo CNN 灯框 + 颜色
      └──► SceneWorker  MapAnything (2 视图)  深度 / 内参 / 相对位姿
                │
        Fusion 线程（60 Hz tick，纯 CPU）
          lanes : IPM → BEV 二次拟合 → 匈牙利关联 → Kalman
          lights: IoU + 中心距跟踪 → 每灯滑窗投票 + 状态机（带强制接受兜底）→ 灯箱尺寸测距
          scene : 路面平面尺度对齐 → 点云入 ego 系 → 相对位姿积分（物理合理性过滤）
          assoc : 自车道判定 → 灯组打分（语义 / 方位 / 距离 / 高度）
                │
        WorldState（带 frame_ts / state_ts）
          ├── WebSocket：JSON 状态 + 二进制点云
          └── WebRTC：H.264 视频轨（aiortc）
```

推理频率与输出频率解耦：三个模型各跑各的频率，融合层维护连续状态，前端按自己的刷新率渲染。

## 坐标系与标定

- 唯一主坐标系为 ego 系：x 前、y 左、z 上，原点为相机在地面的投影；世界系为起始时刻的 ego 系，只保留最近 5 s。
- 所有时序滤波在 BEV 米制空间进行，图像坐标只用于输入输出。
- 标定无需棋盘格：内参取 MapAnything 逐帧估计的中位数；俯仰/偏航由自车道对的消失点求得；相机高度由 IPM 后自车道宽度约束到 3.75 m。`scripts/check_calib.py` 用整段视频的车道检测回验平行度和车道宽。

## 模型

| 模块 | 模型 | 输入 | 许可 |
|---|---|---|---|
| 车道线 | CLRerNet DLA34（CULane 权重） | 任意画幅先转为 CULane 几何（1640 宽、底部 590 行） | Apache-2.0 |
| 交通灯检测 | RT-DETR r18 / r50（COCO + Objects365，取 traffic light 类） | 640×640，GPU 预处理，无 NMS 需去重 | Apache-2.0 |
| 交通灯颜色 | Baidu Apollo 横排 / 竖排 / 方形识别 CNN（Caffe 权重，PyTorch 重建直接加载） | 96×32 / 32×96 / 64×64 | Apache-2.0 |
| 交通灯（备选） | OpenLenda YOLOX（日本样式，带箭头，多标签） | 1280 | Apache-2.0 |
| 场景 | MapAnything（DINOv2 ViT-g，多视图前馈） | 518 长边，滑窗 2 帧 | CC-BY-NC-4.0（apache 版可商用） |

国内样式交通灯在 S2TLD（上海，1080p，319 个标注）上的实测：

| 方案 | 召回（IoU≥0.3） | 颜色准确率 |
|---|---|---|
| OpenLenda s / x | 0.15 / 0.16 | — |
| RT-DETR r50 + HSV 投票 | 0.81 | 0.68 |
| RT-DETR r50 + Apollo CNN | 0.81 | 0.87 |

## 融合细节

- **车道**：每条线 IPM 到 BEV 后拟合 `y = a·x² + b·x + c`（x ∈ [3, 40] m），以 5 m 处横向偏移做匈牙利匹配（门限 0.9 m），每条 track 对 `[a, b, c]` 做 Kalman，遮挡外推最多 0.5 s，连续 3 帧才发布。高程由深度图修正，30 m 外置 0。
- **交通灯**：一个检测框是一个灯箱，含多个 lamp（圆灯颜色 + 各方向箭头）。每个 lamp 独立做 5 帧多数投票；只允许 `绿→黄→红→绿` 及 off 的转移，非法转移连续 10 帧后强制接受并标记 `forced`。距离用灯箱实际尺寸反算，置信度随框宽衰减。
- **场景**：取当前帧点云中 IPM 落在自车道内、5 到 25 m 的点拟合平面，平面高度与标定相机高度之比为尺度因子（低通 + 离群剔除）。相对位姿超出 45 m/s、6 m/s² 或 60°/s 视为无效，改按滤波后的速度外推。
- **关联**：自车道为夹住 y = 0 的两条线；转向属性按车道位置启发式给出，置信度 0.5。灯组得分 = 箭头语义匹配（+1 / −1，只有圆灯 +0.6）+ 与自车道 30 m 处切线方向的夹角（<8° 满分，25° 归零）+ 距离与高度合理性，外推中的灯减半；最高分低于 1 或与次高差距小于 0.3 时置信度压到 0.4 以下。

## 性能（RTX 5090，三路同时运行）

| 模块 | 单独耗时 | 并行频率 |
|---|---|---|
| CLRerNet DLA34 | 18 ms | ~19 Hz |
| RT-DETR r18 + Apollo 颜色 | 15 ms | ~18 Hz |
| MapAnything 2 视图 | 73 ms | 限 4 Hz |
| Fusion | < 3 ms | 30 Hz |

三路并行时 GPU 饱和，`scene_period` 用于限制场景模型频率以让出算力。回放模式（离线 dump 后重放）稳定 30 / 30 / 10 Hz。

## 输出契约

WebSocket 文本帧为 WorldState JSON：

```
frame_ts_ns, state_ts_ns, e2e_latency_ms, calib
ego            { T_world_ego 4×4, speed_mps, yaw_rate_rps, odom_confidence }
lanes[]        { id, role(ego_left|ego_right|left|right), confidence, coasting, poly_bev[a,b,c], points_3d, points_2d }
ego_lane_id, ego_lane_turn { allowed[], confidence }
traffic_lights[] { track_id, bbox_2d, orientation, confidence, coasting, distance_m, distance_confidence,
                   position_ego, lamps[{ shape, color, confidence, stable_frames, forced }], transition_ts }
association    { ego_lane_light_track_id, confidence, score_breakdown }
scene          { n_points, voxel_size_m, scale_factor, scale_confidence }
metrics        { decode_hz, lane_hz, light_hz, scene_hz, fusion_hz, *_latency_ms, *_model_ms }
```

二进制帧为点云：`"PCL1" | uint32 N | float32[N,3] xyz（ego 系，米）| uint8[N,3] rgb`，仅场景更新时发送。

HTTP：`POST /api/upload`、`POST /api/start`、`POST /api/stop`、`GET /api/status`、`GET /api/files`、`POST /api/webrtc/offer`。

## 目录

```
backend/perception/
  calib.py, autocalib.py       标定、IPM、投影
  frames.py, workers.py        解码槽位、模型 worker / 回放 worker
  models/                      CLRerNet / RT-DETR + Apollo / OpenLenda / MapAnything 封装
  fusion/                      lanes, lights, scene, assoc
  pipeline.py, service.py, server.py, webrtc.py, runner.py
backend/scripts/               calibrate, check_calib, dump_models, run_pipeline, eval_s2tld, ws_probe ...
backend/tests/                 各融合模块的合成数据测试
backend/patches/               第三方仓库补丁
frontend/src/                  ws.ts / webrtc.ts / api.ts, components/{VideoPanel, Scene3D, Controls}
```

## 运行

后端（Python 3.11，CUDA 12.8，PyTorch ≥ 2.7；依赖见 `backend/requirements.txt`，第三方仓库放在 `backend/third_party/` 并应用 `backend/patches/`，权重放在 `backend/weights/`）：

```bash
uvicorn perception.service:app --host 0.0.0.0 --port 8000
```

前端：

```bash
cd frontend && npm install --legacy-peer-deps && npm run dev
```

命令行方式：

```bash
python -m scripts.calibrate    --video x.mp4 --out x.calib.yaml --band-bottom 0.85
python -m scripts.run_pipeline --video x.mp4 --calib x.calib.yaml --debug-out debug.mp4
python -m scripts.dump_models  --video x.mp4 --out x.replay.pkl
python -m scripts.run_pipeline --video x.mp4 --calib x.calib.yaml --replay x.replay.pkl
python -m scripts.eval_s2tld   --n 150 --detectors rtdetr_r50 openlenda_s
```

WebRTC 视频轨要求浏览器到后端的 UDP 可达；不可达时前端可切换为 WebSocket 内嵌 JPEG。
