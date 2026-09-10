# traffic — 车道线 + 交通灯 + 3D 场景 实时感知后端

方案见 `E:\Workspace\traffic\技术方案-v2.md`。本目录是 WSL2 里的实现，环境为 micromamba env `traffic`（Python 3.11, CUDA 12.8, torch 2.11+cu128, RTX 5090）。

## 目录

```
perception/
  calib.py                 标定 + IPM + 投影（ego 系：x 前 y 左 z 上）
  frames.py                解码线程 + 每消费者一个 LatestFrame 槽位
  workers.py               模型 worker 线程 / 回放 worker
  pipeline.py              Fusion 线程（60 Hz tick）+ WorldState + 调试叠加
  server.py                WebSocket 广播（JSON 状态 + 二进制点云）
  models/
    lights_openlenda.py    OpenLenda 封装：检测框 -> 灯箱 + lamp 列表
    lanes_clrernet.py      CLRerNet 封装：帧 -> 2D 车道点
    scene_mapanything.py   MapAnything 封装：N 帧 -> 深度/内参/相对位姿
  fusion/
    lanes.py               IPM -> BEV 二次拟合 -> 匈牙利关联 -> Kalman
    lights.py              IoU 跟踪 -> 每灯滑窗投票 + 状态机(带兜底) -> 灯箱尺寸测距
    scene.py               尺度对齐 -> 点云入 ego 系 -> 相对里程计
    assoc.py               自车道 <-> 灯组 关联打分
scripts/
  calibrate.py             从视频估 calib.yaml
  dump_models.py           三模型离线跑完整段视频 -> replay.pkl
  run_pipeline.py          live / replay 运行流水线
  ws_probe.py              连 WebSocket 检查契约
  make_synth_video.py      从单张图造合成视频（仅冒烟测试）
tests/                     各模块合成数据测试（python -m tests.test_xxx）
third_party/               openlenda, CLRerNet, map-anything, UnLanedet（含本地补丁）
weights/                   openlenda_s.pth, clrernet_culane_dla34_ema.pth
```

## 交通灯模型选择

| 模型 | 适用 | S2TLD（上海）实测 |
|---|---|---|
| `--light-model rtdetr`（默认）RT-DETR + Apollo 颜色 CNN | 国内 / 通用 | 召回 0.81（conf 0.3）、颜色准确率 0.87~0.93 |
| `--light-model openlenda` | 日本样式，带箭头 | 召回 0.15，不可用 |

Apollo 颜色模型（Apache-2.0）来自 `https://apollo-pkg-beta.cdn.bcebos.com/perception_model/{horizontal,vertical,quadrate}_caffe.zip`，在 `perception/models/apollo_tl_classifier.py` 里用 PyTorch 重建并直接读 caffemodel。RT-DETR 只出颜色，不出箭头。

## 素材

- `data/city_day_84s.mp4`：Pexels 31901298（美国，白天，84 s，已转 1080p30），`data/city_day_84s.calib.yaml` 已用 `scripts/check_calib.py` 验证（自车道宽 3.73 m，平行度 1°）。
- `data/s2tld_seq*_22fps.mp4`：S2TLD 上海序列，每段约 2 s，只用于测试。
- 国内长片段尚缺。

## 服务与前端

- `perception/service.py`：FastAPI，HTTP 控制接口 + WebSocket（同端口 8000）。`perception/runner.py` 管理模型缓存、自动标定（`perception/autocalib.py`）和流水线启停。
- 启动：`mm uvicorn perception.service:app --host 0.0.0.0 --port 8000`。
- 前端在 Windows 侧 `E:\Workspace\traffic\frontend`（React + Vite + react-three-fiber），`npm run dev` 后打开 http://localhost:5173。

## 常用命令

```bash
export MAMBA_ROOT_PREFIX=~/micromamba
alias mm="~/.local/bin/micromamba run -n traffic"
cd ~/traffic

# 1. 标定（直路片段效果最好；--band-bottom 排除引擎盖）
mm python -m scripts.calibrate --video data/x.mp4 --out data/x.calib.yaml --band-bottom 0.85

# 2. live 运行（WebSocket ws://localhost:8765，调试视频可选）
mm python -m scripts.run_pipeline --video data/x.mp4 --calib data/x.calib.yaml --band-bottom 0.85 --debug-out logs/x_debug.mp4

# 3. 离线 dump + 回放（开发融合层用，不占 GPU）+ 标定检查
mm python -m scripts.dump_models --video data/x.mp4 --out data/x.replay.pkl --band-bottom 0.85
mm python -m scripts.check_calib --calib data/x.calib.yaml --replay data/x.replay.pkl
mm python -m scripts.run_pipeline --video data/x.mp4 --calib data/x.calib.yaml --replay data/x.replay.pkl

# 3b. 在 S2TLD 上评测交通灯检测器
mm python -m scripts.eval_s2tld --n 150 --detectors rtdetr_r50 openlenda_s

# 4. 检查输出契约
mm python -m scripts.ws_probe ws://localhost:8765 5
```

## 输出契约

- 文本帧：JSON，字段见 `perception/pipeline.py::Fusion.publish`（lanes / traffic_lights(含 lamps 列表) / association / ego / scene / metrics / frame.jpeg_b64）。
- 二进制帧：`b"PCL1" | uint32 N | float32[N,3] xyz(ego 系, 米) | uint8[N,3] rgb`，仅场景更新时发送。
- 前端按 `state_ts_ns` 插值渲染；`confidence` / `coasting` / `forced` 必须可视化。

## 第三方补丁

- openlenda/tools/demo.py：`torch.load(weights_only=False)`、无界面 waitKey。
- CLRerNet/libs/models/layers/nms/src/nms_kernel.cu：`boxes.type()` -> `boxes.scalar_type()`（新 torch）。
- site-packages/mmdet/__init__.py：mmcv 最大版本断言放宽到 2.3.0（mmcv 2.2.0 为 sm_120 源码编译）。
- site-packages/mmengine/optim/optimizer/builder.py：Adafactor 注册加 `force=True`（与 transformers 共存时导入即报错）。
- torchaudio 已卸载（transformers 顺带装的 cu130 版本会破坏 CUDA 加载）。
