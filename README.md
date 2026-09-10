# traffic — 车道线 + 交通灯 + 3D 场景 实时感知演示

单卡 RTX 5090（WSL2）上的零训练实时感知后端 + React/Three.js 前端。方案见 [技术方案-v2.md](技术方案-v2.md)。

```
backend/    Python 后端（感知、融合、服务），运行在 WSL2；说明见 backend/README.md
frontend/   React + Vite + react-three-fiber 前端，运行在 Windows
scripts/    sync-backend.ps1：WSL 工作副本 <-> 仓库 backend/ 同步
```

后端的运行副本在 WSL `~/traffic`（含 `third_party/`、`weights/`、`data/`，这些不入库；第三方补丁见 `backend/patches/`）。改了 WSL 里的代码后用 `scripts\sync-backend.ps1` 同步回仓库；反向用 `-Reverse`。

## 启动

1. 后端（WSL 内，HTTP + WebSocket + WebRTC 信令同一端口 8000）：

```bash
wsl.exe -e bash -lc 'cd ~/traffic && export MAMBA_ROOT_PREFIX=~/micromamba && ~/.local/bin/micromamba run -n traffic uvicorn perception.service:app --host 0.0.0.0 --port 8000'
```

2. 前端（Windows）：

```bash
cd frontend && npm install --legacy-peer-deps && npm run dev
```

浏览器打开 http://localhost:5173 。WSL 使用 mirrored 网络模式，Windows 侧 `localhost:8000` 直接可达。

## 使用

- 视频源：上传本地视频（保存到 WSL `~/traffic/data/uploads/`）、从下拉框选已有文件、或填 `rtsp://` / `http://` 流地址。
- 启动后自动加载模型（首次约 20 秒）；文件源没有标定时自动标定（约 1 分钟，结果存为 `<视频名>.calib.yaml`，下次直接复用）；流源用前几秒帧做标定。
- 视频通道默认 WebRTC（aiortc，H.264）。Windows 到 WSL 的入站 UDP 需要放开 Hyper-V 防火墙（管理员 PowerShell）：
  `Set-NetFirewallHyperVVMSetting -Name '<Get-NetFirewallHyperVVMSetting 里的 GUID>' -DefaultInboundAction Allow -LoopbackEnabled True`
  没放开时勾选"JPEG 回退"，画面改走 WebSocket。
- 左侧视频叠加：车道（绿 = 自车道，橙 = 其他，灰 = 外推中）、灯框（颜色 = 灯态，虚线 = 外推，白框 = 关联到自车道的灯）、HUD 指标。
- 右侧 3D：ego 系（车在原点朝 -z），点云、车道线、灯球（颜色 = 灯态，白环 = 关联灯）、鼠标拖动旋转、滚轮缩放。

## 接口

- `POST /api/upload`（multipart）→ `{path}`
- `POST /api/start` `{source, calib?, band_bottom, light_model, rtdetr_size, scene_period, loop, send_frame}`
- `POST /api/stop`，`GET /api/status`，`GET /api/files`
- `POST /api/webrtc/offer` `{sdp, type, codec}` → answer
- `WS /ws`：文本帧 = WorldState JSON，二进制帧 = `PCL1` 点云（ego 系）。

## 模型与许可

| 模块 | 模型 | 许可 |
|---|---|---|
| 车道线 | CLRerNet DLA34（CULane 权重） | Apache-2.0 |
| 交通灯检测 | RT-DETR r18/r50（COCO + Objects365） | Apache-2.0 |
| 交通灯颜色 | Baidu Apollo 横排/竖排/方形识别 CNN（Caffe 权重，PyTorch 重建） | Apache-2.0 |
| 交通灯（备选） | OpenLenda（日本样式，带箭头） | Apache-2.0 |
| 场景 3D | MapAnything | CC-BY-NC-4.0（apache 版可商用） |

国内灯在 S2TLD 上实测：RT-DETR 召回 0.81，Apollo 颜色准确率 0.87；OpenLenda 召回 0.15 不可用。
