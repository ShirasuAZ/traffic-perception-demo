import { useEffect, useRef, useState } from "react";
import { getStatus, listFiles, startPipeline, stopPipeline, uploadFile } from "../api";
import type { Status } from "../types";

interface Props {
  base: string;
  setBase: (b: string) => void;
  connected: boolean;
  msgRate: number;
}

export default function Controls({ base, setBase, connected, msgRate }: Props) {
  const [status, setStatus] = useState<Status | null>(null);
  const [files, setFiles] = useState<{ path: string; name: string; size_mb: number; calib: boolean }[]>([]);
  const [source, setSource] = useState("");
  const [uploadPct, setUploadPct] = useState<number | null>(null);
  const [rtdetrSize, setRtdetrSize] = useState<"r18" | "r50">("r18");
  const [scenePeriod, setScenePeriod] = useState(0.25);
  const [bandBottom, setBandBottom] = useState(0.85);
  const [showLog, setShowLog] = useState(false);
  const [jpegFallback, setJpegFallback] = useState(localStorage.getItem("jpeg") === "1");
  const fileInput = useRef<HTMLInputElement>(null);

  const refresh = async () => {
    try {
      setStatus(await getStatus(base));
      setFiles(await listFiles(base));
    } catch {
      setStatus(null);
    }
  };
  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 1500);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [base]);

  const onUpload = async (f: File) => {
    setUploadPct(0);
    try {
      const r = await uploadFile(base, f, setUploadPct);
      setSource(r.path);
      await refresh();
    } finally {
      setUploadPct(null);
    }
  };

  const onStart = async () => {
    if (!source) return;
    localStorage.setItem("jpeg", jpegFallback ? "1" : "0");
    await startPipeline(base, { source, rtdetr_size: rtdetrSize, scene_period: scenePeriod, band_bottom: bandBottom, loop: true, send_frame: jpegFallback });
    refresh();
  };

  const st = status?.state ?? "offline";
  return (
    <div className="controls">
      <div className="row">
        <label>服务</label>
        <input value={base} onChange={(e) => setBase(e.target.value)} style={{ width: 200 }} />
        <span className={`dot ${status ? "ok" : "bad"}`} title="HTTP" />
        <span className={`dot ${connected ? "ok" : "bad"}`} title="WebSocket" />
        <span className="muted">{msgRate} msg/s</span>
        <span className={`state state-${st}`}>{st}</span>
        {status?.error && <span className="err" title={status.error}>错误</span>}
      </div>
      <div className="row">
        <label>视频源</label>
        <input value={source} onChange={(e) => setSource(e.target.value)} placeholder="文件路径 / rtsp:// / http:// 流地址" style={{ flex: 1 }} />
        <select value="" onChange={(e) => e.target.value && setSource(e.target.value)}>
          <option value="">已有文件…</option>
          {files.map((f) => (
            <option key={f.path} value={f.path}>
              {f.name} ({f.size_mb} MB){f.calib ? " ✓标定" : ""}
            </option>
          ))}
        </select>
        <input ref={fileInput} type="file" accept="video/*" hidden onChange={(e) => e.target.files?.[0] && onUpload(e.target.files[0])} />
        <button onClick={() => fileInput.current?.click()} disabled={uploadPct !== null}>
          {uploadPct === null ? "上传视频" : `上传中 ${Math.round(uploadPct * 100)}%`}
        </button>
      </div>
      <div className="row">
        <label>参数</label>
        <span className="muted">灯检测</span>
        <select value={rtdetrSize} onChange={(e) => setRtdetrSize(e.target.value as "r18" | "r50")}>
          <option value="r18">RT-DETR r18（快）</option>
          <option value="r50">RT-DETR r50（准）</option>
        </select>
        <span className="muted">场景周期(s)</span>
        <input type="number" step={0.05} min={0} max={1} value={scenePeriod} onChange={(e) => setScenePeriod(+e.target.value)} style={{ width: 60 }} />
        <span className="muted">路面底边</span>
        <input type="number" step={0.05} min={0.5} max={1} value={bandBottom} onChange={(e) => setBandBottom(+e.target.value)} style={{ width: 60 }} />
        <label className="muted" style={{ width: "auto" }} title="WebRTC 不通时（如 WSL 入站 UDP 被防火墙拦截）改用 WebSocket 内嵌 JPEG">
          <input type="checkbox" checked={jpegFallback} onChange={(e) => setJpegFallback(e.target.checked)} /> JPEG 回退
        </label>
        <button className="primary" onClick={onStart} disabled={!source || st === "loading" || st === "calibrating"}>
          启动
        </button>
        <button onClick={() => stopPipeline(base).then(refresh)} disabled={st !== "running"}>
          停止
        </button>
        <button onClick={() => setShowLog(!showLog)}>{showLog ? "隐藏日志" : "日志"}</button>
      </div>
      {status?.calib && (
        <div className="row muted small">
          标定: fx {status.calib.fx.toFixed(0)} pitch {status.calib.pitch_deg.toFixed(2)}° 高 {status.calib.cam_height_m.toFixed(2)} m
          {status.calib_info && (status.calib_info as { fallback?: boolean }).fallback ? "（未找到车道对，使用先验）" : ""}
          {status.video && ` | ${status.video.width}x${status.video.height} @ ${status.video.fps.toFixed(0)} fps`}
        </div>
      )}
      {showLog && (
        <pre className="log">{(status?.log ?? []).join("\n")}</pre>
      )}
    </div>
  );
}
