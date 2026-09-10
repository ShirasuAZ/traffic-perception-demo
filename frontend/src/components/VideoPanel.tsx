import { useEffect, useRef, useState } from "react";
import type { WorldState } from "../types";
import { connectVideo } from "../webrtc";

const LAMP_COLORS: Record<string, string> = { red: "#ff3b30", yellow: "#ffcc00", green: "#34c759", empty: "#8e8e93" };

function lampLabel(l: { shape: string; color: string; forced: boolean }) {
  const s = l.shape === "circle" ? "●" : l.shape === "arrow_left" ? "←" : l.shape === "arrow_right" ? "→" : "↑";
  return `${s}${l.forced ? "*" : ""}`;
}

/** Live video over WebRTC with the perception overlays drawn on a canvas on top.
 *  Falls back to the JPEG embedded in the state when the backend sends one. */
export default function VideoPanel({ state, base, running }: { state: WorldState | null; base: string; running: boolean }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const [rtc, setRtc] = useState<"idle" | "connecting" | "connected" | "failed">("idle");
  const [videoSize, setVideoSize] = useState<[number, number]>([0, 0]);

  // (re)connect the WebRTC video whenever the pipeline starts running
  useEffect(() => {
    if (!running || !videoRef.current) return;
    let cancelled = false;
    setRtc("connecting");
    connectVideo(base, videoRef.current)
      .then((pc) => {
        if (cancelled) {
          pc.close();
          return;
        }
        pcRef.current = pc;
        pc.onconnectionstatechange = () => {
          if (pc.connectionState === "connected") setRtc("connected");
          if (pc.connectionState === "failed" || pc.connectionState === "disconnected") setRtc("failed");
        };
      })
      .catch(() => setRtc("failed"));
    return () => {
      cancelled = true;
      pcRef.current?.close();
      pcRef.current = null;
      setRtc("idle");
    };
  }, [base, running]);

  // keep the overlay canvas the same size as the displayed video
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const ro = new ResizeObserver(() => setVideoSize([v.clientWidth, v.clientHeight]));
    ro.observe(v);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const v = videoRef.current;
    if (!canvas || !v || !state) return;
    const vw = v.videoWidth || state.calib.width;
    const vh = v.videoHeight || state.calib.height;
    // displayed video rect inside the element (object-fit: contain)
    const scale = Math.min(videoSize[0] / vw, videoSize[1] / vh);
    const dw = vw * scale, dh = vh * scale;
    const ox = (videoSize[0] - dw) / 2, oy = (videoSize[1] - dh) / 2;
    canvas.width = videoSize[0];
    canvas.height = videoSize[1];
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (state.frame && rtc !== "connected") {
      // JPEG fallback
      const img = new Image();
      img.onload = () => {
        ctx.drawImage(img, ox, oy, dw, dh);
        drawOverlay(ctx, state, dw / state.calib.width, ox, oy);
      };
      img.src = "data:image/jpeg;base64," + state.frame.jpeg_b64;
      return;
    }
    drawOverlay(ctx, state, dw / state.calib.width, ox, oy);
  }, [state, videoSize, rtc]);

  return (
    <div className="panel video-panel">
      <video ref={videoRef} autoPlay muted playsInline />
      <canvas ref={canvasRef} className="overlay" />
      {rtc !== "connected" && <div className="badge">{running ? `video: ${rtc}` : "未运行"}</div>}
    </div>
  );
}

function drawOverlay(ctx: CanvasRenderingContext2D, state: WorldState, s: number, ox: number, oy: number) {
  const X = (x: number) => ox + x * s, Y = (y: number) => oy + y * s;
  ctx.lineWidth = 3;
  for (const lane of state.lanes) {
    const pts = lane.points_2d.filter((p) => Number.isFinite(p[0]) && Number.isFinite(p[1]));
    if (pts.length < 2) continue;
    ctx.strokeStyle = lane.coasting ? "#9a9a9a" : lane.role.startsWith("ego") ? "#34c759" : "#ffb340";
    ctx.globalAlpha = 0.4 + 0.6 * Math.min(1, lane.confidence);
    ctx.beginPath();
    ctx.moveTo(X(pts[0][0]), Y(pts[0][1]));
    for (const p of pts) ctx.lineTo(X(p[0]), Y(p[1]));
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillStyle = ctx.strokeStyle;
    ctx.font = "12px system-ui";
    ctx.fillText(`${lane.id}:${lane.role}`, X(pts[0][0]) + 4, Y(pts[0][1]) - 4);
  }
  const assoc = state.association.ego_lane_light_track_id;
  for (const lt of state.traffic_lights) {
    const x1 = X(lt.bbox_2d[0]), y1 = Y(lt.bbox_2d[1]), x2 = X(lt.bbox_2d[2]), y2 = Y(lt.bbox_2d[3]);
    const main = lt.lamps.find((l) => l.shape === "circle") ?? lt.lamps[0];
    const col = lt.coasting ? "#9a9a9a" : main ? LAMP_COLORS[main.color] ?? "#fff" : "#fff";
    ctx.strokeStyle = col;
    ctx.lineWidth = lt.track_id === assoc ? 4 : 2;
    ctx.setLineDash(lt.coasting ? [4, 4] : []);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.setLineDash([]);
    if (lt.track_id === assoc) {
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1;
      ctx.strokeRect(x1 - 4, y1 - 4, x2 - x1 + 8, y2 - y1 + 8);
    }
    const label = `#${lt.track_id} ${lt.lamps.map(lampLabel).join("")} ${lt.distance_m}m`;
    ctx.font = "12px system-ui";
    const w = ctx.measureText(label).width + 6;
    ctx.fillStyle = "rgba(0,0,0,0.6)";
    ctx.fillRect(x1, Math.max(0, y1 - 16), w, 14);
    ctx.fillStyle = col;
    ctx.fillText(label, x1 + 3, Math.max(11, y1 - 5));
  }
  const m = state.metrics;
  const lines = [
    `lane ${m.lane_hz}Hz ${m.lane_latency_ms}ms | light ${m.light_hz}Hz ${m.light_latency_ms}ms | scene ${m.scene_hz}Hz ${m.scene_latency_ms}ms`,
    `decode ${m.decode_hz}Hz fusion ${m.fusion_hz}Hz e2e ${state.e2e_latency_ms}ms speed ${state.ego.speed_mps} m/s`,
    `ego lane ${state.ego_lane_id ?? "-"} turn ${state.ego_lane_turn?.allowed.join("/") ?? "-"} → light ${assoc ?? "-"} (${state.association.confidence})`,
  ];
  ctx.font = "13px ui-monospace, monospace";
  ctx.fillStyle = "rgba(0,0,0,0.55)";
  ctx.fillRect(ox + 6, oy + 6, 640, 20 * lines.length + 6);
  ctx.fillStyle = "#fff";
  lines.forEach((t, i) => ctx.fillText(t, ox + 12, oy + 22 + 20 * i));
}
