import { useEffect, useRef, useState } from "react";
import "./App.css";
import { DEFAULT_BASE, wsUrl } from "./api";
import { PerceptionSocket } from "./ws";
import type { Cloud, WorldState } from "./types";
import Controls from "./components/Controls";
import VideoPanel from "./components/VideoPanel";
import Scene3D from "./components/Scene3D";

export default function App() {
  const [base, setBase] = useState(localStorage.getItem("base") || DEFAULT_BASE);
  const [state, setState] = useState<WorldState | null>(null);
  const [cloud, setCloud] = useState<Cloud | null>(null);
  const [connected, setConnected] = useState(false);
  const [msgRate, setMsgRate] = useState(0);
  const sock = useRef<PerceptionSocket>(new PerceptionSocket());
  const lastMsg = useRef<number>(0);

  useEffect(() => {
    localStorage.setItem("base", base);
    const s = sock.current;
    s.connect(wsUrl(base));
    // throttle React updates to the display refresh rate
    let pending: { st: WorldState | null; cl: Cloud | null } | null = null;
    let raf = 0;
    const flush = () => {
      raf = 0;
      if (!pending) return;
      setState(pending.st);
      setCloud(pending.cl);
      setConnected(s.connected);
      setMsgRate(s.msgRate);
      pending = null;
    };
    const unsub = s.subscribe((st, cl) => {
      if (st) lastMsg.current = Date.now();
      pending = { st, cl };
      if (!raf) raf = requestAnimationFrame(flush);
    });
    return () => {
      unsub();
      if (raf) cancelAnimationFrame(raf);
      s.close();
    };
  }, [base]);

  return (
    <div className="app">
      <Controls base={base} setBase={setBase} connected={connected} msgRate={msgRate} />
      <div className="main">
        <VideoPanel state={state} base={base} running={connected && state !== null && Date.now() - lastMsg.current < 3000} />
        <Scene3D state={state} cloud={cloud} />
      </div>
      <StatusBar state={state} cloud={cloud} />
    </div>
  );
}

function StatusBar({ state, cloud }: { state: WorldState | null; cloud: Cloud | null }) {
  if (!state) return <div className="statusbar muted">未连接到感知服务</div>;
  const a = state.association;
  const light = state.traffic_lights.find((l) => l.track_id === a.ego_lane_light_track_id);
  const lampTxt = light ? light.lamps.map((l) => `${l.shape.replace("arrow_", "→")}:${l.color}${l.forced ? "*" : ""}`).join(" ") : "-";
  return (
    <div className="statusbar">
      <span>自车道 <b>{state.ego_lane_id ?? "-"}</b></span>
      <span>允许 <b>{state.ego_lane_turn?.allowed.join("/") ?? "-"}</b></span>
      <span>关联灯 <b>{a.ego_lane_light_track_id ?? "-"}</b> {a.confidence < 0.5 && a.ego_lane_light_track_id != null ? <em className="warn">不确定</em> : null}</span>
      <span>灯态 <b>{lampTxt}</b></span>
      <span>车速 <b>{state.ego.speed_mps.toFixed(1)}</b> m/s</span>
      <span>点云 <b>{cloud?.n ?? 0}</b> 尺度 {state.scene.scale_factor ?? "-"} ({state.scene.scale_confidence})</span>
      <span>端到端 <b>{state.e2e_latency_ms ?? "-"}</b> ms</span>
    </div>
  );
}
