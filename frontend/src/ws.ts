import type { Cloud, WorldState } from "./types";

type Listener = (state: WorldState | null, cloud: Cloud | null) => void;

/** WebSocket client for the perception service: text frames = WorldState JSON, binary = PCL1 point cloud. */
export class PerceptionSocket {
  private ws: WebSocket | null = null;
  private url = "";
  private listeners = new Set<Listener>();
  private closed = false;
  state: WorldState | null = null;
  cloud: Cloud | null = null;
  connected = false;
  msgRate = 0;
  private msgCount = 0;
  private rateTimer: number | null = null;

  connect(url: string) {
    this.close();
    this.closed = false;
    this.url = url;
    this.open();
    this.rateTimer = window.setInterval(() => {
      this.msgRate = this.msgCount;
      this.msgCount = 0;
    }, 1000);
  }

  private open() {
    if (this.closed) return;
    const ws = new WebSocket(this.url);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      this.connected = true;
      this.emit();
    };
    ws.onclose = () => {
      this.connected = false;
      this.emit();
      if (!this.closed) setTimeout(() => this.open(), 1000);
    };
    ws.onmessage = (ev) => {
      this.msgCount++;
      if (typeof ev.data === "string") {
        this.state = JSON.parse(ev.data) as WorldState;
      } else {
        this.cloud = parseCloud(ev.data as ArrayBuffer);
      }
      this.emit();
    };
    this.ws = ws;
  }

  close() {
    this.closed = true;
    if (this.rateTimer) window.clearInterval(this.rateTimer);
    this.ws?.close();
    this.ws = null;
  }

  subscribe(fn: Listener) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private emit() {
    for (const fn of this.listeners) fn(this.state, this.cloud);
  }
}

export function parseCloud(buf: ArrayBuffer): Cloud | null {
  const magic = new TextDecoder().decode(new Uint8Array(buf, 0, 4));
  if (magic !== "PCL1") return null;
  const n = new DataView(buf).getUint32(4, true);
  const xyz = new Float32Array(buf.slice(8, 8 + n * 12));
  const rgb = new Uint8Array(buf, 8 + n * 12, n * 3);
  return { ts: performance.now(), xyz, rgb: new Uint8Array(rgb), n };
}
