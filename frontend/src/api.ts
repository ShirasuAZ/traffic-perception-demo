import type { Status } from "./types";

export const DEFAULT_BASE = "http://localhost:8000";

export function wsUrl(base: string): string {
  return base.replace(/^http/, "ws") + "/ws";
}

export async function getStatus(base: string): Promise<Status> {
  const r = await fetch(`${base}/api/status`);
  return r.json();
}

export async function listFiles(base: string): Promise<{ path: string; name: string; size_mb: number; calib: boolean }[]> {
  const r = await fetch(`${base}/api/files`);
  return r.json();
}

export async function uploadFile(base: string, file: File, onProgress?: (p: number) => void): Promise<{ path: string }> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${base}/api/upload`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => (xhr.status < 300 ? resolve(JSON.parse(xhr.responseText)) : reject(new Error(xhr.responseText)));
    xhr.onerror = () => reject(new Error("upload failed"));
    const fd = new FormData();
    fd.append("file", file);
    xhr.send(fd);
  });
}

export interface StartOptions {
  source: string;
  band_bottom?: number;
  light_model?: "rtdetr" | "openlenda";
  rtdetr_size?: "r18" | "r50";
  scene_period?: number;
  loop?: boolean;
  send_frame?: boolean;
}

export async function startPipeline(base: string, opts: StartOptions) {
  const r = await fetch(`${base}/api/start`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(opts) });
  return r.json();
}

export async function stopPipeline(base: string) {
  const r = await fetch(`${base}/api/stop`, { method: "POST" });
  return r.json();
}
