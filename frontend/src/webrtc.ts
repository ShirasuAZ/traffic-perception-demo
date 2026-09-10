/** Receive the backend's live video track over WebRTC (signalling via POST /api/webrtc/offer). */
export async function connectVideo(base: string, video: HTMLVideoElement, codec = "H264"): Promise<RTCPeerConnection> {
  const pc = new RTCPeerConnection({ iceServers: [] });
  pc.addTransceiver("video", { direction: "recvonly" });
  pc.ontrack = (ev) => {
    video.srcObject = ev.streams[0] ?? new MediaStream([ev.track]);
    video.play().catch(() => undefined);
  };
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  // wait for ICE gathering so the offer carries host candidates
  await new Promise<void>((resolve) => {
    if (pc.iceGatheringState === "complete") return resolve();
    const check = () => {
      if (pc.iceGatheringState === "complete") {
        pc.removeEventListener("icegatheringstatechange", check);
        resolve();
      }
    };
    pc.addEventListener("icegatheringstatechange", check);
    setTimeout(resolve, 1500);
  });
  const r = await fetch(`${base}/api/webrtc/offer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sdp: pc.localDescription!.sdp, type: pc.localDescription!.type, codec }),
  });
  if (!r.ok) throw new Error(await r.text());
  const answer = await r.json();
  await pc.setRemoteDescription(answer);
  return pc;
}
