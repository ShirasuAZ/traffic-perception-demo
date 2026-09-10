"""WebRTC video output: the decoded frames (with nothing drawn on them) are pushed to the browser as a
live video track; perception results keep flowing over the WebSocket and are overlaid client-side."""
import asyncio
import fractions
import time
import numpy as np
from av import VideoFrame
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.rtcrtpsender import RTCRtpSender
import aioice.ice as _aioice

# Local demo: the browser runs on the same machine (Windows host, WSL2 mirrored networking), so advertise
# loopback first and keep the LAN addresses as fallback. aioice drops 127.0.0.1 by default.
_orig_get_host_addresses = _aioice.get_host_addresses


def _host_addresses_with_loopback(use_ipv4, use_ipv6):
    addrs = _orig_get_host_addresses(use_ipv4, use_ipv6)
    if use_ipv4:
        addrs = ["127.0.0.1"] + [a for a in addrs if ":" not in a]   # ipv4 only keeps the candidate list short
    return addrs


_aioice.get_host_addresses = _host_addresses_with_loopback

VIDEO_CLOCK = 90000
FPS = 30


class LatestFrameTrack(VideoStreamTrack):
    """Serves whatever frame is newest in `slot_getter()` at up to FPS, without blocking the event loop."""
    kind = "video"

    def __init__(self, slot_getter, max_width=1280):
        super().__init__()
        self.slot_getter = slot_getter
        self.max_width = max_width
        self._last_seq = -1
        self._black = None
        self._t0 = time.monotonic()
        self._n = 0

    async def recv(self):
        # pace at FPS
        self._n += 1
        target = self._t0 + self._n / FPS
        delay = target - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        slot = self.slot_getter()
        item = slot.peek() if slot is not None else None
        if item is not None:
            _, _, img = item
            if img.shape[1] > self.max_width:
                import cv2
                s = self.max_width / img.shape[1]
                img = cv2.resize(img, (self.max_width, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
            frame = VideoFrame.from_ndarray(np.ascontiguousarray(img), format="bgr24")
        else:
            if self._black is None:
                self._black = VideoFrame.from_ndarray(np.zeros((360, 640, 3), np.uint8), format="bgr24")
            frame = self._black
        frame.pts = int((time.monotonic() - self._t0) * VIDEO_CLOCK)
        frame.time_base = fractions.Fraction(1, VIDEO_CLOCK)
        return frame


class WebRTCHub:
    def __init__(self, slot_getter):
        self.slot_getter = slot_getter
        self.pcs = set()

    async def offer(self, sdp, type_, prefer="H264"):
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))   # no STUN: host candidates only
        self.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def _on_state():
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await pc.close()
                self.pcs.discard(pc)

        track = LatestFrameTrack(self.slot_getter)
        sender = pc.addTrack(track)
        # prefer the requested codec if the browser offers it
        caps = RTCRtpSender.getCapabilities("video")
        codecs = [c for c in caps.codecs if c.mimeType.lower() == f"video/{prefer}".lower()]
        if codecs:
            for t in pc.getTransceivers():
                if t.sender is sender:
                    t.setCodecPreferences(codecs + [c for c in caps.codecs if c not in codecs])
        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=type_))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        # aiortc completes ICE gathering inside setLocalDescription
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    async def close_all(self):
        for pc in list(self.pcs):
            await pc.close()
        self.pcs.clear()
