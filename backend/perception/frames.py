"""Video decoding into per-consumer latest-frame slots."""
import threading, time
import cv2
import numpy as np


class LatestFrame:
    """Single-slot mailbox: writer overwrites, reader takes the newest unseen frame."""
    def __init__(self):
        self._cv = threading.Condition()
        self._seq = -1
        self._item = None
        self._taken = -1

    def put(self, seq, ts_ns, frame):
        with self._cv:
            self._seq, self._item = seq, (seq, ts_ns, frame)
            self._cv.notify_all()

    def take(self, timeout=1.0):
        """Block until a frame newer than the last taken one exists. Returns (seq, ts_ns, frame) or None."""
        with self._cv:
            if not self._cv.wait_for(lambda: self._seq > self._taken, timeout):
                return None
            self._taken = self._seq
            return self._item

    def peek(self):
        with self._cv:
            return self._item


class Decoder(threading.Thread):
    """Decodes a file or camera with OpenCV and broadcasts frames to all registered slots.
    For a file, playback is paced to the source fps (realtime=True) so the pipeline behaves like a camera."""
    def __init__(self, source, slots, realtime=True, loop=True):
        super().__init__(daemon=True, name="decoder")
        self.source, self.slots, self.realtime, self.loop = source, slots, realtime, loop
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {source}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.seq = 0
        self.decode_hz = 0.0
        self._stop_evt = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def run(self):
        period = 1.0 / self.fps
        t_next = time.perf_counter()
        t_win, n_win = time.perf_counter(), 0
        while not self._stop_evt.is_set():
            ok, frame = self.cap.read()
            if not ok:
                if self.loop and isinstance(self.source, str):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            ts_ns = time.monotonic_ns()
            for s in self.slots:
                s.put(self.seq, ts_ns, frame)
            self.seq += 1
            n_win += 1
            now = time.perf_counter()
            if now - t_win >= 1.0:
                self.decode_hz, t_win, n_win = n_win / (now - t_win), now, 0
            if self.realtime:
                t_next += period
                delay = t_next - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    t_next = time.perf_counter()
