"""Model worker threads. Each worker owns one LatestFrame slot, runs its model as fast as it can on the
newest frame, and exposes the newest result with timing metrics. Replay workers return pre-computed results."""
import threading
import time
from collections import deque


class RateMeter:
    def __init__(self, window=2.0):
        self.window = window
        self.stamps = deque()

    def tick(self):
        now = time.perf_counter()
        self.stamps.append(now)
        while self.stamps and now - self.stamps[0] > self.window:
            self.stamps.popleft()

    @property
    def hz(self):
        if len(self.stamps) < 2:
            return 0.0
        return (len(self.stamps) - 1) / max(self.stamps[-1] - self.stamps[0], 1e-6)


class Worker(threading.Thread):
    """Runs fn(frame) (or fn(frames) for windowed workers) on every new frame from `slot`."""
    def __init__(self, name, slot, fn, window_s=0.0, min_period_s=0.0):
        super().__init__(daemon=True, name=name)
        self.slot, self.fn = slot, fn
        self.window_s, self.min_period_s = window_s, min_period_s
        self._lock = threading.Lock()
        self._latest = None          # (seq, ts_ns, result, frame)
        self._stop_evt = threading.Event()
        self.meter = RateMeter()
        self.latency_ms = 0.0
        self.model_ms = 0.0
        self.error = None
        self._history = deque()      # (seq, ts_ns, frame) for windowed workers

    def stop(self):
        self._stop_evt.set()

    def run(self):
        while not self._stop_evt.is_set():
            item = self.slot.take(timeout=0.5)
            if item is None:
                continue
            seq, ts_ns, frame = item
            t0 = time.perf_counter()
            try:
                if self.window_s > 0:
                    self._history.append((seq, ts_ns, frame))
                    while self._history and (ts_ns - self._history[0][1]) > 2e9:
                        self._history.popleft()
                    prev = [h for h in self._history if (ts_ns - h[1]) >= self.window_s * 1e9]
                    frames = [prev[-1][2], frame] if prev else [frame]
                    result = self.fn(frames)
                else:
                    result = self.fn(frame)
            except Exception as e:  # keep the pipeline alive; surface the error in metrics
                self.error = repr(e)
                continue
            self.model_ms = (time.perf_counter() - t0) * 1000
            with self._lock:
                self._latest = (seq, ts_ns, result, frame)
                self.latency_ms = (time.monotonic_ns() - ts_ns) / 1e6
                self.meter.tick()
            if self.min_period_s > 0:
                rest = self.min_period_s - (time.perf_counter() - t0)
                if rest > 0:
                    time.sleep(rest)

    def poll(self, after_seq):
        """Return (seq, ts_ns, result, frame) if newer than after_seq, else None."""
        with self._lock:
            if self._latest is not None and self._latest[0] > after_seq:
                return self._latest
            return None


class ReplayWorker(Worker):
    """Looks up pre-computed results by frame seq instead of running a model."""
    def __init__(self, name, slot, table, sim_latency_s=0.0):
        super().__init__(name, slot, fn=None)
        self.table = table
        self.sim_latency_s = sim_latency_s

    def run(self):
        while not self._stop_evt.is_set():
            item = self.slot.take(timeout=0.5)
            if item is None:
                continue
            seq, ts_ns, frame = item
            if seq not in self.table:
                continue
            if self.sim_latency_s > 0:
                time.sleep(self.sim_latency_s)
            with self._lock:
                self._latest = (seq, ts_ns, self.table[seq], frame)
                self.latency_ms = (time.monotonic_ns() - ts_ns) / 1e6
                self.meter.tick()
