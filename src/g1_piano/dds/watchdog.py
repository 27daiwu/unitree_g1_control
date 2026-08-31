"""Monotonic-clock DDS receive watchdog and sliding-window rate."""

from collections import deque
import time


class DdsWatchdog:
    def __init__(self, window_seconds: float = 1.0,
                 warn_seconds: float = 0.020,
                 timeout_seconds: float = 0.100):
        self.window_seconds = window_seconds
        self.warn_seconds = warn_seconds
        self.timeout_seconds = timeout_seconds
        self._times = deque()
        self.last_received = None

    def received(self, now=None):
        now = time.monotonic() if now is None else now
        self.last_received = now
        self._times.append(now)
        self._trim(now)

    def _trim(self, now):
        cutoff = now - self.window_seconds
        while self._times and self._times[0] < cutoff:
            self._times.popleft()

    def snapshot(self, now=None):
        now = time.monotonic() if now is None else now
        self._trim(now)
        age = None if self.last_received is None else now - self.last_received
        if age is None:
            status = "WAITING"
        elif age > self.timeout_seconds:
            status = "TIMEOUT"
        elif age > self.warn_seconds:
            status = "WARN"
        else:
            status = "OK"
        rate = 0.0
        if len(self._times) >= 2:
            span = self._times[-1] - self._times[0]
            if span > 0:
                rate = (len(self._times) - 1) / span
        return status, age, rate, len(self._times)
