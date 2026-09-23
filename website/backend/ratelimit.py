import time
from collections import deque
from threading import Lock

MAX_TRACKED_KEYS = 10000


class SlidingWindow:
    def __init__(self, limit: int, window_seconds: float, clock=time.monotonic):
        self.limit = limit
        self.window = window_seconds
        self.clock = clock
        self._hits: dict[str, deque] = {}
        self._lock = Lock()

    def _prune(self, key: str, now: float) -> deque:
        hits = self._hits.setdefault(key, deque())
        while hits and now - hits[0] >= self.window:
            hits.popleft()
        return hits

    def allow(self, key: str = "") -> bool:
        now = self.clock()
        with self._lock:
            if len(self._hits) > MAX_TRACKED_KEYS:
                for k in list(self._hits):
                    if not self._prune(k, now):
                        del self._hits[k]
            hits = self._prune(key, now)
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            return True
