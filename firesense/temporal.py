"""Stage 8b: temporal confirmation (N-frame voting with hysteresis)."""
from collections import deque

LABELS = ("fire", "smoke")


class TemporalConfirmer:
    def __init__(self, cfg):
        """cfg = config['temporal']"""
        self.window = int(cfg["window"])
        self.min_hits = int(cfg["min_hits"])
        self.clear_hits = int(cfg["clear_hits"])
        self.reset()

    def reset(self):
        self.hist = {k: deque(maxlen=self.window) for k in LABELS}
        self.state = {k: False for k in LABELS}

    def hits(self, label):
        return sum(self.hist[label])

    def update(self, flags):
        """flags: {'fire': bool, 'smoke': bool} for the current frame -> confirmed alarms."""
        for k in LABELS:
            self.hist[k].append(bool(flags.get(k, False)))
            hits = self.hits(k)
            if not self.state[k] and hits >= self.min_hits:
                self.state[k] = True
            elif self.state[k] and hits <= self.clear_hits:
                self.state[k] = False
        return dict(self.state)
