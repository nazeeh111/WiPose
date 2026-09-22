"""EMA 평활(§11, α=0.5) — absent 동안 리셋(재등장 시 첫 관측으로 재초기화)."""
import numpy as np


class EmaSmoother:
    def __init__(self, *, alpha=0.5):
        self.a = float(alpha)
        self._y = None

    def update(self, xy, *, present):
        if not present:
            self._y = None
            return None
        x = np.asarray(xy, np.float32)
        self._y = x.copy() if self._y is None else self.a * x + (1 - self.a) * self._y
        return self._y.copy()
