"""리플레이 영상 — /video/t_ns 최근접 프레임 (cam−csi 보정 반영).

순차 접근 전제(t 단조 증가) — VideoCapture를 앞으로만 감는다."""
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


class ReplayVideo:
    def __init__(self, h5_path, mp4_path, pairing_json):
        with h5py.File(h5_path, "r") as h:
            if "video" not in h:
                raise SystemExit(f"/video 없음: {h5_path} — --video는 카메라 동시 "
                                 "기록 세션에서만 가능")
            self._t = h["video/t_ns"][...].astype(np.int64)
            self._idx = h["video/frame_idx"][...].astype(np.int64)
        shift = 0.0
        if pairing_json:
            d = json.loads(Path(pairing_json).read_text(encoding="utf-8"))
            shift = (d["cam_correction_ms"] - d["csi_correction_ms"]) * 1e6
        self._shift_ns = int(shift)                  # t′ = t + (cam−csi)
        self._cap = cv2.VideoCapture(str(mp4_path))
        if not self._cap.isOpened():
            raise SystemExit(f"mp4 열기 실패: {mp4_path}")
        self._pos = -1
        self._frame = None

    def frame_for(self, t_ns):
        want = int(np.clip(np.searchsorted(self._t, t_ns + self._shift_ns),
                           0, len(self._t) - 1))
        if want > 0 and abs(self._t[want - 1] - (t_ns + self._shift_ns)) \
                <= abs(self._t[want] - (t_ns + self._shift_ns)):
            want -= 1
        target = int(self._idx[want])
        # 시계 역행(target<_pos) = 직전 프레임 유지(데모 단조 시계 계약 — 되감기 없음)
        while self._pos < target:
            ok, f = self._cap.read()
            if not ok:
                # 영상 끝 이후 t = 마지막 프레임 고정(의도)
                break
            self._pos += 1
            self._frame = f
        return self._frame
