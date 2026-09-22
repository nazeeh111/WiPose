"""HDF5 세션 스토어 — 스키마(§5.3 + esp_us/noise 확장)의 유일한 쓰기 경로 (스펙 결정 ②).

/links에는 원시값만: 보정·보간·리샘플은 빌드 산출물(/grid, /samples — samples.py)이며
언제든 재실행 가능하다. iq 페어 내 (I,Q) 순서는 §16-6 SC표 확정 시 재검증 —
진폭 계산에는 무영향.
"""
import json
import threading
import time

import h5py
import numpy as np

_LINK_FIELDS = [
    ("t_ns", np.uint64, ()),
    ("esp_us", np.uint64, ()),
    ("iq", np.int8, (56, 2)),
    ("seq", np.uint32, ()),
    ("rssi", np.int8, ()),
    ("noise", np.int8, ()),
    ("boot_id", np.uint8, ()),
]


class SessionWriter:
    def __init__(self, path, *, meta=None, chunk=4096):
        # h5py는 스레드세이프 아님 — 레코더는 paho 콜백 스레드(append)와 메인 스레드(주기
        # flush)가 동시 진입하므로 공개 변이 메서드 전체를 락으로 직렬화.
        # RLock인 이유: close→flush처럼 락 보유 중 내부 재호출(재진입)이 있음.
        self._lock = threading.RLock()
        self._h = h5py.File(path, "w")
        self._meta = self._h.create_group("meta")
        self.set_meta("created_ns", time.time_ns())
        for k, v in (meta or {}).items():
            self.set_meta(k, v)
        self._chunk = chunk
        self._closed = False
        self._buf = {}      # (rx,tx) -> {field: list}
        self._counts = {}   # (rx,tx) -> 기록 완료 수
        self._vid = []      # (t_ns, frame_idx) 쌍
        self._h.create_dataset("video/t_ns", shape=(0,), maxshape=(None,),
                               dtype=np.uint64, chunks=(1024,))
        self._h.create_dataset("video/frame_idx", shape=(0,), maxshape=(None,),
                               dtype=np.uint32, chunks=(1024,))

    def set_meta(self, k, v):
        with self._lock:
            self._meta.attrs[k] = (v if isinstance(v, (int, float, str, np.integer))
                                   else json.dumps(v, ensure_ascii=False))

    def append(self, *, rx_id, tx_idx, t_ns, esp_us, iq, seq, rssi, noise, boot_id):
        with self._lock:
            key = (int(rx_id), int(tx_idx))
            buf = self._buf.get(key)
            if buf is None:
                buf = self._buf[key] = {name: [] for name, _, _ in _LINK_FIELDS}
                self._ensure_link(key)
            iq_arr = np.frombuffer(iq, dtype=np.int8).reshape(56, 2) if isinstance(
                iq, (bytes, bytearray)) else np.asarray(iq, np.int8).reshape(56, 2)
            vals = {"t_ns": t_ns, "esp_us": esp_us, "iq": iq_arr, "seq": seq,
                    "rssi": rssi, "noise": noise, "boot_id": boot_id}
            for name in buf:
                buf[name].append(vals[name])
            if len(buf["t_ns"]) >= self._chunk:
                self._flush_link(key)

    def append_video(self, t_ns, frame_idx):
        with self._lock:
            self._vid.append((int(t_ns), int(frame_idx)))
            if len(self._vid) >= 1024:
                self._flush_video()

    def _ensure_link(self, key):
        g = self._h.require_group(f"links/{key[0]}{key[1]}")
        for name, dt, extra in _LINK_FIELDS:
            g.create_dataset(name, shape=(0, *extra), maxshape=(None, *extra),
                             dtype=dt, chunks=(self._chunk, *extra))
        self._counts[key] = 0

    def _flush_link(self, key):
        buf = self._buf[key]
        m = len(buf["t_ns"])
        if not m:
            return
        g = self._h[f"links/{key[0]}{key[1]}"]
        n = self._counts[key]
        for name, dt, _ in _LINK_FIELDS:
            ds = g[name]
            ds.resize(n + m, axis=0)
            ds[n:] = np.asarray(buf[name], dtype=dt)
            buf[name].clear()
        self._counts[key] = n + m

    def _flush_video(self):
        if not self._vid:
            return
        ts = self._h["video/t_ns"]
        fi = self._h["video/frame_idx"]
        n = ts.shape[0]
        m = len(self._vid)
        ts.resize(n + m, axis=0)
        fi.resize(n + m, axis=0)
        ts[n:] = np.asarray([v[0] for v in self._vid], np.uint64)
        fi[n:] = np.asarray([v[1] for v in self._vid], np.uint32)
        self._vid.clear()

    def flush(self):
        with self._lock:
            for key in list(self._buf):
                self._flush_link(key)
            self._flush_video()
            self._h.flush()

    def close(self):
        with self._lock:
            if self._closed:    # 멱등 — 이중 close 무해 (append-after-close는 계속 예외)
                return
            self._closed = True
            self.flush()
            self.set_meta("frames_total", int(sum(self._counts.values())))
            self.set_meta("links", json.dumps(
                {f"{k[0]}{k[1]}": int(v) for k, v in sorted(self._counts.items())}))
            self._h.close()


class SessionReader:
    def __init__(self, path):
        self._h = h5py.File(path, "r")

    @property
    def meta(self):
        return dict(self._h["meta"].attrs)

    def link_keys(self):
        if "links" not in self._h:
            return []
        return [(int(n[0]), int(n[1])) for n in sorted(self._h["links"])]

    def link(self, i, j):
        g = self._h[f"links/{i}{j}"]
        return {name: g[name][...] for name in g}

    @property
    def video_t_ns(self):
        return self._h["video/t_ns"][...]

    @property
    def video_frame_idx(self):
        """구세션(필드 없음)은 None — 호출 측이 identity 폴백."""
        if "video/frame_idx" not in self._h:
            return None
        return self._h["video/frame_idx"][...]

    def close(self):
        self._h.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
