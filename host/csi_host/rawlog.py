"""append-only 원본 로그 (브리지의 raw 로그가 원본, HDF5 재구축 가능).

파일 = 헤더 8B b"CSIRAW01" + 레코드 반복. 레코드 = t_ns u64 LE + len u16 LE + bytes.
불완전 꼬리(장애 중단)는 읽기에서 무시. 주기 flush로 장애 시 부분 보존.
"""
import struct

HEADER = b"CSIRAW01"
_REC = struct.Struct("<QH")


class RawLogWriter:
    def __init__(self, path, flush_every: int = 64):
        self._f = open(path, "ab")
        self._flush_every = flush_every
        self._since_flush = 0
        if self._f.tell() == 0:
            self._f.write(HEADER)
            self._f.flush()

    def append(self, t_ns: int, data: bytes):
        self._f.write(_REC.pack(t_ns, len(data)))
        self._f.write(data)
        self._since_flush += 1
        if self._since_flush >= self._flush_every:
            self._f.flush()
            self._since_flush = 0

    def close(self):
        self._f.flush()
        self._f.close()


def read_rawlog(path):
    """(t_ns, bytes) 제너레이터 — 불완전 꼬리에서 조용히 종료."""
    with open(path, "rb") as f:
        if f.read(len(HEADER)) != HEADER:
            raise ValueError(f"not a rawlog (bad header): {path}")
        while True:
            hdr = f.read(_REC.size)
            if len(hdr) < _REC.size:
                return
            t_ns, n = _REC.unpack(hdr)
            data = f.read(n)
            if len(data) < n:
                return
            yield t_ns, data
