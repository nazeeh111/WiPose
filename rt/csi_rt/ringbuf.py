"""인과 그리드/윈도 절단 — 배치 align 프리미티브 재사용.

버퍼에 t≤T 패킷만 존재 → grid_block의 'tb > s.t[-1] → 마스크'가 곧 인과 제약.
따라서 rt valid ⊆ 배치 valid (꼬리 결손 보수 처리)가 구조적으로 성립한다.
boot_id 변화 = 리부트 → 해당 링크 버퍼 클리어(에포크 경계 너머 보간 금지)."""
from collections import deque
from dataclasses import dataclass

import numpy as np

from csi_pipe.align import LinkStream, cut_windows, fill_gaps, grid_block
from csi_pipe.samples import STEP_NS

LINKS = [(r, t) for r in range(3) for t in range(3)]
WIN_NS = 5 * STEP_NS


@dataclass(frozen=True, eq=False)
class Pkt:
    rx: int
    tx: int
    boot_id: int
    t_ns: int
    seq: int
    amp: np.ndarray            # (56,) f32 — amplitude(iq) 산출물


@dataclass(frozen=True, eq=False)
class CutResult:
    X: np.ndarray              # (280,3,3) f16 — 배치 cut_windows와 동일 규약
    valid: bool
    bad: np.ndarray            # (3,3) int — 링크별 마스크 슬롯 수


class RingBuf:
    def __init__(self, *, horizon_ns=1_200_000_000):
        self._links = {lk: deque() for lk in LINKS}
        self._boot = {}
        self._h = horizon_ns

    def add(self, p: Pkt):
        d = self._links[(p.rx, p.tx)]
        prev = self._boot.get((p.rx, p.tx))
        if prev is not None and prev != p.boot_id:
            d.clear()
        self._boot[(p.rx, p.tx)] = p.boot_id
        if d and (p.t_ns <= d[-1].t_ns or p.seq <= d[-1].seq):
            return  # 비단조(같은 boot의 seq 역행) = 글리치/재전송 — 폐기 (진짜 TX 재시작은 boot_id 변화로 도착 → clear)
        d.append(p)
        while d and d[0].t_ns < p.t_ns - self._h:
            d.popleft()

    def cut(self, end_ns: int) -> CutResult:
        tb = np.arange(end_ns - WIN_NS, end_ns, STEP_NS, dtype=np.int64)
        amp_blk = np.zeros((5, 56, 3, 3), np.float32)
        mask_blk = np.zeros((5, 3, 3), bool)
        for rx, tx in LINKS:
            d = self._links[(rx, tx)]
            if len(d) < 2:                           # grid_block 전제(n≥2) 미달
                mask_blk[:, rx, tx] = True
                continue
            t = np.fromiter((p.t_ns for p in d), np.int64, len(d))
            seq = np.fromiter((p.seq for p in d), np.int64, len(d))
            amp = np.stack([p.amp for p in d])
            t2, a2, m2, br = fill_gaps(t, seq, amp)
            a, m = grid_block(LinkStream(t=t2, amp=a2, interp=m2, breaks=br), tb)
            # 클램프(+1ns) 쌍이 외삽 분모가 되면 진폭 폭주(f16 overflow) — 물리 상한
            # 클립(int8 iq 최대 |127+127j|≈179.6 < 256, 정상·배치 경로는 무영향)
            amp_blk[:, :, rx, tx], mask_blk[:, rx, tx] = np.clip(a, 0.0, 256.0), m
        X, valid = cut_windows(amp_blk, mask_blk, np.array([0]))
        return CutResult(X=X[0], valid=bool(valid[0]),
                         bad=mask_blk.sum(axis=0).astype(int))
