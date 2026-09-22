"""정렬 1·2단계 — 에포크 분리, ≤2연속 갭 보간(+마스크), 100Hz 그리드, 윈도 절단.

순서 고정: 리샘플 → 정규화. 여기서는 리샘플까지만 — 정규화는 학습 단계.
그리드/샘플의 시각은 전부 클록핏 보정 시각(t_fit) 기준.
"""
from dataclasses import dataclass, field

import numpy as np


def amplitude(iq):
    """iq [n,56,2] int8 → 진폭 [n,56] float32."""
    f = iq.astype(np.float32)
    return np.sqrt(f[..., 0] ** 2 + f[..., 1] ** 2)


_PHASE_K = np.arange(56, dtype=np.float64)
_PHASE_A = np.stack([_PHASE_K, np.ones(56)], axis=1)          # [56,2] = (기울기, 오프셋)
_PHASE_P = np.eye(56) - _PHASE_A @ np.linalg.inv(_PHASE_A.T @ _PHASE_A) @ _PHASE_A.T


def sanitized_phase(iq):
    """iq [n,56,2] → 부반송파 축 unwrap + LS 선형(STO/CFO) 제거 잔차 [n,56] f32 (§6.3).

    I/Q 열 순서가 뒤집혀 있어도 잔차는 전 데이터 일관 변환(반사+상수) 차이뿐 — 학습 중립."""
    f = iq.astype(np.float64)
    phi = np.arctan2(f[..., 1], f[..., 0])
    return (np.unwrap(phi, axis=1) @ _PHASE_P).astype(np.float32)


def split_epochs(seq, boot_id):
    """seq 후퇴(TX 재시작) 또는 boot_id 변화(RX 리부트) 경계로 슬라이스 분리."""
    seq = np.asarray(seq, np.int64)
    boot = np.asarray(boot_id, np.int64)
    cut = np.flatnonzero((np.diff(seq) <= 0) | (np.diff(boot) != 0)) + 1
    bounds = np.concatenate(([0], cut, [len(seq)]))
    return [slice(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def fill_gaps(t_ns, seq, amp, *, max_run=2):
    """단일 에포크 내 seq 결손 ≤max_run 연속을 선형 보간.

    반환: (t2, amp2, interp_mask, breaks) — breaks는 보간하지 않은 (t_left, t_right)
    구간 목록 (그리드 마스크용, §5.2-1 '초과 구간 폐기')."""
    t = np.asarray(t_ns, np.int64)
    seq = np.asarray(seq, np.int64)
    d = np.diff(seq)
    out_t, out_a, out_m, breaks = [], [], [], []
    prev = 0
    for i in np.flatnonzero(d > 1):
        out_t.append(t[prev:i + 1])
        out_a.append(amp[prev:i + 1])
        out_m.append(np.zeros(i + 1 - prev, bool))
        run = int(d[i]) - 1
        if run <= max_run:
            w = (np.arange(1, run + 1, dtype=np.float64) / (run + 1))
            ti = (t[i] + w * (t[i + 1] - t[i])).astype(np.int64)
            ai = amp[i][None, :] * (1 - w[:, None]) + amp[i + 1][None, :] * w[:, None]
            out_t.append(ti)
            out_a.append(ai.astype(amp.dtype))
            out_m.append(np.ones(run, bool))
        else:
            breaks.append((int(t[i]), int(t[i + 1])))
        prev = i + 1
    out_t.append(t[prev:])
    out_a.append(amp[prev:])
    out_m.append(np.zeros(len(t) - prev, bool))
    return (np.concatenate(out_t), np.concatenate(out_a),
            np.concatenate(out_m), breaks)


@dataclass
class LinkStream:
    """갭 채움·에포크 머지가 끝난 링크 스트림 (t 오름차순, breaks 정렬)."""
    t: np.ndarray                  # i64 ns (t_fit)
    amp: np.ndarray                # f32 [n,56]
    interp: np.ndarray             # bool [n]
    breaks: list = field(default_factory=list)   # [(t_left, t_right)]


def grid_bounds(streams, *, step_ns=10_000_000):
    """전 링크 공통 가용 구간 → (g0, g1) — step 정렬, g0=올림, g1=내림."""
    lo = max(int(s.t[0]) for s in streams)
    hi = min(int(s.t[-1]) for s in streams)
    g0 = -(-lo // step_ns) * step_ns
    g1 = (hi // step_ns) * step_ns
    return g0, g1


def grid_block(s: LinkStream, tb):
    """그리드 시각 블록 tb(i64)에서 링크 진폭 선형 보간 + 마스크.

    마스크 True: 스트림 범위 밖 / break 구간 내 / 브래킷 표본이 보간 유래.
    전제: s.t 표본 n≥2. breaks는 서로소(disjoint)·시작 시각 오름차순 —
    searchsorted 판정이 시작이 가장 가까운 break 하나만 검사하므로
    중첩/포개진 break는 조용히 누락된다 (현 호출자 fill_gaps가 보장)."""
    tb = np.asarray(tb, np.int64)
    n = len(s.t)
    idx = np.clip(np.searchsorted(s.t, tb), 1, n - 1)
    tl, tr = s.t[idx - 1], s.t[idx]
    w = ((tb - tl) / np.maximum(tr - tl, 1)).astype(np.float32)[:, None]
    amp = s.amp[idx - 1] * (1 - w) + s.amp[idx] * w
    mask = (tb < s.t[0]) | (tb > s.t[-1]) | s.interp[idx - 1] | s.interp[idx]
    if s.breaks:
        starts = np.asarray([b[0] for b in s.breaks], np.int64)
        ends = np.asarray([b[1] for b in s.breaks], np.int64)
        bi = np.searchsorted(starts, tb, side="right") - 1
        in_break = (bi >= 0) & (tb < ends[np.clip(bi, 0, len(ends) - 1)]) & (tb > starts[np.clip(bi, 0, len(starts) - 1)])
        mask |= in_break
    return amp, mask


WIN = 5  # 5패킷 × 56SC = 280채널


def window_indices(g0, step, G, anchors, *, win=WIN):
    """앵커 t_f → 윈도 시작 그리드 행. [t_f−win·step, t_f) ⊂ 그리드인 것만 오케이 ~ ."""
    a = np.asarray(anchors, np.int64)
    off = a - g0
    q, r = np.divmod(off, step)
    ceil = q + (r > 0)
    start = ceil - win
    ok = (start >= 0) & (ceil <= G)
    return start, ok


def cut_windows(amp, mask, starts, *, win=WIN):
    """그리드 → X[N,280,3,3] f16 + valid[N] (링크별 마스크 ≥2/5 → False)."""
    starts = np.asarray(starts, np.int64)
    idx = starts[:, None] + np.arange(win)[None, :]          # [N,5]
    blk = amp[idx]                                           # [N,5,56,3,3]
    X = blk.reshape(len(starts), win * 56, 3, 3).astype(np.float16)
    bad = mask[idx].sum(axis=1)                              # [N,3,3]
    valid = (bad < 2).all(axis=(1, 2))
    return X, valid
