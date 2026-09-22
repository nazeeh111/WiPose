"""정렬 분해 검증 코어.

cv2/serial/MQTT 무의존 — TDD 대상.
4함수 + verdict + GapEvent 데이터클래스.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class GapEvent:
    """클러스터된 갭 1건 — 중앙값 t_ns 단위(float64 ns)."""
    t_ns: float


# ── detect_gaps ────────────────────────────────────────────────────────────────

def detect_gaps(t_fit_by_rx: dict, *, min_gap_ms: float = 60.0) -> list:
    """tx0 링크의 rx별 클록핏 보정 시각 배열 {rx: np.ndarray} → GapEvent 목록.

    갭 = 연속 샘플 간격 > min_gap_ms. 갭 시작 = 직전 샘플 t_fit.
    3 RX 각각 검출 후 시각 기준 클러스터링(±200ms) → 클러스터당 중앙값 1개.
    """
    min_gap_ns = min_gap_ms * 1_000_000

    # 각 RX에서 갭 시작 시각 수집
    all_starts = []
    for t_arr in t_fit_by_rx.values():
        t = np.asarray(t_arr, dtype=np.float64)
        if len(t) < 2:
            continue
        diffs = np.diff(t)
        gap_idx = np.flatnonzero(diffs > min_gap_ns)
        for i in gap_idx:
            all_starts.append(t[i])  # 직전 샘플이 갭 시작

    if not all_starts:
        return []

    # 시각 기준 클러스터링(±200ms) — 새 원소가 클러스터 첫 원소에서 200ms 이내면 병합
    # 200ms 정확히 = 같은 클러스터 경계 밖 (< 200ms만 병합)
    cluster_win_ns = 199_999_999.0  # <200ms 허용 (200ms는 별개 클러스터)
    all_starts_sorted = sorted(all_starts)
    clusters = []
    current = [all_starts_sorted[0]]
    for s in all_starts_sorted[1:]:
        if s - current[0] <= cluster_win_ns:
            current.append(s)
        else:
            clusters.append(current)
            current = [s]
    clusters.append(current)

    # 클러스터당 중앙값
    return [GapEvent(t_ns=float(np.median(c))) for c in clusters]


# ── csi_absolute_offsets ──────────────────────────────────────────────────────

def csi_absolute_offsets(gap_starts: list, cmd_times) -> dict:
    """페어링(최근접, ±500ms 밖 미매칭 폐기) → 오프셋 통계.

    gap_starts: list[GapEvent]
    cmd_times: array-like of float (ns)
    반환: {n, mean_ms, se_ms, p5, p95, matched, unmatched}
    """
    max_pair_ns = 500_000_000.0  # 500ms
    cmd = np.asarray(cmd_times, dtype=np.float64)

    if len(gap_starts) == 0 or len(cmd) == 0:
        return {"n": 0, "mean_ms": float("nan"), "se_ms": float("nan"),
                "p5": float("nan"), "p95": float("nan"),
                "matched": 0, "unmatched": int(len(cmd))}

    gap_t = np.asarray([g.t_ns for g in gap_starts], dtype=np.float64)

    # 각 갭에 최근접 cmd 페어링 (탐욕 선착 일치 — 앞 갭이 먼저 cmd 점유)
    offsets = []
    used_cmd = set()
    for gt in gap_t:
        dists = np.abs(cmd - gt)
        best = int(np.argmin(dists))
        if dists[best] <= max_pair_ns and best not in used_cmd:
            offsets.append((gt - cmd[best]) / 1_000_000)  # ms
            used_cmd.add(best)

    n = len(offsets)
    unmatched = len(cmd) - len(used_cmd)

    if n == 0:
        return {"n": 0, "mean_ms": float("nan"), "se_ms": float("nan"),
                "p5": float("nan"), "p95": float("nan"),
                "matched": 0, "unmatched": int(unmatched)}

    arr = np.asarray(offsets)
    se = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {
        "n": n,
        "mean_ms": float(arr.mean()),
        "se_ms": se,
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "matched": n,
        "unmatched": int(unmatched),
    }


# ── flip_offsets ──────────────────────────────────────────────────────────────

def flip_offsets(flip_times, frame_t_ns, frame_brightness) -> dict:
    """프레임 밝기 시계열에서 플립 에지 검출(차분 임계) → 에지 프레임 t_ns − flip_time.

    flip_times: array-like int64 ns
    frame_t_ns: array-like int64 ns
    frame_brightness: array-like float
    반환: {n, mean_ms, se_ms, p5, p95, matched, unmatched}
    """
    flip = np.asarray(flip_times, dtype=np.int64)
    ft = np.asarray(frame_t_ns, dtype=np.int64)
    fb = np.asarray(frame_brightness, dtype=np.float64)

    _empty = {"n": 0, "mean_ms": float("nan"), "se_ms": float("nan"),
              "p5": float("nan"), "p95": float("nan"), "matched": 0, "unmatched": 0}

    if len(ft) < 2 or len(fb) < 2 or len(flip) == 0:
        return _empty

    # 차분으로 에지 검출 — 임계: 밝기 범위의 30%
    diffs = np.diff(fb.astype(np.float64))
    thresh = max(30.0, 0.3 * (fb.max() - fb.min()))
    edge_idx = np.flatnonzero(np.abs(diffs) >= thresh)
    if len(edge_idx) == 0:
        return _empty

    edge_t = ft[edge_idx + 1]  # 에지 이후 첫 프레임 t_ns

    # 각 flip_time에 최근접 에지 페어링 (±2초)
    max_pair_ns = 2_000_000_000
    offsets = []
    used_edge = set()
    for ft_flip in flip:
        dists = np.abs(edge_t - ft_flip)
        best = int(np.argmin(dists))
        if dists[best] <= max_pair_ns and best not in used_edge:
            offsets.append((edge_t[best] - ft_flip) / 1_000_000)  # ms
            used_edge.add(best)

    n = len(offsets)
    unmatched = len(flip) - n

    if n == 0:
        return {"n": 0, "mean_ms": float("nan"), "se_ms": float("nan"),
                "p5": float("nan"), "p95": float("nan"), "matched": 0,
                "unmatched": int(unmatched)}

    arr = np.asarray(offsets)
    se = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {
        "n": n,
        "mean_ms": float(arr.mean()),
        "se_ms": se,
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "matched": n,
        "unmatched": int(unmatched),
    }


# ── match_frames_by_idx ───────────────────────────────────────────────────────

def match_frames_by_idx(brightness, video_frame_idx, video_t_ns):
    """mp4 프레임 밝기를 세션 cam/meta 스탬프와 frame_idx로 정렬 → (t_ns, 밝기).

    cam_capture는 handle_frame(발행)과 writer.write(mp4)를 항상 쌍으로 호출
    → mp4 순번 k = frame_idx k. 세션 HDF5는 레코더/MQTT 결손으로 부분집합일 수
    있으므로 교집합만 반환. mp4 길이 밖 frame_idx(레코더가 더 오래 돈 꼬리)는 폐기.
    HDF5 u64/u32 입력 → int64/float64 정규화 (이후 차분 연산 안전).
    """
    b = np.asarray(brightness, dtype=np.float64)
    idx = np.asarray(video_frame_idx, dtype=np.int64)
    t = np.asarray(video_t_ns).astype(np.int64)
    m = (idx >= 0) & (idx < len(b))
    idx, t = idx[m], t[m]
    order = np.argsort(idx, kind="stable")     # 시간순 보장 (밝기 차분 전제)
    idx, t = idx[order], t[order]
    return t, b[idx]


# ── camera_correction_ms ──────────────────────────────────────────────────────

def camera_correction_ms(mean_ms, frame_t_ns, *, display_latency_ms=13.0):
    """카메라 계통 보정값 = mean − 디스플레이 지연 − T_frame/2 (스펙 산식).

    T_frame은 명목 fps가 아닌 실측 video_t_ns 간격의 중앙값 — 레코더 결손이
    만든 2배 간격은 중앙값이 흡수. 프레임 <2개면 T_frame 산출 불가 → NaN.
    """
    t = np.asarray(frame_t_ns).astype(np.int64)
    if len(t) < 2:
        return float("nan")
    t_frame_ms = float(np.median(np.diff(t))) / 1e6
    return float(mean_ms) - float(display_latency_ms) - t_frame_ms / 2.0


# ── jitter_stats ──────────────────────────────────────────────────────────────

def jitter_stats(cam_t_ns, clockfit_residuals_ms) -> dict:
    """cam 간격과 클록핏 잔차로 지터 통계.

    cam 간격: diff → |interval − median| 의 p95 (σ아닌 강건 통계)
    반환: {cam_interval_p95_ms, cam_sigma_ms, clockfit_resid_p95_ms}
    """
    cam_t = np.asarray(cam_t_ns, dtype=np.float64)
    resid = np.asarray(clockfit_residuals_ms, dtype=np.float64)

    cam_interval_p95 = 0.0
    cam_sigma = 0.0
    if len(cam_t) >= 2:
        intervals_ms = np.diff(cam_t) / 1_000_000
        med = float(np.median(intervals_ms))
        dev = np.abs(intervals_ms - med)
        cam_interval_p95 = float(np.percentile(dev, 95))
        cam_sigma = float(intervals_ms.std(ddof=1)) if len(intervals_ms) > 1 else 0.0

    if len(resid) == 0:
        clockfit_p95 = 0.0
    else:
        clockfit_p95 = float(np.percentile(np.abs(resid), 95))

    return {
        "cam_interval_p95_ms": cam_interval_p95,
        "cam_sigma_ms": cam_sigma,
        "clockfit_resid_p95_ms": clockfit_p95,
    }


# ── verdict ───────────────────────────────────────────────────────────────────

def verdict(csi_abs: dict, jitter: dict, *,
            abs_gate_ms: float = 10.0,
            se_gate_ms: float = 2.0,
            jitter_gate_ms: float = 10.0,
            flip_result: dict = None) -> dict:
    """§13 v1.5.1 판정.

    csi_ok  = se < se_gate_ms AND |mean| < abs_gate_ms — 모델 앵커 무의존
              (v1.5.1: 구 +5ms 앵커는 갭시작 의미론 불일치로 폐기 — 스펙 참조.
               mean은 csi_correction_ms로 항상 동봉, 카메라 ②와 동형 취급)
    jitter_ok = cam σ < jitter_gate_ms AND csi_jitter < jitter_gate_ms
              (v1.5.1: csi_jitter = √(max(0, n·se² − T²/12)) — ① 측정 산포에서
               비컨 위상 몫을 뺀 t_fit 지터. T = csi_abs["period_ms"], 부재 시
               NaN → FAIL(fail-loud). 클록핏 잔차 p95는 브리지 청크 배달 산포라
               게이트 제외 — 참고 기록만)
    카메라 오프셋은 판정 없이 correction_ms로 동봉 (flip_result가 있을 때만).
    """
    mean_ms = csi_abs.get("mean_ms", float("nan"))
    se_ms = csi_abs.get("se_ms", float("nan"))
    csi_ok = bool(se_ms < se_gate_ms) and bool(abs(mean_ms) < abs_gate_ms)

    n = csi_abs.get("n", 0)
    period_ms = csi_abs.get("period_ms", float("nan"))
    var_excess = n * se_ms ** 2 - period_ms ** 2 / 12.0  # σ_shot² − 위상 몫
    if var_excess != var_excess:  # NaN (period_ms/se 부재) — fail-loud
        csi_jitter_ms = float("nan")
    else:
        csi_jitter_ms = max(0.0, var_excess) ** 0.5

    cam_sigma = jitter.get("cam_sigma_ms", 0.0)
    jitter_ok = bool(cam_sigma < jitter_gate_ms) and bool(csi_jitter_ms < jitter_gate_ms)

    out = {
        "csi_ok": csi_ok,
        "jitter_ok": jitter_ok,
        "pass": csi_ok and jitter_ok,
        "csi_correction_ms": float(mean_ms),
        "csi_jitter_ms": float(csi_jitter_ms),
    }

    if flip_result is not None:
        # 보정 완료값(correction_ms = mean − 13ms − T_frame/2, CLI 산출) 우선,
        # 구 결과 JSON(보정 전)은 raw mean 폴백 — 게이트 없이 기록만
        out["correction_ms"] = flip_result.get(
            "correction_ms", flip_result.get("mean_ms", float("nan")))

    return out
