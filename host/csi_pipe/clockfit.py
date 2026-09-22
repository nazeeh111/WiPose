"""보드별 esp_timer→t_host 클록 모델.

USB-UART 배칭 지터는 도착 시각을 +방향으로만 왜곡한다(조기 도착 없음) —
(esp, t_host) 산점의 하한 포락선이 진짜 클록선. 발진기 온도 드리프트는
겹침 윈도 구간별 핏 + 윈도 중심 간 예측값 선형 보간으로 흡수 (조각별 연속).

수치 주의: 윈도 내부 좌표는 (초, µs 지연)으로 스케일 — 외적/핏의 float64
정밀도 확보. 입력 t_ns 절대값(~1.8e18)은 int64로 유지, 모델은 epoch 기준점에서
오프셋으로 계산. 핏 출력(t_fit·resid_ns)은 절대 ns의 float64 — 양자화 ~256ns,
ms 스케일(목표 10ms) 대비 무시 가능.
"""
from dataclasses import dataclass, field

import numpy as np

from csi_host.unwrap import TimeUnwrapper

WRAP_US = TimeUnwrapper.WRAP  # esp_timer u32 랩 2^32 µs (71.58분) — 단일 진실원


def lower_hull_idx(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """x 오름차순 점들의 아래 볼록 껍질 꼭짓점 인덱스 (monotone chain 하변).

    floor(x) 단위 버킷별 최솟값 대표점만 후보로 추림 (단측 지터 노이즈 억제).
    """
    bucket = np.floor(x).astype(np.int64)
    o = np.lexsort((y, bucket))
    first = np.concatenate(([True], np.diff(bucket[o]) != 0))
    cand = np.sort(o[first])
    hull = []
    for j in range(len(cand)):
        i = cand[j]
        while len(hull) >= 2:
            i1, i2 = cand[hull[-2]], cand[hull[-1]]
            if (x[i2] - x[i1]) * (y[i] - y[i1]) - (y[i2] - y[i1]) * (x[i] - x[i1]) <= 0:
                hull.pop()
            else:
                break
        hull.append(j)
    return cand[np.asarray(hull, dtype=np.int64)]


@dataclass
class EpochModel:
    boot: int
    esp_lo: float          # 핏 도메인 (unwrap된 µs)
    esp_hi: float
    esp0: float            # 기준점
    t0_ns: int
    centers_s: np.ndarray  # 윈도 중심 (epoch 내 상대 초)
    coef: np.ndarray       # [k,2] (α: µs/s = ppm, β: µs)


@dataclass
class FitReport:
    t_fit: np.ndarray      # f64 ns (invalid 구간은 0)
    resid_ns: np.ndarray   # t - t_fit (invalid는 NaN)
    valid: np.ndarray      # bool
    slopes: list = field(default_factory=list)  # 에포크별 평균 기울기 (µs/s = ppm)

    def stats(self) -> dict:
        r = self.resid_ns[self.valid] / 1e6  # ms
        if len(r) == 0:
            return {"n": 0}
        return {
            "n": int(self.valid.sum()),
            "slope_ppm": self.slopes,
            "resid_p5_ms": float(np.percentile(r, 5)),
            "resid_p50_ms": float(np.percentile(r, 50)),
            "resid_p95_ms": float(np.percentile(r, 95)),
            "resid_max_ms": float(r.max()),
        }


def _eval_piecewise(xs, centers, coefs):
    """윈도 중심 간 예측값 선형 보간 (양끝은 끝 윈도 모델로 외삽) — µs 지연 반환."""
    if len(centers) == 1:
        return coefs[0, 0] * xs + coefs[0, 1]
    j = np.clip(np.searchsorted(centers, xs) - 1, 0, len(centers) - 2)
    c0, c1 = centers[j], centers[j + 1]
    w = np.clip((xs - c0) / np.maximum(c1 - c0, 1e-9), 0.0, 1.0)
    p0 = coefs[j, 0] * xs + coefs[j, 1]
    p1 = coefs[j + 1, 0] * xs + coefs[j + 1, 1]
    return (1 - w) * p0 + w * p1


def _fit_epoch(xs, ys, window_s):
    """xs(s)·ys(µs 지연) → (centers, coefs) 또는 None. 버킷 최소지연 → 하한 포락선 → 윈도 LS.

    버킷 폭 = window_s/20 (최소 1초) — USB 배칭 최대지터 대비 버킷 내 최솟값이
    충분히 0에 수렴해야 포락선 기울기가 수렴한다. 1초 버킷은 30ms 지터에서 부족.
    """
    bw = max(1.0, window_s / 20.0)
    bucket = (xs // bw).astype(np.int64)
    o = np.lexsort((ys, bucket))
    first = np.concatenate(([True], np.diff(bucket[o]) != 0))
    cand = np.sort(o[first])
    hull = cand[lower_hull_idx(xs[cand], ys[cand])]
    xh, yh = xs[hull], ys[hull]
    span = float(xs[-1])
    starts = (np.arange(0.0, span - window_s + 1e-9, window_s / 2)
              if span > window_s else np.array([0.0]))
    centers, coefs = [], []
    for s in starts:
        m = (xh >= s) & (xh <= s + window_s)
        if m.sum() < 3:
            continue  # 포락선 점 부족 윈도 — 이웃 윈도 모델이 연장 커버 (스펙 오류 처리)
        a, b = np.polyfit(xh[m], yh[m], 1)
        centers.append((max(s, 0.0) + min(s + window_s, span)) / 2)
        coefs.append((a, b))
    if not centers:
        if len(hull) < 2:
            return None
        a, b = np.polyfit(xh, yh, 1)
        centers, coefs = [span / 2], [(a, b)]
    return np.asarray(centers), np.asarray(coefs)


class BoardClockModel:
    def __init__(self, epochs):
        self.epochs = epochs

    _MARGIN_US = 60e6  # 도메인 여유 1분

    def predict(self, esp_us, boot_id):
        """(t_fit_ns f64, valid bool). 에포크 매칭 = boot 값 + esp 도메인."""
        esp = np.asarray(esp_us, np.float64)
        boot = np.asarray(boot_id)
        t = np.zeros(len(esp), np.float64)
        ok = np.zeros(len(esp), bool)
        for e in self.epochs:
            m = ((boot == e.boot) & (esp >= e.esp_lo - self._MARGIN_US)
                 & (esp <= e.esp_hi + self._MARGIN_US))
            if not m.any():
                continue
            xs = (esp[m] - e.esp0) / 1e6
            d_us = _eval_piecewise(xs, e.centers_s, e.coef)
            t[m] = e.t0_ns + 1000.0 * (esp[m] - e.esp0) + d_us * 1e3
            ok[m] = True
        return t, ok


def fit_board(esp_us, t_ns, boot_id, *, window_s=600.0, min_epoch=100):
    """보드(rx) 스트림(도착순) → (BoardClockModel, FitReport). 에포크 = boot_id 변화.

    주의: boot_id 불변인 esp 역행(미감지 리부트)은 분리하지 않음 — 해당 에포크는
    하한선이 앞 구간을 따라가 뒤 구간이 거대 양(+) 잔차로 드러난다 (stats로 검출).
    """
    esp = np.asarray(esp_us, np.float64)
    t = np.asarray(t_ns, np.int64)
    boot = np.asarray(boot_id)
    n = len(esp)
    cut = np.flatnonzero(np.diff(boot.astype(np.int64)) != 0) + 1
    bounds = np.concatenate(([0], cut, [n]))
    epochs = []
    t_fit = np.zeros(n, np.float64)
    valid = np.zeros(n, bool)
    slopes = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < min_epoch:
            continue
        order = np.argsort(esp[a:b], kind="stable") + a  # 안전망 — 도착순이면 이미 정렬
        e, ti = esp[order], t[order]
        esp0, t0 = e[0], int(ti[0])
        xs = (e - esp0) / 1e6
        ys = ((ti - t0) - 1000.0 * (e - esp0)) / 1e3     # µs 지연 (int64 차 → f64 안전)
        fit = _fit_epoch(xs, ys, window_s)
        if fit is None:
            continue
        centers, coefs = fit
        d_us = _eval_piecewise(xs, centers, coefs)
        t_fit[order] = t0 + 1000.0 * (e - esp0) + d_us * 1e3
        valid[order] = True
        epochs.append(EpochModel(boot=int(boot[a]), esp_lo=float(e[0]), esp_hi=float(e[-1]),
                                 esp0=float(esp0), t0_ns=t0,
                                 centers_s=centers, coef=coefs))
        slopes.append(float(coefs[:, 0].mean()))
    resid = np.where(valid, t.astype(np.float64) - t_fit, np.nan)
    rep = FitReport(t_fit=t_fit, resid_ns=resid, valid=valid, slopes=slopes)
    return BoardClockModel(epochs), rep


def wrap_continuity(esp_us, resid_ns, valid, *, halfwin_s=30.0):
    """랩 경계(k·2³²µs) 전후 ±halfwin 잔차 중앙값 차 — <1ms면 unwrap 검증 통과.

    리부트로 esp가 리셋된 경우 경계 표본이 없어 '표본 부족'으로 보고된다(정상).
    """
    esp = np.asarray(esp_us, np.float64)
    out = []
    if len(esp) == 0:
        return out
    for k in range(int(esp.min() // WRAP_US) + 1, int(esp.max() // WRAP_US) + 1):
        w = k * float(WRAP_US)
        left = valid & (esp >= w - halfwin_s * 1e6) & (esp < w)
        right = valid & (esp >= w) & (esp < w + halfwin_s * 1e6)
        if left.sum() < 10 or right.sum() < 10:
            out.append({"wrap_at_min": w / 6e7, "delta_ms": None, "ok": False,
                        "note": "표본 부족"})
            continue
        d = abs(float(np.nanmedian(resid_ns[left])) - float(np.nanmedian(resid_ns[right])))
        out.append({"wrap_at_min": w / 6e7, "delta_ms": d / 1e6, "ok": bool(d < 1e6)})
    return out
