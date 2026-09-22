"""PCK@α — 픽셀 공간, GT c<0.3 관절 제외, lying 단축보정 분모.

평가 입력은 presence=1 행만(호출측 필터). 좌표는 정규화 — 내부에서 W·H 픽셀 환산.
lying 판정 = 코어(어깨2·골반2·무릎2·nose) bbox h/w < 0.8 (유효 코어 ≥3),
분모 = max(토르소 대각, κ·코어 bbox 대각). κ는 train 직립(h/w≥1.2)에서 캘리브.
"""
import numpy as np

NOSE, RSHO, LSHO, RHIP, LHIP, RKNEE, LKNEE = 0, 2, 5, 8, 11, 9, 12
CORE = (RSHO, LSHO, RHIP, LHIP, RKNEE, LKNEE, NOSE)
C_MIN = 0.3
LYING_AR, UPRIGHT_AR = 0.8, 1.2


def frame_geometry(gt_px, c):
    """단일 프레임 → (torso, core_diag, aspect, n_core). 무효 = nan."""
    torso = np.nan
    if c[LSHO] >= C_MIN and c[RHIP] >= C_MIN:
        torso = float(np.linalg.norm(gt_px[LSHO] - gt_px[RHIP]))
    pts = gt_px[[j for j in CORE if c[j] >= C_MIN]]
    if len(pts) < 3:
        return torso, np.nan, np.nan, len(pts)
    w = float(pts[:, 0].max() - pts[:, 0].min())
    h = float(pts[:, 1].max() - pts[:, 1].min())
    aspect = h / w if w > 0 else np.inf
    return torso, float(np.hypot(w, h)), aspect, len(pts)


def calibrate_kappa(gt_xy, gt_c, WH):
    """직립 프레임에서 median(토르소/코어 대각) — train GT로 산출."""
    px = gt_xy * WH[:, None, :]
    ratios = []
    for f in range(len(px)):
        torso, core, ar, n = frame_geometry(px[f], gt_c[f])
        if n >= 3 and np.isfinite(torso) and np.isfinite(ar) and ar >= UPRIGHT_AR and core > 0:
            ratios.append(torso / core)
    if not ratios:
        raise ValueError("직립 프레임 없음 — κ 캘리브 불가")
    return float(np.median(ratios))


def evaluate(pred_xy, gt_xy, gt_c, WH, *, kappa, alphas=(0.2, 0.5), stype=None,
             lying_override=None):
    """정규화 좌표 (F,18,2)·c (F,18)·WH (F,2) → 리포트 dict (JSON 직렬화 가능).

    집계는 (frame,joint) 풀링 평균 — 유효 관절 많은 프레임이 가중됨(per-frame 매크로 아님).
    mpjpe_norm은 16:9 비등방 공간 평균 — 보조 지표(px가 정본). per_joint는 게이트 α=0.2 고정.
    """
    F = len(gt_xy)
    if stype is not None and len(stype) != F:
        raise ValueError(f"stype 길이 {len(stype)} ≠ F {F}")
    if lying_override is not None and len(lying_override) != F:
        raise ValueError(f"lying_override 길이 {len(lying_override)} ≠ F {F}")
    pred_px = pred_xy * WH[:, None, :]
    gt_px = gt_xy * WH[:, None, :]
    dist = np.linalg.norm(pred_px - gt_px, axis=2)            # (F,18)
    valid_j = gt_c >= C_MIN
    den = np.full(F, np.nan)
    lying = np.zeros(F, bool)
    for f in range(F):
        torso, core, ar, n = frame_geometry(gt_px[f], gt_c[f])
        lying[f] = bool(n >= 3 and np.isfinite(ar) and ar < LYING_AR)
        if lying_override is not None:
            lying[f] = bool(lying_override[f])
        if lying[f] and np.isfinite(core):
            den[f] = np.nanmax([torso, kappa * core])
        else:
            den[f] = torso
    ok_f = np.isfinite(den) & (den > 0) & valid_j.any(axis=1)
    use = valid_j & ok_f[:, None]                              # (F,18) 평가 대상 관절

    def _pck(rows):
        m = use & rows[:, None]
        out = {}
        for a in alphas:
            hits = dist <= a * den[:, None]
            out[str(a)] = float(hits[m].mean()) if m.any() else None
        return out

    rep = {"pck": _pck(np.ones(F, bool)),
           "pck_lying": _pck(lying),
           "per_joint_0.2": [(float((dist[:, j][use[:, j]] <=
                                     0.2 * den[use[:, j]]).mean()) if use[:, j].any() else None)
                             for j in range(gt_xy.shape[1])],
           "mpjpe_px": float(dist[use].mean()) if use.any() else None,
           "mpjpe_norm": float(np.linalg.norm(pred_xy - gt_xy, axis=2)[use].mean())
                         if use.any() else None,
           "kappa": float(kappa),
           "n_eval": int(ok_f.sum()), "n_lying": int((lying & ok_f).sum()),
           "n_excluded": int(F - ok_f.sum())}
    if stype is not None:
        rep["per_type"] = {str(t): _pck(np.asarray(stype) == t)
                           for t in sorted(set(map(str, stype)))}
    return rep
