"""세션 HDF5 로드·정규화·인메모리 분할.

행 선별 = valid & label_ok. presence는 행 제외가 아니라 손실 마스크.
정규화 순서: 패킷·링크 L2 → 링크 z-score(train 통계만) → f16 보관.
"""
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import yaml

CHANNEL_CONVENTION = "X[c,i,j]: c=p*56+k (p 0..4 과거→현재, k SC 저→고), [i,j]=(rx_i,tx_j)"
IN_CH, N_JOINTS = 280, 18
HYPER_DEFAULTS = {"batch": 64, "epochs": 30, "warmup": 2, "lr": 1e-3, "wd": 1e-4,
                  "knn_k": 5, "seed": 0}
FEATURES = ("phase", "rssi")          # 지원 피처 — config·ckpt에는 정렬 리스트로 기록


@dataclass
class Rows:
    X: np.ndarray         # (N,280,3,3) f16 — load 직후 raw, build_splits 후 정규화(피처 결합 시 560)
    Y: np.ndarray         # (N,4,18,18) f32 — PAM(정규화 좌표)
    presence: np.ndarray  # (N,) bool
    WH: np.ndarray        # (N,2) f32 — 픽셀 환산용
    stype: np.ndarray     # (N,) <U32 — 리포트 분해 태그
    XP: np.ndarray = None  # (N,280,3,3) f16 | None — sanitized phase (M2.5)
    RS: np.ndarray = None  # (N,5,3,3) f16 | None — 윈도 5패킷 RSSI dB (M2.5)


def load_manifest(path):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"설정 파일 없음: {p}")
    man = yaml.safe_load(p.read_text(encoding="utf-8"))
    man["hyper"] = merge_hyper(man.get("hyper") or {})
    return man


def merge_hyper(hyper):
    return {**HYPER_DEFAULTS, **hyper}


def _canon_features(features):
    feats = tuple(sorted(set(features)))
    unknown = set(feats) - set(FEATURES)
    if unknown:
        raise SystemExit(f"미정의 피처 {sorted(unknown)} — 지원: {sorted(FEATURES)}")
    return feats


def load_session(h5_path, features=()):
    """단일 세션 → 필터된 raw dict {X,Y,presence,W,H[,XP,RS]}."""
    feats = _canon_features(features)
    with h5py.File(h5_path, "r") as h:
        if h["samples/X"].shape[1:] != (IN_CH, 3, 3):
            raise SystemExit(f"{h5_path}: X 형상 {h['samples/X'].shape[1:]} ≠ (280,3,3) — §6.1 규약 위반")
        keep = h["samples/valid"][...] & h["samples/label_ok"][...]
        out = {"X": h["samples/X"][...][keep].astype(np.float16),
               "Y": h["samples/Y"][...][keep].astype(np.float32),
               "presence": h["samples/presence"][...][keep],
               "W": int(h["labels"].attrs["W"]), "H": int(h["labels"].attrs["H"])}
        spec = {"phase": ("samples/X_phase", (IN_CH, 3, 3), "XP"),
                "rssi": ("samples/rssi", (5, 3, 3), "RS")}
        for f in feats:
            ds, shp, key = spec[f]
            if ds not in h:
                raise SystemExit(f"{h5_path}: {ds} 없음 — 구형 빌드, "
                                 "build_samples.py --force 재빌드 필요(M2.5 스펙 §2.1)")
            arr = h[ds][...]
            if arr.shape[1:] != shp:
                raise SystemExit(f"{h5_path}: {ds} 형상 {arr.shape[1:]} ≠ {shp}")
            out[key] = arr[keep].astype(np.float16)
        return out


def load_role(manifest, role, features=()):
    """매니페스트의 role 세션들 → 행 결합 Rows (X·XP·RS는 raw f16)."""
    feats = _canon_features(features)
    entries = [s for s in manifest["sessions"] if s["role"] == role]
    if not entries:
        raise SystemExit(f"매니페스트에 role={role} 세션 없음 — configs/train.yaml 확인")
    parts = []
    for e in entries:
        s = load_session(e["h5"], feats)
        n = len(s["X"])
        parts.append((s, np.full((n, 2), [s["W"], s["H"]], np.float32),
                      np.full(n, str(e.get("type", "")), dtype="<U32")))
    rows = Rows(X=np.concatenate([p[0]["X"] for p in parts]),
                Y=np.concatenate([p[0]["Y"] for p in parts]),
                presence=np.concatenate([p[0]["presence"] for p in parts]),
                WH=np.concatenate([p[1] for p in parts]),
                stype=np.concatenate([p[2] for p in parts]))
    if "phase" in feats:
        rows.XP = np.concatenate([p[0]["XP"] for p in parts])
    if "rssi" in feats:
        rows.RS = np.concatenate([p[0]["RS"] for p in parts])
    return rows


def l2_normalize(X):
    """패킷·링크별 56차원 L2 (§6.2). 0벡터(결손 채움)는 0 유지. 반환 f32."""
    X5 = X.astype(np.float32).reshape(len(X), 5, 56, 3, 3)
    n = np.linalg.norm(X5, axis=2, keepdims=True)
    return (X5 / np.where(n == 0, 1, n)).reshape(len(X), IN_CH, 3, 3)


def fit_stats(Xn):
    """L2 후 링크별 z-score 통계 — (mu(3,3), sigma(3,3)) f32."""
    mu = Xn.mean(axis=(0, 1))
    sigma = Xn.std(axis=(0, 1))
    sigma[sigma == 0] = 1.0
    return mu.astype(np.float32), sigma.astype(np.float32)


def apply_stats(Xn, mu, sigma):
    """z-score 후 f16. 주의: Xn(f32)을 인플레이스 변이 — l2_normalize 산출 전용."""
    Xn -= mu
    Xn /= sigma
    return Xn.astype(np.float16)


def rssi_rescale(Xn, RS):
    """§6.2 정석 — L2 후 ×10^(RSSI/20)(패킷·링크별). Xn(f32) 인플레이스 변이."""
    X5 = Xn.reshape(len(Xn), 5, 56, 3, 3)
    X5 *= 10.0 ** (np.asarray(RS, np.float32)[:, :, None] / 20.0)
    return Xn


def normalize_rows(rows, features, mu, sigma, mu_phase=None, sigma_phase=None):
    """Rows → 결합 X(f16) — amp: L2(→rssi 재스케일)→z-score / phase: 자체 z-score."""
    feats = _canon_features(features)
    Xn = l2_normalize(rows.X)
    if "rssi" in feats:
        Xn = rssi_rescale(Xn, rows.RS)
    X = apply_stats(Xn, mu, sigma)
    if "phase" in feats:
        X = np.concatenate([X, apply_stats(rows.XP.astype(np.float32),
                                           mu_phase, sigma_phase)], axis=1)
    return X


def build_splits(manifest, features=()):
    """{"train","val","mu","sigma","mu_phase","sigma_phase","features"} — X 정규화 완료(f16).

    f32 중간본은 분할당 1개만 라이브 (실데이터 피크 억제 — 품질 리뷰 반영).
    train 인라인 경로와 normalize_rows는 같은 수식 — 정규화 순서 변경 시 둘 다 수정."""
    feats = _canon_features(features)
    tr, va = load_role(manifest, "train", feats), load_role(manifest, "val", feats)
    Xtr = l2_normalize(tr.X)
    if "rssi" in feats:
        Xtr = rssi_rescale(Xtr, tr.RS)
    mu, sigma = fit_stats(Xtr)
    mu_p = sigma_p = None
    X16 = apply_stats(Xtr, mu, sigma)
    del Xtr
    if "phase" in feats:
        Ptr = tr.XP.astype(np.float32)
        mu_p, sigma_p = fit_stats(Ptr)
        X16 = np.concatenate([X16, apply_stats(Ptr, mu_p, sigma_p)], axis=1)
        del Ptr
    tr.X = X16
    va.X = normalize_rows(va, feats, mu, sigma, mu_p, sigma_p)
    return {"train": tr, "val": va, "mu": mu, "sigma": sigma,
            "mu_phase": mu_p, "sigma_phase": sigma_p, "features": list(feats)}


def diag_pose(Y):
    """Y (N,4,18,18) → (xy (N,18,2) f32, c (N,18) f32) — PAM 대각."""
    d = np.arange(Y.shape[-1])
    xy = np.stack([Y[:, 0, d, d], Y[:, 1, d, d]], axis=2)
    return xy.astype(np.float32), Y[:, 2, d, d].astype(np.float32)
