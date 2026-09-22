"""무학습 베이스라인 — 게이트 기준치 = max(①②③) PCK@0.2.

전부 CSI 입력·정규화 좌표 공간. 입력 X·GT는 presence=1 행만(호출측 필터).
②′ oracle_centroid는 GT를 보는 오라클 — 진단 전용, 게이트 제외.
"""
import numpy as np
import torch

RHIP, LHIP = 8, 11


def hip_center(gt_xy):
    """(...,18,2) → (...,2) — (RHip+LHip)/2."""
    return (gt_xy[..., RHIP, :] + gt_xy[..., LHIP, :]) / 2


def mean_pose(gt_xy):
    if len(gt_xy) == 0:
        raise ValueError("mean_pose: 빈 GT — presence=1 행이 없음")
    return gt_xy.mean(axis=0)


def predict_mean(mp, n):
    return np.broadcast_to(mp, (n,) + mp.shape).copy()


def knn_indices(train_X, query_X, k, *, device="cpu", chunk=4096):
    """f32 평탄 L2 거리 k-NN — (Q,k) 인덱스. GPU 청크 처리 (f16 입력은 f32 승격 — 2520차원 누적 정밀도)."""
    if len(train_X) == 0 or len(query_X) == 0:
        raise ValueError(f"k-NN 입력 비어 있음: train {len(train_X)}행, query {len(query_X)}행")
    if k > len(train_X):
        raise ValueError(f"k={k} > train {len(train_X)}행")
    tr = torch.from_numpy(np.ascontiguousarray(train_X, dtype=np.float32)
                          .reshape(len(train_X), -1)).to(device)
    q = torch.from_numpy(np.ascontiguousarray(query_X, dtype=np.float32)
                         .reshape(len(query_X), -1))
    out = np.empty((len(query_X), k), np.int64)
    with torch.no_grad():
        for s in range(0, len(q), chunk):
            d = torch.cdist(q[s:s + chunk].to(device), tr)
            out[s:s + chunk] = d.topk(k, largest=False).indices.cpu().numpy()
    return out


def predict_knn_pose(train_X, train_gt_xy, query_X, *, device="cpu"):
    """③ k=1 최근접 train 행의 GT 자세 복사.

    train≠query 전제 — 동일 세트 평가 시 자기 복사로 PCK 부풀림(val 전용).
    """
    idx = knn_indices(train_X, query_X, 1, device=device)[:, 0]
    return train_gt_xy[idx].copy()


def predict_knn_centroid(train_X, train_gt_xy, query_X, *, k=5, device="cpu"):
    """② k-NN hip 중점 평균으로 중심점 추정 → 평균 자세 평행이동."""
    idx = knn_indices(train_X, query_X, k, device=device)
    cent = hip_center(train_gt_xy[idx]).mean(axis=1)           # (Q,2)
    mp = mean_pose(train_gt_xy)
    return mp[None] + (cent - hip_center(mp[None]))[:, None, :]


def predict_oracle_centroid(train_gt_xy, query_gt_xy):
    """②′ GT 중심점 오라클 — 진단 전용."""
    mp = mean_pose(train_gt_xy)
    return mp[None] + (hip_center(query_gt_xy) - hip_center(mp[None]))[:, None, :]
