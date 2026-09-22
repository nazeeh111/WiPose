"""COCO-17 → OpenPose BODY-18 — 순수 함수, M1 게이트 단위 테스트 대상."""
import numpy as np

# (BODY-18 idx, COCO-17 idx) — neck(1)은 합성이라 제외
_DIRECT = np.array([
    (0, 0),    # nose
    (2, 6),    # RSho
    (3, 8),    # RElb
    (4, 10),   # RWri
    (5, 5),    # LSho
    (6, 7),    # LElb
    (7, 9),    # LWri
    (8, 12),   # RHip
    (9, 14),   # RKnee
    (10, 16),  # RAnkle
    (11, 11),  # LHip
    (12, 13),  # LKnee
    (13, 15),  # LAnkle
    (14, 2),   # REye
    (15, 1),   # LEye
    (16, 4),   # REar
    (17, 3),   # LEar
])
L_SHO, R_SHO = 5, 6


def coco17_to_body18(kpts17):
    """(17,3) f32 (x_px, y_px, c) → (18,3). neck = 어깨 중점, c_neck = min.

    NaN 좌표는 그대로 전파 (어깨 한쪽 NaN → neck NaN)."""
    k = np.asarray(kpts17, np.float32)
    if k.shape != (17, 3):
        raise ValueError(f"COCO-17 형상 아님: {k.shape}")
    out = np.empty((18, 3), np.float32)
    out[_DIRECT[:, 0]] = k[_DIRECT[:, 1]]
    out[1, :2] = (k[L_SHO, :2] + k[R_SHO, :2]) / 2.0
    out[1, 2] = np.minimum(k[L_SHO, 2], k[R_SHO, 2])
    return out
