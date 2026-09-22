"""체크포인트 추론 래퍼 — load_ckpt·l2_normalize 재사용, 대각 (x,y,ĉ) 추출.

정규화 = l2(패킷·링크) → (X−μ)/σ (μ·σ는 ckpt 저장본, 재계산 금지).
--random-weights: M2 전 파이프 스모크 전용(seed 0 고정 — 테스트 결정성)."""
import time
from pathlib import Path

import numpy as np
import torch

from csi_train.data import CHANNEL_CONVENTION, IN_CH, l2_normalize
from csi_train.fit import load_ckpt
from csi_train.model import WiSPPN


class PoseEstimator:
    def __init__(self, ckpt_path, *, random_weights=False, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random = bool(random_weights)
        self.infer_ms = 0.0
        if self.random:
            if ckpt_path:
                raise SystemExit("--ckpt와 --random-weights는 동시 지정 불가")
            torch.manual_seed(0)
            self.model = WiSPPN(in_ch=IN_CH).to(self.device).eval()
            self.vector_head = False
            self.mu = np.zeros((3, 3), np.float32)     # 실 ckpt fit_stats와 동형 — (…,3,3) broadcast
            self.sigma = np.ones((3, 3), np.float32)
            return
        if not ckpt_path or not Path(ckpt_path).exists():
            raise SystemExit(f"ckpt 없음: {ckpt_path} — --ckpt 지정 또는 --random-weights"
                             "(파이프 스모크 전용)")
        self.model, ck = load_ckpt(ckpt_path, device=self.device)
        if ck.get("channel_convention") != CHANNEL_CONVENTION:
            raise SystemExit(f"ckpt 채널 규약 불일치: {ck.get('channel_convention')!r}"
                             f" ≠ {CHANNEL_CONVENTION!r}")
        self.vector_head = bool(ck["config"]["vector_head"])
        self.mu = ck["mu"].cpu().numpy().astype(np.float32)
        self.sigma = ck["sigma"].cpu().numpy().astype(np.float32)

    def __call__(self, X_raw):
        """X_raw (280,3,3) f32/f16 → (xy (18,2) f32 정규화 좌표, c (18,) f32)."""
        Xn = l2_normalize(np.asarray(X_raw, np.float32)[None])
        Xn = (Xn - self.mu) / self.sigma
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model(torch.from_numpy(Xn.astype(np.float32)).to(self.device))
            if self.device == "cuda":
                torch.cuda.synchronize()
        self.infer_ms = (time.perf_counter() - t0) * 1e3
        o = out[0].cpu().numpy()
        if self.vector_head:
            v = o                                   # (3,18)
        else:
            d = np.arange(o.shape[-1])
            v = o[:, d, d]                          # (3,18,18) 대각 → (3,18)
        return v[:2].T.astype(np.float32).copy(), v[2].astype(np.float32).copy()
