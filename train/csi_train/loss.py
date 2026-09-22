"""가중 MSE + confidence.

w = 𝟙[presence]·max(C, 0.2) — C는 PAM c행렬(대각 c_r, 비대각 c_r·c_c).
MSE(w⊙ŷ, w⊙y) ≡ mean(w²·se)로 구현. confidence는 대각 18개만 λ·MSE(ĉ,c),
presence 무관 전 샘플 감독(음성 = 타깃 0이 ĉ 학습의 핵심).
"""
import torch

C_FLOOR = 0.2
MODES = ("pam_full", "diag_balanced", "diag_only")


def pose_loss(pred, Y, presence, *, mode="pam_full", lam=1.0):
    """pred (B,3,J,J) 그리드 또는 (B,3,J) vector head → (total, {"coord","conf"})."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} ∉ {MODES}")
    J = Y.shape[-1]
    d = torch.arange(J, device=Y.device)
    C = Y[:, 2]
    c_diag = C[:, d, d]
    pr = presence.to(Y.dtype)
    if pred.dim() == 3:                                  # vector head — 대각 전용
        if mode != "diag_only":
            raise ValueError("vector head는 diag_only 전용 (§8.3-3안)")
        w = pr.view(-1, 1) * c_diag.clamp(min=C_FLOOR)
        se = (pred[:, :2] - Y[:, :2, d, d]) ** 2          # (B,2,J)
        coord = (w.unsqueeze(1) ** 2 * se).mean()
        conf = (pred[:, 2] - c_diag).pow(2).mean()
    else:
        w = pr.view(-1, 1, 1) * C.clamp(min=C_FLOOR)      # (B,J,J)
        eye = torch.eye(J, device=Y.device, dtype=Y.dtype)
        if mode == "pam_full":
            m = torch.ones_like(eye)
        elif mode == "diag_balanced":
            m = 1.0 + 16.0 * eye                          # 대각 ×17 카운트 균형(306/18)
        else:
            m = eye
        se = (pred[:, :2] - Y[:, :2]) ** 2                # (B,2,J,J)
        coord = (m * w.unsqueeze(1) ** 2 * se).mean()
        conf = (pred[:, 2, d, d] - c_diag).pow(2).mean()
    total = coord + lam * conf
    # parts는 detach 텐서 — 스칼라 변환(.item)은 호출측 로깅 시점에(스텝마다 동기 금지)
    return total, {"coord": coord.detach(), "conf": conf.detach()}
