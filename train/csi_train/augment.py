"""GPU 텐서 증강 4종 — 소표본 보강.

z-score 공간 (B,280,3,3) 또는 (B,560,3,3) 입력 전제, train 배치 전용(eval/predict 금지).
드롭·erasing 채움값 = mu0 = (0−mu)/sigma — raw 0(결손 슬롯)의 정규화값으로,
실제 패킷/링크 결손 분포와 정합. 시간축 워핑류는 5패킷 윈도에서 퇴화라 제외.
560(amp‖phase) 입력은 mu0_phase 필수 — 블록별 채움값, SC 드롭은 양 블록 동일 k.
"""
import torch

N_SC, N_PKT = 56, 5


def mu0_from_stats(mu, sigma):
    """링크별 z-score 통계 (3,3) → raw 0의 정규화값 (3,3) f32 텐서."""
    return -(torch.as_tensor(mu, dtype=torch.float32)
             / torch.as_tensor(sigma, dtype=torch.float32))


def augment_batch(x, mu0, *, mu0_phase=None, p=0.5, noise=0.05, scale=0.25,
                  drop_sc=4, drop_link=0.1):
    """샘플별 Bernoulli(p) 게이트 후: ① 가우시안 노이즈 σ=noise ② 편차 스케일
    U(1±scale)(z-score 공간이라 콘트라스트 스케일링 등가) ③ SC 드롭 drop_sc개
    (전 패킷·전 링크 — 협대역 페이딩 모사) ④ 링크 erasing 확률 drop_link.
    560(amp‖phase) 입력은 mu0_phase 필수 — 블록별 채움값, SC 드롭은 양 블록 동일 k."""
    B = x.shape[0]
    C = x.shape[1]
    n_pkt_blocks = C // N_SC                               # 280→5, 560→10
    dev = x.device
    if C == N_SC * N_PKT:
        fill_c = mu0.to(dev)[None].expand(C, 3, 3)
    elif C == 2 * N_SC * N_PKT:
        if mu0_phase is None:
            raise ValueError("560ch(amp‖phase) 증강은 mu0_phase 필수 — phase 블록 채움값")
        fill_c = torch.cat([mu0.to(dev)[None].expand(C // 2, 3, 3),
                            mu0_phase.to(dev)[None].expand(C // 2, 3, 3)])
    else:
        raise ValueError(f"augment_batch: 지원 채널 280|560 ≠ {C}")
    on = x.new_zeros(B, dtype=torch.bool) if p <= 0 else \
        torch.rand(B, device=dev) < p
    gate = on[:, None, None, None]
    x = x.clone()
    fill = fill_c[None].expand_as(x)
    if noise > 0:
        x = torch.where(gate, x + noise * torch.randn_like(x), x)
    if scale > 0:
        s = 1 + scale * (2 * torch.rand(B, 1, 1, 1, device=dev) - 1)
        x = torch.where(gate, x * s, x)
    if drop_sc > 0:
        r = torch.rand(B, N_SC, device=dev)
        sc = r.argsort(dim=1).argsort(dim=1) < drop_sc     # 균등 k개 (rank < k)
        ch = sc.repeat(1, n_pkt_blocks)                    # c = p*56+k 규약(§6.1), 양 블록 동일 k
        x = torch.where(ch[:, :, None, None] & gate, fill, x)
    if drop_link > 0:
        er = (torch.rand(B, device=dev) < drop_link) & on
        idx = torch.randint(0, 9, (B,), device=dev)
        onehot = torch.zeros(B, 9, dtype=torch.bool, device=dev)
        onehot[torch.arange(B, device=dev), idx] = True
        x = torch.where(onehot.view(B, 1, 3, 3) & er[:, None, None, None],
                        fill, x)
    return x
