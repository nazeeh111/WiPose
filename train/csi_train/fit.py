"""학습/평가 루프 — AdamW·cosine+warmup·bf16, best=val PCK@0.2.

체크포인트는 텐서·기본형만 저장 → torch.load 기본(weights_only=True)으로 로드 가능.
loss parts는 detach 텐서(Task 4 리뷰) — 텐서로 누적, epoch 경계에서만 스칼라 변환.
use_compile: 이 환경 torch 2.12+cu130 inductor 사전결함으로 불능 — 기본 False 유지.
"""
import json
import math
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

from .augment import augment_batch, mu0_from_stats
from .data import CHANNEL_CONVENTION, diag_pose
from .loss import pose_loss
from .model import WiSPPN
from .pck import calibrate_kappa, evaluate


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_scheduler(opt, *, total_steps, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def predict(model, X, *, device, batch=64):
    """X (N,C,3,3) f16 numpy (C=280|560) → (xy (N,18,2) f32, ĉ (N,18) f32) — 대각 추출.

    batch 기본 64: forward가 입력을 144×144로 업샘플하므로 (B,150,144,144) f32
    중간 텐서가 B×11.9MB — 512였을 때 5.93GiB 단일 할당으로 16GiB GPU OOM(실측)."""
    model.eval()
    xs, cs = [], []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            x = torch.from_numpy(np.ascontiguousarray(X[s:s + batch])).float().to(device)
            o = model(x)
            if o.dim() == 4:
                d = torch.arange(o.shape[-1], device=o.device)
                xy = torch.stack([o[:, 0, d, d], o[:, 1, d, d]], dim=2)
                ch = o[:, 2, d, d]
            else:
                xy = o[:, :2].transpose(1, 2)
                ch = o[:, 2]
            xs.append(xy.float().cpu().numpy())
            cs.append(ch.float().cpu().numpy())
    return np.concatenate(xs), np.concatenate(cs)


def _git_rev():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def save_ckpt(path, model, *, mu, sigma, config, kappa, best,
              mu_phase=None, sigma_phase=None):
    raw = getattr(model, "_orig_mod", model)               # torch.compile 언랩
    ck = {"state_dict": {k: v.cpu() for k, v in raw.state_dict().items()},
          "mu": torch.from_numpy(np.asarray(mu)),
          "sigma": torch.from_numpy(np.asarray(sigma)),
          "channel_convention": CHANNEL_CONVENTION,
          "config": config, "git_rev": _git_rev(),
          "kappa": float(kappa), "best": best}
    if mu_phase is not None:
        ck["mu_phase"] = torch.from_numpy(np.asarray(mu_phase))
        ck["sigma_phase"] = torch.from_numpy(np.asarray(sigma_phase))
    torch.save(ck, path)


def load_ckpt(path, *, device="cpu"):
    ck = torch.load(path, map_location=device)             # weights_only=True 기본(2.6+)
    cfg = ck["config"]
    model = WiSPPN(in_ch=cfg["in_ch"], vector_head=cfg["vector_head"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def _val_arrays(rows):
    m = rows.presence
    xy, c = diag_pose(rows.Y[m])
    return m, xy, c, rows.WH[m], rows.stype[m]


def train_model(splits, hyper, *, mode="pam_full", vector_head=False, lam=1.0,
                augment=False, name="run", out_root="runs", device=None,
                use_compile=False, progress=print):
    """학습 → {"best_pck02", "ckpt", "epochs"}. best = val PCK@0.2 최대 시점."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(hyper["seed"])
    tr, va = splits["train"], splits["val"]
    in_ch = tr.X.shape[1]
    model = WiSPPN(in_ch=in_ch, vector_head=vector_head).to(device)
    if use_compile:
        model = torch.compile(model)
    Xtr = torch.from_numpy(tr.X).to(device)                # f16 보관, 배치 캐스팅
    Ytr = torch.from_numpy(tr.Y).to(device)
    Ptr = torch.from_numpy(tr.presence).to(device)
    B, epochs = hyper["batch"], hyper["epochs"]
    steps = math.ceil(len(tr.X) / B)
    opt = torch.optim.AdamW(model.parameters(), lr=hyper["lr"], weight_decay=hyper["wd"])
    sched = make_scheduler(opt, total_steps=steps * epochs,
                           warmup_steps=steps * hyper["warmup"])
    # κ — train 직립 GT 캘리브, 합성 등 직립 부재 시 설계 예상치 0.5 폴백(§9)
    mtr, xy_tr, c_tr, WH_tr, _ = _val_arrays(tr)
    try:
        kappa = calibrate_kappa(xy_tr, c_tr, WH_tr)
    except ValueError:
        kappa = 0.5
        progress("κ 캘리브 불가(직립 프레임 없음) — 0.5 폴백")
    mva, xy_va, c_va, WH_va, st_va = _val_arrays(va)
    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"in_ch": in_ch, "vector_head": vector_head, "mode": mode, "lam": lam,
           "features": list(splits.get("features") or []),
           "augment": bool(augment), "hyper": dict(hyper)}
    mu0 = mu0_from_stats(splits["mu"], splits["sigma"]).to(device) if augment \
        else None
    mu0_p = mu0_from_stats(splits["mu_phase"], splits["sigma_phase"]).to(device) \
        if augment and splits.get("mu_phase") is not None else None
    best = {"epoch": -1, "pck02": -1.0}
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") \
        else torch.autocast("cpu", enabled=False)
    with open(out / "log.jsonl", "w", encoding="utf-8") as logf:
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(len(tr.X), device=device)
            tot = {"coord": 0.0, "conf": 0.0}              # detach 텐서 누적(스텝 동기 금지)
            for s in range(steps):
                b = perm[s * B:(s + 1) * B]
                xb = Xtr[b].float()
                if augment:
                    xb = augment_batch(xb, mu0, mu0_phase=mu0_p)
                with amp:
                    pred = model(xb)
                    loss, parts = pose_loss(pred, Ytr[b], Ptr[b], mode=mode, lam=lam)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                tot = {k: tot[k] + parts[k] for k in tot}
            pred_xy, _ = predict(model, va.X, device=device)
            rep = evaluate(pred_xy[mva], xy_va, c_va, WH_va, kappa=kappa, stype=st_va)
            pck02 = rep["pck"]["0.2"] or 0.0
            row = {"epoch": ep, "coord": float(tot["coord"]) / steps,
                   "conf": float(tot["conf"]) / steps,
                   "val_pck02": pck02, "val_pck05": rep["pck"]["0.5"],
                   "lr": opt.param_groups[0]["lr"]}
            logf.write(json.dumps(row) + "\n")
            logf.flush()
            progress(f"ep{ep:02d} coord={row['coord']:.5f} conf={row['conf']:.5f} "
                     f"PCK@0.2={pck02:.3f}")
            if pck02 > best["pck02"]:
                best = {"epoch": ep, "pck02": pck02}
                save_ckpt(out / "best.pt", model, mu=splits["mu"], sigma=splits["sigma"],
                          config=cfg, kappa=kappa, best=best,
                          mu_phase=splits.get("mu_phase"),
                          sigma_phase=splits.get("sigma_phase"))
    return {"best_pck02": best["pck02"], "ckpt": str(out / "best.pt"), "epochs": epochs}
