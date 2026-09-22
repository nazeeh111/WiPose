"""M1.5 프로브 — 세그먼트 조인·태스크 필터·프로브 학습·게이트 판정.
segments.json에서만(teacher·카메라·보정 비의존). 게이트 = max(linear, mlp) ≥ 임계."""
import json
from pathlib import Path

import h5py
import numpy as np

from .data import l2_normalize

THRESHOLDS = {"pos9": 0.85, "posture3": 0.90, "lying_empty": 0.95}
N_CLS = {"pos9": 9, "posture3": 3, "lying_empty": 2}
_POSTURE_CLS = {"stand": 0, "sit": 1, "lie": 2}


def load_segments(path):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"segments 파일 없음: {path}")
    d = json.loads(p.read_text(encoding="utf-8"))
    missing = {"session", "capture", "plan_version", "segments", "aborted"} - d.keys()
    if missing:
        raise SystemExit(f"segments 키 누락 {sorted(missing)}: {path}")
    if d["aborted"]:
        raise SystemExit(f"aborted 캡처 — 재캡처 필요: {path}")
    return d


def check_pair(seg_tr, seg_ev):
    if seg_tr["plan_version"] != seg_ev["plan_version"]:
        raise SystemExit("plan_version 불일치: "
                         f"{seg_tr['plan_version']} vs {seg_ev['plan_version']}")
    if seg_tr["session"] == seg_ev["session"]:
        raise SystemExit(f"train/eval 세션 동일({seg_tr['session']}) — "
                         "교차-세션 게이트 무효 (인자 복붙 실수 의심)")


def gate_pass(acc, thr):
    """§13 게이트 — 경계 포함(≥)."""
    return bool(acc >= thr)


def _seg_label(seg, task):
    """세그먼트 → 태스크 클래스 (해당 없음 = None). 스펙 §태스크 필터 표."""
    if task == "pos9":
        return seg["pos"] - 1 if seg["posture"] == "stand" else None
    if task == "posture3":
        if seg["pos"] == 5 and seg["posture"] in _POSTURE_CLS:
            return _POSTURE_CLS[seg["posture"]]
        return None
    if task == "lying_empty":
        if seg["posture"] == "lie":
            return 1
        return 0 if seg["empty"] else None
    raise SystemExit(f"미정의 태스크 {task}")


def task_rows(t_ns, valid, segdoc, task, trim_ns):
    """윈도 t_ns → (행 인덱스, 클래스). 조인 = [start+trim, end−trim) ∧ valid."""
    t = t_ns.astype(np.int64)
    idx_parts, y_parts = [], []
    for seg in segdoc["segments"]:
        c = _seg_label(seg, task)
        if c is None:
            continue
        m = valid & (t >= seg["t_start_ns"] + trim_ns) \
                  & (t < seg["t_end_ns"] - trim_ns)
        idx_parts.append(np.flatnonzero(m))
        y_parts.append(np.full(int(m.sum()), c, np.int64))
    total = sum(len(i) for i in idx_parts)
    if total == 0:
        raise SystemExit(f"{task}: 조인 0행 — 시계/세션 불일치 또는 트림 과대 의심")
    y = np.concatenate(y_parts)
    missing = sorted(set(range(N_CLS[task])) - set(np.unique(y).tolist()))
    if missing:
        raise SystemExit(f"{task}: 클래스 {missing} 조인 0행 — 세그먼트 로그/트림 확인")
    idx = np.concatenate(idx_parts)
    if len(np.unique(idx)) != len(idx):
        raise SystemExit(f"{task}: 세그먼트 시간 겹침 — 행 {len(idx) - len(np.unique(idx))}개 "
                         "이중 조인 (segments.json 편집/로거 확인)")
    return idx, y


def _flatten(X):
    """(N,280,3,3) f16 → §6.2 l2_normalize 후 (N,2520) f32 — 세션 간 통계 결합 없음."""
    return l2_normalize(X).reshape(len(X), -1)


def _fit_one(kind, Xtr, ytr, n_cls, *, seed, epochs, lr=1e-3):
    """linear | mlp 프로브 — CPU full-batch Adam. 과적합 무해(평가는 교차-세션)."""
    import torch
    from torch import nn
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    model = (nn.Linear(d, n_cls) if kind == "linear" else
             nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, n_cls)))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(np.ascontiguousarray(Xtr)).float()
    yt = torch.from_numpy(ytr)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        lossf(model(Xt), yt).backward()
        opt.step()
    return model


def _acc(model, X, y):
    import torch
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(np.ascontiguousarray(X)).float()) \
            .argmax(1).numpy()
    return float((pred == y).mean()), pred


def _confusion(pred, y, n_cls):
    M = np.zeros((n_cls, n_cls), np.int64)
    np.add.at(M, (y, pred), 1)
    return M.tolist()


def _load_h5(path, *, with_phase=False):
    if not Path(path).exists():
        raise SystemExit(f"세션 h5 없음: {path} — build_samples.py 실행 여부 확인")
    with h5py.File(path, "r") as h:
        X = h["samples/X"][...]
        XP = None
        if with_phase:
            if "samples/X_phase" not in h:
                raise SystemExit(f"{path}: samples/X_phase 없음 — "
                                 "M2.5 빌드(build_samples --force) 필요")
            XP = h["samples/X_phase"][...]
        t_ns = h["samples/t_ns"][...]
        valid = h["samples/valid"][...].astype(bool)
        n_video = len(h["video/t_ns"]) if "video/t_ns" in h else 0
    if valid.mean() < 0.5:
        print(f"경고: {path} valid {valid.mean():.1%} <50% — 손실 점검 권장")
    if n_video:
        print(f"경고: {path} video 앵커 존재 — m15 규약은 --no-mqtt 카메라(동작은 정상)")
    F = _flatten(X)
    if XP is not None:                       # phase는 L2 없이 결합(잔차 유계 — 스펙 §4)
        F = np.concatenate([F, XP.astype(np.float32).reshape(len(XP), -1)], axis=1)
    return F, t_ns, valid


def _check_overlap(t_ns, segdoc, name):
    t = t_ns.astype(np.int64)
    lo = min(s["t_start_ns"] for s in segdoc["segments"])
    hi = max(s["t_end_ns"] for s in segdoc["segments"])
    if not ((t >= lo) & (t < hi)).any():
        raise SystemExit(f"{name}: t_ns·세그먼트 교집합 없음 — 시계/세션 불일치 의심")


def run_probe(train_h5, train_seg, eval_h5, eval_seg, *,
              trim_s=2.0, seed=7, epochs=200, with_phase=False):
    """교차-세션 게이트 — 캡처1 학습 → 캡처2 평가. verdict dict 반환."""
    seg_tr, seg_ev = load_segments(train_seg), load_segments(eval_seg)
    check_pair(seg_tr, seg_ev)
    Xtr, t_tr, v_tr = _load_h5(train_h5, with_phase=with_phase)
    Xev, t_ev, v_ev = _load_h5(eval_h5, with_phase=with_phase)
    _check_overlap(t_tr, seg_tr, "train")
    _check_overlap(t_ev, seg_ev, "eval")
    trim_ns = int(trim_s * 1e9)
    rng = np.random.default_rng(seed)
    tasks = {}
    for task, thr in THRESHOLDS.items():
        itr, ytr = task_rows(t_tr, v_tr, seg_tr, task, trim_ns)
        iev, yev = task_rows(t_ev, v_ev, seg_ev, task, trim_ns)
        A, B = Xtr[itr], Xev[iev]
        accs, preds = {}, {}
        for kind in ("linear", "mlp"):
            m = _fit_one(kind, A, ytr, N_CLS[task], seed=seed, epochs=epochs)
            accs[kind], preds[kind] = _acc(m, B, yev)
        gate_kind = max(accs, key=accs.get)
        # 동일-세션 참고치(§13 "참고용 상한" — 게이트 아님): 캡처1 내부 80/20
        perm = rng.permutation(len(A))
        cut = max(1, int(len(A) * 0.8))
        ref = {}
        for kind in ("linear", "mlp"):
            m = _fit_one(kind, A[perm[:cut]], ytr[perm[:cut]], N_CLS[task],
                         seed=seed, epochs=epochs)
            ref[kind] = _acc(m, A[perm[cut:]], ytr[perm[cut:]])[0]
        tasks[task] = {
            "n_train": int(len(A)), "n_eval": int(len(B)),
            "linear_acc": accs["linear"], "mlp_acc": accs["mlp"],
            "gate_model": gate_kind, "gate_acc": accs[gate_kind],
            "threshold": thr, "pass": gate_pass(accs[gate_kind], thr),
            "same_session_ref": ref,
            "confusion": _confusion(preds[gate_kind], yev, N_CLS[task]),
        }
    return {"plan_version": seg_tr["plan_version"],
            "features": ["amp", "phase"] if with_phase else ["amp"],
            "train_session": seg_tr["session"], "eval_session": seg_ev["session"],
            "trim_s": trim_s, "seed": seed, "epochs": epochs, "tasks": tasks,
            "overall_pass": all(t["pass"] for t in tasks.values())}
