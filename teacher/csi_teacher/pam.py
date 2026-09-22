"""pose18 → PAM (4,18,18).

대각 (x_r, y_r, c_r, c_r), 비대각 (x_r−x_c, y_r−y_c, c_r·c_c, c_r·c_c).
좌표는 저장 전 x/W·y/H 정규화 (설계 §7)."""
import json
from pathlib import Path

import h5py
import numpy as np

from .labels import STATUS_MULTI, STATUS_OK


def pam_from_pose18(pose18, *, W, H):
    """pose18 (18,3) → PAM (4,18,18) float32.

    결측/NaN 프레임은 호출측(build_pam)이 presence=0 → Y=0으로 처리; 이 함수는 NaN을 그대로 전파한다."""
    p = np.asarray(pose18, np.float32)
    if p.shape != (18, 3):
        raise ValueError(f"BODY-18 형상 아님: {p.shape}")
    x = p[:, 0] / float(W)
    y = p[:, 1] / float(H)
    c = p[:, 2]
    Y = np.empty((4, 18, 18), np.float32)
    Y[0] = x[:, None] - x[None, :]
    Y[1] = y[:, None] - y[None, :]
    Y[2] = c[:, None] * c[None, :]
    Y[3] = Y[2]
    d = np.arange(18)
    Y[0, d, d] = x
    Y[1, d, d] = y
    Y[2, d, d] = c
    Y[3, d, d] = c
    return Y


def build_pam(h5_path, *, verdicts=None, force=False, say=print):
    """QA 반영 최종화: /samples/Y·presence·label_ok + /labels/qa_fail.

    presence: 사람 존재(ok) — no_person은 0이되 학습엔 쓰는 음성 샘플.
    label_ok: 학습 사용 가능 — multi(v1 폐기)·QA fail은 0. 폐기 ≠ 음성."""
    with h5py.File(h5_path, "r+") as h:
        if "labels" not in h or "pose18" not in h["labels"]:
            raise SystemExit("/labels 없음 — 먼저 teacher.py label --h5 실행")
        if "samples" not in h or "t_ns" not in h["samples"]:
            raise SystemExit("/samples 없음 — 먼저 host csi_pipe samples 빌드 실행")
        g = h["labels"]
        pose18 = g["pose18"][...]
        status = g["status"][...]
        W, H = int(g.attrs["W"]), int(g.attrs["H"])
        F = len(status)
        qa_fail = np.zeros(F, bool)
        if verdicts:
            vd = json.loads(Path(verdicts).read_text(encoding="utf-8"))
            vd.pop("_total", None)                     # qa.exp()의 완전성 메타 — 판정 아님
            for k, v in vd.items():
                f = int(k)
                if f >= F:
                    raise SystemExit(f"판정 frame {f} ≥ F {F} — 다른 세션의 verdicts?")
                if v == "fail":
                    qa_fail[f] = True
        if "samples/Y" in h or "labels/qa_fail" in h:
            if not force:
                raise SystemExit("기존 pam 산출물(/samples/Y·/labels/qa_fail) 존재 — --force로 재빌드")
            # 재빌드는 비원자(임시그룹 없음) — Y를 먼저 지우므로 중단돼도 존재 가드가 잡고, 재실행이 자가 복구.
            # qa_fail 단독 잔존(= host build --force 직후) 포함 — 없으면 아래 create_dataset 충돌
            for k in ("samples/Y", "samples/presence", "samples/label_ok",
                      "labels/qa_fail"):
                if k in h:
                    del h[k]
        vt = h["video/t_ns"][...].astype(np.int64)
        fi = (h["video/frame_idx"][...].astype(np.int64) if "video/frame_idx" in h
              else np.arange(len(vt), dtype=np.int64))     # 구세션 identity 폴백
        order = np.argsort(vt, kind="stable")
        vt_s, fi_s = vt[order], fi[order]
        st = h["samples/t_ns"][...].astype(np.int64)
        rows = np.searchsorted(vt_s, st)
        bad = (rows >= len(vt_s)) | (vt_s[np.minimum(rows, len(vt_s) - 1)] != st)
        if bad.any():
            raise SystemExit(
                f"앵커 {int(bad.sum())}개가 /video/t_ns에 없음 — 빌드·라벨의 세션 불일치")
        frames = fi_s[rows]
        if len(frames) and int(frames.max()) >= F:
            raise SystemExit(f"frame_idx {int(frames.max())} ≥ 라벨 F {F} — mp4/세션 불일치")
        sf = status[frames]
        qf = qa_fail[frames]
        presence = (sf == STATUS_OK) & ~qf
        label_ok = ~qf & (sf != STATUS_MULTI)
        N = len(frames)
        Y = np.zeros((N, 4, 18, 18), np.float16)
        for n in np.flatnonzero(presence):
            Y[n] = pam_from_pose18(pose18[frames[n]], W=W, H=H).astype(np.float16)
        sg = h["samples"]
        sg.create_dataset("Y", data=Y)
        sg.create_dataset("presence", data=presence)
        sg.create_dataset("label_ok", data=label_ok)
        g.create_dataset("qa_fail", data=qa_fail)
        sg.attrs["pam_build"] = json.dumps(
            {"verdicts": str(verdicts) if verdicts else None, "N": N,
             "presence": int(presence.sum()), "discard": int((~label_ok).sum())},
            ensure_ascii=False)
        say(f"PAM N={N} presence={int(presence.sum())} 폐기={int((~label_ok).sum())}")
        return {"N": N, "presence": int(presence.sum()),
                "discarded": int((~label_ok).sum())}
