#!/usr/bin/env python3
"""m15 세그먼트 캡처 → presence=0 네가티브 학습 세션.

세그먼트 JSON(empty=true)의 구간만 [start+trim, end−trim)으로 잘라
csi_train.data.load_session이 요구하는 형태(samples/X·Y·presence·label_ok·
t_ns·valid + labels attrs W·H·F)로 새 h5를 만든다. Y는 PAM 규약 (4,18,18)
zeros — presence=0 행은 손실 w=𝟙[presence]·…=0이라 좌표항 미사용(§8.2).
트림 규약은 프로브 조인(csi_train.probe.task_rows)과 동일.

예) python3 train/empty_session.py \\
        --h5 host/sessions/CAP-YYYYMMDD-HHMMSS.h5 \\
        --segments host/logs/CAP-segments.json \\
        --out ~/data/CAP-empty.h5
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

Y_PAM_SHAPE = (4, 18, 18)              # pam 빌더 산출 규약 (teacher-conn-01 실데이터 확인)


def build_empty_session(h5_path, segdoc_path, out_path, *, trim_s, wh):
    """empty 세그먼트 행만 잘라 음성 세션 h5 기록 → 행수 반환."""
    segdoc = json.loads(Path(segdoc_path).read_text(encoding="utf-8"))
    if segdoc.get("aborted"):
        raise SystemExit(f"{segdoc_path}: aborted 세그먼트 기록 — 사용 불가")
    empties = [s for s in segdoc["segments"] if s.get("empty")]
    if not empties:
        raise SystemExit(f"{segdoc_path}: empty 세그먼트 없음")

    trim_ns = int(trim_s * 1e9)
    with h5py.File(h5_path, "r") as hi:
        X = hi["samples/X"]
        if X.shape[1:] != (280, 3, 3):
            raise SystemExit(f"{h5_path}: X 형상 {X.shape[1:]} ≠ (280,3,3) — §6.1 규약 위반")
        t = hi["samples/t_ns"][...].astype(np.int64)
        rows = np.zeros(len(t), bool)
        for s in empties:
            rows |= (t >= s["t_start_ns"] + trim_ns) & (t < s["t_end_ns"] - trim_ns)
        n = int(rows.sum())
        if n == 0:
            raise SystemExit("트림 후 0행 — 세그먼트 길이·trim-s 확인")
        with h5py.File(out_path, "w") as ho:
            g = ho.create_group("samples")
            g.create_dataset("X", data=X[...][rows])
            g.create_dataset("t_ns", data=hi["samples/t_ns"][...][rows])
            g.create_dataset("valid", data=hi["samples/valid"][...][rows])
            g.create_dataset("Y", data=np.zeros((n, *Y_PAM_SHAPE), np.float16))
            g.create_dataset("presence", data=np.zeros(n, bool))
            g.create_dataset("label_ok", data=np.ones(n, bool))
            lab = ho.create_group("labels")     # load_session이 W·H attrs를 요구
            lab.attrs["W"], lab.attrs["H"] = int(wh[0]), int(wh[1])
            lab.attrs["F"] = n
            lab.attrs["source"] = "empty_session(무교사 음성 — 세그먼트 empty 구간)"
            ho.attrs["empty_source"] = str(h5_path)
            ho.attrs["empty_segdoc"] = str(segdoc_path)
            ho.attrs["empty_seg_idx"] = [int(s["idx"]) for s in empties]
            ho.attrs["empty_trim_s"] = trim_s
            ho.attrs["plan_version"] = segdoc.get("plan_version", "")
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--segments", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trim-s", type=float, default=2.0)
    ap.add_argument("--wh", type=int, nargs=2, default=(1280, 720),
                    help="labels attrs W H (캠페인 카메라 해상도와 일치)")
    a = ap.parse_args(argv)
    for p, lbl in ((a.h5, "입력 h5"), (a.segments, "세그먼트 JSON")):
        if not Path(p).exists():
            raise SystemExit(f"{lbl} 없음: {p}")
    out = Path(a.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    n = build_empty_session(a.h5, a.segments, out, trim_s=a.trim_s, wh=tuple(a.wh))
    print(f"완료: {out}  rows={n} (presence=0 전행)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
