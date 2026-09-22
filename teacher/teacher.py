#!/usr/bin/env python3
"""교사 라벨링 CLI — label | qa | gate | pam.

WSL2의 teacher/ 디렉터리에서:
  python3 teacher.py label ../host/sessions/S.mp4 --h5 ../host/sessions/S.h5
  python3 teacher.py qa --h5 ../host/sessions/S.h5 --out qa_s01      # S04는 --all
  python3 teacher.py gate qa_s01/verdicts.json qa_s04/verdicts.json
  python3 teacher.py pam --h5 ../host/sessions/S.h5 --verdicts qa_s01/verdicts.json
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import h5py

from csi_teacher import labels as L
from csi_teacher import qa as Q
from csi_teacher.pam import build_pam


def iter_frames(mp4):
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        raise SystemExit(f"mp4 열기 실패: {mp4}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame
    finally:
        cap.release()


def cmd_label(a, make_runner=None):
    if make_runner is None:
        from csi_teacher.runner import make_runner   # 지연 — rtmlib 미설치가 타 서브커맨드 안 막게
    runner = make_runner(a.device)
    t0 = time.monotonic()
    res = L.run_label(
        iter_frames(a.mp4), runner, det_thr=a.det_thr,
        progress=lambda n: print(
            f"  {n}프레임 {n / max(time.monotonic() - t0, 1e-9):.1f}fps", flush=True))
    res.attrs.update(L.base_attrs(mp4=a.mp4, det_thr=a.det_thr,
                                  det_model=getattr(runner, "det_model", "?"),
                                  pose_model=getattr(runner, "pose_model", "?")))
    F = len(res.status)
    if a.h5:
        L.write_h5(a.h5, res, force=a.force)
        dst = a.h5
    else:
        side = Path(a.mp4).with_suffix(".labels.npz")
        if side.exists() and not a.force:
            raise SystemExit(f"기존 {side} 존재 — --force로 재라벨")
        L.save_npz(side, res)
        dst = side
    dt = max(time.monotonic() - t0, 1e-9)
    print(f"기록: {dst} (F={F}, {F / dt:.1f}fps) — "
          f"ok={int((res.status == L.STATUS_OK).sum())} "
          f"no_person={int((res.status == L.STATUS_NO_PERSON).sum())} "
          f"multi={int((res.status == L.STATUS_MULTI).sum())}")


def cmd_qa(a):
    if a.h5:
        with h5py.File(a.h5, "r") as h:
            if "labels" not in h:
                raise SystemExit("/labels 없음 — 먼저 label 실행")
            g = h["labels"]
            pose18, status = g["pose18"][...], g["status"][...]
            det = g["det_score"][...]
            mp4 = a.mp4 or g.attrs.get("mp4")
    else:
        res = L.load_npz(a.npz)
        pose18, status, det = res.pose18, res.status, res.det_score
        mp4 = a.mp4 or res.attrs.get("mp4")
    if not mp4 or not Path(mp4).exists():
        raise SystemExit(f"mp4 없음: {mp4!r} — --mp4로 명시")
    idxs = Q.pick_frames(len(status), k=a.sample, seed=a.seed, all_frames=a.all)
    page = Q.build_gallery(a.out, mp4, pose18, status, det, idxs,
                           gid=Path(a.out).name)
    print(f"갤러리 {len(idxs)}장: {page}")
    print("Windows 브라우저로 열어 o/x 판정 → 'JSON 내보내기' → teacher.py gate")


def cmd_gate(a):
    per, judged, fails, rate = Q.aggregate(a.verdicts)
    for p, n, f in per:
        print(f"  {p}: {f}/{n} fail")
    ok = Q.gate_pass(rate)
    print(f"실패율 {rate:.2%} ({fails}/{judged}) — §7 게이트(<{Q.GATE_MAX:.0%}): "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def cmd_pam(a):
    r = build_pam(a.h5, verdicts=a.verdicts, force=a.force)
    print(f"완료: N={r['N']} presence={r['presence']} 폐기={r['discarded']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="교사 라벨링 — 설계 §7/§5.3")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("label", help="mp4 추론 → /labels 또는 사이드카")
    p.add_argument("mp4")
    p.add_argument("--h5")
    p.add_argument("--det-thr", type=float, default=0.5, dest="det_thr",
                   help="검출 임계 (S04 세션은 완화 — 설계 §7)")
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_label)
    p = sub.add_parser("qa", help="수동 감사 HTML 갤러리")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--h5")
    src.add_argument("--npz")
    p.add_argument("--mp4", help="기본: /labels attrs의 경로")
    p.add_argument("--sample", type=int, default=200)
    p.add_argument("--all", action="store_true", help="전 프레임 (S04 전수 감사)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_qa)
    p = sub.add_parser("gate", help="판정 JSON 합산 → §7 <2%% 게이트")
    p.add_argument("verdicts", nargs="+")
    p.set_defaults(fn=cmd_gate)
    p = sub.add_parser("pam", help="QA 반영 — /samples/Y·presence·label_ok 확정")
    p.add_argument("--h5", required=True)
    p.add_argument("--verdicts")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_pam)
    a = ap.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
