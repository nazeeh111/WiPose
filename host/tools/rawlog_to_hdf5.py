#!/usr/bin/env python3
"""rawlog 3개(rx0/1/2) → 세션 HDF5 재구축 (설계 §5 — raw 로그가 원본).

  python3 rawlog_to_hdf5.py --rx0 "/tmp/csilogs/rx0-*.rawlog" --rx1 "..." --rx2 "..." \
      --out ../sessions/soak-20260610.h5 --session soak-20260610
"""
import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csi_pipe.rebuild import rebuild_session  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    for i in (0, 1, 2):
        ap.add_argument(f"--rx{i}", default=None, help=f"rx{i} rawlog 글롭")
    ap.add_argument("--out", required=True)
    ap.add_argument("--session", default=None, help="기본값: out 파일명 stem")
    args = ap.parse_args()

    rx_paths = {}
    for i in (0, 1, 2):
        pat = getattr(args, f"rx{i}")
        if pat:
            # 사전순==시간순: 브리지의 0패딩 rx{id}-%Y%m%d-%H%M%S 파일명 규약 전제 (unwrap 순서 의존)
            m = sorted(glob.glob(pat))
            if not m:
                sys.exit(f"일치 없음: --rx{i} {pat}")
            rx_paths[i] = m
    if not rx_paths:
        sys.exit("--rx0/--rx1/--rx2 중 최소 1개 필요")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stats = rebuild_session(rx_paths, out, session=args.session or out.stem,
                            progress=lambda s: print(s, flush=True))
    for rx, s in sorted(stats.items()):
        print(f"rx{rx}: frames={s['frames']} crc={s['crc']} mismatch={s['mismatch']}")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
