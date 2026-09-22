#!/usr/bin/env python3
"""야간 소크 rawlog → M0 판정.

  python3 soak_report.py /tmp/csilogs/*.rawlog --window 22:00-07:00 --json verdict.json

9P가 느리므로 rawlog는 /tmp 복사본 사용 권장 (분석 노하우).
"""
import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csi_pipe.soak import analyze_soak, render_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="rawlog 경로 (와일드카드 허용)")
    ap.add_argument("--window", default=None, help="무인 구간 KST, 예: 22:00-07:00")
    ap.add_argument("--fit-window", type=float, default=600.0, help="클록핏 윈도(초)")
    ap.add_argument("--json", default=None, help="판정 JSON 저장 경로")
    args = ap.parse_args()

    expanded = []
    for p in args.paths:
        m = sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p]
        if not m:
            print(f"경고: 일치 없음 — {p}", file=sys.stderr)
        expanded.extend(m)
    if not expanded:
        sys.exit(1)

    rep = analyze_soak(expanded, window=args.window, fit_window_s=args.fit_window)
    print(render_report(rep))
    if args.json:
        Path(args.json).write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
        print(f"JSON 저장: {args.json}")


if __name__ == "__main__":
    main()
