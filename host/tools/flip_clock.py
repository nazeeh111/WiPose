#!/usr/bin/env python3
"""모니터 흑백 플립 표시기 — 카메라 계통 오프셋 측정용.

cv2 전화면 창으로 흑(0) ↔ 백(255) 전환, 플립 직전 time.time_ns() 기록.
간격 0.7s ± 30% 지터, N=40회 플립.

  python3 flip_clock.py --out /tmp/flips.json
  python3 flip_clock.py --n 20 --interval 0.7 --jitter 0.3 --out flips.json
"""  # noqa: E501
import argparse
import json
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="flip_times.json", help="플립 시각 JSON 저장 경로")
    ap.add_argument("--n", type=int, default=40, help="플립 횟수 (기본 40)")
    ap.add_argument("--interval", type=float, default=0.7, help="평균 간격 초 (기본 0.7)")
    ap.add_argument("--jitter", type=float, default=0.3, help="간격 지터 비율 (기본 0.3 = ±30%%)")
    ap.add_argument("--seed", type=int, default=None, help="난수 시드 (재현용)")
    args = ap.parse_args()

    # cv2는 모드 실행 시 지연 임포트 (--help가 cv2 없이 동작)
    import random
    import numpy as np

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    try:
        import cv2
    except ImportError:
        print("오류: cv2 미설치. pip install opencv-python", file=sys.stderr)
        sys.exit(1)

    # 전화면 검은 창 초기화
    WIN = "flip_clock"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    state = 0  # 0=black, 1=white
    flip_times_ns = []

    # 지터 간격 사전 생성
    rng = np.random.default_rng(args.seed)
    half = args.interval * args.jitter
    waits = rng.uniform(args.interval - half, args.interval + half, args.n)

    try:
        for i, wait_s in enumerate(waits):
            # 대기
            deadline = time.monotonic() + wait_s
            while time.monotonic() < deadline:
                img = np.zeros((100, 100, 3), np.uint8) if state == 0 else np.full((100, 100, 3), 255, np.uint8)
                cv2.imshow(WIN, img)
                if cv2.waitKey(1) == 27:  # ESC → 중단 (finally가 저장·창 정리)
                    print(f"ESC 중단 ({i}/{args.n} 완료)", file=sys.stderr)
                    return

            # 플립 직전 시각 기록 → 전환
            t_ns = time.time_ns()
            flip_times_ns.append(t_ns)
            state = 1 - state
            img = np.zeros((100, 100, 3), np.uint8) if state == 0 else np.full((100, 100, 3), 255, np.uint8)
            cv2.imshow(WIN, img)
            cv2.waitKey(1)
            print(f"플립 {i+1:02d}/{args.n}", end="\r", flush=True)

        print(f"\n{args.n}회 완료")
    finally:
        # 예외·ESC·정상 종료 모든 경로에서 전화면 창 정리 + (부분) 데이터 저장 보증
        # — N 미달 여부는 분석 쪽이 JSON의 n으로 판별
        cv2.destroyAllWindows()
        _save(args.out, flip_times_ns)


def _save(path, flip_times_ns):
    data = {"flip_times_ns": flip_times_ns, "n": len(flip_times_ns)}
    import pathlib
    pathlib.Path(path).write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"저장: {path}  ({len(flip_times_ns)}회)")


if __name__ == "__main__":
    main()
