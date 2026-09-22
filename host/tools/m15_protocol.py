#!/usr/bin/env python3
"""M1.5 미니캡처 운영자 로거 — 세그먼트 경계 기록.

  python m15_protocol.py --capture 1 --session m15-cap1 --out logs\\m15-cap1-segments.json

빈 Enter = 정착(시작)/이동(끝) 토글, u+Enter = 직전 경계 취소, q+Enter = 중단.
권장 시간은 안내일 뿐 — 키가 진실. 실시간 카운터 없음(끝 경계 확정 시 실측 출력).
레코더와 같은 머신에서 실행(time.time_ns 동일 epoch 시계 — 무변환 조인 전제)."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csi_pipe.m15_protocol import M15Session  # noqa: E402


def _say(seg):
    if seg["empty"]:
        return "활동영역 비우기 — 피험자 퇴장 (매트는 그대로 둠)"
    if seg["posture"] == "stand":
        return f"위치 {seg['pos']} 서기 — 진행 중 ~10s마다 회전 큐(N→E→S→W)"
    return {"sit": "매트(5번 칸) 앉기",
            "lie": "매트(5번 칸) 눕기 — 머리 방향은 두 캡처 동일"}[seg["posture"]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", type=int, required=True, choices=(1, 2),
                    help="캡처 회차 — 1=순차 순회, 2=셔플 순회(위치 순서가 다름)")
    ap.add_argument("--session", required=True, help="세션 라벨 (레코더 --session과 동일하게)")
    ap.add_argument("--out", required=True, help="segments.json 출력 경로")
    a = ap.parse_args(argv)
    sess = M15Session(a.capture, a.session)
    print(f"[m15] 캡처{a.capture} 13세그먼트 — Enter=시작/끝 토글, u=직전 경계 취소, q=중단")
    while not sess.done:
        seg = sess.current()
        state = "진행 중 → 이동 지시와 함께 Enter" if sess.settled \
            else "대기 → 피험자 정착 확인 후 Enter"
        print(f"\n[{seg['idx'] + 1}/13] {_say(seg)} (권장 {seg['hint_s']}s) — {state}")
        try:
            cmd = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()                                   # ^C 줄바꿈 정리
            cmd = "q"
        if cmd == "":
            try:
                if sess.mark(time.time_ns()) == "end":
                    s, e = sess.bounds[-1]
                    print(f"  세그먼트 {seg['idx'] + 1}/13 확정: {(e - s) / 1e9:.1f}s")
                else:
                    print(f"  ▶▶▶ {seg['idx'] + 1}/13 시작됨 — 기록 중! (끝낼 때 다시 Enter)")
            except ValueError as e:
                print(f"  무시: {e}")
        elif cmd == "u":
            print(f"  undo: {sess.undo() or '취소할 경계 없음'}")
        elif cmd == "q":
            sess.abort()
            print("  중단 — 확정분만 기록")
            break
        else:
            print("  무시 — Enter/u/q만 인식")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sess.result(), ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"[m15] 기록: {out} (segments={len(sess.bounds)} aborted={sess.aborted})")
    return 1 if sess.aborted else 0


if __name__ == "__main__":
    sys.exit(main())
