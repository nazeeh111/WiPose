#!/usr/bin/env python3
"""M1.5 프로브 게이트 CLI — 교차-세션 go/no-go (m15-probe).

예) python3 train/probe.py \\
        --train-h5 host/sessions/CAP1-….h5 \\
        --train-seg host/logs/CAP1-segments.json \\
        --eval-h5  host/sessions/CAP2-….h5 \\
        --eval-seg host/logs/CAP2-segments.json \\
        --out logs/m15_verdict.json

게이트 = max(linear, mlp) ≥ 임계(pos9 0.85 / posture3 0.90 / lying_empty 0.95).
판정 FAIL은 도구 실패가 아님 — 완주 시 exit 0, 입력 오류만 비0(SystemExit).
입력 ~35MB·1회 로드 — /mnt/c 직독 허용(§15 ext4 규칙은 epoch 반복 I/O 대상)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from csi_train.probe import run_probe  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-h5", required=True)
    ap.add_argument("--train-seg", required=True)
    ap.add_argument("--eval-h5", required=True)
    ap.add_argument("--eval-seg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trim-s", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--with-phase", action="store_true",
                    help="X_phase 결합 5040 피처 — M2.5 진단용(게이트 의미는 기본 모드)")
    a = ap.parse_args(argv)
    v = run_probe(a.train_h5, a.train_seg, a.eval_h5, a.eval_seg,
                  trim_s=a.trim_s, seed=a.seed, epochs=a.epochs,
                  with_phase=a.with_phase)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n== §13 M1.5 게이트: {v['train_session']} 학습 → {v['eval_session']} 평가 ==")
    for name, t in v["tasks"].items():
        print(f"{name:12s} gate={t['gate_acc']:.3f}({t['gate_model']}) "
              f"thr={t['threshold']:.2f} → {'PASS' if t['pass'] else 'FAIL'}"
              f"  [linear {t['linear_acc']:.3f} / mlp {t['mlp_acc']:.3f}"
              f" / 동일세션ref {max(t['same_session_ref'].values()):.3f}]")
    print("overall:", "PASS — 풀 수집 진행 가능" if v["overall_pass"] else
          "FAIL — §13 예정 조치(pos9=40cm 재배치+재실행 / lying=높이 엇갈림 재검토)")
    print(f"verdict: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
