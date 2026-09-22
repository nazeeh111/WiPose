#!/usr/bin/env python3
"""세션 HDF5에 정렬 빌드 (/grid + /samples/X) 추가.

  python3 build_samples.py --h5 ../sessions/soak-20260610.h5 --anchor-rate 20

영상 앵커 세션은 보정값 필수 — 기본은
<repo>/configs/pairing.json에서 로드, --csi-corr-ms/--cam-corr-ms로 오버라이드,
ablation은 --no-correction(명시적 0·0). 적용값은 /meta attrs build.corrections 기록.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csi_pipe.samples import build, resolve_corrections  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--anchor-rate", type=float, default=20.0,
                    help="영상 없을 때 합성 앵커 Hz (기본 20)")
    ap.add_argument("--fit-window", type=float, default=600.0)
    ap.add_argument("--force", action="store_true", help="기존 /grid·/samples 재빌드")
    ap.add_argument("--pairing-config", default=None,
                    help="보정값 JSON (기본: <repo>/configs/pairing.json)")
    ap.add_argument("--csi-corr-ms", type=float, default=None,
                    help="CSI 계통 보정값 ms — config 오버라이드")
    ap.add_argument("--cam-corr-ms", type=float, default=None,
                    help="카메라 계통 보정값 ms — config 오버라이드")
    ap.add_argument("--no-correction", action="store_true",
                    help="명시적 무보정(0·0) — ablation/디버그용")
    args = ap.parse_args()
    if args.pairing_config is not None and not args.pairing_config.strip():
        ap.error("--pairing-config가 빈 경로 — 기본 config는 플래그 생략, "
                 "무보정은 --no-correction (미설정 셸 변수 의심)")
    if args.pairing_config and not Path(args.pairing_config).exists():
        ap.error(f"--pairing-config 파일 없음: {args.pairing_config}")
    cfg = (Path(args.pairing_config) if args.pairing_config
           else Path(__file__).resolve().parents[2] / "configs" / "pairing.json")
    corr, src = resolve_corrections(csi_ms=args.csi_corr_ms, cam_ms=args.cam_corr_ms,
                                    no_correction=args.no_correction, config_path=cfg)
    if corr is not None:
        corr = {**corr, "source": src}
        print(f"보정: csi={corr['csi_ms']}ms cam={corr['cam_ms']}ms ({src})")
    info = build(Path(args.h5), anchor_rate=args.anchor_rate,
                 fit_window_s=args.fit_window, force=args.force,
                 corrections=corr, progress=lambda s: print(s, flush=True))
    print(f"완료: G={info['G']} N={info['N']} valid={info['valid_ratio']:.1%}")


if __name__ == "__main__":
    main()
