#!/usr/bin/env python3
"""정렬 분해 검증 CLI — (STOP×20·모니터 플립·지터 세 성분).

모드:
  --mode csi     tx0 포트에 STOP/START 발사 → 클록핏 갭 vs 발사 시각 통계
  --mode cam     플립 JSON + mp4 + 세션 HDF5 → 카메라 계통 오프셋 산출
                 (cam_capture MQTT 발행 + 레코더 세션 기록 중 flip_clock 실행 —
                  학습 페어링에 쓰이는 바로 그 cam/meta t_ns 스탬프를 검증)
  --mode jitter  기존 세션 HDF5 + 클록핏 잔차 → 지터 통계
  --mode report  세 결과 JSON 병합 → §13 판정 출력

주의: --mode csi 검증 런의 rawlog는 소크 rawlog와 분리 저장할 것
     (검증 갭은 의도된 것 — 소크 채점 파일과 혼용 금지).

예:
  python3 align_verify.py --mode csi --port COM34 \\
      --rawlog /tmp/v_rx0.rawlog /tmp/v_rx1.rawlog /tmp/v_rx2.rawlog
  python3 align_verify.py --mode cam --video cam.mp4 --session session.h5 \\
      --flips flip_times.json
  python3 align_verify.py --mode jitter --hdf5 session.h5
  python3 align_verify.py --mode report --csi-json csi_result.json \\
      --cam-json cam_result.json --jitter-json jitter_result.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csi_pipe.align_verify import (  # noqa: E402
    camera_correction_ms,
    csi_absolute_offsets,
    detect_gaps,
    flip_offsets,
    jitter_stats,
    match_frames_by_idx,
    verdict,
)


def _out_path(args, suffix):
    """--out 기본 접두사 + suffix → 저장 경로."""
    base = getattr(args, "out", None) or "align"
    return Path(str(base) + suffix)


# ── mode csi ──────────────────────────────────────────────────────────────────

def mode_csi(args):
    """tx0 시리얼 포트에 STOP/START를 N회 발사, rawlog 3개에서 클록핏 갭 추출, 통계."""
    # pyserial은 실연결 시에만 필요 — 지연 임포트
    import serial
    import random
    import numpy as np
    from csi_pipe.soak import collect_rawlog
    from csi_pipe.clockfit import fit_board

    port = args.port
    baud = args.baud
    n_shots = args.n
    rawlog_paths = [Path(p) for p in args.rawlog]
    rate = args.rate
    interval = args.interval
    jitter_ratio = args.shot_jitter

    print(f"포트: {port}  rawlog: {[str(p) for p in rawlog_paths]}  N={n_shots}", flush=True)

    cmd_times_ns = []

    with serial.Serial(port, baud, timeout=0.2) as ser:
        try:
            for i in range(n_shots):
                # 발사 직전 시각 기록
                t_ns = time.time_ns()
                cmd_times_ns.append(t_ns)
                ser.write(b"STOP\n")
                time.sleep(0.1)
                ser.write(f"START rate={rate}\n".encode())
                print(f"  발사 {i+1:02d}/{n_shots}", end="\r", flush=True)

                # 다음 발사까지 지터 대기
                half = interval * jitter_ratio
                wait = random.uniform(interval - half, interval + half)
                time.sleep(wait)
        finally:
            # 어떤 종료 경로(Ctrl-C·SerialException 포함)에서도 송신 재개 보증 —
            # STOP과 START 사이 중단 시 tx0 침묵 방치 방지
            try:
                time.sleep(0.1)
                ser.write(f"START rate={rate}\n".encode())
            except Exception as e:
                print(f"경고: 종료 START 재주입 실패 — 수동으로 'START rate={rate}' "
                      f"주입 필요: {e}", file=sys.stderr)

    print(f"\n발사 완료 — rawlog {len(rawlog_paths)}개 분석", flush=True)

    # rawlog(=RX 보드)별 → 전 프레임 클록핏 → tx0 링크 보정 시각 → 3 RX 중앙값 클러스터링
    t_fit_by_rx = {}
    for p in rawlog_paths:
        board = collect_rawlog(p)
        if board.frames < 200:
            print(f"경고: 표본 부족 (<200 프레임) — {p}", file=sys.stderr)
            continue
        # 핏은 전 프레임으로 (STOP 중에도 tx1/2가 흘러 핏 도메인이 끊기지 않음),
        # 갭 검출은 tx0 링크의 보정 시각만
        _, rep = fit_board(
            board.esp_us.astype(np.float64),
            board.t_ns,
            board.boot,
        )
        rx_id = int(np.bincount(board.rx_ids).argmax())  # 지배 rx_id (soak 패턴)
        m = rep.valid & (board.tx == 0)
        if rx_id in t_fit_by_rx:
            print(f"경고: rx{rx_id} 중복 rawlog — 나중 파일로 대체: {p}", file=sys.stderr)
        t_fit_by_rx[rx_id] = rep.t_fit[m]

    cmd_arr = np.asarray(cmd_times_ns, dtype=np.float64)
    gaps = detect_gaps(t_fit_by_rx)
    result = csi_absolute_offsets(gaps, cmd_arr)
    # 비컨 주기 — verdict의 csi_jitter(√(n·se²−T²/12)) 산출에 필요
    result["rate"] = rate
    result["period_ms"] = 1000.0 / rate

    # 결과 출력
    print(f"\n[CSI 절대 오프셋]  n={result['n']}  mean={result['mean_ms']:.2f}ms"
          f"  se={result['se_ms']:.2f}ms  p5/p95={result['p5']:.1f}/{result['p95']:.1f}ms"
          f"  matched={result['matched']}  unmatched={result['unmatched']}")
    gate = result["se_ms"] < 2.0 and abs(result["mean_ms"]) < 10.0
    print(f"  게이트 SE<2ms ∧ |mean|<10ms (v1.5.1): {'PASS' if gate else 'FAIL'}"
          f"  — mean은 CSI 계통 보정값으로 기록")

    # cmd_times JSON 저장 (후속 report 모드에서 재사용)
    out_cmd = _out_path(args, "_cmd_times.json")
    out_csi = _out_path(args, "_csi_result.json")
    out_cmd.write_text(json.dumps({"cmd_times_ns": cmd_times_ns}, indent=1), encoding="utf-8")
    out_csi.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"저장: {out_csi}")
    return result


# ── mode cam ──────────────────────────────────────────────────────────────────

def mode_cam(args):
    """플립 JSON + mp4 + 세션 HDF5 → 절대 t_ns 정렬 → flip_offsets 통계.

    프레임 시각은 세션 cam/meta 스탬프(video_t_ns)를 그대로 사용 — mp4 순번 k =
    frame_idx k (cam_capture가 발행·기록을 쌍으로 호출)로 정렬한다. 학습 페어링에
    쓰이는 바로 그 스탬프를 검증하므로 측정 타당성이 정확해진다.
    """
    import cv2  # 지연 임포트 — 이 모드에서만 필요
    import numpy as np
    from csi_pipe.store import SessionReader

    flip_data = json.loads(Path(args.flips).read_text(encoding="utf-8"))
    flip_times = np.asarray(flip_data["flip_times_ns"], dtype=np.int64)

    # mp4에서 프레임별 평균 밝기 추출 (순번 = frame_idx)
    cap = cv2.VideoCapture(str(args.video))
    brightness_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_list.append(float(gray.mean()))
    cap.release()
    brightness = np.asarray(brightness_list, dtype=np.float64)

    # 세션 HDF5의 절대 스탬프 (레코더 결손 = 부분집합 가능)
    with SessionReader(args.session) as sr:
        video_t_ns = sr.video_t_ns
        video_idx = sr.video_frame_idx
    if video_idx is None:  # 구세션 폴백 — identity (store.py 계약)
        video_idx = np.arange(len(video_t_ns))

    frame_t, fb = match_frames_by_idx(brightness, video_idx, video_t_ns)
    print(f"mp4 {len(brightness)}프레임 / 세션 {len(video_t_ns)}스탬프 "
          f"→ 정렬 {len(frame_t)}쌍", flush=True)

    result = flip_offsets(flip_times, frame_t, fb)

    # 보정 산식: mean − 디스플레이 지연 − T_frame/2 (T_frame = 실측 간격 중앙값)
    if result["n"] > 0:
        result["correction_ms"] = camera_correction_ms(
            result["mean_ms"], frame_t, display_latency_ms=args.display_latency)

    print(f"[카메라 오프셋]  n={result['n']}  raw_mean={result.get('mean_ms', float('nan')):.1f}ms"
          f"  correction={result.get('correction_ms', float('nan')):.1f}ms"
          f"  잔여 불확실성 ±15ms (디스플레이 지연 불확실성 지배)")

    out = _out_path(args, "_cam_result.json")
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"저장: {out}")
    return result


# ── mode jitter ───────────────────────────────────────────────────────────────

def mode_jitter(args):
    """HDF5 cam/meta t_ns + 클록핏 잔차 → jitter_stats."""
    import numpy as np

    try:
        from csi_pipe.store import SessionReader
    except ImportError:
        print("오류: h5py 미설치. pip install h5py", file=sys.stderr)
        sys.exit(1)

    with SessionReader(args.hdf5) as sr:
        cam_t = sr.video_t_ns

    # 클록핏 잔차: rawlog(복수 가능)가 있으면 보드별 재계산 후 합산, 없으면 빈 배열
    from csi_pipe.soak import collect_rawlog
    from csi_pipe.clockfit import fit_board

    resid_parts = []
    for p in (args.rawlog or []):
        board = collect_rawlog(Path(p))
        if board.frames < 200:
            continue
        _, rep = fit_board(
            board.esp_us.astype(np.float64),
            board.t_ns,
            board.boot,
        )
        resid_parts.append(rep.resid_ns[rep.valid] / 1_000_000)
    resid_ms = np.concatenate(resid_parts) if resid_parts else np.zeros(0)

    result = jitter_stats(cam_t, resid_ms)

    cam_ok = result["cam_sigma_ms"] < 10.0
    print(f"[지터]  cam_σ={result['cam_sigma_ms']:.2f}ms (게이트 <10ms: "
          f"{'PASS' if cam_ok else 'FAIL'})"
          f"  cam_p95={result['cam_interval_p95_ms']:.2f}ms"
          f"  clockfit_resid_p95={result['clockfit_resid_p95_ms']:.2f}ms"
          f" (참고 — 브리지 청크 산포, v1.5.1 게이트 제외. CSI측 판정은 --mode report)")

    out = _out_path(args, "_jitter_result.json")
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"저장: {out}")
    return result


# ── mode report ───────────────────────────────────────────────────────────────

def mode_report(args):
    """세 결과 JSON 병합 → 판정."""
    csi = json.loads(Path(args.csi_json).read_text(encoding="utf-8"))
    jit = json.loads(Path(args.jitter_json).read_text(encoding="utf-8"))

    flip_res = None
    if args.cam_json and Path(args.cam_json).exists():
        flip_res = json.loads(Path(args.cam_json).read_text(encoding="utf-8"))

    v = verdict(csi, jit, flip_result=flip_res)

    print("\n=== §13 M1 v1.5.1 정렬 판정 ===")
    print(f"  CSI 절대  : {'PASS' if v['csi_ok'] else 'FAIL'}"
          f"  (mean={csi.get('mean_ms', float('nan')):.2f}ms,"
          f" se={csi.get('se_ms', float('nan')):.2f}ms,"
          f" 게이트 SE<2 ∧ |mean|<10 — mean은 CSI 보정값 기록)")
    print(f"  지터      : {'PASS' if v['jitter_ok'] else 'FAIL'}"
          f"  (cam_σ={jit.get('cam_sigma_ms', 0):.2f}ms,"
          f" csi_jitter={v.get('csi_jitter_ms', float('nan')):.2f}ms — 둘 다 <10ms;"
          f" 클록핏 잔차 p95={jit.get('clockfit_resid_p95_ms', 0):.1f}ms는 참고)")
    if flip_res:
        corr = v.get("correction_ms", flip_res.get("correction_ms", float("nan")))
        print(f"  카메라 오프셋: 보정값={corr:.1f}ms  (잔여 불확실성 ±15ms, 게이트 없음)")
    print(f"  전체: {'PASS' if v['pass'] else 'FAIL'}")

    out = _out_path(args, "_verdict.json")
    out.write_text(json.dumps(v, indent=1), encoding="utf-8")
    print(f"저장: {out}")
    return v


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["csi", "cam", "jitter", "report"],
                    help="동작 모드")
    ap.add_argument("--out", default="align", help="출력 파일 접두사 (기본: align)")

    # csi 모드
    g = ap.add_argument_group("--mode csi")
    g.add_argument("--port", default=None, help="tx0 시리얼 포트 (예: COM34)")
    g.add_argument("--baud", type=int, default=921600)
    g.add_argument("--n", type=int, default=20, help="STOP/START 발사 횟수 (기본 20)")
    g.add_argument("--rate", type=int, default=103, help="START 후 전송 rate (기본 103)")
    g.add_argument("--interval", type=float, default=1.7, help="발사 간격 초 (기본 1.7)")
    g.add_argument("--shot-jitter", type=float, default=0.3, help="간격 지터 비율 (기본 0.3)")
    g.add_argument("--rawlog", nargs="+", default=None,
                   help="CSI rawlog 경로 (rx0/1/2 — 3개 주면 3 RX 중앙값 클러스터링)")

    # cam 모드
    g2 = ap.add_argument_group("--mode cam")
    g2.add_argument("--flips", default=None, help="flip_clock.py 출력 JSON")
    g2.add_argument("--video", default=None, help="촬영 mp4 경로")
    g2.add_argument("--session", default=None,
                    help="레코더 세션 HDF5 (cam/meta 절대 t_ns 스탬프 원천)")
    g2.add_argument("--display-latency", type=float, default=13.0,
                    help="디스플레이 지연 차감 ms (기본 13)")

    # jitter 모드
    g3 = ap.add_argument_group("--mode jitter")
    g3.add_argument("--hdf5", default=None, help="세션 HDF5 경로")
    # --rawlog 공유

    # report 모드
    g4 = ap.add_argument_group("--mode report")
    g4.add_argument("--csi-json", default=None, help="csi 결과 JSON")
    g4.add_argument("--cam-json", default=None, help="cam 결과 JSON (선택)")
    g4.add_argument("--jitter-json", default=None, help="jitter 결과 JSON")

    args = ap.parse_args()

    if args.mode == "csi":
        if not args.port:
            ap.error("--mode csi 에는 --port 필요")
        if not args.rawlog:
            ap.error("--mode csi 에는 --rawlog 필요")
        mode_csi(args)
    elif args.mode == "cam":
        if not args.flips:
            ap.error("--mode cam 에는 --flips 필요")
        if not args.video:
            ap.error("--mode cam 에는 --video 필요")
        if not args.session:
            ap.error("--mode cam 에는 --session 필요")
        mode_cam(args)
    elif args.mode == "jitter":
        if not args.hdf5:
            ap.error("--mode jitter 에는 --hdf5 필요")
        mode_jitter(args)
    elif args.mode == "report":
        if not args.csi_json or not args.jitter_json:
            ap.error("--mode report 에는 --csi-json, --jitter-json 필요")
        mode_report(args)


if __name__ == "__main__":
    main()
