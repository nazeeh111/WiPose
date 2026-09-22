"""rt 데모 CLI — 리플레이/라이브 조립 + 성능 요약 JSON.

절단 구동: 리플레이 = 데이터 시계(패킷 t_ns), 라이브 = 벽시계. 절단 경계 B는
50ms 정수배(10ms 슬롯 위상 일치), (now − settle) ≥ B에서 B 윈도 절단."""
import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for sub in ("host", "train", "teacher", "rt"):
    p = str(_HERE.parent / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from csi_rt.config import load_rt_config            # noqa: E402
from csi_rt.fall import FallDetector                # noqa: E402
from csi_rt.infer import PoseEstimator              # noqa: E402
from csi_rt.overlay import CANVAS_WH, render        # noqa: E402
from csi_rt.presence import MotionEnergy, PresenceGate  # noqa: E402
from csi_rt.ringbuf import RingBuf                  # noqa: E402
from csi_rt.smooth import EmaSmoother               # noqa: E402
from csi_rt.sources import LiveSource, ReplaySource  # noqa: E402
from csi_rt.video import ReplayVideo                # noqa: E402

TICK_NS = 50_000_000
_WND = "rt"
_STOP_KEYS = (27, ord("q"))        # ESC·q — 라이브 창 종료 키


class Engine:
    def __init__(self, est, cfg, *, force_present, video=None):
        self.est, self.cfg, self.video = est, cfg, video
        self._wall0 = time.perf_counter()
        self.ring = RingBuf()
        self.motion = MotionEnergy(window_s=cfg["motion_window_s"])
        self.gate = PresenceGate(cfg["tau_presence"], force=force_present)
        self.ema = EmaSmoother(alpha=cfg["ema_alpha"])
        self.fall = FallDetector(cfg["fall"], CANVAS_WH)
        self.frames_in = self.windows = self.valid_windows = 0
        self.dropped = self.alarms = 0
        self.catchup_windows = 0       # 라이브 따라잡기 틱(e2e 미표본) 카운트
        self.infer_ms, self.e2e_ms = [], []
        self._last = (None, None, False, "IDLE")     # xy, c, present, state

    def add(self, pkt):
        self.frames_in += 1
        self.ring.add(pkt)
        self.motion.add(pkt.rx, pkt.tx, pkt.t_ns, pkt.amp)

    def tick(self, B_ns, *, wall_ns=None):
        self.windows += 1
        cut = self.ring.cut(B_ns)
        xy, c, present, state = self._last
        if cut.valid:
            self.valid_windows += 1
            xy_r, c = self.est(cut.X.astype(np.float32))
            self.infer_ms.append(self.est.infer_ms)
            present = self.gate.update(c)
            xy = self.ema.update(xy_r, present=present)
            out = self.fall.update(B_ns / 1e9, xy if xy is not None else xy_r,
                                   c, present, self.motion.energy())
            state = out.state
            self.alarms += int(out.fired)
            self._last = (xy, c, present, state)
        else:
            self.dropped += 1
        if wall_ns is not None:
            # e2e는 t̂ 기준 — 청크 배칭 지터(p50 ~55ms)만큼 보수(과대). 과소보고 없음 — 게이트 안전측
            self.e2e_ms.append((wall_ns - B_ns) / 1e6)
        hud = {"fps": self.windows / max(time.perf_counter() - self._wall0, 1e-9),
               "infer_ms": float(np.mean(self.infer_ms[-20:]) if
                                 self.infer_ms else 0.0),
               "e2e_ms": float(self.e2e_ms[-1]) if self.e2e_ms else 0.0,
               "drop": self.dropped, "motion": self.motion.energy(),
               "random": self.est.random}
        vf = self.video.frame_for(B_ns) if self.video else None
        return render(vf, xy, c, present=present, fall_state=state, hud=hud)


def _pct(v, q):
    return float(np.percentile(v, q)) if v else 0.0


def _emit(engine, args, wall_s):
    d = {"mode": "live" if args.live else "replay",
         "random_weights": engine.est.random, "frames_in": engine.frames_in,
         "windows": engine.windows, "valid_windows": engine.valid_windows,
         "dropped_windows": engine.dropped,
         "fps_mean": engine.windows / wall_s if wall_s > 0 else 0.0,
         "infer_ms_p50": _pct(engine.infer_ms, 50),
         "infer_ms_p95": _pct(engine.infer_ms, 95),
         "e2e_ms_p50": _pct(engine.e2e_ms, 50),
         "e2e_ms_p95": _pct(engine.e2e_ms, 95), "alarms": engine.alarms,
         "catchup_windows": engine.catchup_windows}
    Path(args.perf_out).write_text(json.dumps(d, indent=1, ensure_ascii=False),
                                   encoding="utf-8")
    print(f"[rt] perf → {args.perf_out}: {d}")
    return d


def _user_stop(key, visible):
    """사용자 종료 요청 — ESC/q 키 또는 창 X 닫힘(visible<1)."""
    return key in _STOP_KEYS or visible < 1


def _show(args, frame, writer):
    """표시/녹화 1틱. 반환 True = 사용자 종료 요청 — 호출측은 루프 탈출(마무리 경로 보존)."""
    if writer is not None:
        writer.write(frame)
    if args.headless:
        return False
    import cv2
    try:
        visible = cv2.getWindowProperty(_WND, cv2.WND_PROP_VISIBLE)
    except cv2.error:              # Qt: 마지막 창 X 닫힘 → guiReceiver 소멸 → 조회 자체가 throw
        return True
    if visible >= 1:
        cv2.imshow(_WND, frame)    # X 닫힘(visible<1) 시 재생성 금지 — imshow가 창을 되살림
    return _user_stop(cv2.waitKey(1) & 0xFF, visible)


def _make_writer(args, frame):
    if not args.save:
        return None
    import cv2
    return cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 20,
                           (frame.shape[1], frame.shape[0]))


def run(argv=None):
    ap = argparse.ArgumentParser(description="rt 실시간/리플레이 데모 (M3)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--replay", metavar="H5")
    src.add_argument("--live", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--random-weights", action="store_true")
    ap.add_argument("--config", default=str(_HERE.parent / "configs" / "rt.yaml"))
    ap.add_argument("--video", metavar="MP4")
    ap.add_argument("--pairing", default=str(_HERE.parent / "configs" / "pairing.json"))
    ap.add_argument("--speed", type=float, default=None)
    ap.add_argument("--fast", action="store_true", help="리플레이 무페이싱(테스트)")
    ap.add_argument("--mqtt-host")
    ap.add_argument("--save", metavar="MP4")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--perf-out", default=str(_HERE.parent / "host" / "logs"
                                              / "rt_perf.json"))
    ap.add_argument("--duration", type=float, default=0.0,
                    help="라이브 실행 시간(s, 0=무한)")
    args = ap.parse_args(argv)
    if args.live and args.fast:
        ap.error("--fast는 리플레이 전용")
    if args.fast and args.speed is not None:
        ap.error("--speed는 --fast와 동시 지정 불가")
    if not args.random_weights and not args.ckpt:
        ap.error("--ckpt 필요 (M2 전 파이프 스모크는 --random-weights)")  # rc 2
    if args.video and args.live:
        raise SystemExit("--video 라이브: cam jpg 토픽 미발행 — 캔버스 모드 사용"
                         " (스펙 §오버레이)")

    cfg = load_rt_config(args.config)
    est = PoseEstimator(args.ckpt, random_weights=args.random_weights)
    video = ReplayVideo(args.replay, args.video, args.pairing) if args.video else None
    engine = Engine(est, cfg, force_present=args.random_weights, video=video)
    settle_ns = int(cfg["settle_ms"] * 1e6)
    writer, wall0 = None, time.perf_counter()
    if not args.headless:
        import cv2
        cv2.namedWindow(_WND)      # 첫 틱 전 생성 — getWindowProperty 기반 X 감지 전제

    if args.replay:
        source = ReplaySource(args.replay,
                              speed=args.speed if args.speed is not None else 1.0,
                              fast=args.fast)
        B, stop = None, False
        for pkt in source:
            engine.add(pkt)
            if B is None:
                B = (pkt.t_ns // TICK_NS + 2) * TICK_NS
            while not stop and pkt.t_ns - settle_ns >= B:
                frame = engine.tick(B)
                if writer is None:
                    writer = _make_writer(args, frame)
                stop = _show(args, frame, writer)
                B += TICK_NS
            if stop:
                break
    else:
        source = LiveSource(host=args.mqtt_host or cfg["mqtt"]["host"],
                            port=cfg["mqtt"]["port"])
        B, t_end = None, (time.time() + args.duration if args.duration else None)
        stop = False
        try:
            while not stop and (t_end is None or time.time() < t_end):
                for pkt in source.drain():
                    engine.add(pkt)
                    if B is None:
                        B = (pkt.t_ns // TICK_NS + 2) * TICK_NS
                now = time.time_ns()
                while not stop and B is not None and now - settle_ns >= B:
                    current = (now - settle_ns) < B + TICK_NS   # 마지막(현재) 틱만 e2e 표본
                    if not current:
                        engine.catchup_windows += 1             # stale 틱 — e2e 미표본(p95 보호)
                    frame = engine.tick(B, wall_ns=time.time_ns() if current else None)
                    if writer is None:
                        writer = _make_writer(args, frame)
                    stop = _show(args, frame, writer)
                    B += TICK_NS
                time.sleep(0.01)
        except KeyboardInterrupt:
            pass
        finally:
            source.close()

    if writer is not None:
        writer.release()
    _emit(engine, args, time.perf_counter() - wall0)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except SystemExit as e:                          # 메시지형 fail-loud → rc 2 통일
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr)
            sys.exit(2)
        raise
