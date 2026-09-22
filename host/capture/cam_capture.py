#!/usr/bin/env python3
"""웹캠 채록 — mp4 로컬 직저장 + MQTT cam/meta {frame_idx,t_ns} 발행.

  python cam_capture.py --out ..\\sessions --session s01-r1 [--duration 600] [--no-mqtt]

t_ns는 grab 직후 호스트 시계(브리지 동일 원칙) — 노출 시점과의 오프셋은 LED 정렬 검증이 측정.
자동 노출/포커스/WB는 비활성을 시도하고 요청vs실측을 로그(§3.3 — 일부 웹캠은 set 무시).
필요 패키지: opencv-python, msgpack, paho-mqtt.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # host/
from csi_host.cam_core import CamCore  # noqa: E402


# ---------------------------------------------------------------------------
# MQTT sink — bridge.py 동형 (NullSink / MqttSink)
# ---------------------------------------------------------------------------

class NullSink:
    def publish(self, topic, payload):
        pass

    def close(self):
        pass


class MqttSink:
    def __init__(self, host, port):
        import paho.mqtt.client as mqtt
        try:  # paho 2.x
            self._c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except (AttributeError, TypeError):  # paho 1.x
            self._c = mqtt.Client()
        self._c.enable_logger()          # 연결 실패·재연결을 stderr로 가시화
        self._c.connect(host, port)      # 실패 시 예외 — 파일 없음(recorder.py 동형)
        self._c.loop_start()

    def publish(self, topic, payload):
        self._c.publish(topic, payload, qos=0)

    def close(self):
        self._c.loop_stop()


# ---------------------------------------------------------------------------
# 카메라 프롭 설정 헬퍼
# ---------------------------------------------------------------------------

def _set_and_log(cap, prop_id, prop_name, req_val):
    """set → get read-back 결과를 출력. 실패해도 진행 (일부 웹캠은 set 무시)."""
    import cv2
    ok = cap.set(prop_id, req_val)
    got = cap.get(prop_id)
    status = "ok" if ok else "ignored"
    print(f"[cam] {prop_name}: req={req_val} got={got} ({status})", flush=True)


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--camera", type=int, default=0, help="카메라 인덱스 (기본 0)")
    ap.add_argument("--backend", choices=["msmf", "dshow", "any"], default="msmf",
                    help="캡처 백엔드 (기본 msmf — 720p 30fps 협상 실측)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", default="sessions", help="출력 디렉터리")
    ap.add_argument("--session", required=True, help="세션 라벨 (파일명 일부)")
    ap.add_argument("--duration", type=float, default=None, help="녹화 시간(초) — 생략 시 Ctrl-C까지")
    ap.add_argument("--status-period", type=float, default=5.0)
    ap.add_argument("--mqtt-host", default="127.0.0.1")
    ap.add_argument("--mqtt-port", type=int, default=1883)
    ap.add_argument("--no-mqtt", action="store_true", help="MQTT 발행 생략 (mp4만 저장)")
    args = ap.parse_args()

    import cv2  # cv2는 여기서만 import — 테스트는 이 파일을 import하지 않음

    # ① sink 준비 — no-mqtt면 NullSink, 아니면 paho 연결
    #   (실패 시 예외 즉사 — 파일 없음, recorder.py 동형)
    if args.no_mqtt:
        sink = NullSink()
    else:
        sink = MqttSink(args.mqtt_host, args.mqtt_port)

    # ② CamCore
    core = CamCore(sink)

    # ③ VideoCapture 열기 — 기본 MSMF (2026-06-11 실측: DSHOW는 MJPG 거부로
    #    720p YUY2 10fps에 갇힘, MSMF는 30fps 협상). 카메라별 차이는 --backend로
    backends = {"msmf": "CAP_MSMF", "dshow": "CAP_DSHOW", "any": None}
    bk = backends[args.backend]
    if bk and hasattr(cv2, bk):
        cap = cv2.VideoCapture(args.camera, getattr(cv2, bk))
    else:
        cap = cv2.VideoCapture(args.camera)

    # ④ 프롭 설정 + 로그 — MJPG를 해상도보다 먼저 (USB2 무압축 YUY2 720p는
    #    대역폭 한계로 ~10fps 제한 — 백엔드가 거부해도 무해)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    # 버퍼 최소화 — 최신 프레임 우선
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    # 실측 read-back
    w_actual = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h_actual = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps_actual = cap.get(cv2.CAP_PROP_FPS)
    fourcc_got = int(cap.get(cv2.CAP_PROP_FOURCC)) & 0xFFFFFFFF
    fourcc_str = "".join(chr((fourcc_got >> 8 * i) & 0xFF) for i in range(4))
    print(f"[cam] 해상도: req={args.width}x{args.height} got={int(w_actual)}x{int(h_actual)}", flush=True)
    print(f"[cam] fps: req={args.fps} got={fps_actual} fourcc: req=MJPG got={fourcc_str}", flush=True)

    # 자동 노출/포커스/WB 비활성화 시도 (실패해도 진행)
    # 0.25=수동(DSHOW 관례; 0.75=자동) — 백엔드별 상이
    _set_and_log(cap, cv2.CAP_PROP_AUTO_EXPOSURE, "auto_exposure", 0.25)
    _set_and_log(cap, cv2.CAP_PROP_AUTOFOCUS, "autofocus", 0)
    _set_and_log(cap, cv2.CAP_PROP_AUTO_WB, "auto_wb", 0)

    # ⑤ 첫 프레임 read 성공 후 mp4 열기
    #   첫 read 30회 연속 실패면 에러 출력 후 exit 1
    first_frame = None
    t_first = None
    writer = None
    out_path = None
    exit_code = 0

    try:
        for attempt in range(30):
            ret, frame = cap.read()
            if ret:
                t_first = time.time_ns()  # grab 직후 즉시 — VideoWriter 초기화 지연 배제
                first_frame = frame
                break
            # 임계 = 연속 30회 × ~프레임주기 — read가 즉시 반환해도 ~1초 보장 (스핀 방지)
            time.sleep(1.0 / max(args.fps, 1.0))
        else:
            print("[cam] 오류: 카메라에서 첫 프레임을 읽지 못했습니다 (30회 실패)", file=sys.stderr, flush=True)
            exit_code = 1
            return

        # 실측 shape로 VideoWriter 생성
        h_frame, w_frame = first_frame.shape[:2]
        fps_write = fps_actual if fps_actual > 0 else args.fps  # mp4 fps는 재생 참고치 — 타이밍 진실원본은 cam/meta t_ns
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.session}-{time.strftime('%Y%m%d-%H%M%S')}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_write, (w_frame, h_frame))
        if not writer.isOpened():
            # 열기 실패 시 write()가 조용히 버려져 cam/meta만 발행되는 무증상 비대칭 실패 방지
            print(f"[cam] 오류: VideoWriter 열기 실패 — 코덱(mp4v)/경로 확인: {out_path}",
                  file=sys.stderr, flush=True)
            out_path = None              # 파일 미생성 — 종료 요약이 가리키지 않게
            exit_code = 1
            return
        print(f"[cam] 기록: {out_path}", flush=True)

        # 첫 프레임도 정상 경로로 처리 (t_first는 grab 직후에 이미 확보)
        core.handle_frame(t_first)
        writer.write(first_frame)

        # 메인 루프
        t0 = time.monotonic()
        last_status = t0
        frames_at_last = core.status()["frames"]

        while True:
            ret, frame = cap.read()
            t = time.time_ns()  # grab 직후 즉시 시각 기록

            if ret:
                core.handle_frame(t)
                writer.write(frame)
            else:
                consec = core.note_drop()
                if consec >= 30:
                    print(f"[cam] 오류: 연속 드롭 {consec}회 — 카메라 연결 끊김", file=sys.stderr, flush=True)
                    exit_code = 1
                    break
                # 임계 = 연속 30회 × ~프레임주기 — read가 즉시 반환해도 ~1초 보장 (스핀·글리치 과민 방지)
                time.sleep(1.0 / max(args.fps, 1.0))

            now = time.monotonic()

            # 주기 status 출력
            if now - last_status >= args.status_period:
                st = core.status()
                elapsed = now - last_status
                frames_delta = st["frames"] - frames_at_last
                live_fps = frames_delta / elapsed if elapsed > 0 else 0.0
                print(f"[cam] {st} fps_live={live_fps:.1f}", flush=True)
                last_status = now
                frames_at_last = st["frames"]

            # duration 만료
            if args.duration is not None and now - t0 >= args.duration:
                break

    except KeyboardInterrupt:
        print("\n[cam] stopping", flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        sink.close()
        st = core.status()
        print(f"[cam] 종료: frames={st['frames']} drops={st['drops']} → {out_path}", flush=True)
        if exit_code != 0:
            sys.exit(exit_code)


if __name__ == "__main__":
    main()
