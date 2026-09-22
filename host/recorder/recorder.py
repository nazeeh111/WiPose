#!/usr/bin/env python3
"""라이브 레코더 — MQTT(csi/rx*, cam/meta) 구독 → 세션 HDF5 (윈도우 네이티브).

  python recorder.py --out ..\\sessions --session s01-r1 [--duration 600]

원본은 여전히 브리지 rawlog — 레코더가 죽어도 rawlog_to_hdf5로 재구축 가능.
필요 패키지: paho-mqtt, numpy, h5py (requirements.txt 참고).
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # host/
from csi_pipe.mqtt_recorder import RecorderCore, wire_client  # noqa: E402
from csi_pipe.store import SessionWriter  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mqtt-host", default="127.0.0.1")
    ap.add_argument("--mqtt-port", type=int, default=1883)
    ap.add_argument("--out", default="sessions", help="세션 디렉터리")
    ap.add_argument("--session", required=True, help="세션 라벨 (파일명)")
    ap.add_argument("--duration", type=float, default=None, help="초 — 생략 시 Ctrl-C까지")
    ap.add_argument("--status-period", type=float, default=5.0)
    args = ap.parse_args()

    import paho.mqtt.client as mqtt

    try:  # paho 2.x
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except (AttributeError, TypeError):  # paho 1.x
        client = mqtt.Client()
    client.enable_logger()                  # 연결 실패·재연결을 stderr로 가시화
    client.connect(args.mqtt_host, args.mqtt_port)   # 파일 생성 전 — 실패 시 빈 .h5 안 남김

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.session}-{time.strftime('%Y%m%d-%H%M%S')}.h5"
    writer = SessionWriter(path, meta={"session": args.session})
    core = RecorderCore(writer, on_event=lambda k, v: print(f"[rec] {k}: {v}", flush=True))

    # paho는 콜백 예외를 삼킨다 — handle()이 실패하면 무증상 무기록 세션이 됨.
    # 첫 예외만 상세 출력 후 종료 플래그를 세워 메인 루프가 비정상 종료하도록 함.
    _error_flag = [None]   # [예외] 또는 [None] — 리스트로 스레드 간 공유

    def _on_message(client, userdata, msg):
        if _error_flag[0] is not None:
            return                          # 이미 종료 예정 — 추가 처리 생략
        try:
            core.handle(msg.topic, msg.payload, time.time_ns())
        except Exception as exc:           # noqa: BLE001
            # 첫 예외만 자세히 출력 — paho가 삼키기 전에 stderr에 기록
            print("\n[rec] 치명적 오류 — handle() 예외 (HDF5 쓰기 실패 등):", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            _error_flag[0] = exc

    # wire_client는 connect() 후, loop_start() 전에 호출함.
    # CONNACK은 loop_start()의 네트워크 스레드에서 처리되므로,
    # 그 시점에 on_connect → subscribe 발화 — 이 순서를 지켜야 첫 연결 구독 누락을 막음.
    wire_client(client, _on_message, log=lambda msg: print(msg, flush=True))
    client.loop_start()
    print(f"[rec] 기록: {path}", flush=True)

    t0 = time.monotonic()
    last = t0
    exit_code = 0
    try:
        while True:
            time.sleep(0.2)
            # handle() 예외 → 즉시 비정상 종료
            if _error_flag[0] is not None:
                print("[rec] handle() 예외 감지 — 비정상 종료", file=sys.stderr, flush=True)
                exit_code = 1
                break
            now = time.monotonic()
            if now - last >= args.status_period:
                last = now
                writer.flush()                      # 주기 flush (스펙 오류 처리)
                print(f"[rec] {core.status()}", flush=True)
            if args.duration is not None and now - t0 >= args.duration:
                break    # is not None: --duration 0도 즉시 종료 (falsy 가드는 무한 기록 버그)
    except KeyboardInterrupt:
        print("\n[rec] stopping", flush=True)
    finally:
        client.loop_stop()
        writer.set_meta("recorder_status", str(core.status()))
        writer.close()
        print(f"[rec] 종료: frames={core.frames} crc_drops={core.crc_drops} → {path}",
              flush=True)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
