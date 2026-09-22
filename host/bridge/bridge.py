#!/usr/bin/env python3
"""시리얼 → raw 로그 + MQTT 브리지 (Windows 네이티브 실행, 설계 §5).

RX 보드 COM 포트 1개당 1프로세스:
  python bridge.py --port COM24 --rx-id 0 [--no-mqtt] [--raw-dir logs] [--auto-start]

원칙(§5): 타임스탬프는 시리얼 read 직후의 time.time_ns() — 단일 호스트 시계.
append-only raw 로그가 원본 — MQTT/레코더가 죽어도 HDF5 재구축 가능.
COM 단절 시 1초 백오프 자동 재접속, 상태는 5초마다 sys/status/rxN 발행.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # host/ → csi_host
from csi_host.bridge_core import BridgeCore  # noqa: E402


class NullSink:
    def publish(self, topic, payload):
        pass


class MqttSink:
    def __init__(self, host, port):
        import paho.mqtt.client as mqtt
        try:  # paho 2.x
            self._c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except (AttributeError, TypeError):  # paho 1.x
            self._c = mqtt.Client()
        self._c.connect(host, port)
        self._c.loop_start()

    def publish(self, topic, payload):
        self._c.publish(topic, payload, qos=0)  # csi/* QoS 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True, help="예: COM24")
    ap.add_argument("--rx-id", type=int, required=True, choices=[0, 1, 2])
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--mqtt-host", default="127.0.0.1")
    ap.add_argument("--mqtt-port", type=int, default=1883)
    ap.add_argument("--no-mqtt", action="store_true", help="raw 로그만 (MQTT 생략)")
    ap.add_argument("--raw-dir", default="logs")
    ap.add_argument("--auto-start", action="store_true",
                    help="접속(재접속 포함) 시 START 송신 — seq 리셋은 reset 이벤트로 처리됨")
    ap.add_argument("--status-period", type=float, default=5.0)
    args = ap.parse_args()

    import serial  # pyserial

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"rx{args.rx_id}-{time.strftime('%Y%m%d-%H%M%S')}.rawlog"

    sink = NullSink() if args.no_mqtt else MqttSink(args.mqtt_host, args.mqtt_port)

    def on_event(kind, val):
        if kind in ("text", "reboot", "wrap"):
            print(f"[rx{args.rx_id}] {kind}: {val}", flush=True)

    core = BridgeCore(rx_id=args.rx_id, raw_path=raw_path, sink=sink, on_event=on_event)
    print(f"[rx{args.rx_id}] raw log: {raw_path}", flush=True)

    reconnects = 0
    last_status = time.monotonic()
    ser = None
    try:
        while True:
            try:
                ser = serial.Serial(args.port, args.baud, timeout=0.05)
            except (serial.SerialException, OSError) as e:
                print(f"[rx{args.rx_id}] open failed: {e}; retry 1s", flush=True)
                reconnects += 1
                time.sleep(1)
                continue
            try:
                # Windows 드라이버 수신 버퍼 증설 — 호스트 일시 멈춤(~수 초)에도
                # 바이트 유실 방지 (2026-06-10 소크: 3포트 동시 CRC = 호스트 정체).
                # 39KB/s × 256KB ≈ 6.5s 내성. 비Windows는 미지원이라 무시.
                ser.set_buffer_size(rx_size=262144)
            except (AttributeError, OSError):
                pass
            print(f"[rx{args.rx_id}] {args.port} open", flush=True)
            if args.auto_start:
                ser.write(b"START\n")
            try:
                while True:
                    chunk = ser.read(4096)
                    if chunk:
                        core.ingest(time.time_ns(), chunk)  # 직후 단일 시계
                    now = time.monotonic()
                    if now - last_status >= args.status_period:
                        last_status = now
                        st = core.status()
                        st["reconnects"] = reconnects
                        line = json.dumps(st, separators=(",", ":"))
                        sink.publish(f"sys/status/rx{args.rx_id}", line.encode())
                        print(f"[rx{args.rx_id}] {line}", flush=True)
            except (serial.SerialException, OSError) as e:
                print(f"[rx{args.rx_id}] serial error: {e}; reconnect", flush=True)
                reconnects += 1
                try:
                    ser.close()
                except Exception:
                    pass
                ser = None
                time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n[rx{args.rx_id}] stopping", flush=True)
        if ser is not None:
            try:
                ser.write(b"STOP\n")
                ser.close()
            except Exception:
                pass
    finally:
        core.close()
        print(json.dumps(core.status(), indent=1))


if __name__ == "__main__":
    main()
