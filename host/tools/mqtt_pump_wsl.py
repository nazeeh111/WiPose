"""Windows→WSL MQTT 중계 — WSL NAT 인바운드 차단 우회 (rt 라이브 데모용).

배경: mosquitto 서비스는 127.0.0.1 전용 바인딩이고 Windows 방화벽이
WSL→호스트 인바운드를 막아(실측 시), WSL rt 데모가 직접 구독할 수 없다.
Windows→WSL 아웃바운드는 허용되므로 방향을 뒤집는다: 이 스크립트가
mosquitto(127.0.0.1:1883)의 csi/#를 구독해 WSL 측 브로커(amqtt)로 재발행.

사용 (Windows, Espressif python — bridge.py와 동일 env):
    python tools\\mqtt_pump_wsl.py --wsl-host <WSL eth0 IP>

WSL 측 브로커는 /tmp/mqtt-broker-venv/run_broker.py (임시 — WSL 재시작 시 소멸).
영구 해법은 .wslconfig networkingMode=mirrored (rt.yaml 주석 전제) 또는
mosquitto conf 리스너 추가+방화벽 규칙(관리자).
"""
import argparse
import sys

import paho.mqtt.client as mqtt


def make_client():
    try:                                         # paho 2.x
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except (AttributeError, TypeError):          # paho 1.x
        return mqtt.Client()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wsl-host", required=True, help="WSL eth0 IP (wsl에서 `ip addr show eth0`로 확인)")
    ap.add_argument("--src-host", default="127.0.0.1", help="mosquitto 호스트")
    ap.add_argument("--topic", default="csi/#")
    args = ap.parse_args()

    sub, pub = make_client(), make_client()
    relayed = [0]

    def on_connect(client, userdata, flags, rc, *extra):
        print(f"[pump] sub connected rc={rc} → subscribe {args.topic}", flush=True)
        client.subscribe(args.topic, qos=0)

    def on_message(client, userdata, msg):
        pub.publish(msg.topic, msg.payload, qos=0)
        relayed[0] += 1
        if relayed[0] % 3000 == 0:
            print(f"[pump] relayed {relayed[0]}", flush=True)

    sub.on_connect = on_connect
    sub.on_message = on_message

    try:
        pub.connect(args.wsl_host, 1883, keepalive=30)
    except OSError as e:
        sys.exit(f"[pump] WSL 브로커 접속 실패 {args.wsl_host}:1883 ({e}) — "
                 "WSL에서 run_broker.py 가동 여부 확인")
    pub.loop_start()

    try:
        sub.connect(args.src_host, 1883, keepalive=30)
    except OSError as e:
        sys.exit(f"[pump] mosquitto 접속 실패 {args.src_host}:1883 ({e})")
    print("[pump] start", flush=True)
    sub.loop_forever()


if __name__ == "__main__":
    main()
