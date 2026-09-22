"""RecorderCore — MQTT 메시지 → SessionWriter.

MQTT 클라이언트는 주입(브리지 BridgeCore 패턴) — handle(topic, payload)만 호출하면
되므로 가짜 클라이언트로 테스트. CRC는 parse_frame에서 재검증(전송 경로 불신)."""
from csi_host.bridge_core import unpack_csi
from csi_host.framing import parse_frame
from csi_host.gap import LinkTracker
from csi_host.unwrap import TimeUnwrapper

try:
    import msgpack
except ImportError:
    msgpack = None


# 레코더가 구독할 기본 토픽 목록 — wire_client의 기본값이자 외부 참조용 상수.
SUBSCRIPTIONS = [("csi/#", 0), ("cam/meta", 0)]


def wire_client(client, on_message, *, subscriptions=None, log=None):
    """클라이언트에 콜백을 설치한다. loop_start() 전에 호출할 것
    (CONNACK은 네트워크 루프에서 처리되므로 connect() 전후 모두 가능).

    구독을 on_connect 안에서 수행해야 브로커 재시작 후 자동 재연결 시
    재구독이 보장된다 — connect() 한 번만 호출되는 곳에서 subscribe()를
    직접 호출하면 재연결 후 구독이 복원되지 않아 조용히 0프레임 세션이 된다.

    paho 1.x(인자 4개)·2.x VERSION2(인자 5개) 시그니처를 모두 지원한다.
    _on_connect 내부에서 예외를 던지면 안 된다: paho는 콜백 예외를 재전파해
    네트워크 스레드가 조용히 죽는다(레코더의 _error_flag 경로를 타지 않음).
    """
    _subs = subscriptions if subscriptions is not None else SUBSCRIPTIONS

    def _on_connect(client, userdata, flags, rc, properties=None):
        # paho는 rc≠0(연결 거부)에도 on_connect를 호출 — 거부 시 subscribe는 무의미.
        # rc는 1.x=int, 2.x=ReasonCode지만 ReasonCode.__eq__가 int 비교를 지원해
        # rc != 0이 양쪽 호환.
        if rc != 0:
            if log is not None:
                log(f"[rec] on_connect 거부 rc={rc} — 재구독 생략")
            return
        # 연결·재연결 시마다 재구독 — 브로커 재시작 내성의 핵심
        client.subscribe(_subs)
        if log is not None:
            log(f"[rec] on_connect: 재구독 {_subs}")

    client.on_connect = _on_connect
    client.on_message = on_message


class RecorderCore:
    def __init__(self, writer, *, on_event=None):
        self.writer = writer
        self.on_event = on_event
        self._unwrap = {}    # rx_id -> TimeUnwrapper
        self._links = {}     # (rx,tx) -> LinkTracker
        self.frames = 0
        self.crc_drops = 0
        self.cam_frames = 0
        self.cam_errors = 0
        self.unknown = 0
        self.reboots = 0
        self.wraps = 0       # u32 랩 누적 횟수 (rx별 합산)

    def handle(self, topic, payload, t_recv_ns=0):
        # t_recv_ns: 레코더 수신 시각 — 브리지가 찍은 호스트 시각(unpack_csi 페이로드에 포함)을
        # 쓰므로 이 값은 보조 참고용일 뿐이며 저장 경로에 사용되지 않음.
        if topic.startswith("csi/"):
            self._on_csi(payload)
        elif topic == "cam/meta":
            self._on_cam(payload)
        elif topic.startswith("sys/"):
            pass                                    # 하트비트 — 저장 대상 아님
        else:
            self.unknown += 1

    def _on_csi(self, payload):
        try:
            t_ns, raw = unpack_csi(payload)
            f = parse_frame(raw)            # CRC 재검증 포함 — 실패 시 None 또는 예외
        except Exception:
            self.crc_drops += 1
            return
        if f is None:
            self.crc_drops += 1
            return
        uw = self._unwrap.setdefault(f.rx_id, TimeUnwrapper())
        u, ev = uw.update(boot_id=f.boot_id, t_us=f.esp_timer_us)
        if ev == "reboot":
            self.reboots += 1
            for (i, _), tr in self._links.items():
                if i == f.rx_id:
                    tr.rebaseline()
            self._emit("reboot", (f.rx_id, f.boot_id))
        elif ev == "wrap":
            self.wraps += 1
        tr = self._links.setdefault((f.rx_id, f.tx_idx), LinkTracker())
        tr.update(f.seq)
        self.writer.append(rx_id=f.rx_id, tx_idx=f.tx_idx, t_ns=t_ns, esp_us=u,
                           iq=f.iq, seq=f.seq, rssi=f.rssi, noise=f.noise_floor,
                           boot_id=f.boot_id)
        self.frames += 1

    def _on_cam(self, payload):
        if msgpack is None:
            self.cam_errors += 1
            return
        try:
            d = msgpack.unpackb(payload)
            if not isinstance(d, dict):
                raise ValueError("cam/meta가 dict 아님")
            t = d.get("t_ns", d.get(b"t_ns"))
            fi = d.get("frame_idx", d.get(b"frame_idx"))
            if t is None or fi is None:
                raise ValueError("t_ns/frame_idx 누락")
            self.writer.append_video(int(t), int(fi))
            self.cam_frames += 1
        except Exception:
            self.cam_errors += 1

    def _emit(self, kind, val):
        if self.on_event:
            self.on_event(kind, val)

    def status(self):
        # links 키는 "rx-tx" — 브리지는 1브리지=1rx라 tx만 키, 리코더는 멀티 rx라 둘 다 필요
        return {
            "frames": self.frames, "crc_drops": self.crc_drops,
            "cam_frames": self.cam_frames, "cam_errors": self.cam_errors,
            "unknown": self.unknown, "reboots": self.reboots,
            "wraps": self.wraps,
            "links": {f"{k[0]}-{k[1]}": {"rx": tr.received, "lost": tr.lost,
                                         "resets": tr.resets,
                                         "loss": round(tr.loss_ratio, 5)}
                      for k, tr in sorted(self._links.items())},
        }
