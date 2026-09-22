"""입력 어댑터 — Replay(h5 /links 병합·페이싱) / Live(MQTT, csi_host 파서 재사용).

공통 산출 = ringbuf.Pkt. Replay는 이터레이터(__iter__), Live는 drain() 폴링
(라이브 루프가 벽시계로 절단을 구동 — demo.py).

시간 재구성: 브리지는 시리얼 청크 단위로 동일 t_ns를 찍음(실측 ~11pkt·105ms 주기).
CausalOffset으로 t̂ = esp_ns + rolling-min(t_ns − esp_ns)를 RX별로 산출, Pkt.t_ns에
대입한다. 링크 내 단조성은 링크별 직전 t̂+1ns 클램프로 보장.
heapq.merge는 링크 내 단조를 가정 — 링크 간 오프셋 차로 약간 어긋날 수 있으나 문제없음."""
import heapq
import queue
import time

import h5py
import numpy as np

from csi_host.bridge_core import unpack_csi
from csi_host.framing import parse_frame
from csi_pipe.align import amplitude
from csi_pipe.mqtt_recorder import wire_client

from .ringbuf import Pkt
from .timefit import CausalOffset

_BLK = 65536


def _link_iter(h5_path, key, rx, tx, causal_offset, last_t):
    """링크 1개 이터레이터 — t̂ 재구성 + 링크 내 단조 클램프 적용.

    causal_offset: CausalOffset 인스턴스(RX별 공유 — 같은 RX의 3링크는 동일 시리얼 경로).
    last_t: dict, 링크 키 → 직전 t̂ 추적(클램프용, 호출자가 관리).
    esp_us가 없는 h5(레거시)는 t_ns 직사용으로 폴백."""
    with h5py.File(h5_path, "r") as h:
        g = h[f"links/{key}"]
        has_esp = "esp_us" in g
        n = len(g["t_ns"])
        for b0 in range(0, n, _BLK):
            sl = slice(b0, min(b0 + _BLK, n))
            t, seq = g["t_ns"][sl], g["seq"][sl]
            boot = g["boot_id"][sl]
            amp = amplitude(g["iq"][sl])
            esp = g["esp_us"][sl] if has_esp else None
            for i in range(len(t)):
                t_ns = int(t[i])
                if has_esp:
                    esp_ns = int(esp[i]) * 1000
                    t_hat = causal_offset.estimate(int(boot[i]), t_ns, esp_ns)
                else:
                    t_hat = t_ns
                # 링크 내 단조 클램프
                prev = last_t.get(key, None)
                if prev is not None and t_hat <= prev:
                    t_hat = prev + 1
                last_t[key] = t_hat
                yield Pkt(rx=rx, tx=tx, boot_id=int(boot[i]), t_ns=t_hat,
                          seq=int(seq[i]), amp=amp[i])


class ReplaySource:
    def __init__(self, h5_path, *, speed=1.0, fast=False):
        self.h5_path, self.speed, self.fast = h5_path, float(speed), bool(fast)
        with h5py.File(h5_path, "r") as h:
            if "links" not in h or not len(h["links"]):
                raise SystemExit(f"/links 없음: {h5_path} — 레코더 h5인지 확인")
            self._keys = sorted(h["links"])

    def __iter__(self):
        # RX별 CausalOffset — 같은 RX의 3링크는 동일 시리얼 경로이므로 공유
        fits = {rx: CausalOffset() for rx in range(3)}
        last_t = {}  # 링크 key → 직전 t̂ (단조 클램프용)
        its = [_link_iter(self.h5_path, k, int(k[0]), int(k[1]),
                          fits[int(k[0])], last_t)
               for k in self._keys]
        # heapq.merge: 링크 내 단조는 클램프로 보장; 링크 간 순서는 RingBuf에 무관
        merged = heapq.merge(*its, key=lambda p: p.t_ns)
        t0 = wall0 = None
        for p in merged:
            if not self.fast:
                if t0 is None:
                    t0, wall0 = p.t_ns, time.perf_counter()
                lag = (p.t_ns - t0) / 1e9 / self.speed - (time.perf_counter() - wall0)
                if lag > 0.001:
                    time.sleep(lag)
            yield p


def _iq_to_amp(iq):
    """CsiFrame.iq (bytes 또는 ndarray) → amplitude (56,) f32.

    실시간 경로: parse_frame이 반환하는 CsiFrame.iq는 112 bytes(i8×112).
    테스트 경로: mock F.iq는 np.ndarray (56,2) int8. 양쪽 모두 처리."""
    if isinstance(iq, (bytes, bytearray)):
        arr = np.frombuffer(iq, dtype=np.int8).reshape(56, 2)
    else:
        arr = np.asarray(iq, dtype=np.int8)
        if arr.ndim == 1:
            arr = arr.reshape(56, 2)
    return amplitude(arr[None])[0]


class LiveSource:
    def __init__(self, *, host, port=1883):
        import paho.mqtt.client as mqtt
        self._init_queue()
        try:                                         # paho 2.x
            self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except (AttributeError, TypeError):          # paho 1.x
            self._client = mqtt.Client()
        self._client.enable_logger()
        wire_client(self._client, self._on_message,
                    subscriptions=[("csi/#", 0)])
        try:
            self._client.connect(host, port, keepalive=30)
        except OSError as e:
            raise SystemExit(f"MQTT 접속 실패 {host}:{port} ({e}) — Windows "
                             "mosquitto면 --mqtt-host에 호스트 IP 지정")
        self._client.loop_start()

    def _init_queue(self):
        from csi_host.unwrap import TimeUnwrapper
        self._q = queue.Queue()
        self.crc_drops = 0
        # RX별 인과 시간 재구성 상태 — Replay와 동일 전략
        self._uw = {}           # rx_id → TimeUnwrapper (u32 랩 해제)
        self._fit = {}          # rx_id → CausalOffset
        self._last_t = {}       # (rx, tx) → 직전 t̂ (링크 내 단조 클램프용)
        self._TimeUnwrapper = TimeUnwrapper

    def _on_message(self, client, userdata, msg):
        if not msg.topic.startswith("csi/"):
            return
        try:
            t_ns, raw = unpack_csi(msg.payload)
            f = parse_frame(raw)
        except Exception:
            self.crc_drops += 1
            return
        if f is None:
            self.crc_drops += 1
            return
        rx = f.rx_id
        # esp_timer_us 언랩
        uw = self._uw.setdefault(rx, self._TimeUnwrapper())
        u, _ = uw.update(boot_id=f.boot_id, t_us=f.esp_timer_us)
        esp_ns = u * 1000
        # RX별 CausalOffset으로 t̂ 산출
        fit = self._fit.setdefault(rx, CausalOffset())
        t_hat = fit.estimate(f.boot_id, t_ns, esp_ns)
        # 링크 내 단조 클램프
        lk = (f.rx_id, f.tx_idx)
        prev = self._last_t.get(lk)
        if prev is not None and t_hat <= prev:
            t_hat = prev + 1
        self._last_t[lk] = t_hat
        self._q.put(Pkt(rx=f.rx_id, tx=f.tx_idx, boot_id=f.boot_id, t_ns=t_hat,
                        seq=f.seq, amp=_iq_to_amp(f.iq)))

    def drain(self):
        out = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                return out

    def close(self):
        self._client.loop_stop()
        self._client.disconnect()
