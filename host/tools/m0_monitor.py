#!/usr/bin/env python3
"""MQTT 9링크 모니터 + M0 수용 판정.

  python m0_monitor.py --duration 600 --strict                    # 10분 소크
  python m0_monitor.py --duration 4800 --strict --require-wrap    # 80분 — u32 랩 경계 통과

판정(--strict): 9링크 전부 평균 pps ≥ 95, loss < 5% (6+연속·폐기율 포함
전체 판정은 soak_report 전담), CRC 오류 0 → exit 0/1.
--require-wrap: 추가로 rx 3대 모두 esp_timer 랩 ≥ 1회 관측 + unwrap 단조 위반 0건.
"""
import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csi_host.bridge_core import unpack_csi  # noqa: E402
from csi_host.framing import parse_frame     # noqa: E402
from csi_host.gap import LinkTracker         # noqa: E402
from csi_host.unwrap import TimeUnwrapper    # noqa: E402
from csi_pipe.mqtt_recorder import wire_client  # noqa: E402


class Monitor:
    def __init__(self):
        self.lock = threading.Lock()
        self.links = {}        # (rx, tx) -> LinkTracker
        self.interval_cnt = {} # (rx, tx) -> count (주기 출력용)
        self.unwrappers = {}   # rx -> TimeUnwrapper
        self.last_unwrapped = {}
        self.mono_violations = {}
        self.crc_errors = 0
        self.t0 = time.monotonic()

    def on_frame(self, payload):
        try:
            _, raw = unpack_csi(payload)
        except Exception:
            return
        f = parse_frame(raw)
        with self.lock:
            if f is None:
                self.crc_errors += 1
                return
            key = (f.rx_id, f.tx_idx)
            self.links.setdefault(key, LinkTracker()).update(f.seq)
            self.interval_cnt[key] = self.interval_cnt.get(key, 0) + 1
            u = self.unwrappers.setdefault(f.rx_id, TimeUnwrapper())
            t_unw, ev = u.update(boot_id=f.boot_id, t_us=f.esp_timer_us)
            if ev == "reboot":
                self.last_unwrapped.pop(f.rx_id, None)
                for (rx, _), tr in self.links.items():
                    if rx == f.rx_id:
                        tr.rebaseline()
            last = self.last_unwrapped.get(f.rx_id)
            if last is not None and t_unw < last:
                self.mono_violations[f.rx_id] = self.mono_violations.get(f.rx_id, 0) + 1
            self.last_unwrapped[f.rx_id] = t_unw

    def table(self, interval):
        with self.lock:
            lines = ["link   " + "".join(f"tx{j}      " for j in range(3)) + "lost  loss%   wraps"]
            for i in range(3):
                cells = []
                for j in range(3):
                    pps = self.interval_cnt.get((i, j), 0) / interval
                    cells.append(f"{pps:7.1f}p ")
                lost = sum(tr.lost for (rx, _), tr in self.links.items() if rx == i)
                rxn = sum(tr.received for (rx, _), tr in self.links.items() if rx == i)
                loss = lost / (lost + rxn) * 100 if (lost + rxn) else 0.0
                wraps = self.unwrappers[i].wraps if i in self.unwrappers else 0
                lines.append(f"rx{i}   " + "".join(cells) + f"{lost:5d} {loss:6.2f}%  {wraps}")
            self.interval_cnt.clear()
            lines.append(f"crc_err={self.crc_errors} elapsed={time.monotonic() - self.t0:.0f}s")
            return "\n".join(lines)

    def verdict(self, *, pps_min, loss_max, require_wrap, elapsed):
        with self.lock:
            problems = []
            for i in range(3):
                for j in range(3):
                    tr = self.links.get((i, j))
                    if tr is None:
                        problems.append(f"link rx{i}-tx{j}: no frames")
                        continue
                    avg = tr.received / elapsed
                    if avg < pps_min:
                        problems.append(f"link rx{i}-tx{j}: avg pps {avg:.1f} < {pps_min}")
                    if tr.loss_ratio >= loss_max:
                        problems.append(f"link rx{i}-tx{j}: loss {tr.loss_ratio:.3%} >= {loss_max:.0%}")
            if self.crc_errors:
                problems.append(f"crc errors: {self.crc_errors}")
            if require_wrap:
                for i in range(3):
                    u = self.unwrappers.get(i)
                    if u is None or u.wraps < 1:
                        problems.append(f"rx{i}: esp_timer wrap 미관측 (가동 71.6분 미만?)")
                    if self.mono_violations.get(i):
                        problems.append(f"rx{i}: unwrap 단조 위반 {self.mono_violations[i]}건")
            return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mqtt-host", default="127.0.0.1")
    ap.add_argument("--mqtt-port", type=int, default=1883)
    ap.add_argument("--duration", type=float, default=0, help="초 (0=무한, 판정 없음)")
    ap.add_argument("--strict", action="store_true", help="종료 시 M0 판정 → exit 0/1")
    ap.add_argument("--require-wrap", action="store_true",
                    help="v1.3: rx 3대 wrap≥1 + unwrap 단조 위반 0 요구 (75분+ 소크)")
    ap.add_argument("--pps", type=float, default=95.0)
    ap.add_argument("--loss", type=float, default=0.05)
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args()

    import paho.mqtt.client as mqtt

    mon = Monitor()

    def on_message(client, userdata, msg):
        if msg.topic.startswith("csi/"):
            mon.on_frame(msg.payload)
        elif msg.topic.startswith("sys/status/"):
            pass  # 브리지 자체 보고 — 모니터는 프레임에서 독립 집계 (교차 검증)

    try:  # paho 2.x
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except (AttributeError, TypeError):
        client = mqtt.Client()
    client.connect(args.mqtt_host, args.mqtt_port)
    # on_connect 안에서 재구독 → 브로커 재시작 후 재연결 시 자동 재구독 보장
    wire_client(client, on_message, subscriptions=[("csi/#", 0), ("sys/status/#", 0)])
    client.loop_start()

    t_end = time.monotonic() + args.duration if args.duration > 0 else None
    try:
        while t_end is None or time.monotonic() < t_end:
            time.sleep(args.interval)
            print(mon.table(args.interval), flush=True)
            print(flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()

    elapsed = time.monotonic() - mon.t0
    if args.strict:
        problems = mon.verdict(pps_min=args.pps, loss_max=args.loss,
                               require_wrap=args.require_wrap, elapsed=elapsed)
        if problems:
            print("M0 FAIL:")
            for p in problems:
                print(f"  - {p}")
            sys.exit(1)
        print(f"M0 PASS ({elapsed:.0f}s, 9링크 pps≥{args.pps}, loss<{args.loss:.0%}, crc=0"
              + (", wrap 검증 포함" if args.require_wrap else "") + ")")
        sys.exit(0)


if __name__ == "__main__":
    main()
