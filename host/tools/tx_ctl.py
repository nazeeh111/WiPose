#!/usr/bin/env python3
"""보드 일괄 시리얼 명령 (MAC 채록, SET_IDX/SET_CH/START/STOP/SCAN).

예:
  python tx_ctl.py --ports COM34 COM35 COM36 --cmd HELLO
  python tx_ctl.py --ports COM34 --cmd "SET_IDX 0"
  python tx_ctl.py --ports COM34 COM35 COM36 --cmd "START rate=100"
"""
import argparse
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ports", nargs="+", required=True)
    ap.add_argument("--cmd", required=True)
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--wait", type=float, default=1.0, help="응답 대기 초 (SCAN은 5 권장)")
    args = ap.parse_args()

    import serial

    failed = []
    for port in args.ports:
        try:
            with serial.Serial(port, args.baud, timeout=0.2) as ser:
                ser.reset_input_buffer()
                ser.write(args.cmd.encode() + b"\n")
                deadline = time.monotonic() + args.wait
                while time.monotonic() < deadline:
                    line = ser.readline()
                    if line:
                        print(f"[{port}] {line.decode(errors='replace').rstrip()}")
        except (serial.SerialException, OSError) as e:
            print(f"[{port}] ERROR: {e}", file=sys.stderr)
            failed.append(port)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
