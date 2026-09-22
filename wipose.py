"""WiPose: WiFi channel measurements into 18-joint pose estimates."""
import argparse
from pathlib import Path
import subprocess
import sys

COMMANDS = {'train': 'train/train.py', 'infer': 'rt/demo.py', 'label': 'teacher/teacher.py', 'probe': 'train/probe.py'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    script = root / COMMANDS[args.command]
    # The existing entry point expects its documented working directory.
    cwd = script.parent if args.command == "run" else root
    return subprocess.call([sys.executable, str(script), *args.arguments], cwd=cwd)


if __name__ == "__main__":
    raise SystemExit(main())
