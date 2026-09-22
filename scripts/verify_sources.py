"""Verify source syntax and unchanged numerical/firmware/media files."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--baseline-dir", required=True, type=Path)
args = parser.parse_args()
checked = {}
python_files = 0
extensions = {".py", ".c", ".h", ".cpp", ".hpp", ".npz", ".png", ".gif", ".jpg", ".svg", ".csv", ".json", ".yaml", ".yml"}
for source in sorted(args.baseline_dir.rglob("*")):
    if not source.is_file() or source.suffix not in extensions:
        continue
    relative = source.relative_to(args.baseline_dir)
    if ".git" in relative.parts:
        continue
    current = ROOT / relative
    assert source.read_bytes() == current.read_bytes(), str(relative)
    if source.suffix == ".py":
        ast.parse(current.read_text(), filename=str(relative))
        python_files += 1
    checked[str(relative)] = hashlib.sha256(current.read_bytes()).hexdigest()
assert checked, "No baseline files found"
print(json.dumps({"unchanged_source_config_media_files": len(checked), "python_files_parsed": python_files, "sha256": checked}, indent=2))
