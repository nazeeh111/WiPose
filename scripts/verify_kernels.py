"""Compare CPU kernels with an explicit baseline; never capture or download models."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def snapshot(root):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    sys.path[:0] = [str(root / "train"), str(root / "rt"), str(root / "host")]
    from csi_train.model import WiSPPN
    from csi_rt.presence import PresenceGate, MotionEnergy
    from csi_host.crc16 import crc16_ccitt
    record = {"crc16_reference": crc16_ccitt(b"123456789")}
    assert record["crc16_reference"] == 0x29B1
    for vector in (False, True):
        torch.manual_seed(42)
        model = WiSPPN(in_ch=8, width=8, layers=(1, 1, 1, 1), vector_head=vector).eval()
        sample = torch.randn(1, 8, 3, 3)
        with torch.no_grad():
            output = model(sample).numpy()
        assert np.isfinite(output).all()
        record[f"head_{vector}"] = {"shape": list(output.shape), "sha256": hashlib.sha256(output.tobytes()).hexdigest()}
    gate = PresenceGate(0.5)
    record["presence"] = [gate.update(np.array(v)) for v in ([0.0, 0.1], [0.5, 0.5], [0.8, 0.9])]
    energy = MotionEnergy()
    for index in range(8):
        energy.add(0, 0, index * 100000000, np.full(56, index, dtype=float))
    record["motion_energy"] = energy.energy()
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    if args.snapshot:
        print(json.dumps(snapshot(args.snapshot.resolve()), sort_keys=True))
    else:
        if not args.baseline_dir:
            parser.error("--baseline-dir is required")
        original = subprocess.check_output([sys.executable, __file__, "--snapshot", str(args.baseline_dir.resolve())])
        branded = subprocess.check_output([sys.executable, __file__, "--snapshot", str(ROOT)])
        assert original == branded, "Kernel outputs differ"
        print(json.dumps({"exact_kernel_parity": True, "scope": "Seeded random weights, reduced-width CPU model; not trained inference accuracy", "outputs": json.loads(branded)}, indent=2))
