# Verification

Executed on 2026-09-22. Python 3.12.13, PyTorch 2.14.0, NumPy 1.26.4 on macOS ARM64.

## Passed

- 107 tracked numerical-source, firmware, configuration, and media files remained byte-identical to the baseline; 67 original Python files passed syntax parsing. [File hashes](source-verification.json).
- The new command dispatch test passed, checking argument preservation, paths containing spaces, expected working directories, and child exit-code propagation for every exposed command. Dispatch was mocked to avoid launching hardware routes.
- Two seeded CPU model variants (PAM and vector heads) produced byte-identical arrays against the baseline. Presence-gate and motion-energy outputs matched; CRC16 passed the standard 123456789 fixture. Models used reduced width and random weights, not trained checkpoints. [Kernel evidence](kernel-verification.json).
- New command help works without importing optional hardware or model dependencies.

## Reproduce

```bash
python -m unittest discover -s tests -v
python scripts/verify_sources.py --baseline-dir /path/to/previous-checkout
python scripts/verify_kernels.py --baseline-dir /path/to/previous-checkout
```

Parity commands take an explicit separate prior checkout; they do not depend on unpublished historical Git objects. Kernel dependencies are separated from the full application requirements. Source syntax checks do not prove that optional dependencies or hardware work.

## Not executed

ESP32 hardware/firmware build and flashing, serial links, MQTT, teacher labeling, model downloads, training datasets, full trained inference, fall-detection accuracy, and cross-session evaluation were not executed. The reference figures/results are recorded project material, not new measurements.

These are bounded source and synthetic-runtime checks, not universal correctness or research-replication claims. Original numerical code, calibration, and recorded scientific images were preserved.
