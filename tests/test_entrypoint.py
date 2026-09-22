import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("entry", ROOT / "wipose.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)

class EntrypointTests(unittest.TestCase):
    def test_dispatch_preserves_arguments_and_failure_code(self):
        for mode, path in entry.COMMANDS.items():
            with self.subTest(mode=mode), patch.object(entry.subprocess, "call", return_value=7) as call:
                args = ["--input", "path with spaces/input.dat"]
                self.assertEqual(entry.main([mode, *args]), 7)
                self.assertEqual(call.call_args.args[0][2:], args)
                self.assertEqual(Path(call.call_args.args[0][1]), ROOT / path)
                expected = (ROOT / path).parent if mode == "run" else ROOT
                self.assertEqual(call.call_args.kwargs["cwd"], expected)

if __name__ == "__main__":
    unittest.main()
