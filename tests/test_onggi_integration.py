"""Opt-in real file restoration; no NPU/SDK needed (ONGGI_BIN=/path/to/onggi)."""
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "scripts"))
from swap_bench import run_trial, summarize


@unittest.skipUnless(os.environ.get("ONGGI_BIN"), "set ONGGI_BIN for actual Onggi restore integration")
class OnggiIntegration(unittest.TestCase):
    def test_real_full_delta_restore_produces_same_target_with_less_store_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = {"device": "atom", "device_index": 0,
                   "onggi": os.environ["ONGGI_BIN"], "store": str(root / "store")}
            payload = random.Random(42).randbytes(4 << 20)
            variants = []
            for name, header in (("b1", b"ONE!"), ("b4", b"FOUR")):
                source = root / name
                source.mkdir()
                (source / "model.rbln").write_bytes(header + payload[4:])
                proc = subprocess.run([cfg["onggi"], "push", "--json", "--store", cfg["store"],
                                       "--ref", "fixture@" + name, str(source)],
                                      check=True, capture_output=True, text=True)
                variants.append({"name": name, "ref": json.loads(proc.stdout)["manifest"],
                                 "runtime": {}, "request": {}})

            class Reader:
                timings = {}

                def __init__(self, directory):
                    self.path = directory / "model.rbln"

                def request(self, _):
                    return {"kind": "file-sha256", "value": hashlib.sha256(self.path.read_bytes()).hexdigest()}

                def close(self):
                    pass

            records = [run_trial(cfg, *variants, mode, root / mode,
                                 opener=lambda device, index, directory, runtime: Reader(directory))
                       for mode in ("full", "delta")]
            self.assertTrue(summarize(records)["ok"], records)
            self.assertEqual(records[0]["response"]["value"],
                             hashlib.sha256((root / "b4" / "model.rbln").read_bytes()).hexdigest())
            self.assertGreater(records[1]["pull_stats"]["reused"], 0)
            self.assertLess(records[1]["pull_stats"]["read_bytes"], records[0]["pull_stats"]["read_bytes"])
