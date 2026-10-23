import copy
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "experiments" / "scripts"
sys.path.insert(0, str(SCRIPTS))


def config():
    return {
        "device": "atom", "device_index": 0, "onggi": "onggi", "store": "store",
        "repetitions": 2, "modes": ["full", "delta", "resident"],
        "variants": [
            {"name": "b1", "ref": "sha256:" + "a" * 64,
             "runtime": {"graph": "model.rbln"}, "request": {}},
            {"name": "b4", "ref": "sha256:" + "b" * 64,
             "runtime": {"graph": "model.rbln"}, "request": {}},
        ],
        "transitions": [{"source": "b1", "target": "b4"}],
    }


class BenchTests(unittest.TestCase):
    def setUp(self):
        try:
            self.bench = importlib.import_module("swap_bench")
        except ModuleNotFoundError:
            self.fail("common swap benchmark has not been implemented")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = config()
        self.source, self.target = self.cfg["variants"]
        self.events = []
        self.fail_close = False
        self.fail_request = False

    def pull(self, cfg, variant, dest, base=None):
        self.events.append(("pull", variant["name"], base is not None))
        dest.mkdir()
        (dest / "model.rbln").write_text(variant["name"])
        if base:
            self.assertTrue((base[1] / "model.rbln").exists())
            (base[1] / "model.rbln").unlink()
        return {"bytes": 100, "read_bytes": 10 if base else 100, "reused": 90 if base else 0}

    def open_model(self, device, index, directory, runtime):
        name = (directory / "model.rbln").read_text()
        self.events.append(("open", name, index))
        outer = self

        class Model:
            timings = {"host_parse_s": 0.01, "runtime_create_s": 0.02}

            def request(self, request):
                outer.events.append(("request", name))
                if outer.fail_request and name == "b4":
                    raise RuntimeError("request failed")
                return {"kind": "fixture", "value": request.get("value", name)}

            def close(self):
                outer.events.append(("close", name))
                if outer.fail_close:
                    raise RuntimeError("close failed")

        return Model()

    def trial(self, mode="delta"):
        return self.bench.run_trial(
            self.cfg, self.source, self.target, mode, self.root / "trial",
            opener=self.open_model, puller=self.pull,
        )

    def test_delta_releases_old_runtime_before_consuming_private_base(self):
        result = self.trial()
        self.assertEqual(result["status"], "measured")
        self.assertLess(self.events.index(("close", "b1")),
                        self.events.index(("pull", "b4", True)))
        self.assertEqual(result["pull_stats"]["read_bytes"], 10)
        self.assertEqual(result["response"], {"kind": "fixture", "value": "b4"})
        self.assertGreaterEqual(result["end_to_end_s"], result["restore_s"])
        self.assertEqual(result["cleanup"], "ok")
        self.assertFalse((self.root / "trial").exists())

    def test_full_does_not_offer_delta_base(self):
        self.assertEqual(self.trial("full")["status"], "measured")
        self.assertIn(("pull", "b4", False), self.events)

    def test_release_failure_stops_target_restore_and_retains_directory(self):
        self.fail_close = True
        result = self.trial()
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["stage"], "release")
        self.assertNotIn(("pull", "b4", True), self.events)
        self.assertTrue((self.root / "trial" / "base" / "model.rbln").exists())

    def test_request_failure_is_error_and_cleanup_still_runs(self):
        self.fail_request = True
        result = self.trial()
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["stage"], "first_response")
        self.assertIn(("close", "b4"), self.events)
        self.assertFalse((self.root / "trial").exists())

    def test_cleanup_failure_is_not_a_success(self):
        original = self.open_model

        def opener(*args):
            model = original(*args)
            if (args[2] / "model.rbln").read_text() == "b4":
                def fail():
                    raise RuntimeError("target cleanup failed")
                model.close = fail
            return model

        result = self.bench.run_trial(self.cfg, self.source, self.target, "full",
                                     self.root / "trial", opener=opener, puller=self.pull)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["cleanup"], "error")

    def test_existing_workspace_is_never_removed(self):
        trial = self.root / "trial"
        trial.mkdir()
        (trial / "keep").write_text("user data")
        result = self.trial()
        self.assertEqual(result["status"], "error")
        self.assertEqual((trial / "keep").read_text(), "user data")

    def test_constructor_failure_retains_files_and_reports_uncertain_cleanup(self):
        for failing_name in ("b1", "b4"):
            with self.subTest(failing_name=failing_name):
                workspace = self.root / failing_name
                owners = []

                def opener(*args):
                    model = self.open_model(*args)
                    if (args[2] / "model.rbln").read_text() == failing_name:
                        owners.append(model)
                        raise RuntimeError("allocated before constructor failure")
                    return model

                result = self.bench.run_trial(self.cfg, self.source, self.target, "delta",
                                             workspace, opener=opener, puller=self.pull)
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["cleanup"], "unknown")
                self.assertTrue(workspace.exists())
                self.assertEqual(len(owners), 1)
                owners[0].close()

    def test_different_artifacts_cannot_reuse_resident_runtime(self):
        result = self.trial("resident")
        self.assertEqual(result["status"], "unsupported")
        self.assertEqual(self.events, [])

    def test_resident_uses_same_runtime_for_new_request(self):
        self.target["ref"] = self.source["ref"]
        self.target["request"] = {"value": "new-shape"}
        result = self.trial("resident")
        self.assertEqual(result["status"], "measured")
        self.assertEqual(result["response"]["value"], "new-shape")
        self.assertEqual(sum(e[0] == "open" for e in self.events), 1)
        self.assertEqual(result["restore_s"], 0)
        self.assertEqual(result["load_s"], 0)

    def test_schema_rejects_ambiguous_or_unusable_inputs(self):
        mutations = [
            ("device", "gpu"), ("device_index", -1), ("device_index", True),
            ("repetitions", 0), ("repetitions", 1.5), ("modes", ["delta"]),
            ("modes", ["full", "pretend-partial"]), ("modes", ["full", "full"]),
        ]
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                cfg = config()
                cfg[key] = value
                with self.assertRaises(ValueError):
                    self.bench.validate_config(cfg)
        for change in ("mutable-ref", "duplicate-name", "unknown-source", "empty-transition"):
            with self.subTest(change=change):
                cfg = config()
                if change == "mutable-ref":
                    cfg["variants"][0]["ref"] = "model@latest"
                elif change == "duplicate-name":
                    cfg["variants"][1]["name"] = "b1"
                elif change == "unknown-source":
                    cfg["transitions"][0]["source"] = "typo"
                else:
                    cfg["transitions"] = []
                with self.assertRaises(ValueError):
                    self.bench.validate_config(cfg)

    def measured(self, mode, value, seconds):
        return {"source": "b1", "target": "b4", "mode": mode, "status": "measured",
                "cleanup": "ok", "response": {"kind": "fixture", "value": value},
                "end_to_end_s": seconds, "release_s": seconds / 4,
                "restore_s": seconds / 4, "load_s": seconds / 4,
                "first_response_s": seconds / 4}

    def test_summary_excludes_mismatch_and_missing_reference(self):
        records = [self.measured("full", "right", 10),
                   self.measured("delta", "right", 2),
                   self.measured("delta", "wrong", 0.01)]
        report = self.bench.summarize(records)
        self.assertFalse(report["ok"])
        group = next(g for g in report["groups"] if g["mode"] == "delta")
        self.assertEqual(group["samples"], 1)
        self.assertEqual(group["failed"], 1)
        self.assertEqual(group["end_to_end_s"]["p50"], 2)
        missing = self.bench.summarize([records[1]])
        self.assertFalse(missing["ok"])
        self.assertEqual(missing["groups"][0]["samples"], 0)

    def test_inconsistent_full_results_invalidate_entire_transition(self):
        report = self.bench.summarize([
            self.measured("full", "a", 10), self.measured("full", "b", 11),
            self.measured("delta", "a", 1),
        ])
        self.assertFalse(report["ok"])
        self.assertTrue(all(g["samples"] == 0 for g in report["groups"]))

    def test_nearest_rank_percentiles_and_sample_counts(self):
        records = [self.measured("full", "a", i) for i in range(1, 21)]
        report = self.bench.summarize(records)
        self.assertTrue(report["ok"])
        self.assertEqual(report["groups"][0]["samples"], 20)
        self.assertEqual(report["groups"][0]["end_to_end_s"], {"p50": 10, "p95": 19})

    def test_pull_command_is_argv_and_delta_uses_only_provided_base(self):
        executable = self.root / "fake_onggi.py"
        executable.write_text(
            'import json,sys\n'
            'from pathlib import Path\n'
            'Path(sys.argv[-1]).mkdir()\n'
            'print(json.dumps({"manifest":sys.argv[-2],"stats":{"bytes":100,"read_bytes":10,"reused":90}}))\n'
        )
        self.cfg["onggi"] = [sys.executable, str(executable)]
        dest = self.root / "restored with spaces"
        stats = self.bench.restore(self.cfg, self.target, dest,
                                   (self.source, self.root / "base with spaces"))
        self.assertEqual(stats["read_bytes"], 10)
        self.assertTrue(dest.is_dir())

    def test_pull_failure_and_wrong_manifest_are_not_accepted(self):
        executable = self.root / "fail.py"
        self.cfg["onggi"] = [sys.executable, str(executable)]
        for body in ('raise SystemExit(7)',
                     'print(\'{"manifest":"wrong","stats":{}}\')'):
            executable.write_text(body)
            with self.subTest(body=body), self.assertRaises((ValueError, RuntimeError)):
                self.bench.restore(self.cfg, self.target, self.root / "target")

    def test_cli_invalid_config_exits_nonzero_without_sdk_import(self):
        cfg_path = self.root / "config.json"
        bad = config()
        bad["device_index"] = -1
        cfg_path.write_text(json.dumps(bad))
        proc = subprocess.run([sys.executable, str(SCRIPTS / "swap_bench.py"),
                               str(cfg_path), "--output", str(self.root / "result.json"),
                               "--work-dir", str(self.root / "work")],
                              capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("device_index", proc.stderr)
        self.assertNotIn("No module named", proc.stderr)

    def test_cli_runs_isolated_workers_and_keeps_correctness_gated_report(self):
        sdk = self.root / "sdk"
        sdk.mkdir()
        (sdk / "rebel.py").write_text(
            'from pathlib import Path\nimport numpy as np\n'
            'class RBLNCompiledModel:\n'
            ' @staticmethod\n'
            ' def inspect(path): return {"inputs":[{"shape":[1],"dtype":"int64"}]}\n'
            ' def __init__(self,path): self.value=Path(path).read_bytes()[0]\n'
            ' def create_runtime(self,**kwargs): return self\n'
            ' def run(self,*args): return np.array([self.value],dtype="int64")\n'
        )
        onggi = self.root / "onggi.py"
        onggi.write_text(
            'import sys,json\nfrom pathlib import Path\n'
            'dest=Path(sys.argv[-1]); dest.mkdir()\n'
            '(dest/"model.rbln").write_text(sys.argv[-2])\n'
            'print(json.dumps({"manifest":sys.argv[-2],"stats":{"bytes":100,"read_bytes":100,"reused":0}}))\n'
        )
        cfg = config()
        cfg["onggi"] = [sys.executable, str(onggi)]
        cfg_path, output, work = self.root / "config.json", self.root / "result.json", self.root / "work"
        cfg_path.write_text(json.dumps(cfg))
        env = dict(os.environ, PYTHONPATH=str(sdk))
        proc = subprocess.run([sys.executable, str(SCRIPTS / "swap_bench.py"), str(cfg_path),
                               "--output", str(output), "--work-dir", str(work)],
                              env=env, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        report = json.loads(output.read_text())
        self.assertTrue(report["complete"])
        self.assertTrue(report["ok"])
        self.assertEqual(len(report["trials"]), 6)
        self.assertEqual(len(set(r["worker_pid"] for r in report["trials"])), 6)
        self.assertTrue(all(r["status"] in ("ok", "unsupported") for r in report["trials"]))
        self.assertEqual(list(work.glob("swap-*/trial-*")), [])


if __name__ == "__main__":
    unittest.main()
