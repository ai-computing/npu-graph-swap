"""SDK boundary checks; hardware SDKs are replaced only within each test."""

import gc
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "experiments" / "scripts"
sys.path.insert(0, str(SCRIPTS))


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "graph.rbln").write_bytes(b"compiled")
        (self.root / "model").mkdir()
        (self.root / "tokenizer").mkdir()
        (self.root / "graph.fxb").write_bytes(b"compiled")

    def open(self, device="atom", runtime=None, index=3):
        self.assertIsNotNone(importlib.util.find_spec("swap_adapters"),
                             "SDK adapter module must implement open_model")
        module = importlib.import_module("swap_adapters")
        return module.open_model(device, index, self.root,
                                 runtime or {"graph": "graph.rbln"})

    def patch_modules(self, modules):
        patcher = patch.dict(sys.modules, modules)
        patcher.start()
        self.addCleanup(patcher.stop)

    def atom_sdk(self, output=None, events=None, destructor_error=False):
        events = events if events is not None else []
        state = SimpleNamespace(inputs=None, device=None, output=(
            np.array([1, 2], dtype="float32") if output is None else output))

        class Runtime:
            def run(self, *inputs):
                state.inputs = inputs
                return state.output

            def __del__(self):
                events.append("runtime destroyed")
                if destructor_error:
                    raise RuntimeError("SDK runtime destructor failed")

        class Compiled:
            def __init__(self, path):
                if path != str(self_outer.root / "graph.rbln"):
                    raise AssertionError("wrong graph path")
                state.compiled = weakref.ref(self)

            @staticmethod
            def inspect(path):
                return {"inputs": [{"shape": [1, 2], "dtype": "int64"},
                                   {"shape": [2], "dtype": "float32"}]}

            def create_runtime(self, *, device, tensor_type):
                if tensor_type != "np":
                    raise AssertionError("wrong tensor type")
                state.device = device
                runtime = Runtime()
                state.runtime = weakref.ref(runtime)
                return runtime

            def __del__(self):
                events.append("compiled destroyed")

        self_outer = self
        module = ModuleType("rebel")
        module.RBLNCompiledModel = Compiled
        self.patch_modules({"rebel": module})
        return state

    def rngd_sdk(self, tokens=None, shutdown_error=False):
        state = SimpleNamespace(tokens=[[7], [8]] if tokens is None else tokens,
                                shutdowns=0)

        class SamplingParams:
            def __init__(self, *, max_tokens, temperature):
                state.sampling = (max_tokens, temperature)

        class LLM:
            def __init__(self, model, *, fxb, devices, max_model_len):
                state.arguments = (model, fxb, devices, max_model_len)

            def generate(self, prompts, sampling_params):
                state.prompts = prompts
                return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=t)])
                        for t in state.tokens]

            def shutdown(self):
                state.shutdowns += 1
                if shutdown_error:
                    raise RuntimeError("SDK shutdown failed")

        module = ModuleType("furiosa_llm")
        module.LLM, module.SamplingParams = LLM, SamplingParams
        self.patch_modules({"furiosa_llm": module})
        return state

    def optimum_sdk(self, output=None, destructor_error=False):
        state = SimpleNamespace(output=np.array([[11, 12, 91], [0, 13, 92]])
                                if output is None else output)

        class Model:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                state.model_arguments = (path, kwargs)
                return cls()

            def generate(self, **kwargs):
                state.generation = kwargs
                return state.output

            def __del__(self):
                if destructor_error:
                    raise RuntimeError("SDK optimum destructor failed")

        class Tokenizer:
            pad_token, eos_token = None, "<eos>"

            @classmethod
            def from_pretrained(cls, path, **kwargs):
                state.tokenizer_path = path
                state.tokenizer = cls()
                return state.tokenizer

            def __call__(self, prompts, *, return_tensors, padding):
                if (return_tensors, padding) != ("pt", True):
                    raise AssertionError("expected padded PyTorch inputs")
                return {"input_ids": np.array([[11, 12], [0, 13]]),
                        "attention_mask": np.array([[1, 1], [0, 1]])}

        optimum = ModuleType("optimum")
        rbln = ModuleType("optimum.rbln")
        rbln.RBLNLlamaForCausalLM = Model
        transformers = ModuleType("transformers")
        transformers.AutoTokenizer = Tokenizer
        self.patch_modules({
            "optimum": optimum, "optimum.rbln": rbln,
            "transformers": transformers})
        return state

    def test_import_does_not_import_vendor_sdks_or_numpy(self):
        code = ("import sys; sys.path.insert(0, sys.argv[1]); import swap_adapters; "
                "assert not any(n in sys.modules for n in "
                "['rebel', 'numpy', 'furiosa_llm', 'optimum', 'transformers'])")
        result = subprocess.run([sys.executable, "-c", code, str(SCRIPTS)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_atom_uses_explicit_device_and_metadata_zero_inputs(self):
        state = self.atom_sdk()
        model = self.open()
        result = model.request({})
        self.assertEqual(state.device, 3)
        self.assertEqual([(i.shape, str(i.dtype), i.tolist()) for i in state.inputs],
                         [((1, 2), "int64", [[0, 0]]), ((2,), "float32", [0, 0])])
        self.assertEqual(result["kind"], "atom-step-sha256")
        self.assertEqual(len(result["value"]), 64)
        self.assertEqual(set(model.timings), {"host_parse_s", "runtime_create_s"})
        self.assertTrue(all(t >= 0 for t in model.timings.values()))
        model.close()

    def test_atom_fingerprint_includes_all_arrays_all_bytes_dtype_and_shape(self):
        state = self.atom_sdk()
        model = self.open()
        fingerprints = []
        cases = [
            [np.arange(1000, dtype="int32"), np.array([1], dtype="int32")],
            [np.arange(1000, dtype="int32"), np.array([2], dtype="int32")],
            [np.arange(1000, dtype="int32").reshape(10, 100), np.array([1], dtype="int32")],
            [np.arange(1000, dtype="int32").view("float32"), np.array([1], dtype="int32")],
            [np.concatenate((np.arange(999, dtype="int32"), [2000])).astype("int32"),
             np.array([1], dtype="int32")],
        ]
        for output in cases:
            state.output = output
            fingerprints.append(model.request({})["value"])
        self.assertEqual(len(set(fingerprints)), len(cases))
        state.output = cases[0]
        self.assertEqual(model.request({})["value"], fingerprints[0])
        model.close()

    def test_atom_rejects_empty_nonfinite_and_non_numeric_outputs(self):
        state = self.atom_sdk()
        model = self.open()
        for output in ([], np.array([]), [np.array([1]), np.array([])],
                       np.array([np.nan]), np.array([np.inf]),
                       np.array([object()]), None):
            with self.subTest(output=repr(output)):
                state.output = output
                with self.assertRaises((ValueError, TypeError)):
                    model.request({})
        model.close()

    def test_atom_close_collects_runtime_before_compiled_model(self):
        events = []
        state = self.atom_sdk(events=events)
        model = self.open()
        real_collect = gc.collect

        def collect():
            events.append("gc")
            return real_collect()

        with patch("swap_adapters.gc.collect", side_effect=collect):
            model.close()
        self.assertIsNone(state.runtime())
        self.assertIsNone(state.compiled())
        self.assertLess(events.index("runtime destroyed"), events.index("gc"))
        self.assertLess(events.index("gc"), events.index("compiled destroyed"))

    def test_atom_close_reports_destructor_failure_and_restores_unraisable_hook(self):
        self.atom_sdk(destructor_error=True)
        model = self.open()
        delegated = []
        with patch.object(sys, "unraisablehook", delegated.append):
            original_hook = sys.unraisablehook
            with self.assertRaisesRegex(RuntimeError, "destructor"):
                model.close()
            self.assertIs(sys.unraisablehook, original_hook)
        self.assertEqual(len(delegated), 1)
        self.assertEqual(str(delegated[0].exc_value), "SDK runtime destructor failed")

    def test_optimum_close_reports_destructor_failure_and_restores_unraisable_hook(self):
        self.optimum_sdk(destructor_error=True)
        model = self.open(runtime={"backend": "optimum", "model": "model"})
        delegated = []
        with patch.object(sys, "unraisablehook", delegated.append):
            original_hook = sys.unraisablehook
            with self.assertRaisesRegex(RuntimeError, "destructor"):
                model.close()
            self.assertIs(sys.unraisablehook, original_hook)
        self.assertEqual(len(delegated), 1)
        self.assertEqual(str(delegated[0].exc_value), "SDK optimum destructor failed")

    def test_rngd_one_greedy_token_per_prompt_with_explicit_device(self):
        state = self.rngd_sdk()
        model = self.open("rngd", {"model": "model", "fxb": "graph.fxb"})
        result = model.request({"prompts": ["first", "second"]})
        self.assertEqual(state.arguments, (str(self.root / "model"),
                         str(self.root / "graph.fxb"), "npu:3", 1024))
        self.assertEqual(result, {"kind": "rngd-token-ids", "value": [[7], [8]]})
        self.assertEqual(state.sampling, (1, 0))
        json.dumps(result)
        model.close()
        self.assertEqual(state.shutdowns, 1)

    def test_rngd_generation_limit_changes_per_request_and_allows_early_eos(self):
        state = self.rngd_sdk(tokens=[[7, 8, 9], [10]])
        model = self.open("rngd", {"model": "model", "fxb": "graph.fxb"})
        result = model.request({"prompts": ["first", "second"], "max_new_tokens": 3})
        self.assertEqual(result, {"kind": "rngd-token-ids",
                                  "value": [[7, 8, 9], [10]]})
        self.assertEqual(state.sampling, (3, 0))
        state.tokens = [[7], [8]]
        self.assertEqual(model.request({"prompts": ["first", "second"]})["value"],
                         [[7], [8]])
        self.assertEqual(state.sampling, (1, 0))
        model.close()

    def test_optimum_generation_limit_returns_all_tokens_and_allows_early_eos(self):
        state = self.optimum_sdk(output=np.array([[11, 12, 91, 92], [0, 13, 93, 94]]))
        model = self.open(runtime={"backend": "optimum", "model": "model"})
        result = model.request({"prompts": ["first", "second"], "max_new_tokens": 2})
        self.assertEqual(result, {"kind": "atom-token-ids",
                                  "value": [[91, 92], [93, 94]]})
        self.assertEqual(state.generation["max_new_tokens"], 2)
        state.output = np.array([[11, 12, 91], [0, 13, 92]])
        self.assertEqual(model.request({"prompts": ["first", "second"],
                                       "max_new_tokens": 32})["value"], [[91], [92]])
        self.assertEqual(state.generation["max_new_tokens"], 32)
        model.close()

    def test_token_adapters_reject_invalid_generation_limits_and_over_limit_outputs(self):
        self.rngd_sdk(tokens=[[7, 8, 9], [10]])
        self.optimum_sdk(
            output=np.array([[11, 12, 91, 92, 93], [0, 13, 94, 95, 96]]))
        models = [self.open("rngd", {"model": "model", "fxb": "graph.fxb"}),
                  self.open(runtime={"backend": "optimum", "model": "model"})]
        for model in models:
            for limit in (True, False, 0, -1, 1.5, "2", None):
                with self.subTest(model=type(model).__name__, limit=limit):
                    with self.assertRaises(ValueError):
                        model.request({"prompts": ["first", "second"],
                                       "max_new_tokens": limit})
            with self.assertRaises(ValueError):
                model.request({"prompts": ["first", "second"], "max_new_tokens": 2})
            model.close()

    def test_rngd_honors_max_model_len_and_propagates_shutdown_failure(self):
        state = self.rngd_sdk(shutdown_error=True)
        model = self.open("rngd", {"model": "model", "fxb": "graph.fxb",
                                   "max_model_len": 256})
        self.assertEqual(state.arguments[-1], 256)
        with self.assertRaisesRegex(RuntimeError, "shutdown failed"):
            model.close()

    def test_rngd_rejects_missing_empty_invalid_or_wrong_count_tokens(self):
        state = self.rngd_sdk()
        model = self.open("rngd", {"model": "model", "fxb": "graph.fxb"})
        for tokens in ([], [[7]], [[], [7]], [[1.5], [7]], [[True], [7]],
                       [[-1], [7]], [[7, 8], [9]]):
            with self.subTest(tokens=tokens):
                state.tokens = tokens
                with self.assertRaises((TypeError, ValueError)):
                    model.request({"prompts": ["first", "second"]})
        model.close()

    def test_token_adapters_reject_invalid_prompts(self):
        self.rngd_sdk()
        model = self.open("rngd", {"model": "model", "fxb": "graph.fxb"})
        for request in ({}, {"prompts": []}, {"prompts": "x"},
                        {"prompts": [""]}, {"prompts": [1]}):
            with self.subTest(request=request):
                with self.assertRaises((TypeError, ValueError)):
                    model.request(request)
        model.close()

    def test_optimum_returns_only_generated_tokens_and_left_pads(self):
        state = self.optimum_sdk()
        model = self.open(runtime={"backend": "optimum", "model": "model",
                                   "tokenizer": "tokenizer"})
        result = model.request({"prompts": ["first", "second"]})
        self.assertEqual(result, {"kind": "atom-token-ids", "value": [[91], [92]]})
        self.assertEqual(state.model_arguments,
                         (str(self.root / "model"), {"export": False,
                          "rbln_config": {"device": 3}}))
        self.assertEqual(state.tokenizer_path, str(self.root / "tokenizer"))
        self.assertEqual(state.tokenizer.padding_side, "left")
        self.assertEqual(state.tokenizer.pad_token, "<eos>")
        self.assertEqual(state.generation["max_new_tokens"], 1)
        self.assertIs(state.generation["do_sample"], False)
        model.close()

    def test_optimum_defaults_tokenizer_to_model_and_rejects_bad_generation(self):
        state = self.optimum_sdk()
        model = self.open(runtime={"backend": "optimum", "model": "model"})
        self.assertEqual(state.tokenizer_path, str(self.root / "model"))
        for output in (np.array([[11, 12]]), np.array([[11, 12], [0, 13]]),
                       np.array([[11, 12, 9, 8], [0, 13, 8, 9]])):
            state.output = output
            with self.assertRaises(ValueError):
                model.request({"prompts": ["first", "second"]})
        model.close()

    def test_artifact_paths_reject_escape_absolute_missing_and_wrong_types(self):
        self.atom_sdk()
        for graph in ("../outside.rbln", str(self.root / "graph.rbln"),
                      "missing.rbln", "model", "graph.fxb"):
            with self.subTest(graph=graph):
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.open(runtime={"graph": graph})
        self.rngd_sdk()
        for runtime in ({"model": "graph.fxb", "fxb": "graph.fxb"},
                        {"model": "model", "fxb": "model"},
                        {"model": "../outside", "fxb": "graph.fxb"}):
            with self.assertRaises(ValueError):
                self.open("rngd", runtime)

    def test_symlink_cannot_escape_artifact_tree(self):
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / "outside.rbln"
            target.write_bytes(b"compiled")
            try:
                (self.root / "link.rbln").symlink_to(target)
            except OSError as exc:
                self.skipTest("symlink unavailable: " + str(exc))
            with self.assertRaises(ValueError):
                self.open(runtime={"graph": "link.rbln"})

    def test_invalid_devices_backend_and_lengths_fail_before_sdk_use(self):
        for device, index, runtime in (
            ("other", 0, {"graph": "graph.rbln"}),
            ("atom", -1, {"graph": "graph.rbln"}),
            ("atom", True, {"graph": "graph.rbln"}),
            ("atom", 0, {"backend": "other"}),
            ("rngd", 0, {"model": "model", "fxb": "graph.fxb", "max_model_len": 0}),
        ):
            with self.subTest(device=device, index=index, runtime=runtime):
                with self.assertRaises(ValueError):
                    self.open(device, runtime, index)


if __name__ == "__main__":
    unittest.main()
