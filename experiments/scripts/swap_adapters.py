"""Lazy SDK adapters for artifact replacement experiments, not a serving API."""

from contextlib import contextmanager
import gc
import hashlib
import json
from numbers import Integral
from pathlib import Path
import sys
import time


def _artifact(root, value, *, directory=False, suffix=None):
    if not isinstance(value, str) or not value:
        raise ValueError("artifact path must be a nonempty relative string")
    relative = Path(value)
    if relative.is_absolute() or relative.drive or ".." in relative.parts:
        raise ValueError("artifact path must stay within the restored tree")
    root = Path(root).resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("artifact path escapes the restored tree")
    if directory and not path.is_dir():
        raise ValueError("model/tokenizer artifact must be a directory")
    if not directory and not path.is_file():
        raise ValueError("graph artifact must be a file")
    if suffix and path.suffix != suffix:
        raise ValueError("graph artifact must have suffix " + suffix)
    return str(path)


def _token_request(request):
    if (not isinstance(request, dict)
            or set(request) - {"prompts", "max_new_tokens"}):
        raise ValueError("token requests accept only prompts and max_new_tokens")
    prompts = request.get("prompts")
    if (not isinstance(prompts, list) or not prompts
            or any(not isinstance(p, str) or not p.strip() for p in prompts)):
        raise ValueError("prompts must be a nonempty list of nonempty strings")
    limit = request.get("max_new_tokens", 1)
    if type(limit) is not int or limit <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    return prompts, limit


def _tokens(rows, count, limit):
    if len(rows) != count:
        raise ValueError("generation response count differs from prompt count")
    result = []
    for row in rows:
        values = list(row)
        if not 1 <= len(values) <= limit:
            raise ValueError("generated token count must be within the requested limit")
        if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 0
               for v in values):
            raise ValueError("generated token IDs must be nonnegative integers")
        result.append([int(v) for v in values])
    return result


@contextmanager
def _check_destructor_errors():
    previous = sys.unraisablehook
    failures = []

    def capture(error):
        failures.append(error.exc_value)
        previous(error)

    sys.unraisablehook = capture
    try:
        yield
    finally:
        sys.unraisablehook = previous
    if failures:
        raise RuntimeError("SDK destructor failed during release") from failures[0]


class _AtomRaw:
    def __init__(self, path, device):
        import numpy as np
        import rebel

        metadata = rebel.RBLNCompiledModel.inspect(path)
        self._inputs = [np.zeros(i["shape"], dtype=i["dtype"])
                        for i in metadata["inputs"]]
        start = time.perf_counter()
        self._compiled = rebel.RBLNCompiledModel(path)
        parsed = time.perf_counter()
        self._runtime = self._compiled.create_runtime(device=device, tensor_type="np")
        created = time.perf_counter()
        self.timings = {"host_parse_s": parsed - start,
                        "runtime_create_s": created - parsed}

    def request(self, request):
        import numpy as np

        if not isinstance(request, dict) or request:
            raise ValueError("raw ATOM requests use the empty zero-input fixture")
        outputs = self._runtime.run(*self._inputs)
        if isinstance(outputs, np.ndarray):
            outputs = [outputs]
        if not isinstance(outputs, (tuple, list)) or not outputs:
            raise ValueError("ATOM returned no output arrays")
        digest = hashlib.sha256()
        digest.update(len(outputs).to_bytes(8, "big"))
        for output in outputs:
            if (not isinstance(output, np.ndarray) or not output.size
                    or output.dtype.kind not in "biufc"):
                raise ValueError("ATOM outputs must be nonempty numeric arrays")
            if not np.isfinite(output).all():
                raise ValueError("ATOM output contains nonfinite values")
            metadata = json.dumps([output.dtype.str, list(output.shape)],
                                  separators=(",", ":")).encode("utf-8")
            digest.update(len(metadata).to_bytes(8, "big"))
            digest.update(metadata)
            digest.update(output.tobytes(order="C"))
        return {"kind": "atom-step-sha256", "value": digest.hexdigest()}

    def close(self):
        # Match Q4 ownership order; GC alone cannot prove firmware buffer release.
        with _check_destructor_errors():
            self._runtime = None
            gc.collect()
            self._compiled = None
            self._inputs = None
            gc.collect()


class _AtomOptimum:
    def __init__(self, model, tokenizer, device):
        from optimum.rbln import RBLNLlamaForCausalLM
        from transformers import AutoTokenizer

        start = time.perf_counter()
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer, local_files_only=True)
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = RBLNLlamaForCausalLM.from_pretrained(
            model, export=False, rbln_config={"device": device})
        self.timings = {"runtime_create_s": time.perf_counter() - start}

    def request(self, request):
        prompts, limit = _token_request(request)
        encoded = self._tokenizer(prompts, return_tensors="pt", padding=True)
        output = self._model.generate(**encoded, max_new_tokens=limit, do_sample=False)
        rows = output[:, encoded["input_ids"].shape[1]:].tolist()
        return {"kind": "atom-token-ids", "value": _tokens(rows, len(prompts), limit)}

    def close(self):
        with _check_destructor_errors():
            self._model = None
            gc.collect()
            self._tokenizer = None


class _Rngd:
    def __init__(self, model, fxb, device, max_model_len):
        from furiosa_llm import LLM

        start = time.perf_counter()
        self._model = LLM(model, fxb=fxb, devices="npu:" + str(device),
                          max_model_len=max_model_len)
        self.timings = {"runtime_create_s": time.perf_counter() - start}

    def request(self, request):
        from furiosa_llm import SamplingParams

        prompts, limit = _token_request(request)
        output = self._model.generate(prompts, SamplingParams(max_tokens=limit, temperature=0))
        if any(len(item.outputs) != 1 for item in output):
            raise ValueError("expected one completion per prompt")
        rows = [item.outputs[0].token_ids for item in output]
        return {"kind": "rngd-token-ids", "value": _tokens(rows, len(prompts), limit)}

    def close(self):
        if self._model is not None:
            self._model.shutdown()
            self._model = None
            gc.collect()


def open_model(device: str, device_index: int, artifact_dir: Path, runtime: dict):
    """Open restored artifacts on exactly the requested device; imports stay lazy."""
    if (device not in ("atom", "rngd") or type(device_index) is not int
            or device_index < 0):
        raise ValueError("device must be atom/rngd with a nonnegative integer index")
    if not isinstance(runtime, dict):
        raise ValueError("runtime must be an object")
    if device == "atom":
        backend = runtime.get("backend", "raw")
        if backend == "raw":
            graph = _artifact(artifact_dir, runtime.get("graph"), suffix=".rbln")
            return _AtomRaw(graph, device_index)
        if backend == "optimum":
            model = _artifact(artifact_dir, runtime.get("model"), directory=True)
            tokenizer = _artifact(artifact_dir, runtime.get("tokenizer", runtime.get("model")),
                                  directory=True)
            return _AtomOptimum(model, tokenizer, device_index)
        raise ValueError("unsupported ATOM backend")
    length = runtime.get("max_model_len", 1024)
    if type(length) is not int or length <= 0:
        raise ValueError("max_model_len must be a positive integer")
    model = _artifact(artifact_dir, runtime.get("model"), directory=True)
    fxb = _artifact(artifact_dir, runtime.get("fxb"), suffix=".fxb")
    return _Rngd(model, fxb, device_index, length)
