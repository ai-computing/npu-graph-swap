#!/usr/bin/env python3
"""Correctness-gated artifact replacement experiments; see experiments/README.md."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time

MODES = ("full", "delta", "resident")
STAGES = ("release_s", "restore_s", "load_s", "first_response_s", "end_to_end_s")


def validate_config(cfg):
    if not isinstance(cfg, dict):
        raise ValueError("config must be an object")
    if cfg.get("device") not in ("atom", "rngd"):
        raise ValueError("device must be atom or rngd")
    for key, minimum in (("device_index", 0), ("repetitions", 1)):
        value = cfg.get(key)
        if type(value) is not int or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    executable = cfg.get("onggi")
    if not (isinstance(executable, str) and executable.strip() or
            isinstance(executable, list) and executable and
            all(isinstance(x, str) and x for x in executable)):
        raise ValueError("onggi must be an executable string or argv list")
    if not isinstance(cfg.get("store"), str) or not cfg["store"]:
        raise ValueError("store must be a nonempty path/URL")
    modes = cfg.get("modes")
    if (not isinstance(modes, list) or not modes or
            any(m not in MODES for m in modes) or
            len(modes) != len(set(modes)) or "full" not in modes):
        raise ValueError("modes must be unique full/delta/resident values and include full")
    variants = cfg.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("variants must be a nonempty list")
    names = set()
    for variant in variants:
        if not isinstance(variant, dict):
            raise ValueError("variant must be an object")
        name = variant.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("variant names must be nonempty and unique")
        names.add(name)
        if not isinstance(variant.get("ref"), str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", variant["ref"]):
            raise ValueError(f"{name}: ref must be an immutable sha256 manifest")
        if not isinstance(variant.get("runtime"), dict) or not isinstance(variant.get("request"), dict):
            raise ValueError(f"{name}: runtime and request must be objects")
    pairs = cfg.get("transitions")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("transitions must be a nonempty list")
    seen = set()
    for pair in pairs:
        if (not isinstance(pair, dict) or pair.get("source") not in names or
                pair.get("target") not in names):
            raise ValueError("transition source/target must name known variants")
        key = (pair["source"], pair["target"])
        if key in seen:
            raise ValueError("duplicate transition")
        seen.add(key)


def restore(cfg, variant, dest, base=None):
    executable = cfg["onggi"]
    command = [executable] if isinstance(executable, str) else list(executable)
    command += ["pull", "--json", "--store", cfg["store"]]
    if base is not None:
        command += ["--base", base[0]["ref"], "--base-dir", str(base[1])]
    command += [variant["ref"], str(dest)]
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"onggi pull exited {proc.returncode}: {proc.stderr[-2000:]}")
    result = json.loads(proc.stdout)
    if result.get("manifest") != variant["ref"] or not isinstance(result.get("stats"), dict):
        raise ValueError("onggi returned a different manifest or missing stats")
    stats = result["stats"]
    for key in ("bytes", "read_bytes", "reused"):
        if type(stats.get(key)) is not int or stats[key] < 0:
            raise ValueError(f"onggi returned invalid {key}")
    if stats["reused"] > stats["bytes"]:
        raise ValueError("onggi reused bytes exceed logical bytes")
    return stats


def run_trial(cfg, source, target, mode, workspace, opener=None, puller=None):
    result = {"source": source["name"], "target": target["name"], "mode": mode,
              "status": "error", "cleanup": "not-needed", "worker_pid": os.getpid()}
    if mode == "resident" and (source["ref"] != target["ref"] or
                                source["runtime"] != target["runtime"]):
        result.update(status="unsupported", reason="resident requires the same manifest and runtime settings")
        return result
    if mode not in MODES:
        result.update(stage="validate", error="unknown mode")
        return result
    if opener is None:
        from swap_adapters import open_model
        opener = open_model
    puller = puller or restore
    workspace = Path(workspace)
    created, close_failed, opening, model = False, False, False, None
    stage = "prepare_base"
    try:
        # Never adopt or delete an existing directory supplied by a caller.
        workspace.mkdir()
        created = True
        base_dir, target_dir = workspace / "base", workspace / "target"
        puller(cfg, source, base_dir)
        opening = True
        model = opener(cfg["device"], cfg["device_index"], base_dir, source["runtime"])
        opening = False
        model.request(source["request"])
        result["base_prepared"] = True
        started = time.perf_counter()
        if mode == "resident":
            result.update(release_s=0.0, restore_s=0.0, load_s=0.0, adapter_timings={})
        else:
            stage = "release"
            before = time.perf_counter()
            try:
                model.close()
            except Exception:
                close_failed = True
                raise
            model = None
            result["release_s"] = time.perf_counter() - before
            stage = "restore"
            before = time.perf_counter()
            result["pull_stats"] = puller(cfg, target, target_dir,
                                         (source, base_dir) if mode == "delta" else None)
            result["restore_s"] = time.perf_counter() - before
            stage = "load"
            before = time.perf_counter()
            opening = True
            model = opener(cfg["device"], cfg["device_index"], target_dir, target["runtime"])
            opening = False
            result["load_s"] = time.perf_counter() - before
            result["adapter_timings"] = dict(model.timings)
        stage = "first_response"
        before = time.perf_counter()
        response = model.request(target["request"])
        result["first_response_s"] = time.perf_counter() - before
        result["end_to_end_s"] = time.perf_counter() - started
        if (not isinstance(response, dict) or not isinstance(response.get("kind"), str) or
                not response["kind"] or "value" not in response or response["value"] is None):
            raise ValueError("adapter returned no response fingerprint")
        json.dumps(response, allow_nan=False)
        result.update(status="measured", response=response)
    except Exception as exc:
        result.update(status="error", stage=stage, error=f"{type(exc).__name__}: {exc}")
    finally:
        if model is not None and not close_failed:
            try:
                model.close()
                model = None
            except Exception as exc:
                close_failed = True
                result.update(status="error", stage="cleanup", error=f"{type(exc).__name__}: {exc}")
        if created:
            if opening or close_failed:
                # A failed constructor can own buffers without returning a handle.
                result.update(cleanup="unknown" if opening else "error", retained_workspace=str(workspace))
            else:
                try:
                    shutil.rmtree(workspace)
                    result["cleanup"] = "ok"
                except OSError as exc:
                    result.update(status="error", stage="cleanup", cleanup="error", error=str(exc))
    return result


def summarize(records):
    references = {}
    for record in records:
        if record["mode"] == "full" and record["status"] == "measured":
            key = (record["source"], record["target"])
            references.setdefault(key, set()).add(json.dumps(record["response"], sort_keys=True, allow_nan=False))
    groups, verified = {}, []
    for raw in records:
        record = dict(raw)
        pair = (record["source"], record["target"])
        key = (*pair, record["mode"])
        group = groups.setdefault(key, {"source": pair[0], "target": pair[1], "mode": key[2],
                                        "samples": 0, "failed": 0, "unsupported": 0})
        if record["status"] == "measured":
            refs = references.get(pair, set())
            value = json.dumps(record["response"], sort_keys=True, allow_nan=False)
            if len(refs) != 1:
                record.update(status="error", correctness="missing-or-inconsistent-full-reference")
            elif value not in refs:
                record.update(status="error", correctness="mismatch")
            else:
                record.update(status="ok", correctness="match")
        if record["status"] == "ok":
            group["samples"] += 1
        elif record["status"] == "unsupported":
            group["unsupported"] += 1
        else:
            group["failed"] += 1
        verified.append(record)
    for key, group in groups.items():
        matches = [r for r in verified if (r["source"], r["target"], r["mode"]) == key and r["status"] == "ok"]
        for stage in STAGES:
            values = sorted(r[stage] for r in matches)
            group[stage] = ({f"p{q}": values[math.ceil(q / 100 * len(values)) - 1] for q in (50, 95)}
                            if values else None)
    return {"ok": bool(verified) and any(r["status"] == "ok" for r in verified) and
                  all(r["status"] in ("ok", "unsupported") for r in verified),
            "trials": verified, "groups": list(groups.values())}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--trial", nargs=3, metavar=("SOURCE", "TARGET", "MODE"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        cfg = json.loads(args.config.read_text(encoding="utf-8"))
        validate_config(cfg)
        variants = {v["name"]: v for v in cfg["variants"]}
        if args.output.exists():
            raise ValueError("output already exists; use a new result path")
        if args.trial:
            source, target, mode = args.trial
            record = run_trial(cfg, variants[source], variants[target], mode, args.work_dir)
            write_json(args.output, record)
            return 0 if record["status"] in ("measured", "unsupported") else 1
        args.work_dir.mkdir(parents=True, exist_ok=True)
        session = Path(tempfile.mkdtemp(prefix="swap-", dir=args.work_dir.resolve()))
        records = []
        evidence = {"schema_version": 1, "config": cfg,
                    "cache_policy": "OS page cache as found, primed by base restore; no Onggi chunk cache",
                    "timing_scope": "old runtime release through first checked target response; base preparation excluded",
                    "metadata": {"python": platform.python_version(), "platform": platform.platform(),
                                 "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                 "adapters_sha256": hashlib.sha256(Path(__file__).with_name("swap_adapters.py").read_bytes()).hexdigest()},
                    "session_dir": str(session), "complete": False}
        for pair in cfg["transitions"]:
            for repetition in range(cfg["repetitions"]):
                modes = cfg["modes"]
                offset = repetition % len(modes)
                for mode in modes[offset:] + modes[:offset]:
                    number = len(records)
                    trial_output = session / f"{number}.json"
                    log = session / f"{number}.log"
                    command = [sys.executable, str(Path(__file__).resolve()), str(args.config.resolve()),
                               "--output", str(trial_output), "--work-dir", str(session / f"trial-{number}"),
                               "--trial", pair["source"], pair["target"], mode]
                    # Do not forcibly kill a vendor owner holding live device buffers.
                    with log.open("wb") as stream:
                        proc = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
                    if trial_output.exists():
                        record = json.loads(trial_output.read_text(encoding="utf-8"))
                    else:
                        record = dict(pair, mode=mode, status="error", stage="worker", cleanup="unknown",
                                      error=f"worker exited {proc.returncode} without a result; see {log.name}")
                    if proc.returncode and record["status"] != "error":
                        record.update(status="error", stage="worker", error=f"worker exited {proc.returncode}")
                    record.update(repetition=repetition, worker_exit=proc.returncode, log=log.name)
                    records.append(record)
                    report = summarize(records)
                    evidence.update(trials=report["trials"], summary=report["groups"], ok=report["ok"])
                    write_json(args.output, evidence)
                    print(f"{pair['source']}->{pair['target']} {mode} #{repetition}: {record['status']}", flush=True)
                    if record.get("cleanup") in ("error", "unknown"):
                        print("Stopping: inspect uncertain device cleanup before another trial", file=sys.stderr)
                        return 1
        evidence["complete"] = True
        write_json(args.output, evidence)
        print(json.dumps({"ok": evidence["ok"], "summary": evidence["summary"]}, indent=2))
        return 0 if evidence["ok"] else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"swap-bench: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
