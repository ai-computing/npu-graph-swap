# Common ATOM/RNGD replacement benchmark

Issue: #12. Implements the comparison approved in the conversation: full restore,
Onggi delta restore, and supported resident selection, through the first correct
response. This remains an experiment, not a serving engine.

## Approach and scope

Extend this repository's Q1–Q4 scripts rather than add vendor dependencies to
Onggi or put device code in research-hub. A Python 3.10+ standard-library runner
uses lazy SDK adapters on the allocated device. ATOM raw compiled graphs report
first **step**, while Optimum ATOM and furiosa-llm report a generated **response**; these are never
pooled. Engine integrations and arbitrary resident-weight rebinding are outside
this change. Vendor and original experiment files remain untouched.

Alternatives: importing Q4 logs cannot measure the connected path; a production
engine patch crosses the engine ownership boundary. A small runner with explicit
SDK adapters is the chosen implementation decision.

## Input and trial

JSON specifies device (`atom` or `rngd`), nonnegative `device_index`, Onggi
executable/store, repetitions, modes, named variants, and source→target pairs.
Each variant specifies an immutable `sha256:` manifest, a `runtime` object, and a
`request` object. Relative artifact paths must stay within the restored tree.
No shell commands, implicit device discovery, or global page-cache drops.

Modes are `full`, `delta`, `resident`. Each trial runs in a fresh child process
and its own temporary directory, without a shared Onggi chunk cache. Base restore,
base load and a base request establish the starting state outside the measurement.
The timed interval is old-runtime release → target restore → target construction
→ first target response. Delta uses `--base`/`--base-dir` only after release and
may consume only its private base. Resident is supported only for the same
manifest and equal runtime settings, with a different request shape if desired;
it reuses the base runtime. Different artifacts are explicitly unsupported.
Modes rotate per repetition to reduce systematic ordering bias. OS page cache is
as found, primed by base preparation; no cold-cache claim is made.

The adapter interface is `open_model(device, device_index, artifact_dir, runtime)`.
Its returned object has `timings` (host parse/runtime construction where exposed),
`request(request) -> {kind, value}`, and `close()`. SDK imports are lazy. ATOM's
`runtime.graph` names a `.rbln`; `request` is empty and inputs follow the Q4 zero
fixture. RNGD's `runtime.model` and `runtime.fxb` name restored model/FXB paths;
ATOM `runtime.backend=optimum` instead loads `runtime.model`, with an optional
artifact-relative `runtime.tokenizer` (defaults to model), through Optimum's
existing public Llama adapter and returns generated token IDs.
`runtime.max_model_len` defaults to 1024; request `prompts` is a nonempty string
list. Token requests are greedy and accept positive integer `max_new_tokens`
(default 1), with early EOS permitted. Decoder probes use more than one generated
token; one-token generation measures prefill only. Whole-batch response wall
time is not labelled streaming single-request TTFT. ATOM fingerprints must include output dtype, shape and all
bytes, rejecting nonfinite outputs; RNGD records nonempty token IDs per prompt.

## Evidence and failure semantics

Record source/target, mode, repetition, scope, release/restore/load/first-response
and end-to-end seconds, exposed adapter timings, Onggi byte counters, response
fingerprint, errors, and cleanup outcome. Compare each target's results with
successful full-mode results for the same target. Missing reference, response
mismatch, command error, or cleanup error cannot count as success. Unsupported
resident transitions remain visible and are excluded from latency aggregates.
Emit per-transition/mode p50/p95 using nearest rank, sample counts and failed/
unsupported counts. An all-failed or unverified run exits nonzero; partial
failures also exit nonzero. Store partial results after each trial.

Only disposable runner directories may be removed. No automatic forced timeout
of an SDK process owning device memory: run in an isolated, resource-allocated
job and inspect a stuck owner before teardown. Hardware data must identify SDK,
driver, artifact/config/source hashes and device alias, not publish site paths.

## Acceptance

CPU tests exercise real subprocess CLI failures, restore arguments/order,
private-directory isolation, request/cleanup failures, missing correctness
reference, mismatches, unsupported resident mode and quantiles. Run available
hardware checks without recompiling into existing contexts or touching live
services. Record the distinction between actual model steps and serving tokens,
and disclose unsupported or unavailable paths.
