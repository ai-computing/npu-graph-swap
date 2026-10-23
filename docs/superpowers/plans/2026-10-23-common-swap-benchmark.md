# Common swap benchmark implementation plan

> **For agentic workers:** Use executing-plans for the runner and independent adapter implementation, followed by whole-branch review.

**Goal:** Compare full and delta artifact replacement through a correct first response on ATOM/RNGD.

**Architecture:** One standard-library runner with lazy vendor SDK adapters, private fresh-process trials, explicit immutable artifact configuration, and correctness-gated summaries.

**Tech Stack:** Python 3.10+, unittest; installed Onggi/rebel/furiosa-llm only for integration runs.

**Spec:** ../specs/2026-10-23-common-swap-design.md

## Global constraints

- Experiments only; vendor and engine files stay untouched.
- No shell commands, implicit device selection, shared mutable base, or cold-cache claims.
- Distinguish raw ATOM model steps, generated-batch completion and native prompt TTFT.
- Any failure, unverified output or cleanup error must produce nonzero exit.

## Review focus

- A device release failure must stop delta consumption of the base.
- Different artifacts must never enter resident mode.
- Incorrect/missing full references cannot yield successful latency samples.
- Relative artifact paths, immutable digests and nonfinite outputs require validation.
- Repetitions and p95 must expose small sample counts and per-transition scope.

### Task 1: runner and evidence schema

Files: `experiments/scripts/swap_bench.py`, `tests/test_swap_bench.py`.

- [x] Write failing unittest cases for validation, stage ordering, isolation, failures and correctness-gated summaries; observe RED.
- [x] Implement `validate_config`, `run_trial`, `summarize` and CLI with private child-process trials and incremental JSON evidence.
- [x] Run `python -m unittest discover -s tests -v`; inspect errors and assertions, including actual CLI subprocess execution.

### Task 2: SDK adapters

Files: `experiments/scripts/swap_adapters.py`, `tests/test_swap_adapters.py`.

Interface: `open_model(device, device_index, artifact_dir, runtime)` returns an object with `timings`, `request(dict) -> {kind,value}`, `close()`.

- [x] Write failing cases using scoped fake SDK modules to assert public API calls, explicit device selection, complete fingerprinting and invalid-output failures; observe RED.
- [x] Implement ATOM raw-step/Optimum and RNGD generation adapters with lazy imports and synchronous release.
- [x] Run the complete unittest suite.

### Task 3: integration and reproducibility

Files: `experiments/configs/*`, experiment record/data, English/Korean READMEs, CI workflow.

- [x] Add placeholder configs, exact invocation/stage definitions, failure and cache-policy documentation.
- [x] Exercise real Onggi full/delta restoration with byte-checked fixtures.
- [x] Run available device-allocated hardware trials; record unsupported/environment limitations and clean temporary resources. Completed raw/Optimum ATOM and RNGD O0→O3 profiles, plus a separately scoped native ATOM run; RNGD resident/native paths were not run after the objective changed to resident-weight updates.
- [x] Fresh independent review, meaningful fixes with regression tests, full suite, sensitive-data and diff checks; open a reviewable PR with evidence. PR #13 remains draft because the broader RNGD resident/native acceptance coverage is explicitly unverified; further device-update work is separate.

## Execution evidence

- Baseline: no tests existed. Initial runner cases: 15 expected missing-feature failures → 15 passed.
- Initial adapters: missing-feature assertions → 13 passed, one Windows symlink-permission skip.
- Fresh independent review identified constructor ownership uncertainty, destructor errors hidden by Python, and one-token probes failing to exercise decode.
- Constructor ownership regression: both base and target cases failed with cleanup incorrectly `ok` → passed with `unknown`, retained files, and parent stop.
- Destructor regressions: raw and Optimum close incorrectly succeeded → passed with captured errors and restored hook.
- Generation-limit regressions: both adapters rejected multi-token requests → passed with per-request limits and early EOS handling.
- Python 3.10 test portability fixed after Linux exposed `TestCase.enterContext` as 3.11-only.
- CLI integration proves six distinct worker PIDs, correctness-gated output and private artifact cleanup.
- Full suite: 37 cases; Windows 35 passed/2 skipped, Linux Python 3.10 36 passed/1 opt-in Onggi test skipped. With actual Onggi enabled on Linux: all 37 passed, no skips, 1.555 s. Python 3.10/3.12 CPU CI passed. Completed hardware evidence is recorded in `experiments/2026-10-23-file-versus-runtime-swap.md`; no further benchmark matrix is planned under this file-restoration task.
- Review boundary: no firmware-release guarantee inferred from Python GC; disjoint hardware source/target prompt prefixes and explicit cache-scope disclosure are used instead of modifying vendor cache behavior.
