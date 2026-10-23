# Experiment records

One Markdown file per run: `YYYY-MM-DD-<question>-<slug>.md`, with the Issue, hypothesis, hardware (device, driver,
firmware, SDK and compiler versions), model and settings, commands, raw numbers (or a `data/` file), and the
conclusion, including when the answer is no. Node names use aliases (`npu-node-1`).

## Common replacement benchmark (Issue #12)

`scripts/swap_bench.py` connects artifact materialization to the first checked
response on ATOM or RNGD. Python 3.10+ is required; the runner uses the standard
library. Raw ATOM uses installed NumPy/rebel, Optimum ATOM uses optimum-rbln and
transformers, and RNGD uses furiosa-llm. No SDK is imported by `--help` or invalid
config handling. CPU adapter tests install NumPy but replace vendor SDK modules.

1. Use a dedicated Onggi store. Push each complete, precompiled variant, recording
   the immutable manifest in the JSON output:

   ```sh
   onggi push --json --store STORE --ref experiment@variant-b1 BUILD_B1
   onggi push --json --store STORE --ref experiment@variant-b4 BUILD_B4
   ```

   For raw ATOM, each build directory contains `model.rbln`. Optimum uses an
   exported bundle including tokenizer files; RNGD examples use `model/` for the
   checkpoint/tokenizer and `model.fxb` for the matching precompiled FXB. Symlinks
   that escape an artifact are rejected by adapters. Stage source trees or
   hardlinks may be used for ingest only, never as a mutable delta base.

2. Copy a [config example](configs/), replace the store, executable, device index,
   runtime paths, and **placeholder manifests**. The default examples use five
   repetitions and both directions. Use an allocated idle device; the runner
   never discovers or reserves devices. Validate batch/sequence compatibility
   against the compiler metadata before loading. Different request lengths can
   exercise precompiled buckets, but a requested shape is not proof of the
   internal bucket selected by a scheduler.

3. Run on the device node in the matching SDK environment:

   ```sh
   python experiments/scripts/swap_bench.py config.json \
     --output new-result.json --work-dir disposable-work
   python -m unittest discover -s tests -v
   ```

   To enable the actual Onggi file-restore integration test as well:

   ```sh
   ONGGI_BIN=/absolute/path/to/onggi python -m unittest discover -s tests -v
   ```

Each trial is a fresh process with private restored files. Preparing the source
artifact, loading it and making its first request are excluded. The measured
interval is old-runtime **release → restore → load → first response**, including
fingerprinting for raw steps. It ends before target cleanup. The per-stage
`adapter_timings` separate host parse/device construction where the public SDK
exposes them; these are components of `load_s`, not additional time to add.
Python/SDK imports during base preparation are excluded from the interval.

- `full`: restore every target byte into a new directory with Onggi, then reload.
- `delta`: after source release, give Onggi the private base with
  `--base`/`--base-dir`, then reload. `pull_stats.read_bytes` counts compressed
  store payload, `bytes` is logical artifact size, and `reused` is data left in
  place. These counters exclude manifest/index/locator traffic and are not total
  network traffic. Delta reuse changes files, not resident NPU memory.
- `resident`: keep the existing runtime and make the target request. Supported
  only when manifest and runtime configuration are identical. ATOM's bundle must
  already include compatible shared-context graphs. Separately compiled graphs
  are outside this runner's resident adapter. No context IDs are forced or patched.

The Optimum/RNGD examples deliberately share one manifest across batch sizes:
their delta path is a zero-artifact-change control, while resident requests test
selection within the existing bundle/engine. To compare genuinely different
compiled artifacts, supply their actual manifests; resident will be unsupported.
Token adapters default to one greedy token; request `max_new_tokens` accepts a
positive integer for decoder probes, allowing early EOS. The token examples use
eight tokens so generation reaches decode. Use a separate config with one token
to measure first-token response; one-token generation alone cannot establish
decoder graph switching. `first_response_s` always measures the whole batch
completion, not streaming per-request TTFT, and includes all requested decode
steps when the limit is greater than one.
Raw ATOM reports a full-array SHA256 for a zero-input model step, not a serving
token. Do not pool those scopes or compare absolute times across devices/models.

Full-mode target responses establish a per-transition correctness reference.
Every successful full run must agree; mismatches, missing references, operation
failures and cleanup failures are excluded from successful latency samples and
make the command exit nonzero. Unsupported resident cases are counted separately.
The JSON is updated after every trial and marked `complete` only at the end.
Per-transition/mode summaries use nearest-rank p50/p95 and expose sample/failure
counts. Five trials are a smoke test, not a reliable tail-latency estimate.

Modes rotate per repetition. There is no separate warm-up exclusion: raw first
use per trial remains measured. OS page cache is **as found**, primed by source
preparation, with no Onggi chunk cache; store and destination should be on the
same declared filesystems throughout a comparison. No global cache is dropped,
and no cold-machine or durable-write latency claim is made. Resident requests
can benefit from vendor prefix/KV caches; the runner does not disable them or
prove a particular internal bucket was used. Keep the prompts/config and report
that distinction when interpreting the difference.

Logs remain in a uniquely named session under `--work-dir`; disposable trial
artifacts are removed after successful release. Existing output files and trial
directories are never overwritten. SDK worker processes have no forced timeout:
run inside an isolated, resource-allocated job with operator-controlled lifetime,
inspect a stuck device owner before teardown, and preserve failure evidence.
ATOM release follows the existing Q4 reference-drop/GC path and captures Python
destructor errors, but cannot prove firmware-level release. Constructor failures
retain files and report unknown cleanup; the parent stops before another trial.
Publishing results requires sanitized site paths, model,
SDK/driver versions, source/config/artifact hashes and a cleanup record.

For actual checkpoint-tensor updates and custom shared-weight graph swapping,
see the existing [rngd-pd implementation and verification](https://github.com/ai-computing/rngd-pd/pull/22)
and its native `rngd_pd.swap.bench_rngd` / `bench_atom_bundles` harnesses.
This file-materialization experiment does not replace those engine tests or
establish a missing hardware capability. An unsupported resident case here means
this adapter does not bind a different artifact, not that every NPU runtime is
incapable of retaining weights. Tensor updates, graph selection, file restoration
and vendor-engine reconstruction have distinct timing boundaries.
