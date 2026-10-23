# Q12: file restoration versus runtime replacement

- Issue: [#12](https://github.com/ai-computing/npu-graph-swap/issues/12).
- Hypothesis: Onggi delta restoration reduces replacement-through-response
  latency compared with full restoration; resident selection avoids those
  restoration and construction stages where the runtime already supports it.
- Implementation: `0d7016a4332ebcab2ae9b067a99d6e2b264d80a6`.
- Artifact/code hashes, software versions and CPU checks:
  [validation.json](data/2026-10-23-common-swap/validation.json).

## Method

The [common runner](README.md#common-replacement-benchmark-issue-12) uses the
same immutable source and destination manifests in fresh-process trials. It
restores/loads the source and makes a source request before the timed interval.
Timing then includes old-runtime release, target materialization, new runtime
construction and the first checked target response. Target cleanup follows the
interval but must succeed for a valid sample. Imports and source preparation
are excluded. Each mode has two repetitions, rotated in order. Nearest-rank
p50/p95 are shown with sample counts; two samples are a smoke test, not a
statistically reliable tail estimate.

`full` means **full Onggi file restoration plus runtime reconstruction**, not a
runtime-only load from an already local artifact. `delta` offers the private
source tree to Onggi after releasing its runtime, then reconstructs the target
runtime. `resident` retains that same runtime and changes the requested workload;
this adapter supports only an identical manifest/runtime configuration. It does
not implement another engine's external-weight binding API.

Hardware: one ATOM RBLN-CA22 (15.7 GiB) or one RNGD (47.5 GiB) on `npu-node-1`,
2 × AMD EPYC 9254, 96 logical CPUs. Each allocated Pod is limited to 4 CPUs and
64 GiB RAM. Linux 6.8.0-40; ATOM KMD/FW/SMC 3.0.0, RNGD FW 2026.4.0 (module
version query exposed no driver version). Source, store and destinations use
the same ext4/LVM filesystem; backing disks report nonrotational, without
establishing NVMe. OS page cache is as found and primed by source preparation,
with no Onggi chunk cache. These are local filesystem measurements, not network
delivery or cold-machine measurements.

All large artifact ingestion and primer restoration finished before formal
measurements. An earlier ATOM full-mode functional pilot overlapped ingestion
and passed output/cleanup checks; its timing is excluded. Source ingestion
views use hardlinks only for reading. Every consumed delta base is independently
restored, and inode separation from original artifacts was checked.

## ATOM: separately compiled raw B1 → B4

Llama-3.1-8B, S512; independent decode artifacts of 14,002,676,629 and
14,152,348,510 bytes, rebel-compiler 0.11.0. Inputs are the existing Q4 zero-input
fixture, including host KV. This is a model **step**, not a full prompt or a
serving-token measurement. All output arrays are finite and fingerprinted using
dtype, shape and every output byte.

| Mode | Successful samples | Restore p50 / p95 (s) | Load p50 / p95 (s) | Replacement → checked step p50 / p95 (s) |
|---|---:|---:|---:|---:|
| Full restoration | 2 | 35.269 / 38.131 | 13.842 / 14.130 | **49.606 / 52.860** |
| Delta restoration | 2 | 38.267 / 49.175 | 11.962 / 14.122 | **50.719 / 63.812** |
| Resident adapter | 0; 2 unsupported | — | — | Different manifests; no binding attempted |

The delta path reduced compressed store-payload reads from **9,222,267,653 B**
to **3,434,136 B**, but read **13,959,156,356 B** from the local base. No bytes
were left at the same file offsets (`reused=0`, `base_best=0`); Onggi instead
reconstructed the target from shifted local data (`local_base`/`local_bytes`).
This is substantial local reuse, not a failure to deduplicate. It still requires
local reads/writes and the subsequent runtime parse.

There is **no demonstrated latency win** for this delta path on this local
filesystem. Both delta observations were slower than the faster full observation;
the small sample and shared-node load do not establish a universal slowdown.
Network delivery could value the saved store payload differently and is not
measured here. Host parsing accounts for roughly 11.45–13.58 s of delta loading;
device runtime construction itself was about 0.42–0.44 s. File byte savings do
not imply that either host parsing or device weight upload was skipped.

All four measured outputs match the same full-mode fingerprint, and cleanup
succeeded. [Raw numeric evidence and reproducible config](data/2026-10-23-common-swap/atom-raw.json)
retain both samples; the excluded pilot was not selected for a favorable number.

## ATOM: one Optimum bundle, B1 → B4 request

Llama-3.1-8B, S512, one co-compiled Optimum bundle with decoder batches 1 and 4,
18,347,319,308 artifact bytes including configuration/tokenizer. rebel 0.11.0,
optimum-rbln 0.11.0.post1, torch-rbln 0.2.2 and PyTorch 2.11.0+cpu. The source
and target use **the same manifest and runtime settings**; their batch requests
differ. This is a zero-artifact-change restoration control and an intra-bundle
resident request, not a demonstration of binding a separately compiled bundle.

All target requests use four distinct prompts and up to eight greedy generated
tokens per prompt; the source prompt has a different first word to limit shared
prefix-cache effects. These runs produced eight tokens in every target row.
Times below end at **whole batch completion**, not at its first token. More than
one token is requested so execution includes decode as well as prefill.

| Mode | Successful samples | Restore p50 / p95 (s) | Load p50 / p95 (s) | Replacement → batch completion p50 / p95 (s) |
|---|---:|---:|---:|---:|
| Full restoration | 2 | 33.422 / 46.062 | 15.536 / 23.886 | **50.570 / 71.415** |
| Delta restoration | 2 | 8.109 / 13.153 | 20.124 / 23.880 | **29.824 / 39.115** |
| Resident runtime | 2 | 0 / 0 | 0 / 0 | **0.7576 / 0.7601** |

Delta left all 18,347,319,308 bytes in place and read 0 B of compressed store
payload, compared with 10,637,033,436 B for full restoration. Metadata/base
processing still took 8.1–13.2 s and Optimum model construction 20.1–23.9 s.
Retaining the already loaded runtime avoided both stages. The small sample,
OS/vendor caches and shared host prohibit a general speedup or SLO guarantee.

All six target token sequences equal the full-mode reference, and every trial
released its owner and removed its private artifacts. See
[all samples and exact requests](data/2026-10-23-common-swap/atom-optimum.json).
This positive control does not overturn the preceding shifted-artifact result:
same-offset file reuse and resident runtime selection differ from reconstructing
a separately compiled file whose matching bytes moved.

## RNGD: compiled O0 → O3 artifact replacement

Llama-3.1-8B-Instruct, the same checkpoint and maximum model length 2048, TP8;
the O0 and O3 FXBs are genuinely different compiled artifacts. The source uses
one prompt and the target four distinct prompts, with disjoint first words.
Each target generated eight greedy tokens per row. furiosa-llm 2026.3.0,
furiosa-torch 2026.2.0 and PyTorch 2.10.0+cu128 were installed; this adapter
uses the vendor LLM API, not the custom furiosa-torch decode engine.

| Mode | Successful samples | Restore p50 / p95 (s) | Load p50 / p95 (s) | Replacement → batch completion p50 / p95 (s) |
|---|---:|---:|---:|---:|
| Full restoration | 2 | 30.613 / 34.911 | 9.172 / 10.266 | **41.380 / 46.733** |
| Delta restoration | 2 | 0.481 / 0.486 | 8.449 / 8.759 | **10.671 / 10.924** |

The target contains **16,081,399,306 B**. Each delta trial left
**16,069,722,669 B** at the same offsets and read **2,716,384 B** of compressed
store payload, compared with **10,707,247,845 B** for each full restore. This
same-offset file reuse is a positive result for local artifact restoration.
It does **not** patch resident NPU weights: both modes release the source LLM,
materialize target files and construct a new vendor LLM, including weight load.
Runtime construction still took 8.4–8.8 s on the delta path.

All four trial responses match the full-mode destination reference; all cleanup
checks passed, the owner exited with code 0, and the post-run context listing
was empty. [Every trial, token sequence and exact config](data/2026-10-23-common-swap/rngd-o0-o3.json)
is retained. The two observations per mode are a smoke test; nearest-rank p95
is simply the larger observation. No streaming TTFT or exact internal bucket
selection is established by these whole-batch calls.

## Native ATOM: prepared shared-session selection and prompt TTFT

This separate run used the existing `rngd_pd.swap.bench_atom_bundles` harness,
not the common file-restoration runner. The independently copied source is
[`rngd-pd` revision da85c80](https://github.com/ai-computing/rngd-pd/commit/da85c80ba5f0eb979d32bbcbbbcd4b1de13b0c13),
published on `validation/prepared-serving-20261023`
([draft PR #27](https://github.com/ai-computing/rngd-pd/pull/27)). It loaded the
same complete S512 Optimum bundle with co-compiled B1/B4 decoders and retained
the prefill weight owner and all runtimes. The selected request batch determines
the matching decoder. No independently compiled graph was rebound to another
bundle's weights.

One round used prompt IDs `[128000, 791, 6864, 315, 9822, 374]`, repeated across
the selected batch, and eight greedy generated tokens per row. Each fresh
baseline loaded the same complete bundle in a separate process. The native
streamer measures the first generated batch token; subsequent tokens are checked
for parity but excluded from TTFT.

| Selected shape | Observations | Prepared selection (µs) | Prompt TTFT (ms) | Selection → first token (ms) | Fresh load → first token (s) |
|---|---:|---:|---:|---:|---:|
| B1/S512 | 1 | 26.4 | 94.5 | 94.5 | 21.155 |
| B4/S512 | 1 | 53.4 | 283.7 | 283.8 | 22.244 |

The B1 row reselects the initially loaded B1 shape; B4 is the actual B1 → B4
transition. All 40 generated tokens across the two selected batches exactly
match their respective fresh baselines. SDK allocation reports attribute
15,011,414,016 B of Kernel weights to the prefill owner, with no Kernel entry
on either decoder. Selection uploaded no weights. This establishes this
bundle's prepared selection path, not changed-weight updates or arbitrary
graph binding. One observation per shape cannot estimate a distribution.

The native worker exited with code 0 and its post-run device-context listing
was empty. [All fresh/selected tokens, timings, allocations and path-free source provenance](data/2026-10-23-common-swap/native-atom.json)
are retained. Native teardown drops references without the common adapter's
destructor-error guard; neither proves firmware-level release. These TTFT
numbers must not be pooled with the common runner's whole-batch completion
times. Fresh native timing starts at bundle loading and excludes process
startup/imports.

To reproduce this separate native measurement from the pinned source snapshot:

```sh
PYTHONPATH=SNAPSHOT/src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
  python -m rngd_pd.swap.bench_atom_bundles --bundle RESTORED_BUNDLE \
  --variants B1,S512 B4,S512 --device 0 --tokens 8 --rounds 1 \
  --out native-atom.json
```

## Reproduction and boundaries

Use the published config in each JSON evidence file, recreate its immutable
manifests with the same artifact bytes, and replace only the store/executable
paths and the allocated device index. Example invocation:

```sh
python experiments/scripts/swap_bench.py atom-raw-config.json \
  --output new-result.json --work-dir disposable-work
ONGGI_BIN=/absolute/path/to/onggi python -m unittest discover -s tests -v
```

The actual Onggi build used `05c97605b07882f9ceaaeb90b94878f8923f81cf` and Go
1.27.3. All 37 Linux CPU tests passed with the actual binary, including real
full/delta restoration of a byte-checked 4 MiB fixture. Windows passed 35 with
two explicit opt-in/platform skips; Python 3.10/3.12 CPU CI passed.

The companion [rngd-pd tensor-update implementation](https://github.com/ai-computing/rngd-pd/pull/22)
has different APIs, ownership and timing scopes. This experiment measures
file materialization and vendor-runtime reconstruction; it does not conclude
that all NPU runtimes lack partial device updates. The common runner's raw step
hash and generated-batch response metrics are distinct from the native swap
harnesses' graph selection and prompt TTFT.

Python destruction errors and construction failures stop the common runner
conservatively, but GC cannot establish firmware-level release. No fatal DMA
fault was injected on live hardware. Concurrent HTTP admission/draining,
streaming TTFT, production rollout and arbitrary KV migration are outside this
common-runner experiment. The separate native ATOM measurement above has its
own explicit first-token timing scope.

The RNGD same-artifact resident control and custom native RNGD benchmark were
**not run**: further benchmark expansion stopped when the requested objective
was clarified as actual updates to NPU-resident weights. Neither path is claimed
as verified here. That device-update work is pending separately; file restore
savings and unchanged-weight decoder selection do not satisfy it. This record
preserves the completed measurements without extending them into an in-place
NPU-update claim.
