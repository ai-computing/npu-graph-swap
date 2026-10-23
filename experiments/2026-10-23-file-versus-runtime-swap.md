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
experiment.
