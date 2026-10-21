# Q1 (ATOM): one compile with decoder batch sizes {1, 4, 8} — shared weights, memory, switch latency

- Issue: #1
- Date: 2026-10-21
- Hypothesis: optimum-rbln compiled with prefill plus several decoder batch sizes in one build keeps one copy of the
  weights on the device; switching between its decoder graphs costs milliseconds, and per-step time equals the
  separately compiled variants.

## Setup

| Item | Value |
|---|---|
| Node | `npu-node-1` (shared; 2 × AMD EPYC 9254, 96 CPUs). Load average **350–510** during compiles and runs (other users' jobs) |
| Device | Rebellions ATOM RBLN-CA22, 15.7 GiB, PCIe 32 GT/s x16; one card per build (`rbln0` for 8B, `rbln0`/`rbln1` for 1B) |
| Driver / firmware | KMD 3.0.0 / FW 3.0.0 |
| SDK | rebel-compiler 0.11.0, optimum-rbln 0.11.0.post1, torch 2.11.0+cpu, transformers 5.8.1, Python 3.10 |
| Models | Llama-3.1-8B-Instruct (main), Llama-3.2-1B-Instruct (method check) |
| Settings | `max_seq_len=512`, `tensor_parallel_size=1`, default attention (eager, KV block 512), default dtype |

The vendor option is `rbln_decoder_batch_sizes` (`RBLNDecoderOnlyModelForCausalLMConfig.decoder_batch_sizes`): one
`decoder_batch_<b>` graph per listed size, all compiled in one `rebel.CompileContext(use_weight_sharing=True)` with
the prefill graph; the largest must equal `batch_size`. The model then routes a decode call with batch `b` to
`decoders[b]`.

Builds (all new, under one scratch directory, deleted at the end):

| Build | Command | Compile time | Peak host RSS | Files |
|---|---|---:|---:|---|
| 8B combined | `compile_opt.py <8B> out 8 512 8,4,1` | 1006.6 s (+22.6 s save) | 55 GB | prefill.rbln 16.1 GB, decoder_batch_{1,4,8}.rbln 1.09 / 1.15 / 1.10 GB |
| 8B separate B1 | `compile_opt.py <8B> out 1 512 1` | 533.8 s | 57 GB | 17 GB |
| 8B separate B4 | `compile_opt.py <8B> out 4 512 4` | 546.0 s | 54 GB | 17 GB |
| 8B separate B8 | `compile_opt.py <8B> out 8 512 8` | 650.0 s | 55 GB | 17 GB |
| 1B combined | `compile_opt.py <1B> out 8 512 8,4,1` | 640.4 s | 11.7 GB | 4.4 GB |
| 1B separate B1/B4/B8 | as above | 303.6 / 167.3 / 215.1 s | 11–11.6 GB | 3.4 GB each |

The four 8B builds were compiled concurrently (each at ≈ 135 % CPU), so compile times are inflated by contention.
A first 8B combined compile failed at the end (`SYS_EBUSY: Lack of device memory`) because `from_pretrained` loads the
result on device 0 by default and another of my runs held that card; all later compiles use
`rbln_create_runtimes=False`.

"Separately compiled variants" here are optimum-rbln builds with a single decoder batch size, so the comparison is
like for like. The older custom decode builds in the user's directories take the KV cache as a host input and are
not comparable for step time; one of them was used for Q4.

## Method

Scripts: [`scripts/compile_opt.py`](scripts/compile_opt.py), [`scripts/q1_measure.py`](scripts/q1_measure.py)
(`RBLN_RUNTIME_TIMER=1` for per-call device time).

```sh
HF_HUB_OFFLINE=1 python3 compile_opt.py <Llama-3.1-8B-Instruct dir> builds/l8_s512_multi 8 512 8,4,1
HF_HUB_OFFLINE=1 python3 compile_opt.py <Llama-3.1-8B-Instruct dir> builds/l8_s512_b4 4 512 4      # also b1, b8
python3 q1_measure.py builds/l8_s512_multi 0 results/q1_l8_multi.json <Llama-3.1-8B-Instruct dir>
python3 q1_measure.py builds/l8_s512_b4 0 results/q1_l8_b4.json <Llama-3.1-8B-Instruct dir>
```

1. Load the build with `RBLNLlamaForCausalLM.from_pretrained(dir, export=False)`; `rbln-stat` device memory before
   and after; per-graph allocation from `runtime.get_alloc_per_node_by_key()` and the context id of every graph.
2. Decode-graph timing on the raw runtimes (`model.model[i].run`) with fixed dummy inputs (`input_ids` 0,
   `cache_position` 5, one KV block per row): first call of each graph after load (B8, B4, B1 order), second call,
   50 calls of the same graph (after 5 warm-up calls), and for every ordered pair a→c 50 times "run a, then time c".
   Switch overhead = median(c after a) − median(c after c). Both host wall time and device time (runtime timer) are
   recorded.
3. Correctness: greedy `generate` (32 new tokens, left padding) on fixed prompts with batch 1, 4 and 8, which routes
   to `decoder_batch_1/4/8`; token ids compared with the separately compiled build of the same batch size.

## Results — Llama-3.1-8B, S=512

Device memory after load (`rbln-stat`):

| Build | Device memory | prefill alloc (weights `Kernel`) | decoder graph alloc |
|---|---:|---:|---|
| **combined {8, 4, 1}** | **14.7 GiB** | 14.54 GiB (13.98 GiB) | B8 50 MiB, B4 96 MiB, B1 42 MiB, no `Kernel` |
| separate B1 | 14.2 GiB | 14.17 GiB | 42 MiB |
| separate B4 | 14.4 GiB | 14.29 GiB | 96 MiB |
| separate B8 | 14.6 GiB | 14.54 GiB | 50 MiB |

All four graphs of the combined build have the same context id. The prefill differences between separate builds are
the KV cache (`DramTensor`, one 512-token block per batch row). Keeping the three separate builds resident would need
43.2 GiB (three cards); the combined build needs 0.1 GiB more than the B8 build alone.

Decode step, device time (ms, p10 / p50 / p90 of 50 calls):

| Batch | combined, same graph | combined, after switching from another batch | separate build |
|---:|---|---|---|
| 1 | 65.02 / **65.13** / 68.65 | from 4: 65.10, from 8: 65.07 (p50) | 65.01 / **65.06** / 67.59 |
| 4 | 66.45 / **66.52** / 72.00 | from 1: 66.53, from 8: 66.52 | 66.41 / **66.48** / 68.71 |
| 8 | 69.50 / **69.57** / 71.99 | from 1: 69.55, from 4: 69.53 | 69.48 / **69.53** / 71.13 |

Switch overhead on the device (p50, after − same): **−0.06 to +0.01 ms** for all six directions, i.e. none
measurable. First call of each graph after load: B8 69.6 ms, B4 66.5 ms, B1 66.9 ms device time (vs. 65.1–69.6 steady):
no warm-up penalty beyond ≈ 2 ms.

Host wall time per call: B1 65.5 ms (≈ device). For B4 and B8 the wall time is 150–820 ms in **both** the combined and
the separate builds (separate B8 p50 209 ms, combined B8 p50 514 ms, with large run-to-run spread), i.e. host-side
work in the runtime that grows with batch and with the host load (≈ 350–490 here). It is not caused by sharing and
was not investigated further; the switch numbers above use device time for that reason. Wall-time switch "overheads"
are negative or within that noise.

Correctness — greedy tokens, combined vs separate build:

| Batch | tokens compared | identical |
|---:|---:|---|
| 1 | 32 | **32 / 32** |
| 4 | 128 | **128 / 128** |
| 8 | 256 | **256 / 256** |

Load time (`from_pretrained`, warm page cache, host load ≈ 350–380): combined 53.0 s, separate 20.6–25.9 s.

## Results — Llama-3.2-1B, S=512 (method check)

Same picture: combined build 2.6 GiB vs separate B8 2.5 GiB, decoders without `Kernel` (12–26 MiB each), one context
id; device step p50 B1 11.14 ms (separate 11.18), B4 13.98 ms (11.56; p10 11.54 vs 11.53), B8 13.35 ms (13.30);
device switch overhead −2.4 to +0.3 ms (noise); tokens identical for batch 1, 4 and 8 (32/128/256).

Raw data: [`data/2026-10-21-q1/`](data/2026-10-21-q1/) (`q1_<model>_<build>.json`, with every call's wall and device time
in `all_calls`).

## Conclusion

**Yes**, for graphs compiled together. On ATOM with optimum-rbln 0.11, one build with prefill and decoder batch sizes
{1, 4, 8} keeps **one** weight copy on the card (14.7 GiB vs 14.6 GiB for the B8-only build; each extra decoder adds
42–96 MiB), switching between its decoder graphs costs **nothing measurable on the device** (|Δ| ≤ 0.06 ms against a
65–70 ms step, also on the first call), per-step device time equals the separately compiled builds (within 0.1 ms),
and greedy outputs are **token-exact** to the separate builds.

Conditions and limits:

- All variants must be known at compile time and compiled in one session; adding a batch size later means compiling
  in that session again (see Q3 for compiling a graph next to loaded weights).
- The KV cache is sized for the largest batch (8 blocks of 512 tokens here); the B1 graph uses one block of it.
- The decoder `.rbln` files are ≈ 1.1 GB each on disk although they allocate only 42–96 MiB on the device.
- S=2048 was not built: at batch 8 the KV cache alone would be 2 GiB, which with 14 GiB of weights does not fit one
  15.7 GiB card; it needs fewer KV blocks or two cards.
- Host wall time for batch > 1 is dominated by host-side runtime work on this loaded machine (see above).
- One card, one run per build; node shared with heavy CPU load.
