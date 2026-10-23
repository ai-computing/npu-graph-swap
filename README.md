# npu-graph-swap

**Can an NPU switch between compiled graphs (sequence length, batch) while the weights stay loaded?**

English · [한국어](README.ko.md)

> Status: **feasibility study.** Experiments only; nothing here is a supported library yet.

## Why

NPUs run ahead-of-time compiled graphs, one per sequence-length bucket and batch size. A server that sees changing
load wants to switch graphs: batch 1 when traffic is light, batch 8 when it queues; a short-sequence graph for short
prompts. GPUs do this today by capturing one CUDA graph per batch size over the same weights.

The [onggi](https://github.com/ai-computing/onggi/blob/main/docs/artifact-formats.md) measurements show why it should
be possible on NPUs too:

- Rebellions ATOM decode builds for different sequence lengths and batch sizes share **98.5–99.7 %** of their bytes;
  each variant adds only 50–190 MB on top of ≈ 15 GB of weights. The weights are laid out the same way; only the
  program differs.
- FuriosaAI RNGD already keeps graphs (`.edf`, KB–MB) apart from the weights, and furiosa-llm selects among buckets
  that share weights.

But today, on ATOM, each separately compiled variant embeds its own weight copy: switching means reloading ≈ 14 GB,
and keeping two variants warm doubles device memory.

## Questions

| # | Question | Device |
|---|---|---|
| 1 | Does a single compile with several decoder batch sizes (and prefill) share one weight copy on the device? Memory, switch latency between its graphs, step time vs separate compiles. | ATOM |
| 2 | How fast does furiosa-llm switch between buckets, and how does device memory grow with the number of buckets? | RNGD |
| 3 | Can a graph compiled separately attach to weights already on the device (runtime API, weight binding, shared buffers)? | ATOM, RNGD |
| 4 | Baseline: full reload time of one variant (load + first step). | ATOM, RNGD |
| 5 | If 1–3 allow it: a load-driven graph selector (hysteresis, warm pool, prefetch) and its effect on throughput and SLO. | both |

Each question is an Issue with a hypothesis; each run is a record in `experiments/`.

The [common replacement benchmark](experiments/README.md#common-replacement-benchmark-issue-12)
compares full restore, Onggi delta restore, and supported resident selection on
ATOM/RNGD, separating file restore, runtime loading and the first checked response.
It reports unsupported paths and correctness failures explicitly; file delta
restoration does not imply device-memory delta loading.

The [completed measurements](experiments/2026-10-23-file-versus-runtime-swap.md)
retain the raw ATOM delta path's lack of a demonstrated local latency win,
positive file-reuse results for ATOM Optimum and RNGD, and a separate native ATOM
selection/TTFT run. All samples and tokens are linked there. Actual changes to
NPU-resident weights remain separate work; the unrun RNGD resident/native paths
are explicitly disclosed.

## Where results go

- Possible → a spec for the engine: graph switching in [Optimus Prime](https://github.com/ai-computing/aicomp) as a
  feature request (this repository does not change engine or vendor code), and graph-delta metadata in onggi.
- Not possible → the runtime requirement written down for the vendors, with the measured cost of the alternatives.

## Related

[ai-rm](https://github.com/ai-computing/ai-rm) · [onggi](https://github.com/ai-computing/onggi) ·
[rngd-pd](https://github.com/ai-computing/rngd-pd) · [npu-train](https://github.com/ai-computing/npu-train)

License: [Apache-2.0](LICENSE).
