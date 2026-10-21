# Q3: can a separately compiled graph attach to weights already on the device?

- Issue: #3
- Date: 2026-10-21
- Hypothesis: at least one runtime exposes a way (weight binding, external/shared device buffers, re-using a
  loaded model's tensors) to run a separately compiled graph against weights already loaded, without a second copy.

## Setup

| Item | Value |
|---|---|
| Node | `npu-node-1` (shared; load average 450–500 on 96 CPUs during the runs) |
| ATOM | RBLN-CA22 15.7 GiB, KMD 3.0.0, FW 3.0.0; one card per probe |
| ATOM SDK | rebel-compiler 0.11.0, optimum-rbln 0.11.0.post1, vllm-rbln 0.11.0, torch 2.11.0+cpu, transformers 5.8.1 |
| RNGD SDK (survey only) | furiosa-llm 2026.4.0, furiosa-torch 2026.4.0, RNGD FW 2026.4.0 |
| Models | toy: 8 × Linear(4096, 4096) without bias, fp32 host weights (268 MiB on the device); Llama-3.2-1B-Instruct; Llama-3.1-8B-Instruct |

Raw data: [`data/2026-10-21-q3/`](data/2026-10-21-q3/) (`q3_toy_round1.log` has a wrong memory column, fixed in
`q3_toy.log`; the A3/A4 files were saved in round 1). No vendor file was modified. Scripts: [`scripts/q3_toy.py`](scripts/q3_toy.py), [`scripts/q3_tc.py`](scripts/q3_tc.py),
[`scripts/q3_late.py`](scripts/q3_late.py). `q3_late.py` keeps optimum-rbln's compile context and wrapped model by
wrapping two classmethods at runtime (no file change).

```sh
DEV=2 python3 q3_toy.py same_session_late     # also global_late, save_share_1, save_share_4, load_share,
                                               # save_global_1, save_global_4, load_global, cached_then_fresh2, global_two_sessions
DEV=2 python3 q3_tc.py fresh && DEV=2 python3 q3_tc.py cached && DEV=2 python3 q3_tc.py partial
HF_HUB_OFFLINE=1 python3 q3_late.py <Llama-3.1-8B-Instruct dir> 2 results/q3_late_8b.json
```

## How weights are shared on ATOM today (from the public Python layer of `rebel`)

- Every compiled executor carries a **context id** stamped at compile time (`executor.get_context_id()`). At
  `create_runtime`, the runtime looks up a device `Context` by `(context id, device)` and creates it only if it does
  not exist yet. Runtimes with the same context id in the same process share that context: its constant (weight)
  buffer pool and its static DRAM tensors (the KV cache marked with `CompileContext.mark_static_address`).
- The context id comes from the `rebel.CompileContext` used at compile time: a fresh UUID per context, or a fixed
  per-device id (`kGlobalCtx<N>`) with `use_global_ctx=True`.
- With `use_weight_sharing=True`, a graph compiled in the same context as an earlier graph refers to the earlier
  graph's constant buffers instead of embedding its own. The weight identity is the host tensor (its storage) in the
  compiling process. `get_alloc_per_node_by_key()` shows it: the first graph has a `Kernel` entry (the weights), later
  graphs in the same context have none.
- optimum-rbln builds prefill and every `decoder_batch_<b>` in one `CompileContext(use_weight_sharing=True)`; this is
  what makes Q1's combined build share weights.
- The vendor `torch.compile` backend (`backend="rbln"`, used by vllm-rbln) compiles each new input shape in a global
  context with weight sharing, can cache compiled graphs on disk (`cache_dir`), and on a cache hit calls
  `RBLNCompiledModel.reserve_const_buf_indices(session)` so that later fresh compiles do not reuse its buffer indices.
  It streams weights from the live host module into the runtime (`apply_weight_free_weights`).
- Public API that does **not** exist: exporting a loaded runtime's or context's constant buffers by name, compiling
  a graph whose weights are references to named external buffers, or binding a compiled model to the constant
  buffers of an existing context. `rebel.Runtime`, `RBLNCompiledModel.create_runtime`, `Context` and `CompileContext`
  have no such parameter or method.

## Probes and results (ATOM)

### A. Toy model, one card

| # | Scenario | Device memory after graph 1 → after graph 2 | Graph 2 `Kernel` alloc | Outputs | Result |
|---|---|---|---:|---|---|
| A1 | one process, one `CompileContext(use_weight_sharing=True)`: compile B1, load and run it, **then** compile B4 in the same context, load and run | 268 MiB → **280 MiB** | none | match CPU (max abs err 5e-4) | **shared** |
| A2 | same with `use_global_ctx=True` | 268 MiB → **280 MiB** | none | match | **shared** |
| A3 | B1 and B4 compiled in two separate processes (UUID contexts), saved, loaded into one process | 268 MiB → **536 MiB** | 268 MiB | match | two copies |
| A4 | B1 and B4 compiled in separate processes with `use_global_ctx=True` (same context id), loaded into one process | 268 MiB → – | 268 MiB | B4 run: `SYS_TASK_ABORTED` | **crash** |
| A5 | load cached B1 (global ctx), `reserve_const_buf_indices(new session)`, compile B4 fresh, create both runtimes | 270 MiB | 268 MiB | B1 run: `SYS_TASK_ABORTED` | **crash** |
| A6 | one process, two different global-ctx sessions, one graph each | 270 MiB | 268 MiB | B1 run: `SYS_TASK_ABORTED` | **crash** |

### B. Vendor `torch.compile` path with an on-disk cache (toy model)

| # | Scenario | Memory after B1 → after B4 | Outputs | Result |
|---|---|---|---|---|
| B1 | fresh cache: B1 and B4 compiled in one process | 268 → **280 MiB** | match | shared |
| B2 | **new process, both graphs from the cache** | 268 → **280 MiB** | match | **shared** (B1 4.3 s, B4 0.77 s to first output) |
| B3 | new process, B1 from the cache, B4 compiled fresh | 268 → **536 MiB** | match | two copies |

### C. Real LLM: decoder graph compiled after the weights are on the device

`q3_late.py`: build Llama with prefill + decoder batch 8 (optimum-rbln, S=512), load it on one card, generate; then,
in the same process and the same compile context, compile a `decoder_batch_1` graph, create its runtime, attach it
as `decoders[1]`, and generate with one prompt.

| Model | Build (prefill + B8) | Memory after load | Late B1 compile | Late B1 alloc | Memory after late B1 | Late B1 tokens vs separately compiled B1 build | B8 tokens before/after |
|---|---:|---|---:|---|---|---|---|
| Llama-3.2-1B | 314 s | 2.5 GiB | 132 s | 20 MiB, no `Kernel` | **2.5 GiB** | **identical (32/32 tokens)**, also identical to the combined build's B1 | identical |
| Llama-3.1-8B | 469 s | 14.6 GiB | 141 s | 42 MiB, no `Kernel` | **14.6 GiB** | **identical (32/32 tokens)**, also identical to the combined build's B1 | identical (and identical to the separate B8 build) |

Same context id for the loaded prefill and the late graph in both models; late runtime creation 0.75–0.92 s. The
late graph needs the host model (fp32 for 8B: 65 GB peak RSS) and the compile context kept alive in the serving
process.

## Conclusion

**Yes, with conditions; no in general.**

- **Yes, in the same process and compile session.** A graph compiled after the weights are already on the device
  uses them, with no second copy and token-exact output (toy A1/A2; Llama-3.2-1B and Llama-3.1-8B in C). The cost is
  a compile in the serving process (132 s for a 1B and 141 s for an 8B decoder graph here, on a loaded host) and
  keeping the original host model and the compile context alive.
- **Yes, across processes, only for graphs that were compiled together.** Through the vendor `torch.compile` cache,
  graphs compiled in one session and reloaded in a new process share one copy (B2). This is the same "compile all
  variants together" condition as Q1, in a different packaging.
- **No for a graph compiled in a different session.** It always brings its own copy (A3, B3), and forcing it into the
  same device context without the vendor's reservation crashes the device task (A4–A6). There is no public API to
  bind a compiled graph to constant buffers that are already resident.

## Required runtime capability (draft feature request, not filed)

To: Rebellions (rebel-compiler / runtime). Also relevant to Optimus Prime once it serves ATOM.

> **Bind a separately compiled graph to weight buffers already resident in a device context.**
>
> Today weight sharing between graphs on ATOM exists only inside one compile session: the second graph refers to the
> first graph's constant buffers by the host tensor identity in the compiling process. A graph compiled in another
> session (another process, another day, another batch size or sequence length) always uploads its own copy, and
> two separately compiled models placed in the same global context abort with `SYS_TASK_ABORTED`.
>
> Needed, in any form:
> 1. A stable weight identity in the compiled artifact: each constant buffer keyed by parameter name + shape + dtype
>    + layout/transform + content hash, not by a session-local buffer index or host pointer.
> 2. Compile option "external weights": emit a graph whose constant buffers are references to such keys (no weight
>    data in the `.rbln`), plus the layout it expects.
> 3. Runtime API: `create_runtime(..., bind_to=<existing runtime or context>)` that resolves the graph's weight keys
>    against the constant buffers already resident in that context, uploads only what is missing (or fails
>    clearly), and never collides buffer indices.
> 4. Optional: the inverse, releasing one graph's program while keeping the shared weights resident.
>
> Evidence: ATOM probes A1–A6, B1–B3 and C in this record (rebel-compiler 0.11.0, KMD/FW 3.0.0). Use case: load-driven
> switching of batch and sequence-length graphs (Issue #5) without reloading ≈ 14 GB per variant (26–56 s measured in
> Q4) or doubling device memory.

## RNGD (survey only, no card was free)

All three RNGD cards on the node were held by Kubernetes pods (1 + 2 `furiosa.ai/rngd`) during this work, so nothing
was run on RNGD.

- **furiosa-llm 2026.4** (`LLM(model, fxb=...)` → `NativeLLMEngine(model_path, fxb_path, ...)`): one `.fxb` holds all
  pipelines (buckets) compiled together; the engine takes the `.fxb` path at construction. There is no public method
  to add a bucket/pipeline to a running engine or to point a second `.fxb` at parameters another engine has loaded.
  `NextGenArtifact.override_with(...)` edits artifacts offline. Bucket switching inside one engine is Q2.
- **furiosa-torch 2026.4** exposes device tensors (`furiosa.torch.empty`, `copy_from_tensor`, streams, memory types)
  and a `torch.compile` backend. With weights held as device tensors and passed to compiled graphs as inputs (the
  pattern npu-lora uses for training), a graph compiled later for a new shape would read the same device weights.
  This is the candidate RNGD "yes" path; it is untested here and needs a free card.
