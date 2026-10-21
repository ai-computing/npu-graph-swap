# Q2 (RNGD): furiosa-llm bucket switching cost and device memory per bucket

- Issue: #2
- Date: 2026-10-21
- Hypothesis: furiosa-llm buckets (sequence length × batch) share one weight copy; switching between buckets is per
  request and costs < 1 ms; device memory grows only by the graphs.

## Setup

| Item | Value |
|---|---|
| Node / card | `npu-node-1`, one FuriosaAI RNGD (47.5 GiB, 8 PEs), firmware 2026.4.0; card assigned by the Kubernetes device plugin |
| Software | image `furiosaai/furiosa-llm:latest`: furiosa-llm 2026.3.0, furiosa-torch 2026.2.0, torch 2.10.0+cpu |
| Model | Llama-3.1-8B-Instruct, bf16 weights (15.0 GiB) read from the lab's shared model store (network share) |
| Build | `fxb build <model> out.fxb --max-model-len 1024 -O <level> --build-report --concurrency 4` (tensor parallel 8 = whole card, the preset), optimisation levels O0–O3 select how many buckets of the Llama-3.1-8B preset are kept |
| How it ran | Kubernetes Jobs. Builds in a job **without** an RNGD (the compiler needs no device, only an aarch64 cross toolchain, which the job installs). Measurements in one job with one RNGD that runs all four FXBs one after another (each in a fresh process) and exits; `activeDeadlineSeconds` and `ttlSecondsAfterFinished` set. A non-privileged init container records the card the device plugin assigned and the engine is started with that card only (`devices="npu:N"`). |
| Host load | load average 250–1000 on 96 CPUs during the builds (four builds in parallel plus other users), 250–430 during the measurements |

Scripts: [`scripts/rngd/build.sh`](scripts/rngd/build.sh), [`scripts/rngd/q2_measure.py`](scripts/rngd/q2_measure.py),
[`scripts/rngd/q2_all.sh`](scripts/rngd/q2_all.sh), Job manifests from [`scripts/rngd/mkjob.py`](scripts/rngd/mkjob.py).
Raw data: [`data/2026-10-21-q2/`](data/2026-10-21-q2/).

Bucket sets (from `fxb build --dry-run`; kv-cache blocks of 1024 tokens; "tokenwise" kernels run the layers for
up to N tokens, "attention" kernels per batch/attention/kv shape):

| FXB | buckets | kernels | prefill (batch, len) | decode (batch, len) | append | tokenwise | file |
|---|---:|---:|---|---|---:|---|---:|
| O0 | 5 | 9 | (1, 1024) | (1, 1024) | 1 | 1, 1024 | 2.9 MB |
| O1 | 9 | 15 | (1, 256), (1, 1024) | (1, 1024) | 3 | 1, 16, 1024 | 5.0 MB |
| O2 | 15 | 25 | (1, 128/256/512/1024) | (1, 1024) | 5 | 1, 4, 16, 64, 1024 | 7.4 MB |
| O3 | 20 | 38 | (1, 128/256/512/1024) | (1, 1024), **(8, 1024)** | 5 | 1, 4, 8, 16, 32, 64, 128, 256, 1024 | 11.2 MB |

The FXB holds graphs only (2.9–11.2 MB); the weights come from the Hugging Face checkpoint at load time.

## Method

For each FXB, in a fresh process on the same card: construct `LLM(model, fxb=..., devices="npu:N", max_model_len=1024)`,
read `furiosa-smi status` and the engine's own start-up log (KV-cache size and block count), then:

- **Prefill bucket switch:** one request with ≈ 100 prompt tokens ("short") or ≈ 900 ("long"), `max_tokens=1`
  (time to the first token). 15 × short, 15 × long, then 15 × (short, long) alternating. Every request gets a different
  first sentence, so no request reuses the prefix cache of another.
- **Decode batch switch:** 1 request vs 8 concurrent requests (`generate([p] * 8)`), 32 new tokens each; 15 × 1,
  15 × 8, then 15 alternating.
- First call of each shape after load, and greedy tokens for 3 fixed prompts (32 tokens) to compare the four FXBs.

Switch cost = median(request after the other shape) − median(request after the same shape).

## Results

Memory and load (engine log and `furiosa-smi`):

| FXB | buckets / kernels | weights loaded | KV cache reserved | KV blocks | `furiosa-smi` used | engine load (s) |
|---|---|---|---:|---:|---|---:|
| O0 | 5 / 9 | 15.0 GiB in 182 s | 29 225 MB | 233 804 | 45.50 / 47.50 GiB | 188.3 |
| O1 | 9 / 15 | 15.0 GiB in 235 s | 29 223 MB | 233 790 | 45.50 GiB | 240.4 |
| O2 | 15 / 25 | 15.0 GiB in 205 s | 29 222 MB | 233 777 | 45.50 GiB | 213.7 |
| O3 | 20 / 38 | 15.0 GiB in 210 s | 29 219 MB | 233 758 | 45.50 GiB | 217.1 |

The engine reserves what is left after weights and graphs for the KV cache, so the graphs' cost shows as the KV cache
shrinking: **6 MB in total from 5 to 20 buckets (9 to 38 kernels)**, ≈ 0.4 MB per added bucket. One weight copy in
every case.

Prefill bucket switch (seconds to the first token, medians of 15):

| FXB | short after short | short after long | long after long | long after short | switch cost |
|---|---:|---:|---:|---:|---|
| O0 | 0.1407 | 0.1363 | 2.7690 | 2.7760 | −4.4 / +7.0 ms |
| O1 | 0.1262 | 0.1239 | 0.6791 | 0.6855 | −2.3 / +6.4 ms |
| O2 | 0.1062 | 0.1101 | 0.5973 | 0.5933 | +3.9 / −4.0 ms |
| O3 | **0.0267** | **0.0270** | **0.3050** | **0.3095** | **+0.3 / +4.5 ms** |

Decode batch switch (seconds for 32 new tokens, medians of 15):

| FXB | 1 after 1 | 1 after 8 | 8 after 8 | 8 after 1 | switch cost |
|---|---:|---:|---:|---:|---|
| O0 | 0.6784 | 0.6786 | 4.9144 | 4.8344 | +0.2 / −80 ms |
| O1 | 0.6785 | 0.6711 | 1.1651 | 1.1401 | −7.4 / −25 ms |
| O2 | 0.6565 | 0.6609 | 1.1156 | 1.1187 | +4.4 / +3.1 ms |
| O3 | **0.5652** | **0.5732** | **0.7806** | **0.7747** | **+8.0 / −5.9 ms** |

First call of a shape after load (O3): short prefill 0.053 s (steady 0.027), long 0.360 s (0.305), 1 × 32 tokens
0.565 s (0.565), 8 × 32 tokens 1.017 s (0.781): a one-time 0–240 ms per shape.

Greedy tokens for the 3 fixed prompts are **identical across O0–O3** (96/96).

Compile time (`--build-report`, per kernel; the four builds ran concurrently with a shared compiler cache, so a
kernel compiled by one build shows ≈ 0.2 s in another and the per-build totals overlap):

| FXB | wall time | sum of kernel times | largest kernels |
|---|---:|---:|---|
| O0 | 24.2 min | 71.8 min | `mid_tokenwise` tw1024 24.2 min, `last_tokenwise_with_lm_head` tw1024 23.0 min, `first_tokenwise` tw1024 12.1 min, attention b1 a1024 6.9 min |
| O1 | 28.9 min | 93.0 min | + tw16 kernels 3–17 min each, attention b1 a256 ≈ 1 min |
| O2 | 33.3 min | 120.2 min | + tw4, tw64 kernels 1–16 min each, attention a128/a512 0.8–1.8 min |
| O3 | 38.4 min | 142.6 min | + tw8, tw32, tw128, tw256 kernels 1–13 min each, decode attention b8 a1024 15 s |

So an added bucket costs what its new kernels cost: an attention-shape bucket that reuses existing tokenwise kernels
compiles in seconds to a few minutes; a new tokenwise size costs 1–25 minutes per kernel (three kernels per size).

## Conclusion

**Yes.** furiosa-llm keeps one weight copy for all buckets of an FXB; going from 5 to 20 buckets (9 to 38 kernels)
costs ≈ 6 MB of device memory in total. Switching buckets between requests has no measurable cost at the request
level: the differences between "after the same shape" and "after the other shape" are within ±8 ms on requests of
27 ms – 4.9 s, of both signs, i.e. noise; a below-1-ms claim cannot be resolved by request-level timing. Outputs are
identical across bucket sets.

What the buckets do change is speed: with only a batch-1 decode bucket and a 1024-token prefill bucket (O0), 8
concurrent requests take 4.9 s instead of 0.78 s (O3) and a 900-token prompt takes 2.77 s instead of 0.31 s.

Conditions and limits:

- Buckets are fixed at build time (one FXB per engine); furiosa-llm 2026.3 has no public call to add a bucket to a
  running engine, so "switching graphs" means choosing among buckets compiled together — the RNGD counterpart of the
  ATOM combined build in Q1.
- Engine load takes 188–240 s here, almost all of it reading 15 GiB of weights from the network model store (see Q4).
- Memory is read from the engine log (KV-cache size) and `furiosa-smi`; per-graph device allocations are not exposed.
- One card, one run per FXB, shared and heavily loaded host. Compile times are from concurrent builds on that host.
