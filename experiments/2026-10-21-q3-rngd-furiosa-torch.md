# Q3 (RNGD part): graphs compiled later, run against weights already on the device (furiosa-torch)

- Issue: #3 (ATOM part: [2026-10-21-q3-weight-binding.md](2026-10-21-q3-weight-binding.md))
- Date: 2026-10-21
- Hypothesis: on RNGD, weights held as device tensors can be passed as inputs to graphs compiled at different times
  (here: for batch 1, then batch 4), so a new graph does not bring a second weight copy.

## Setup

| Item | Value |
|---|---|
| Node / card | `npu-node-1`, one FuriosaAI RNGD (47.5 GiB), firmware 2026.4.0; card assigned by the Kubernetes device plugin |
| Software | image `furiosaai/furiosa-llm:latest`: furiosa-torch 2026.2.0, torch 2.10.0+cpu (furiosa-llm 2026.3.0) |
| How it ran | Kubernetes Jobs, one RNGD at a time, each job runs the measurement and exits. Compile-only checks ran in jobs without a card. A non-privileged init container records the card the device plugin assigned; the measurement opens only that card (privileged containers see every card on the node). |
| Host load | load average 490–640 on 96 CPUs during the runs (shared node) |
| Model | toy: 16 × (8192 × 8192) bf16 matrices = 2.0 GiB of weights, `x = tanh(x @ W_i)` |

Script: [`scripts/rngd/q3_ftorch.py`](scripts/rngd/q3_ftorch.py), public furiosa-torch API only
(`CompileModule.from_module`, `.to("rngd:N")`, device tensors). Raw output: [`data/2026-10-21-q3-rngd/`](data/2026-10-21-q3-rngd/).

```sh
python3 q3_ftorch.py compile-only   # job without a card: both graphs compile on the CPU (no device needed)
python3 q3_ftorch.py A              # job with one card: weights as graph inputs
python3 q3_ftorch.py B              # job with one card: weights inside each compiled module
```

Two variants, each in its own process:

- **A — weights as inputs.** The 16 weight matrices are copied to the card once (`w.to(device)`). A graph
  `f(x, w0..w15)` is compiled for batch 1, loaded and run with those device tensors; then a second graph is compiled
  for batch 4 and run with the **same** device tensors; then batch 1 runs again.
- **B — weights inside.** The weights are module parameters; each compiled module is moved to the card with
  `.to(device)`, once for batch 1 and once for batch 4.

`furiosa-smi status` shows 0.00 GiB used for furiosa-torch allocations (it shows the furiosa-llm engine's memory,
see Q2), so device memory is measured by capacity: allocate 256 MiB device tensors until allocation fails, free them,
and compare the number before and after each step (resolution 0.25 GiB). The runtime's allocator reports the card's
capacity as 51,002,736,640 bytes (47.5 GiB).

## Results

| Step | A: free capacity (GiB) | B: free capacity (GiB) |
|---|---:|---:|
| start | 47.25 | 47.25 |
| weights copied to the card (A only) | 45.25 (−2.00) | – |
| batch-1 graph compiled, loaded, run | 45.00 (−0.25) | 45.00 (−2.25) |
| batch-4 graph compiled, loaded, run | **45.00 (±0)** | **43.00 (−2.00)** |
| batch-1 graph run again | 45.00 | 43.00 |

| | A | B |
|---|---|---|
| compile batch 1 / batch 4 (CPU, loaded host) | 66.7 s / 274.7 s | 69.7 s / 247.5 s |
| weight upload / module `.to(device)` | 0.59 s for 2 GiB | 0.68 s / 0.75 s |
| first run batch 1 / batch 4 | 0.011 s / 0.013 s | – |
| max abs error vs CPU (bf16) | 0.051 / 0.052 | 0.051 / 0.052 |

Compiling also succeeds in a job without a card (`q3_ftorch_compileonly.json`, from an earlier version of the script
with a 0.25 GiB toy: 5.5 s / 42.6 s), so compiling a new graph does not need the device.

## Conclusion

**Yes on RNGD, with furiosa-torch and weights passed as device-tensor inputs.** A graph compiled after the weights are
on the card runs against them; the second graph costs no measurable device memory (< 0.25 GiB) and gives the same
results as the first. When the weights live inside each compiled module (variant B), every graph brings its own copy
(−2.0 GiB per graph).

Conditions and limits:

- Shown on a toy model (2 GiB of weights) with furiosa-torch 2026.2, not on an LLM served by furiosa-llm. furiosa-llm
  2026.3's engine takes one FXB at start-up and has no public call to add buckets to a running engine (see Q2).
- The weights must be explicit graph inputs and stay alive in the process that owns the device tensors; a graph
  compiled in another process cannot reach them (no public API to share device tensors across processes was found).
- Compiling a graph took 1–5 minutes on the loaded host; a server would compile ahead of need.
- Memory resolution is 0.25 GiB.
