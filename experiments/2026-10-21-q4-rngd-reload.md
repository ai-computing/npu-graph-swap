# Q4 (RNGD part): full reload time of one furiosa-llm model

- Issue: #4 (ATOM part: [2026-10-21-q4-atom-reload.md](2026-10-21-q4-atom-reload.md))
- Date: 2026-10-21
- Hypothesis: reloading a variant costs seconds and sets the bar a switching mechanism has to beat.

## Setup

| Item | Value |
|---|---|
| Node / card | `npu-node-1`, one FuriosaAI RNGD (47.5 GiB), firmware 2026.4.0; card assigned by the Kubernetes device plugin |
| Software | image `furiosaai/furiosa-llm:latest`: furiosa-llm 2026.3.0 |
| Model | Llama-3.1-8B-Instruct, 16.06 GB of safetensors (15.0 GiB loaded to the card), FXB `l8_len1024_O0` from Q2 (5 buckets, 2.9 MB) |
| How it ran | one Kubernetes Job with one RNGD, run-and-exit, deadline and TTL set; each load in a fresh process so the card is released in between; engine pinned to the assigned card |
| Host load | load average 180–250 on 96 CPUs |

Script: [`scripts/rngd/q4_rngd.py`](scripts/rngd/q4_rngd.py). Raw output: [`data/2026-10-21-q4-rngd/q4_rngd.json`](data/2026-10-21-q4-rngd/q4_rngd.json).

```sh
python3 q4_rngd.py <Llama-3.1-8B-Instruct dir on the model store> l8_len1024_O0.fxb q4_rngd.json
```

Per load: `import furiosa_llm`, `LLM(model, fxb=..., devices="npu:N", max_model_len=1024)`, then two one-token
requests. Five loads: twice from the lab's shared model store (a network share, where the job mounts the
checkpoints), then the checkpoint is copied to the container's local disk, its pages are dropped with
`posix_fadvise(DONTNEED)` (cold, `fincore` 0 bytes resident), and it is loaded three times (cold, warm, warm; warm =
all 16.06 GB resident).

## Results

| Load | weights resident before | engine construction (s) | first token (s) | **load → first token (s)** | process wall incl. import (s) |
|---|---:|---:|---:|---:|---:|
| network model store, 1st | 0 B | 214.5 | 0.284 | **214.8** | 221.9 |
| network model store, 2nd | 0 B | 212.1 | 0.273 | **212.4** | 219.6 |
| local disk, cold | 0 B | 16.1 | 0.228 | **16.3** | 37.9 |
| local disk, warm | 16.06 GB | 6.6 | 0.264 | **6.9** | 13.6 |
| local disk, warm | 16.06 GB | 7.3 | 0.331 | **7.6** | 14.2 |

Copying the checkpoint from the share to local disk took 198 s (≈ 81 MB/s), the same rate the engine loads at
from the share. The share's pages are not kept in the page cache between loads (0 B resident before the second load),
so every load from it is a cold read. In Q2 the engine log reported "parameters loaded: 15.0 GiB in 182–235 s" for
the same store. The cold local run's 19 s Python import is extra file-cache cost too (later imports took 4 s).

## Conclusion

A full reload of one Llama-3.1-8B model on RNGD costs **≈ 7 s** to the first token when the checkpoint is local and
in the page cache, **≈ 16 s** from local disk with a cold cache, and **≈ 213 s** from the lab's network model store.
The device side is small: with the weights in host memory the whole engine start takes 6.6–7.3 s. Against Q2's bucket
switch (no measurable cost) and the ≈ 0.4 MB device memory per extra bucket, reloading is the expensive option on
RNGD as well, and serving should keep model checkpoints on local disk.

Limits: one card, one model, one FXB; the network-store numbers depend on that share and its load; "cold" means a
cold file, not a cold machine.
