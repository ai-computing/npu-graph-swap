# Q4 (ATOM part): full reload time of one separately compiled decode build

- Issue: #4
- Date: 2026-10-21
- Hypothesis: reloading a variant costs seconds (ATOM: ≈ 14 GB per decode build) and sets the bar any switching
  mechanism has to beat.

## Setup

| Item | Value |
|---|---|
| Node | `npu-node-1` (shared, 2 × AMD EPYC 9254, 96 CPUs, 1.5 TB RAM) |
| Device | Rebellions ATOM RBLN-CA22, 15.7 GiB, PCIe 32 GT/s x16; one card (`rbln3`) |
| Driver / firmware | KMD 3.0.0 / FW 3.0.0 |
| SDK | rebel-compiler 0.11.0 (Python runtime `rebel`), Python 3.10, kernel 6.8 |
| Build | an existing separately compiled Llama-3.1-8B decode graph, S=512, batch 1, compiled with rebel-compiler 0.11.0 (13.1 GB file, `alloc_per_node` 14.21 GB). It takes the KV cache as host inputs (it is a custom PD build, not an optimum-rbln one). Not modified; a copy was used for the cold run and deleted afterwards. |
| Host load | load average 460–490 on 96 CPUs during every run (other users' jobs). All host-side numbers are inflated by this. |

## Method

Scripts: [`scripts/q4_reload.py`](scripts/q4_reload.py), [`scripts/q4_swap.py`](scripts/q4_swap.py),
[`scripts/dropcache.py`](scripts/dropcache.py). Raw output: [`data/2026-10-21-q4/`](data/2026-10-21-q4/).

```sh
cp <decode build>.rbln q4/ && sync && python3 dropcache.py q4/<decode build>.rbln && fincore q4/<decode build>.rbln
python3 q4_reload.py q4/<decode build>.rbln 3      # cold, then run twice more for warm
python3 q4_swap.py q4/<B1 build>.rbln <B4 build>.rbln 3 3
```

1. **Full reload** = `rebel.RBLNCompiledModel(path)` (read and parse the file on the host) → `create_runtime(device)`
   (allocate on the device and upload) → first `run()` with zero inputs → second `run()` → 10 more runs.
2. **Cold vs warm file cache.** Dropping the page cache needs root. Instead the build was copied into a new file,
   and that file's pages were dropped with `posix_fadvise(POSIX_FADV_DONTNEED)` (allowed for a file one owns).
   `fincore` confirmed 0 B resident before the cold run and 13 GB resident before the warm runs.
3. **Host-resident swap** (an alternative found on the way): keep two parsed compiled models (decode S512 B1 and
   B4, 13.1 GB and 13.2 GB files) in host RAM, and alternate which one is on the device by destroying one runtime
   and creating the other. The two do not fit on one card together (13.2 + 13.7 GiB > 15.7 GiB).

## Results

Full reload, one decode build (13.2 GiB on the device after load):

| Run | Page cache | host parse (s) | create_runtime (s) | first run (s) | **load → first step (s)** | steady step (s) |
|---|---|---:|---:|---:|---:|---:|
| 1 | cold (0 B resident) | 54.72 | 0.71 | 0.39 | **55.8** | 0.35 |
| 2 | warm | 31.65 | 0.57 | 0.37 | **32.6** | 0.23 |
| 3 | warm | 25.66 | 0.50 | 0.39 | **26.5** | 0.31 |

The step time here (0.23–0.35 s) includes copying the KV cache (≈ 128 MB) from the host every step, because this
custom build takes it as an input; it is not the optimum-rbln decode step time (see Q1).

Host-resident swap (parse both once: 49.3 s; then per swap):

| Round | Graph | destroy previous runtime (s) | create_runtime (s) | first run (s) | device memory loaded / after destroy |
|---|---|---:|---:|---:|---|
| 0 | B1 | – | 0.49 | 0.17 | 13.2 GiB / 0 B |
| 0 | B4 | 0.58 | 0.49 | 0.21 | 13.7 GiB / 0 B |
| 1 | B1 | 0.77 | 0.42 | 0.13 | 13.2 GiB / 0 B |
| 1 | B4 | 1.13 | 0.56 | 0.25 | 13.7 GiB / 0 B |
| 2 | B1 | 0.62 | 0.49 | 0.15 | 13.2 GiB / 0 B |
| 2 | B4 | 0.90 | 0.53 | 0.27 | 13.7 GiB / 0 B |

Swap = destroy + create + first run = **1.2–1.9 s** for ≈ 13–14 GiB.

## Conclusion

- A full reload of one ≈ 14 GB ATOM decode build from file costs **26–33 s warm and ≈ 56 s cold** (to the first
  step) on this shared, heavily loaded host. Almost all of it is host-side file reading and parsing; the device part
  (allocate + upload ≈ 13 GiB + first step) is **≈ 1 s**.
- So the bar is not "14 GB over PCIe" but "parse a 14 GB file". If the parsed compiled models stay in host RAM
  (≈ 13 GB each), swapping which one is on the card costs **1.2–1.9 s**. This is a usable fallback when nothing
  better exists, at the price of host RAM and of having only one variant on the card at a time.
- Both are far above what Q1 measured for graphs compiled together (no reload; switch cost below measurement noise).

Limits: one card, one model, one run per cache state for the full reload; the host was at load ≈ 470, so the host
parse times are pessimistic. The cold run uses a new copy whose pages were dropped, which is equivalent to a cold
file but not to a cold machine (the driver and Python modules were warm). RNGD part of Q4: not run (see Issue #4).
