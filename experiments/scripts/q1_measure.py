# Q1: load an optimum-rbln build, record device memory, decoder switch latency, per-step time, greedy tokens.
# usage: q1_measure.py BUILD_DIR DEVICE OUT_JSON [MODEL_FOR_TOKENIZER]
import sys, os, time, json, subprocess, statistics
os.environ.setdefault("RBLN_RUNTIME_TIMER", "1")  # per-run host/device timer reports from the rebel runtime
import torch
from optimum.rbln import RBLNLlamaForCausalLM
from transformers import AutoTokenizer

build, dev, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
tok_src = sys.argv[4] if len(sys.argv) > 4 else build
R = {"build": os.path.basename(build.rstrip("/")), "device": dev, "loadavg_start": open("/proc/loadavg").read().split()[:3]}

def stat():
    o = subprocess.run(["rbln-stat"], capture_output=True, text=True).stdout
    dev_mem = [l.split("|")[8].strip() for l in o.splitlines() if l.startswith("| %d " % dev)]
    proc = [l.split("|")[7].strip() for l in o.splitlines() if (" %d " % os.getpid()) in l]
    return {"device_used": dev_mem[0] if dev_mem else None, "process_memalloc": proc}

R["mem_before"] = stat()
t0 = time.perf_counter()
m = RBLNLlamaForCausalLM.from_pretrained(build, export=False, rbln_config={"device": dev})
R["load_s"] = round(time.perf_counter() - t0, 2)
R["mem_after_load"] = stat()
cfg = m.rbln_config
R["batch_size"], R["decoder_batch_sizes"], R["max_seq_len"] = cfg.batch_size, cfg.decoder_batch_sizes, cfg.max_seq_len
R["kvcache_num_blocks"], R["kvcache_block_size"] = cfg.kvcache_num_blocks, cfg.kvcache_block_size
names = cfg.expected_compiled_model_names
R["alloc_by_key"] = {n: {k: v[0] for k, v in m.model[i].get_alloc_per_node_by_key().items()} for i, n in enumerate(names)}
R["alloc_total"] = {n: m.model[i].get_alloc_per_node()[0] for i, n in enumerate(names)}
R["context_ids"] = {n: m.model[i]._executor.get_context_id() if hasattr(m.model[i], "_executor") else None for i, n in enumerate(names)}

DT = {"int64": torch.int64, "int32": torch.int32, "int16": torch.int16, "float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
def dummy(name):
    cc = [c for c in cfg.compile_cfgs if c.compiled_model_name == name][0]
    xs = []
    for n, shape, dt in cc.input_info:
        if n.startswith("past_key_values"):
            continue
        x = torch.zeros(shape, dtype=DT[dt])
        if n == "block_tables" and len(shape) == 2:
            x = torch.arange(shape[0], dtype=DT[dt]).unsqueeze(1).expand(shape).contiguous() % max(cfg.kvcache_num_blocks, 1)
        if n == "cache_position":
            x.fill_(5)
        xs.append(x)
    return xs

dec = {b: names.index(f"decoder_batch_{b}") for b in cfg.decoder_batch_sizes}
inp = {b: dummy(f"decoder_batch_{b}") for b in dec}
TIMES = []  # (batch, wall_ms, runtime_total_ms, device_ms) for every call
def call(b):
    rt = m.model[dec[b]]
    a = time.perf_counter(); rt.run(*inp[b]); w = (time.perf_counter() - a) * 1e3
    rep = [x for x in rt.get_reports() if x.get("type") == "timer"]
    rt.flush_reports()
    tot = rep[-1]["total"] / 1e3 if rep else None
    devt = rep[-1]["total_device"] / 1e3 if rep else None
    TIMES.append((b, round(w, 3), tot, devt))
    return w
def last_dev():
    return TIMES[-1][3]
def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 3) if xs else None

# First call of each decoder graph after load (largest first, the order a server warms them).
R["first_call"] = {}
for b in sorted(dec, reverse=True):
    w = call(b); R["first_call"][b] = {"wall_ms": round(w, 2), "device_ms": last_dev()}
R["second_call"] = {}
for b in sorted(dec, reverse=True):
    w = call(b); R["second_call"][b] = {"wall_ms": round(w, 2), "device_ms": last_dev()}
N = 50
R["same_graph"] = {}
for b in sorted(dec):
    for _ in range(5):
        call(b)
    k0 = len(TIMES)
    for _ in range(N):
        call(b)
    seg = TIMES[k0:]
    R["same_graph"][b] = {"wall_p10_p50_p90_ms": [pct([t[1] for t in seg], q) for q in (0.1, 0.5, 0.9)],
                          "device_p10_p50_p90_ms": [pct([t[3] for t in seg], q) for q in (0.1, 0.5, 0.9)]}
if len(dec) > 1:
    bs = sorted(dec)
    sw = {}
    for a in bs:
        for c in bs:
            if a == c:
                continue
            seg = []
            for _ in range(N):
                call(a); call(c); seg.append(TIMES[-1])
            sw[f"{a}->{c}"] = {"wall_p10_p50_p90_ms": [pct([t[1] for t in seg], q) for q in (0.1, 0.5, 0.9)],
                               "device_p10_p50_p90_ms": [pct([t[3] for t in seg], q) for q in (0.1, 0.5, 0.9)]}
    R["after_switch"] = sw
    def p50(d, kind):
        return d[kind + "_p10_p50_p90_ms"][1]
    R["switch_overhead_p50_ms"] = {}
    for k, v in sw.items():
        same = R["same_graph"][int(k.split("->")[1])]
        R["switch_overhead_p50_ms"][k] = {kind: (round(p50(v, kind) - p50(same, kind), 3) if p50(v, kind) is not None else None)
                                          for kind in ("wall", "device")}
R["all_calls"] = TIMES
R["mem_after_runs"] = stat()

# Greedy generation through the HF generate path: batch b uses decoder_batch_b.
tok = AutoTokenizer.from_pretrained(tok_src); tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
prompts = ["The capital of France is", "Explain in one sentence why the sky is blue.", "def fibonacci(n):",
           "Write a haiku about autumn leaves.", "List three prime numbers greater than 100:", "Translate to Korean: Good morning, everyone.",
           "The theory of relativity states that", "Once upon a time in a small village,"]
R["generate"] = {}
for b in sorted(dec):
    enc = tok(prompts[:b], return_tensors="pt", padding=True)
    t = time.perf_counter()
    o = m.generate(**enc, max_new_tokens=32, do_sample=False)
    el = time.perf_counter() - t
    gen = o[:, enc["input_ids"].shape[1]:].tolist()
    R["generate"][b] = {"tokens": gen, "seconds": round(el, 3), "text0": tok.decode(gen[0], skip_special_tokens=True)}
R["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
json.dump(R, open(out, "w"), indent=1)
print("RESULT " + json.dumps({k: v for k, v in R.items() if k not in ("generate", "all_calls")}), flush=True)
