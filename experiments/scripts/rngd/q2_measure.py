# Q2 on RNGD: furiosa-llm with one FXB (a bucket set) — load time, device memory, cost of requests that change
# bucket vs repeat the same bucket, greedy tokens. Runs only on the card the device plugin assigned.
# usage: q2_measure.py MODEL_DIR FXB OUT_JSON
import os, sys, time, json, subprocess, statistics
from furiosa_llm import LLM, SamplingParams
from transformers import AutoTokenizer

model, fxb, out = sys.argv[1], sys.argv[2], sys.argv[3]
NAME = open("/run/ngs/rngd").read().strip()          # e.g. "npu1"
DEVICES = f"npu:{NAME[3:]}"

def smi():
    # raw furiosa-smi status rows (all cards); the assigned card is recorded separately and picked out later
    o = subprocess.run(["furiosa-smi", "status"], capture_output=True, text=True).stdout
    return [" ".join(l.split()) for l in o.splitlines() if "RNGD" in l or "npu" in l]

R = {"fxb": os.path.basename(fxb), "fxb_bytes": os.path.getsize(fxb), "device": NAME, "devices_arg": DEVICES,
     "loadavg_start": open("/proc/loadavg").read().split()[:3], "smi_before": smi()}
tok = AutoTokenizer.from_pretrained(model)
base = ("The history of computing is a long story of machines that calculate, store and communicate. " * 200)
ids = tok(base)["input_ids"]
def prompt(n):
    return tok.decode(ids[1:n])                         # about n tokens with BOS
P_SHORT, P_LONG = prompt(100), prompt(900)
R["prompt_tokens"] = {"short": len(tok(P_SHORT)["input_ids"]), "long": len(tok(P_LONG)["input_ids"])}

t = time.time()
llm = LLM(model, fxb=fxb, devices=DEVICES, max_model_len=1024)
R["load_s"] = round(time.time() - t, 2)
R["smi_after_load"] = smi()

CALL = [0]
def uniq(p):
    # a different first line per request, so no request hits the prefix cache of an earlier one
    CALL[0] += 1
    return f"Request {CALL[0]} of the test run. " + p

def run(prompts, n, fresh=True):
    sp = SamplingParams(max_tokens=n, temperature=0.0)
    if fresh:
        prompts = [uniq(p) for p in prompts]
    t = time.perf_counter(); o = llm.generate(prompts, sp); el = time.perf_counter() - t
    return el, [list(x.outputs[0].token_ids) for x in o]

# first call per shape after load
R["first_call_s"] = {}
for k, p in (("short", P_SHORT), ("long", P_LONG)):
    R["first_call_s"][k] = round(run([p], 1)[0], 4)
for k, b in (("dec_b1", 1), ("dec_b8", 8)):
    R["first_call_s"][k] = round(run([P_SHORT] * b, 32)[0], 4)

N = 15
def series(seq):
    return [round(run(*x)[0], 4) for x in seq]
def med(xs):
    return round(statistics.median(xs), 4)

# prefill bucket: short (~100 tok) vs long (~900 tok), max_tokens=1 (time to first token)
same_s = series([([P_SHORT], 1)] * N); same_l = series([([P_LONG], 1)] * N)
alt = series([([P_SHORT], 1), ([P_LONG], 1)] * N)
R["prefill"] = {"short_after_short": med(same_s), "long_after_long": med(same_l),
                "short_after_long": med(alt[0::2][1:]), "long_after_short": med(alt[1::2]),
                "raw": {"same_short": same_s, "same_long": same_l, "alternating": alt}}
# decode batch: 1 request vs 8 requests, 32 new tokens each
same_1 = series([([P_SHORT], 32)] * N); same_8 = series([([P_SHORT] * 8, 32)] * N)
alt = series([([P_SHORT], 32), ([P_SHORT] * 8, 32)] * N)
R["decode"] = {"b1_after_b1": med(same_1), "b8_after_b8": med(same_8),
               "b1_after_b8": med(alt[0::2][1:]), "b8_after_b1": med(alt[1::2]),
               "raw": {"same_b1": same_1, "same_b8": same_8, "alternating": alt}}
R["smi_after_runs"] = smi()
prompts = ["The capital of France is", "def fibonacci(n):", "Explain in one sentence why the sky is blue."]
R["tokens"] = run(prompts, 32, fresh=False)[1]
R["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
json.dump(R, open(out, "w"), indent=1)
print("RESULT " + json.dumps({k: v for k, v in R.items() if k not in ("tokens",)}), flush=True)
try:
    llm.shutdown()
except Exception:
    pass
