# Q4 on RNGD: full reload of one furiosa-llm model (Llama-3.1-8B weights + an FXB): engine construction and the
# first generated token, from the shared model store and from a local copy with cold and warm page cache.
# Each load runs in a fresh process so the card is released between loads.
# usage: q4_rngd.py MODEL_DIR FXB OUT_JSON      (child mode: q4_rngd.py child MODEL_DIR FXB)
import os, sys, time, json, shutil, subprocess

def child(model, fxb):
    t0 = time.perf_counter()
    from furiosa_llm import LLM, SamplingParams
    t1 = time.perf_counter()
    name = open("/run/ngs/rngd").read().strip()
    llm = LLM(model, fxb=fxb, devices=f"npu:{name[3:]}", max_model_len=1024)
    t2 = time.perf_counter()
    o = llm.generate(["The capital of France is"], SamplingParams(max_tokens=1, temperature=0.0))
    t3 = time.perf_counter()
    o2 = llm.generate(["The capital of France is"], SamplingParams(max_tokens=1, temperature=0.0))
    t4 = time.perf_counter()
    print("CHILD " + json.dumps({"import_s": round(t1 - t0, 2), "engine_s": round(t2 - t1, 2),
                                  "first_token_s": round(t3 - t2, 3), "second_request_s": round(t4 - t3, 3),
                                  "load_to_first_token_s": round(t3 - t1, 2), "token": list(o[0].outputs[0].token_ids)}), flush=True)
    try:
        llm.shutdown()
    except Exception:
        pass

def drop(path):
    for d, _, fs in os.walk(path):
        for f in fs:
            fd = os.open(os.path.join(d, f), os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)

def resident(path):
    r = subprocess.run("fincore -b -n " + " ".join(os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith(".safetensors")),
                       shell=True, capture_output=True, text=True)
    return sum(int(l.split()[0]) for l in r.stdout.splitlines() if l.strip() and l.split()[0].isdigit()) if r.returncode == 0 else r.stderr.strip()[:200]

def run(label, model, fxb, R):
    s = {"label": label, "resident_bytes_before": resident(model), "loadavg": open("/proc/loadavg").read().split()[:3]}
    t = time.perf_counter()
    p = subprocess.run([sys.executable, __file__, "child", model, fxb], capture_output=True, text=True)
    s["process_wall_s"] = round(time.perf_counter() - t, 2)
    line = [l for l in p.stdout.splitlines() if l.startswith("CHILD ")]
    s.update(json.loads(line[0][6:]) if line else {"error": (p.stderr or p.stdout)[-1500:]})
    R["runs"].append(s); print(s, flush=True)

if __name__ == "__main__" and sys.argv[1] == "child":
    child(sys.argv[2], sys.argv[3]); sys.exit(0)

model, fxb, out = sys.argv[1], sys.argv[2], sys.argv[3]
R = {"fxb": os.path.basename(fxb), "device": open("/run/ngs/rngd").read().strip(), "runs": [],
     "weights_bytes": sum(os.path.getsize(os.path.join(model, f)) for f in os.listdir(model) if f.endswith(".safetensors"))}
run("model store (network share), as found", model, fxb, R)
run("model store (network share), warm", model, fxb, R)
local = "/tmp/model"
t = time.perf_counter(); shutil.copytree(model, local, symlinks=False); R["copy_local_s"] = round(time.perf_counter() - t, 1)
drop(local)
run("local disk, cold (pages dropped)", local, fxb, R)
run("local disk, warm", local, fxb, R)
run("local disk, warm", local, fxb, R)
shutil.rmtree(local)
json.dump(R, open(out, "w"), indent=1)
print("RESULT " + json.dumps(R), flush=True)
