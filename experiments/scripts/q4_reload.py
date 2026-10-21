# Q4: full reload of one separately compiled decode build: host load, device runtime creation, first and second step.
import sys, time, json, subprocess, os
import numpy as np
import rebel
path, dev = sys.argv[1], int(sys.argv[2])
def stat():
    out = subprocess.run(["rbln-stat"], capture_output=True, text=True).stdout
    return [l for l in out.splitlines() if l.startswith("| %d " % dev) or (" %d " % os.getpid()) in l]
t0 = time.perf_counter()
cm = rebel.RBLNCompiledModel(path)
t1 = time.perf_counter()
rt = cm.create_runtime(device=dev, tensor_type="np")
t2 = time.perf_counter()
meta = rebel.RBLNCompiledModel.inspect(path)
ins = [np.zeros(i["shape"], dtype=i["dtype"]) for i in meta["inputs"]]
o = rt.run(*ins)
t3 = time.perf_counter()
o = rt.run(*ins)
t4 = time.perf_counter()
steps = []
for _ in range(10):
    a = time.perf_counter(); rt.run(*ins); steps.append(time.perf_counter() - a)
print(json.dumps({"file": os.path.basename(path), "size_gb": round(os.path.getsize(path) / 1e9, 2),
                  "host_load_s": round(t1 - t0, 3), "create_runtime_s": round(t2 - t1, 3),
                  "first_run_s": round(t3 - t2, 3), "second_run_s": round(t4 - t3, 4),
                  "steady_median_s": round(sorted(steps)[5], 4),
                  "load_to_first_token_s": round(t3 - t0, 3), "alloc_per_node": meta["alloc_per_node"],
                  "rbln_stat": stat(), "loadavg": open("/proc/loadavg").read().split()[:3]}), flush=True)
