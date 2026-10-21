# Q4b: keep two separately compiled ~14 GB decode builds parsed in host RAM and swap which one is on the device
# by destroying one runtime and creating the other. usage: q4_swap.py A.rbln B.rbln DEVICE ROUNDS
import sys, os, time, json, gc, subprocess
import numpy as np
import rebel
pa, pb, dev, rounds = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
def mem():
    o = subprocess.run(["rbln-stat"], capture_output=True, text=True).stdout
    return [l.split("|")[8].strip() for l in o.splitlines() if l.startswith("| %d " % dev)][0]
R = {"loadavg": open("/proc/loadavg").read().split()[:3], "events": []}
t = time.perf_counter(); A = rebel.RBLNCompiledModel(pa); B = rebel.RBLNCompiledModel(pb)
R["host_parse_both_s"] = round(time.perf_counter() - t, 2)
ins = {}
for k, p in (("A", pa), ("B", pb)):
    meta = rebel.RBLNCompiledModel.inspect(p)
    ins[k] = [np.zeros(i["shape"], dtype=i["dtype"]) for i in meta["inputs"]]
cms = {"A": A, "B": B}
for r in range(rounds):
    for k in ("A", "B"):
        t0 = time.perf_counter(); rt = cms[k].create_runtime(device=dev, tensor_type="np")
        t1 = time.perf_counter(); rt.run(*ins[k]); t2 = time.perf_counter(); rt.run(*ins[k]); t3 = time.perf_counter()
        m_on = mem()
        del rt; gc.collect(); t4 = time.perf_counter(); m_off = mem()
        R["events"].append({"round": r, "graph": k, "create_runtime_s": round(t1 - t0, 3), "first_run_s": round(t2 - t1, 3),
                            "second_run_s": round(t3 - t2, 3), "destroy_s": round(t4 - t3, 3), "mem_loaded": m_on, "mem_after_destroy": m_off})
        print(R["events"][-1], flush=True)
print("RESULT " + json.dumps(R), flush=True)
