# Q3 probe via the vendor torch.compile path (the one vllm-rbln uses): global compile context with weight sharing
# and an on-disk cache of compiled graphs. Checks whether graphs loaded from the cache in a NEW process, or
# compiled fresh next to cached ones, share one weight copy on the device.
# usage: q3_tc.py MODE   (MODE: fresh | cached | partial)
import sys, os, json, time, glob, subprocess
import torch, rebel
from rebel.compile_context import CompileContext

DEV = int(os.environ.get("DEV", "2"))
CACHE = os.path.expanduser(os.environ.get("CACHE", "~/ngs-work/q3/tc_cache"))
os.makedirs(CACHE, exist_ok=True)

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.layers = torch.nn.ModuleList(torch.nn.Linear(4096, 4096, bias=False) for _ in range(8))
    def forward(self, x):
        for l in self.layers:
            x = torch.tanh(l(x))
        return x

def mem():
    o = subprocess.run(["rbln-stat"], capture_output=True, text=True).stdout
    return [l.split("|")[8].strip() for l in o.splitlines() if l.startswith("| %d " % DEV)][0]

mode = sys.argv[1]
if mode == "fresh":
    for f in glob.glob(CACHE + "/*"):
        os.remove(f)
model = Toy().eval()
ctx = CompileContext(use_weight_sharing=True, use_global_ctx=True, device_id=DEV)
f = torch.compile(model, backend="rbln", dynamic=False,
                  options={"compile_context": ctx, "cache_dir": CACHE, "device": DEV})
res = {"mode": mode, "cache_files_before": sorted(os.path.basename(p) for p in glob.glob(CACHE + "/*.rbln")), "mem0": mem()}
if mode == "partial":
    # keep only the oldest cached graph (batch 1), drop the other so it is compiled fresh in this process
    files = sorted(glob.glob(CACHE + "/*.rbln"), key=os.path.getmtime)
    for p in files[1:]:
        os.remove(p)
    res["cache_files_kept"] = [os.path.basename(p) for p in files[:1]]
with torch.inference_mode():
    for b in (1, 4, 1, 4):
        x = torch.randn(b, 4096, generator=torch.Generator().manual_seed(b))
        t = time.time()
        y = f(x)
        el = time.time() - t
        err = float((y.cpu() - model(x)).abs().max())
        res.setdefault("steps", []).append({"batch": b, "s": round(el, 3), "max_abs_err": err, "mem": mem()})
        print(res["steps"][-1], flush=True)
res["cache_files_after"] = sorted(os.path.basename(p) for p in glob.glob(CACHE + "/*.rbln"))
print("RESULT " + json.dumps(res), flush=True)
