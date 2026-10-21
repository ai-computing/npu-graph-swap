# Q3 probe: can a graph compiled separately run against weights already on the device?
# Toy model: 8 Linear(4096,4096) layers, fp32 weights (~0.5 GB) so weight copies are visible in rbln-stat.
import sys, os, time, json, subprocess
import torch, rebel
from rebel.compile_context import CompileContext

DEV = int(os.environ.get("DEV", "2"))
OUT = os.environ.get("OUT", os.path.expanduser("~/ngs-work/q3"))
os.makedirs(OUT, exist_ok=True)

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
    out = subprocess.run(["rbln-stat"], capture_output=True, text=True).stdout
    for l in out.splitlines():
        if l.startswith("| %d " % DEV):
            return l.split("|")[8].strip()
    return "?"

def ctx(kind):
    if kind == "none":
        return None
    if kind == "share":
        return CompileContext(use_weight_sharing=True)
    if kind == "global":
        return CompileContext(use_weight_sharing=True, use_global_ctx=True, device_id=DEV)

def comp(model, b, c):
    t = time.time()
    cm = rebel.compile_from_torch(model, [("x", [b, 4096], "float32")], compile_context=c)
    return cm, time.time() - t

def run(rt, b, model):
    x = torch.randn(b, 4096, generator=torch.Generator().manual_seed(b))
    y = rt.run(x.numpy())
    y = y[0] if isinstance(y, (list, tuple)) else y
    ref = model(x).detach().numpy()
    return float(abs(y - ref).max())

def info(cm):
    return {"alloc": cm.get_alloc_per_node(), "ctx": cm._compiled_model.get_executor_by_idx(0).get_context_id(),
            "by_key": {k: v for k, v in cm.get_alloc_per_node_by_key().items()}}

mode = sys.argv[1]
model = Toy().eval()
res = {"mode": mode, "mem0": mem()}
if mode in ("same_session_late", "global_late"):
    # One process: compile B1, load and run it, then compile B4 in the SAME session after B1's weights are on the device.
    c = ctx("share" if mode == "same_session_late" else "global")
    a, ta = comp(model, 1, c); ra = a.create_runtime(device=DEV); res["err_b1"] = run(ra, 1, model); res["mem_after_b1"] = mem()
    b, tb = comp(model, 4, c); rb = b.create_runtime(device=DEV); res["err_b4"] = run(rb, 4, model); res["mem_after_b4"] = mem()
    res["err_b1_again"] = run(ra, 1, model)
    res.update(compile_s=[round(ta, 1), round(tb, 1)], info_b1=info(a), info_b4=info(b))
elif mode.startswith("save_"):
    # Separate process compiles one graph and saves it. save_<ctxkind>_<batch>
    _, kind, bs = mode.split("_")
    c = ctx(kind); m, t = comp(model, int(bs), c); m.save(f"{OUT}/toy_{kind}_b{bs}.rbln")
    res.update(compile_s=round(t, 1), info=info(m))
elif mode.startswith("load_"):
    # Load two separately compiled files into one process. load_<ctxkind>
    kind = mode.split("_")[1]
    a = rebel.RBLNCompiledModel(f"{OUT}/toy_{kind}_b1.rbln"); ra = a.create_runtime(device=DEV)
    res["err_b1"] = run(ra, 1, model); res["mem_after_b1"] = mem(); print(res, flush=True)
    b = rebel.RBLNCompiledModel(f"{OUT}/toy_{kind}_b4.rbln"); rb = b.create_runtime(device=DEV)
    res["err_b4"] = run(rb, 4, model); res["mem_after_b4"] = mem(); res["err_b1_again"] = run(ra, 1, model)
    res.update(info_b1=info(a), info_b4=info(b))
elif mode == "cached_then_fresh":
    # vllm-rbln partial-cache path: load a cached B1 compiled in another process, reserve its const bufs
    # in a new global session, compile B4 fresh in this process.
    c = ctx("global")
    a = rebel.RBLNCompiledModel(f"{OUT}/toy_global_b1.rbln"); a.reserve_const_buf_indices(c)
    ra = a.create_runtime(device=DEV); res["err_b1"] = run(ra, 1, model); res["mem_after_b1"] = mem()
    b, tb = comp(model, 4, c); rb = b.create_runtime(device=DEV); res["err_b4"] = run(rb, 4, model)
    res["mem_after_b4"] = mem(); res["err_b1_again"] = run(ra, 1, model)
    res.update(info_b1=info(a), info_b4=info(b))
elif mode in ("cached_then_fresh2", "global_two_sessions"):
    # cached_then_fresh2: reserve B1's const bufs, compile B4 fresh, create both runtimes, then run.
    # global_two_sessions: one process, two different global sessions (no reservation), each compiles one graph.
    if mode == "cached_then_fresh2":
        c = ctx("global"); a = rebel.RBLNCompiledModel(f"{OUT}/toy_global_b1.rbln"); a.reserve_const_buf_indices(c)
        b, tb = comp(model, 4, c)
    else:
        a, ta = comp(model, 1, ctx("global")); b, tb = comp(model, 4, ctx("global"))
    res.update(info_b1=info(a), info_b4=info(b))
    ra = a.create_runtime(device=DEV); rb = b.create_runtime(device=DEV); res["mem_after_rt"] = mem(); print(res, flush=True)
    res["err_b1"] = run(ra, 1, model); print("b1 ok", res["err_b1"], flush=True)
    res["err_b4"] = run(rb, 4, model); print("b4 ok", res["err_b4"], flush=True)
    res["mem_after_b4"] = mem(); res["err_b1_again"] = run(ra, 1, model)
print("RESULT " + json.dumps(res), flush=True)
