# Q3 on RNGD via furiosa-torch 2026.2: do graphs compiled later run against weights already on the device?
#   mode A: weights are device tensors created once and passed to every compiled graph as inputs.
#   mode B: weights live inside each compiled module (CompileModule.to(device) per graph).
# furiosa-smi does not show furiosa-torch allocations, so device memory is measured by capacity: allocate
# 256 MiB device tensors until allocation fails, free them, and compare the counts before and after a step.
# Runs only on the card the device plugin assigned (privileged containers see every card).
# usage: q3_ftorch.py A|B|compile-only
import os, sys, time, json, re, gc, subprocess
import torch
import furiosa.torch  # noqa: F401  (after torch, before other furiosa imports)
from furiosa.torch.custom_ops import CompileModule
from furiosa.torch import native_device as nd
try:
    from furiosa.torch.compiler import Config as CompilerConfig, TacticHintConfig
except ImportError:
    from furiosa.torch._C.config.compiler import CompilerConfig, TacticHintConfig

MODE = sys.argv[1]
COMPILE_ONLY = MODE == "compile-only"
OUT = "/mnt/store/ngs/results"; os.makedirs(OUT, exist_ok=True)

def assigned():
    name = open("/run/ngs/rngd").read().strip()
    ids = sorted({int(m.group(1)) for f in os.listdir("/dev/rngd") if (m := re.match(r"npu(\d+)", f))})
    return name, ids.index(int(name[3:]))

NAME, ORD = ("", 0) if COMPILE_ONLY else assigned()
DEV = f"rngd:{ORD}"
CHUNK = 256 << 20

def capacity():
    """Number of 256 MiB device tensors that still fit (allocated, then freed)."""
    held = []
    try:
        while len(held) < 400:
            held.append(torch.empty(CHUNK // 2, dtype=torch.bfloat16, device=DEV))
    except Exception as e:  # out of device memory
        err = type(e).__name__
    else:
        err = "cap"
    n = len(held); del held; gc.collect()
    return {"chunks": n, "gib": round(n * CHUNK / 2**30, 2), "stop": err}

nd.set_fusion(8)
nd.set_mesh_kind(nd.MeshKind.Mesh(2))
cfg = CompilerConfig(allow_external_operators=True, allow_unlowered_operators=True,
                     tactic_hint=TacticHintConfig.ForMiscModel, max_num_unique_shapes=8)
H, L = 8192, 16                      # 16 x 8192 x 8192 bf16 = 2 GiB of weights
torch.manual_seed(0)
Ws = [torch.randn(H, H, dtype=torch.bfloat16) * 0.02 for _ in range(L)]

args = ", ".join(f"w{i}" for i in range(L))
ns = {"torch": torch}
exec(f"class WeightsAsInputs(torch.nn.Module):\n    def forward(self, x, {args}):\n        for w in ({args},):\n"
     f"            x = torch.tanh(x @ w)\n        return x\n", ns)
WeightsAsInputs = ns["WeightsAsInputs"]

class WeightsInside(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ws = torch.nn.ParameterList(torch.nn.Parameter(w.clone(), requires_grad=False) for w in Ws)
    def forward(self, x):
        for w in self.ws:
            x = torch.tanh(x @ w)
        return x

def ref(x):
    for w in Ws:
        x = torch.tanh(x.float() @ w.float()).to(torch.bfloat16)
    return x

def compile_graph(mod, sample):
    gm = torch.fx.symbolic_trace(mod)
    t = time.time(); m = CompileModule.from_module(gm, tuple(sample), compiler_config=cfg)
    return m, round(time.time() - t, 1)

def out0(y):
    return (y[0] if isinstance(y, (tuple, list)) else y).detach().cpu()

def err(y, x):
    return float((y.float() - ref(x).float()).abs().max())

R = {"mode": MODE, "device": NAME, "ordinal": ORD, "weights_gib": round(sum(w.numel() * 2 for w in Ws) / 2**30, 2),
     "versions": {p: __import__("importlib.metadata").metadata.version(p) for p in ("furiosa-torch", "torch")},
     "loadavg": open("/proc/loadavg").read().split()[:3], "steps": []}
xs = {b: torch.randn(b, H, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(b)) for b in (1, 4)}
def step(name, **kw):
    s = {"step": name, **kw}
    if not COMPILE_ONLY:
        s["capacity"] = capacity()
    R["steps"].append(s); print(s, flush=True)

step("start")
if MODE in ("A", "compile-only"):
    dws = Ws
    if not COMPILE_ONLY:
        t = time.time(); dws = [w.to(DEV) for w in Ws]; step("weights uploaded", s=round(time.time() - t, 2))
    G = {}
    for b in (1, 4):
        m, cs = compile_graph(WeightsAsInputs(), [xs[b]] + Ws)
        if COMPILE_ONLY:
            step(f"compiled b{b}", compile_s=cs); continue
        m.to(DEV)
        t = time.time(); y = out0(m(xs[b].to(DEV), *dws)); fr = round(time.time() - t, 3)
        G[b] = m
        step(f"graph b{b} compiled, loaded, run", compile_s=cs, first_run_s=fr, max_abs_err=err(y, xs[b]))
    if not COMPILE_ONLY:
        y = out0(G[1](xs[1].to(DEV), *dws)); step("b1 again", max_abs_err=err(y, xs[1]))
if MODE == "B":
    G = {}
    for b in (1, 4):
        m, cs = compile_graph(WeightsInside(), [xs[b]])
        t = time.time(); m.to(DEV); ld = round(time.time() - t, 2)
        y = out0(m(xs[b].to(DEV)))
        G[b] = m
        step(f"graph b{b} (weights inside) compiled, loaded, run", compile_s=cs, to_device_s=ld, max_abs_err=err(y, xs[b]))
    y = out0(G[1](xs[1].to(DEV))); step("b1 again", max_abs_err=err(y, xs[1]))
json.dump(R, open(f"{OUT}/q3_ftorch_{MODE}.json", "w"), indent=1)
print("RESULT " + json.dumps(R), flush=True)
