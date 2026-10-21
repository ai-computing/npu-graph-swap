# Q3 on a real LLM: build Llama with decoder batch 8 only, load it, then compile a batch-1 decoder in the SAME
# compile session after the weights are already on the device, and run it. No vendor code is modified: the
# compile context and wrapped model are kept by wrapping two optimum-rbln classmethods at runtime.
# usage: q3_late.py MODEL DEVICE OUT_JSON
import sys, os, time, json, copy, subprocess
import torch, rebel
from transformers import AutoTokenizer
from optimum.rbln import RBLNLlamaForCausalLM
from optimum.rbln.transformers.models.decoderonly.decoderonly_runtime_utils import RBLNRuntimeModel

model_id, dev, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
C = RBLNLlamaForCausalLM
stash = {}
_ctx, _wrap = C._get_compile_context.__func__, C._wrap_model_if_needed.__func__
def ctx_keep(cls, cc, ex):
    c, st = _ctx(cls, cc, ex); stash["ctx"], stash["static"] = c, st; return c, st
def wrap_keep(cls, model, cfg):
    w = _wrap(cls, model, cfg); stash["wrapped"] = w; return w
C._get_compile_context = classmethod(ctx_keep)
C._wrap_model_if_needed = classmethod(wrap_keep)

def stat():
    o = subprocess.run(["rbln-stat"], capture_output=True, text=True).stdout
    return [l.split("|")[8].strip() for l in o.splitlines() if l.startswith("| %d " % dev)][0]

R = {"model": os.path.basename(model_id.rstrip("/")), "loadavg": open("/proc/loadavg").read().split()[:3], "mem0": stat()}
t = time.time()
m = C.from_pretrained(model_id, export=True, rbln_batch_size=8, rbln_max_seq_len=512, rbln_tensor_parallel_size=1,
                      rbln_config={"device": dev})
R["compile_b8_build_s"] = round(time.time() - t, 1)
R["mem_after_b8_load"] = stat()
tok = AutoTokenizer.from_pretrained(model_id); tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
prompts = ["The capital of France is", "Explain in one sentence why the sky is blue.", "def fibonacci(n):",
           "Write a haiku about autumn leaves.", "List three prime numbers greater than 100:", "Translate to Korean: Good morning, everyone.",
           "The theory of relativity states that", "Once upon a time in a small village,"]
def gen(b):
    enc = tok(prompts[:b], return_tensors="pt", padding=True)
    o = m.generate(**enc, max_new_tokens=32, do_sample=False)
    return o[:, enc["input_ids"].shape[1]:].tolist()
R["tokens_b8"] = gen(8)
R["mem_after_b8_run"] = stat()

# Late graph: decoder for batch 1, compiled now, in the same session, against the resident weights.
cfg = m.rbln_config
cc = copy.deepcopy(cfg.compile_cfgs[-1])
cc.compiled_model_name = "decoder_batch_1"
cc.input_info = C.get_input_info(batch_size=1, query_length=1, rbln_config=cfg, model_config=m.config)
ex = cc.get_dummy_inputs(fill=0, static_tensors=stash["static"])
t = time.time()
with torch.inference_mode():
    cm = C._compile_model(stash["wrapped"], cc, ex, stash["ctx"], cfg, cfg.quantization, phase="decode")
R["late_compile_s"] = round(time.time() - t, 1)
R["late_alloc_by_key"] = {k: v[0] for k, v in cm.get_alloc_per_node_by_key().items()}
R["b8_alloc_by_key"] = {k: v[0] for k, v in m.model[1].get_alloc_per_node_by_key().items()}
R["prefill_alloc_by_key"] = {k: v[0] for k, v in m.model[0].get_alloc_per_node_by_key().items()}
R["ctx_ids"] = [m.model[0]._executor.get_context_id(), cm._compiled_model.get_executor_by_idx(0).get_context_id()]
t = time.time()
rt = rebel.Runtime(cm, tensor_type="pt", device=dev)
R["late_runtime_create_s"] = round(time.time() - t, 3)
R["mem_after_late_runtime"] = stat()
d8 = m.decoders[8]
m.decoders[1] = RBLNRuntimeModel(runtime=rt, phase="decode", batch_size=1, dec_attn_mask=d8.dec_attn_mask,
                                 page_table_manager=d8.page_table_manager, rbln_config=cfg, config=m.config,
                                 main_input_name="input_ids", embed_tokens=None)
t = time.time()
R["tokens_b1_late"] = gen(1)
R["late_first_generate_s"] = round(time.time() - t, 3)
R["mem_after_late_run"] = stat()
R["tokens_b8_again"] = gen(8)
R["b8_unchanged"] = R["tokens_b8_again"] == R["tokens_b8"]
R["b1_late_equals_b8_row0"] = R["tokens_b1_late"][0] == R["tokens_b8"][0]
json.dump(R, open(out, "w"), indent=1)
print("RESULT " + json.dumps({k: v for k, v in R.items() if not k.startswith("tokens")}), flush=True)
