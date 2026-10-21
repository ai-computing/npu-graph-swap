# Compile a Llama model with optimum-rbln. usage: compile_opt.py MODEL OUT BATCH SEQ DECODER_BATCHES(comma)
import sys, time, json, os
from optimum.rbln import RBLNLlamaForCausalLM
model, out, bs, seq, dbs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), [int(x) for x in sys.argv[5].split(",")]
t0 = time.time()
kw = dict(export=True, rbln_batch_size=bs, rbln_max_seq_len=seq, rbln_tensor_parallel_size=1, rbln_create_runtimes=False)
if len(dbs) > 1 or dbs != [bs]:
    kw["rbln_decoder_batch_sizes"] = dbs
m = RBLNLlamaForCausalLM.from_pretrained(model, **kw)
t1 = time.time()
m.save_pretrained(out)
t2 = time.time()
print(json.dumps({"compile_s": round(t1 - t0, 1), "save_s": round(t2 - t1, 1), "out": out, "bs": bs, "seq": seq, "decoder_batch_sizes": dbs}), flush=True)
