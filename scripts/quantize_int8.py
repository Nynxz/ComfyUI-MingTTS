"""int8 + ConvRot quantization of a converted Ming TTS checkpoint (mainly for the 16 GB tiny).

    <comfy>/.venv/bin/python scripts/quantize_int8.py ming-3.0-tts-tiny_bf16.safetensors ming-3.0-tts-tiny_int8_convrot.safetensors \
        --tool <comfy-model-tools dir>

Uses quant_int8_convrot.py's recipe (fp32 upcast, block-Hadamard rotation, per-channel absmax
scale, `<layer>.comfy_quant` config), which skips 3-D tensors; here every MoE expert matrix is
quantized the same way and stacked into an int8 bank that ComfyUI's expert banks load
([E, out, in] int8 + [E, out, 1] scale). Quantized: the LLM's attention/MLP linears and all
experts. Kept bf16/fp32: embeddings, the router (fp32 in the reference), the KDA decay/beta
gates (their error accumulates in the recurrent state), the flow head, aggregator and heads.
"""

import argparse
import json
import os
import re
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert import write_safetensors  # noqa: E402

# model.model.layers (TTS) or model.layers (audio-tiny thinker)
LLM_LINEAR = re.compile(r"^model\.(?:model\.)?layers\.\d+\.(attention|mlp(\.shared_experts)?)\.\w+_proj\.weight$|"
                        r"^model\.(?:model\.)?layers\.\d+\.attention\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|dense)\.weight$")
KEEP = re.compile(r"\.attention\.(f_proj|b_proj|g_proj)\.weight$")  # KDA decay/beta gates (and MLA's 16-row gate)
BANK = re.compile(r"\.mlp\.experts\.(gate_up_proj|down_proj)\.weight$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--tool", default=os.path.join(os.path.dirname(__file__), "..", "..", "references", "comfy-model-tools"),
                    help="directory with quant_int8_convrot.py")
    ap.add_argument("--min-gemm", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.tool))
    import quant_int8_convrot as q

    entries, errs = [], []
    f = safe_open(args.src, "pt")
    meta = f.metadata() or {}

    def quantized(key, shape, stacked):
        """-> (group size, get(kind)); the matrix is quantized once, on the first get, and freed
        after its scale (written right after the weight) is taken."""
        gs = q.best_gs(shape[-1])
        memo = {}

        def get(kind):
            if not memo:
                w = f.get_tensor(key)
                qs, ss = [], []
                for m in (w if stacked else w.unsqueeze(0)):
                    qd, scale = q.quantize_convrot(m, gs, mseclip=False, device=args.device)
                    errs.append(q.recon_metrics(qd, scale, m, gs, device=args.device)[1])
                    qs.append(qd.cpu())
                    ss.append(scale.cpu())
                memo["weight"], memo["scale"] = (torch.stack(qs), torch.stack(ss)) if stacked else (qs[0], ss[0])
            out = memo[kind]
            if kind == "scale":
                memo.clear()
            return out
        return gs, get

    for key in f.keys():
        s = f.get_slice(key)
        shape = tuple(s.get_shape())
        dtype = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16, "I64": torch.int64}[s.get_dtype()]
        base = key[: -len(".weight")]
        stacked = bool(BANK.search(key))
        eligible = stacked or (LLM_LINEAR.match(key) and not KEEP.search(key) and len(shape) == 2 and min(shape) >= args.min_gemm)
        if eligible and q.best_gs(shape[-1]):
            gs, run = quantized(key, shape, stacked)
            scale_shape = shape[:-1] + (1,)
            entries.append((key, shape, torch.int8, lambda run=run: run("weight")))
            entries.append((f"{base}.weight_scale", scale_shape, torch.float32, lambda run=run: run("scale")))
            cfg = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": gs}
            if stacked:
                cfg["num_experts"] = shape[0]
            blob = torch.tensor(list(json.dumps(cfg).encode()), dtype=torch.uint8)
            entries.append((f"{base}.comfy_quant", tuple(blob.shape), torch.uint8, lambda b=blob: b))
        else:
            entries.append((key, shape, dtype, lambda k=key: f.get_tensor(k)))
    write_safetensors(args.dst, entries, meta)
    if errs:
        errs = torch.tensor(errs)
        print(f"wrote {args.dst}: {len(errs)} quantized matrices, weight relerr mean {errs.mean():.2f}% max {errs.max():.2f}%")


if __name__ == "__main__":
    main()
