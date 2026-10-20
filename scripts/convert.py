"""Convert a released Ming-3.0 TTS checkpoint into ComfyUI files.

    python scripts/convert.py <hf checkpoint dir> --name ming-3.0-tts-0.5b --out <dir>
    python scripts/convert.py <hf checkpoint dir> --name ming-3.0-tts-tiny --out <dir> --no-vae

Writes:
  <name>_bf16.safetensors          -> models/diffusion_models (LLM + flow head + aggregator + heads)
  ming-3.0_audio_vae.safetensors   -> models/vae (AudioVAE + CAM++ speaker encoder from campplus.onnx;
                                      identical in every Ming-3.0 release, so it is needed once)

Key names are unchanged except the tiny's MoE, whose 128 per-expert weights per layer are
stacked into ComfyUI expert banks. Tensors are streamed to disk one at a time, so converting
the 18 GB tiny needs little RAM. ``onnx`` is needed for campplus.onnx (conversion only).
"""

import argparse
import json
import os
import re
import struct
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DTYPES = {torch.float32: "F32", torch.bfloat16: "BF16", torch.float16: "F16", torch.int8: "I8", torch.uint8: "U8",
          torch.int32: "I32", torch.int64: "I64", torch.float8_e4m3fn: "F8_E4M3"}
EXPERT = re.compile(r"^(.*\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
# model.lm_head is the TTS models' unused text head; the audio-tiny thinker's top-level lm_head is kept
DROP = re.compile(r"(rotary_embed\.inv_freq|istft\.window)$|^model\.lm_head\.weight$")


def write_safetensors(path, entries, metadata=None):
    """entries: [(name, shape, dtype, produce)], ``produce()`` returning the tensor when it is written."""
    header, offset = {}, 0
    for name, shape, dtype, _ in entries:
        size = torch.tensor([], dtype=dtype).element_size()
        for d in shape:
            size *= d
        header[name] = {"dtype": DTYPES[dtype], "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    if metadata:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for name, shape, dtype, produce in entries:
            t = produce().to(dtype).contiguous()
            if tuple(t.shape) != tuple(shape):
                raise ValueError(f"{name}: produced {tuple(t.shape)}, declared {tuple(shape)}")
            f.write(t.reshape(-1).view(torch.uint8).numpy().tobytes())


def campplus_state_dict(onnx_path):
    import onnx
    from onnx import numpy_helper

    from ming.speaker import CAMPPlus, conv_modules_in_onnx_order

    graph = onnx.load(onnx_path).graph
    inits = {t.name: torch.from_numpy(numpy_helper.to_array(t).copy()) for t in graph.initializer}
    model = CAMPPlus()
    names = {m: n for n, m in model.named_modules()}
    sd = {}

    convs = [n for n in graph.node if n.op_type == "Conv"]
    modules = conv_modules_in_onnx_order(model)
    if len(convs) != len(modules):
        raise ValueError(f"campplus.onnx has {len(convs)} convs, expected {len(modules)}")
    for node, module in zip(convs, modules):
        name = names[module]
        w = inits[node.input[1]]
        if w.shape != module.weight.shape:
            raise ValueError(f"{name}: onnx weight {node.input[1]} {tuple(w.shape)} != {tuple(module.weight.shape)}")
        sd[f"{name}.weight"] = w
        if len(node.input) > 2:
            if module.bias is None:
                raise ValueError(f"{name}: onnx conv has a bias, module has none")
            sd[f"{name}.bias"] = inits[node.input[2]]
        elif module.bias is not None:
            raise ValueError(f"{name}: module expects a folded bias")

    eps = {n.input[1]: next((a.f for a in n.attribute if a.name == "epsilon"), 1e-5) for n in graph.node if n.op_type == "BatchNormalization"}
    if any(abs(e - 1e-5) > 1e-9 for e in eps.values()):
        raise ValueError(f"unexpected BatchNorm epsilon {set(eps.values())}")
    renames = [
        (r"^xvector\.block(\d+)\.tdnnd(\d+)\.nonlinear1\.batchnorm\.(.+)$", lambda m: f"blocks.{int(m[1]) - 1}.layers.{int(m[2]) - 1}.bn.{m[3]}"),
        (r"^xvector\.transit(\d+)\.nonlinear\.batchnorm\.(.+)$", lambda m: f"transits.{int(m[1]) - 1}.bn.{m[2]}"),
        (r"^xvector\.dense\.nonlinear\.batchnorm\.(.+)$", lambda m: f"dense_bn.{m[1]}"),
    ]
    for key, value in inits.items():
        for pattern, fn in renames:
            m = re.match(pattern, key)
            if m:
                sd[fn(m)] = value
    for name, buf in model.state_dict().items():
        if name.endswith("num_batches_tracked"):
            sd[name] = torch.zeros_like(buf)
    model.load_state_dict(sd, strict=True)  # every parameter and buffer covered
    return {k: v.float().contiguous() for k, v in sd.items()}


class Shards:
    def __init__(self, src):
        index = os.path.join(src, "model.safetensors.index.json")
        if os.path.exists(index):
            self.where = {k: os.path.join(src, f) for k, f in json.load(open(index))["weight_map"].items()}
        else:
            path = os.path.join(src, "model.safetensors")
            with safe_open(path, "pt") as f:
                self.where = {k: path for k in f.keys()}
        self.handles = {}

    def handle(self, key):
        path = self.where[key]
        if path not in self.handles:
            self.handles[path] = safe_open(path, "pt")
        return self.handles[path]

    def info(self, key):
        s = self.handle(key).get_slice(key)
        dtype = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}[s.get_dtype()]
        return tuple(s.get_shape()), dtype

    def get(self, key):
        return self.handle(key).get_tensor(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="Hugging Face checkpoint directory")
    ap.add_argument("--name", required=True, help="output name, e.g. ming-3.0-tts-0.5b")
    ap.add_argument("--out", default=".")
    ap.add_argument("--no-vae", action="store_true", help="skip the AudioVAE file (it is the same for every Ming-3.0 model)")
    args = ap.parse_args()

    config = os.path.join(args.src, "config.json")
    config = json.load(open(config)) if os.path.exists(config) else {}
    llm = config.get("llm_config") or config.get("text_config") or {}
    # multi-token-prediction layers sit after the decoder layers (audio-tiny); generation doesn't use them
    mtp = re.compile(r"^model\.(?:model\.)?layers\.(\d+)\.")
    n_layers = llm.get("num_hidden_layers")

    shards = Shards(args.src)
    tts, vae, banks = [], [], {}
    for key in sorted(shards.where):
        m = mtp.match(key)
        if DROP.search(key) or (m and n_layers and int(m[1]) >= n_layers):
            continue
        m = EXPERT.match(key)
        if m:
            banks.setdefault(m[1], {}).setdefault(int(m[2]), {})[m[3]] = key
            continue
        shape, dtype = shards.info(key)
        # fp32 tensors (KDA A_log/dt_bias, MoE expert_bias, ...) stay fp32: the reference keeps them so
        out_dtype = dtype if dtype == torch.float32 else torch.bfloat16
        produce = lambda k=key, d=out_dtype: shards.get(k).to(d)
        if key.startswith("audio."):
            vae.append((key[len("audio."):], shape, out_dtype, produce))
        else:
            tts.append((key, shape, out_dtype, produce))
    for prefix, experts in sorted(banks.items()):
        order = sorted(experts)
        (inter, hidden), _ = shards.info(experts[order[0]]["gate_proj"])
        stack_gate_up = lambda e=experts, o=order: torch.stack([torch.cat([shards.get(e[i]["gate_proj"]), shards.get(e[i]["up_proj"])]) for i in o])
        stack_down = lambda e=experts, o=order: torch.stack([shards.get(e[i]["down_proj"]) for i in o])
        tts.append((f"{prefix}.gate_up_proj.weight", (len(order), 2 * inter, hidden), torch.bfloat16, stack_gate_up))
        tts.append((f"{prefix}.down_proj.weight", (len(order), hidden, inter), torch.bfloat16, stack_down))

    meta = {"ming_source": os.path.basename(os.path.normpath(args.src))}
    if "rope_theta" in llm:
        meta["ming_rope_theta"] = llm["rope_theta"]

    os.makedirs(args.out, exist_ok=True)
    tts_path = os.path.join(args.out, f"{args.name}_bf16.safetensors")
    write_safetensors(tts_path, tts, meta)
    print(f"wrote {tts_path}: {len(tts)} tensors")
    if not args.no_vae:
        for key, value in campplus_state_dict(os.path.join(args.src, "campplus.onnx")).items():
            vae.append((f"speaker_encoder.{key}", tuple(value.shape), torch.float32, lambda v=value: v))  # fp32: BatchNorm statistics
        vae_path = os.path.join(args.out, "ming-3.0_audio_vae.safetensors")
        write_safetensors(vae_path, vae, meta)
        print(f"wrote {vae_path}: {len(vae)} tensors")


if __name__ == "__main__":
    main()
