"""Parity of the Ling V3 port (Ming-3.0-tts-tiny) against layer-by-layer dumps of the official runtime.

    <comfy>/.venv/bin/python tests/parity_tiny.py --comfy <ComfyUI> --checkpoint <Ming-3.0-tts-tiny dir> --dumps <dir>

Runs on the CPU one decoder layer at a time (a few GB of RAM). Per case it checks the prompt
token ids and spliced embeddings, every layer's prefill output given the reference's input, the
same layer driven as prefill + single-token decode steps (KDA recurrent/conv state, MLA cache),
the chained run through all 24 layers, the final hidden states, stop probabilities, and the flow
head given the reference's inputs.
"""

import argparse
import glob
import json
import os
import sys

import torch


def rel(a, b):
    a, b = a.float().flatten().cpu(), b.float().flatten().cpu()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--layers", default="all", help="'all' or comma list of layer indices for the per-layer checks")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"], help="match the dtype the dumps were made in")
    args = ap.parse_args()

    sys.path.insert(0, args.comfy)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.argv = sys.argv[:1] + ["--cpu"]
    import comfy.options

    comfy.options.enable_args_parsing(True)
    import comfy.ops
    from safetensors import safe_open

    from ming import prompts
    from ming.flow import DiT, Aggregator, sample_patch
    from ming.ling import LingLayer, LingV3Config, stack_experts
    from ming.pipeline import Task, lower

    torch.set_grad_enabled(False)
    dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    ops = comfy.ops.disable_weight_init
    cfg = LingV3Config()
    wmap = json.load(open(os.path.join(args.checkpoint, "model.safetensors.index.json")))["weight_map"]
    handles = {}

    def weights(prefix):
        out = {}
        for k, f in wmap.items():
            if k.startswith(prefix):
                h = handles.setdefault(f, safe_open(os.path.join(args.checkpoint, f), "pt"))
                out[k[len(prefix):]] = h.get_tensor(k)
        return out

    def load(module, prefix, transform=None):
        sd = weights(prefix)
        sd = transform(sd) if transform else sd
        sd = {k: v for k, v in sd.items() if not k.endswith("inv_freq")}
        missing, unexpected = module.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, (prefix, missing[:4], unexpected[:4])
        return module.eval()

    embed = weights("model.model.word_embeddings.")["weight"].to(dt)
    spk_head = load(ops.Linear(192, cfg.hidden_size, dtype=dt), "spk_head.")
    agg = load(Aggregator(64, 1024, 8, 16, 4.0, llm_input_dim=cfg.hidden_size, dtype=dt, operations=ops), "linear_proj_audio.")
    dit = load(DiT(64, 1024, 16, 16, 4.0, llm_cond_dim=cfg.hidden_size, dtype=dt, operations=ops), "flowloss.cfm.model.")
    stop_head = load(ops.Linear(cfg.hidden_size, 2, dtype=dt), "stop_head.")
    norm_w = weights("model.model.norm.")["weight"]
    tok = prompts.load_tokenizer("ming_tts_tiny")
    only = None if args.layers == "all" else {int(i) for i in args.layers.split(",")}

    tasks = {"seed": ("speak", {"language": "English"}), "emotion": ("emotion", {"emotion": "sad"}),
             "ip": ("builtin", {"voice": "US_James"}), "freeform": ("instruction", {"instruction": "an encouraging sports coach, energetic and firm"})}
    dumps = {os.path.basename(p)[:-3]: torch.load(p, weights_only=False) for p in sorted(glob.glob(os.path.join(args.dumps, "*.pt")))}
    for name, ref in dumps.items():
        kind, controls = tasks[ref["case"]["task"]]
        low = lower(Task(kind, controls), "ling_v3", ref["case"]["text"])
        T = 0 if ref["prompt_latent"] is None else ref["prompt_latent"].shape[1]
        transcript = ref["case"].get("prompt_text") if T else None
        ids, spos, lpos = prompts.build_input_ids_tiny(tok, ref["case"]["text"], low.instruction, transcript, T, low.language)
        print(f"\n== {name}: ids identical {ids == ref['ids']} ({len(ids)}), speaker slot {spos == [ref['speaker_position']]}, instruction identical {low.instruction == ref['instruction']}")

        x = embed[torch.tensor(ids)].unsqueeze(0).clone()
        x[0, spos[0]] = 0 if ref["speaker_emb"] is None else spk_head(ref["speaker_emb"].to(dt))[0]
        if T:
            x[0, lpos:] = agg(ref["prompt_latent"].to(dt).reshape(-1, 1, 64)).reshape(T, -1)
        P = ref["prompt_len"]
        teach = torch.cat([agg(ref["teacher"][:, j:j + 1].to(dt)) for j in range(ref["teacher"].shape[1])], dim=1)
        x = torch.cat([x, teach], dim=1)
        print(f"   inputs_embeds rel {rel(x, ref['inputs_embeds']):.2e}")

        from ming.ling import KDAState, KVCache, rope_cos_sin
        L = x.shape[1]
        rope_all = rope_cos_sin(torch.arange(L), cfg.qk_rope_head_dim, cfg.rope_theta, dt)
        chained = ref["layers"][0].to(dt)
        worst_local, worst_step = 0.0, 0.0
        for i in range(cfg.num_hidden_layers):
            layer = load(LingLayer(cfg, i, dtype=dt, ops=ops, expert_ops=comfy.ops.mixed_precision_ops({}, dt)),
                         f"model.model.layers.{i}.", stack_experts)
            inp, want = ref["layers"][i].to(dt), ref["layers"][i + 1]
            cache = lambda: KVCache(L) if layer.mla else KDAState()
            if only is None or i in only:
                local = rel(layer(inp.clone(), rope_all, cache()), want)
                # prefill the prompt, then feed the teacher steps one token at a time
                c = cache()
                outs = [layer(inp[:, :P].clone(), (rope_all[0][:P], rope_all[1][:P]), c)]
                for j in range(P, L):
                    outs.append(layer(inp[:, j:j + 1].clone(), (rope_all[0][j:j + 1], rope_all[1][j:j + 1]), c))
                step = rel(torch.cat(outs, 1)[:, P:], want[:, P:])
                worst_local, worst_step = max(worst_local, local), max(worst_step, step)
                print(f"   layer {i:2d} {'MLA' if layer.mla else 'KDA'}: prefill {local:.2e}  decode steps {step:.2e}")
            chained = layer(chained, rope_all, cache())
            del layer
        h = chained.float()
        z = (h * torch.rsqrt(h.square().mean(-1, keepdim=True) + cfg.rms_norm_eps)).to(dt) * norm_w.to(dt)
        print(f"   chained 24 layers + norm: z rel {rel(z, ref['z']):.2e}  (worst per-layer prefill {worst_local:.2e}, decode {worst_step:.2e})")
        stop = stop_head(ref["z"][:, P - 1:].to(dt)).float().softmax(-1)[0, :, 1]
        print(f"   stop probs port {[round(v, 3) for v in stop.tolist()]}  ref {[round(v, 3) for v in ref['stop'].tolist()]}")
        hist, errs = ref["flow_hist0"].float(), []
        for j, (noise, out) in enumerate(zip(ref["flow_noise"], ref["flow_out"])):
            got = sample_patch(dit, ref["z"][:, P - 1 + j:P + j].to(dt), hist, noise.transpose(1, 2).float(), torch.zeros(10, 1, 1, 64),
                               cfg=2.0, cfg_mode="standard", noise_after_last=False)
            errs.append(rel(got, out))
            hist = ref["teacher"][:, j:j + 1].float()
        print(f"   flow head (given ref z/history/noise): max rel {max(errs):.2e}")


if __name__ == "__main__":
    main()
