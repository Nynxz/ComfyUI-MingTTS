"""Compare a quantized Ming TTS file against its bf16 source, both loaded through ComfyUI.

    <comfy>/.venv/bin/python tests/quant_check.py --comfy <ComfyUI> --ref <bf16 .safetensors> --quant <int8 .safetensors>

Runs the same prompt through each model (one at a time) and reports the relative error of the
prompt's final hidden states and of the first latents generated with the same seed.
"""

import argparse
import importlib
import os
import sys

import torch


def run(path, comfy_sd, mm, prompts):
    patcher = comfy_sd.load_diffusion_model(path)
    mm.load_models_gpu([patcher])
    model = patcher.model.diffusion_model
    dev, dt = patcher.load_device, patcher.model.get_dtype_inference()
    tiny = model.backbone == "ling_v3"
    tok = prompts.load_tokenizer("ming_tts_tiny" if tiny else "ming_tts_0.5b")
    text = "The old radio crackled softly as the storm moved across the valley."
    if tiny:
        ids, spos, _ = prompts.build_input_ids_tiny(tok, text, None, None, 0, "English")
    else:
        ids, spos, _ = prompts.build_input_ids(tok, prompts.tts_prompt("English"), text, num_speakers=1)
    with torch.inference_mode():
        embeds = model.embed_prompt(ids, spos, [None], device=dev, dtype=dt)
        state = model.llm_state(len(ids) + 4, dev, dt)
        hidden = model.llm_forward(embeds.clone(), state).float().cpu()
        latents = model.generate(model.embed_prompt(ids, spos, [None], device=dev, dtype=dt), seed=3, max_steps=24, progress=False).float().cpu()
    mm.unload_all_models()
    mm.soft_empty_cache()
    return hidden, latents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--quant", required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.comfy)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.argv = sys.argv[:1]
    import comfy.options

    comfy.options.enable_args_parsing(False)
    import comfy.model_management as mm
    import comfy.sd

    pack = importlib.import_module(os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    prompts = importlib.import_module(pack.__name__ + ".ming.prompts")
    h_ref, l_ref = run(args.ref, comfy.sd, mm, prompts)
    h_q, l_q = run(args.quant, comfy.sd, mm, prompts)
    rel = lambda a, b: ((a - b).norm() / b.norm()).item()
    cos = lambda a, b: torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    print(f"prompt hidden states: rel {rel(h_q, h_ref):.3e}  cos {cos(h_q, h_ref):.5f}")
    n = min(l_q.shape[1], l_ref.shape[1], 8)
    print(f"first {n} latents (same seed): rel {rel(l_q[:, :n], l_ref[:, :n]):.3e}  cos {cos(l_q[:, :n], l_ref[:, :n]):.5f}")


if __name__ == "__main__":
    main()
