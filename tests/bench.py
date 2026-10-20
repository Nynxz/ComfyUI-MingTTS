"""Per-step timing of a Ming TTS model loaded through ComfyUI (LLM step vs flow head).

    <comfy>/.venv/bin/python tests/bench.py --comfy <ComfyUI> --model <tts .safetensors> [--steps 30]
"""

import argparse
import importlib
import os
import sys
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=30)
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
    flow = importlib.import_module(pack.__name__ + ".ming.flow")

    patcher = comfy.sd.load_diffusion_model(args.model)
    mm.load_models_gpu([patcher])
    model = patcher.model.diffusion_model
    dev, dt = patcher.load_device, patcher.model.get_dtype_inference()
    print(f"loaded {model.backbone}: {patcher.loaded_size() / 1e9:.1f} GB on GPU of {patcher.model_size() / 1e9:.1f} GB")
    tiny = model.backbone == "ling_v3"
    tok = prompts.load_tokenizer("ming_tts_tiny" if tiny else "ming_tts_0.5b")
    text = "The old radio crackled softly as the storm moved across the valley."
    if tiny:
        ids, spos, _ = prompts.build_input_ids_tiny(tok, text, None, None, 0, "English")
    else:
        ids, spos, _ = prompts.build_input_ids(tok, prompts.tts_prompt("English"), text, num_speakers=1)
    sync = torch.cuda.synchronize
    with torch.inference_mode():
        embeds = model.embed_prompt(ids, spos, [None], device=dev, dtype=dt)
        state = model.llm_state(len(ids) + args.steps + 1, dev, dt)
        sync(); t = time.time()
        cond = model.llm_forward(embeds, state)[:, -1:]
        sync(); prefill = time.time() - t
        sd = model.settings.state_dtype or dt
        hist = torch.zeros(1, 1, 64, device=dev, dtype=sd)
        llm_t = flow_t = 0.0
        for _ in range(args.steps):
            sync(); t = time.time()
            latent = flow.sample_patch(model.dit, cond, hist, torch.randn(1, 1, 64, device=dev, dtype=sd), torch.zeros(10, 1, 1, 64, device=dev, dtype=sd),
                                       cfg_mode=model.settings.cfg_mode)
            sync(); flow_t += time.time() - t; t = time.time()
            cond = model.llm_forward(model.linear_proj_audio(latent.to(dt)), state)[:, -1:]
            sync(); llm_t += time.time() - t
            hist = latent
    n = args.steps
    print(f"prefill {len(ids)} tokens: {prefill * 1000:.0f} ms;  per step: LLM {llm_t / n * 1000:.1f} ms + flow head {flow_t / n * 1000:.1f} ms "
          f"= {(llm_t + flow_t) / n * 1000:.1f} ms  ({0.08 / ((llm_t + flow_t) / n):.2f}x real time)")


if __name__ == "__main__":
    main()
