"""Parity of the ComfyUI port against dumps of the official Ming-3.0-tts-0.5B runtime.

    <comfy>/.venv/bin/python tests/parity_05b.py --comfy <ComfyUI> --checkpoint <Ming-3.0-tts-0.5B>/model.safetensors \
        --dumps <dir of *.pt written by the reference harness>

Each dump holds the reference's token ids, speaker embeddings, prompt latents, per-step flow
noise and sampled latents, and the decoded waveform. With the recorded noise fed in, the port
must reproduce the token ids exactly and the latents/waveform to bf16 precision.
"""

import argparse
import glob
import os
import sys

import torch


def setup(comfy, cpu):
    sys.path.insert(0, comfy)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.argv = sys.argv[:1] + (["--cpu"] if cpu else [])
    import comfy.options

    comfy.options.enable_args_parsing(cpu)


def sf_read(path, **_):
    import wave
    import numpy as np

    with wave.open(path) as w:
        sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        data = np.frombuffer(w.readframes(n), dtype=np.int16).reshape(-1, ch).astype(np.float32) / 32768
    return data, sr


def split_checkpoint(path):
    from safetensors.torch import load_file

    sd = load_file(path)
    vae = {k[len("audio."):]: v for k, v in sd.items() if k.startswith("audio.")}
    tts = {k: v for k, v in sd.items() if not k.startswith("audio.")}
    return tts, vae


def rel(a, b):
    a, b = a.float().flatten().cpu(), b.float().flatten().cpu()
    n = min(a.numel(), b.numel())
    a, b = a[:n], b[:n]
    return f"max|d| {(a - b).abs().max():.3e}  rel {((a - b).norm() / b.norm().clamp_min(1e-12)):.3e}  cos {torch.nn.functional.cosine_similarity(a, b, dim=0):.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--ref-repo", default=None, help="Ming-3.0-tts checkout, to re-encode the reference prompt clips")
    args = ap.parse_args()
    setup(args.comfy, args.device == "cpu")

    import comfy.ops
    from ming import prompts
    from ming.tts import MingTTS
    from ming.vae import MingAudioVAE

    dev, dt = torch.device(args.device), {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    ops = comfy.ops.disable_weight_init
    tts_sd, vae_sd = split_checkpoint(args.checkpoint)

    model = MingTTS(device=dev, dtype=dt, operations=ops)
    missing, unexpected = model.load_state_dict(tts_sd, strict=False)
    unexpected = [k for k in unexpected if not k.endswith("rotary_embed.inv_freq") and k != "model.lm_head.weight"]
    print("TTS  missing", missing, "unexpected", unexpected)
    model.to(dev, dt).eval()

    vae = MingAudioVAE(device=dev, dtype=dt, operations=ops)
    missing, unexpected = vae.load_state_dict(vae_sd, strict=False)
    print("VAE  missing", missing, "unexpected", unexpected)
    vae.to(dev, dt).eval()

    tok = prompts.load_tokenizer()
    for path in sorted(glob.glob(os.path.join(args.dumps, "*.pt"))):
        ref = torch.load(path, weights_only=False)
        case = ref["case"]
        name = os.path.basename(path)[:-3]
        print(f"\n== {name}")
        with torch.inference_mode():
            # 1. token layout
            if case["task"] == "FREEFORM":
                instruction = case["instruction"]
            elif case["task"] == "IP":
                instruction = prompts.structured_instruction(ip=case["controls"]["ip"])
            else:
                instruction = None
            language = {"en": "English", "zh": "Chinese"}.get(case.get("language_id"))
            prompt_latent = ref["prompt_latent"]
            n_spk = len(ref["spk_emb"])
            ids, spk_pos, lat_pos = prompts.build_input_ids(
                tok, prompts.tts_prompt(language), case["text"], num_speakers=n_spk, instruction=instruction,
                prompt_text=case.get("prompt_text"), prompt_latent_steps=0 if prompt_latent is None else prompt_latent.shape[1])
            same = ids == ref["input_ids"][0].tolist()
            print("input_ids identical:", same, len(ids))
            assert same

            # 2. embeddings with spliced speaker / prompt latents
            spk = ref["spk_emb"]
            if case.get("use_zero_spk_emb"):
                spk = [torch.zeros(1, 192)]
            embeds = model.embed_prompt(ids, spk_pos, spk, lat_pos, prompt_latent, device=dev, dtype=dt)
            if case.get("use_zero_spk_emb"):
                embeds[0, spk_pos[0]] = 0
            print("inputs_embeds:", rel(embeds, ref["inputs_embeds"]))

            # 3. AR decode with the reference's noise
            noise = lambda step: (ref["noise"][step], ref["sde"][step]) if step < len(ref["noise"]) else (torch.randn(1, 1, 64), torch.randn(10, 1, 1, 64))
            latents = model.generate(embeds, prompt_latent=prompt_latent, noise_fn=noise, progress=False)
            print(f"steps port {latents.shape[1]} ref {ref['tokens'].shape[1]}")
            print("step0 latent :", rel(latents[:, :1], ref["out"][0]))
            print("all latents  :", rel(latents, ref["tokens"]))

            # 4. VAE decode of the reference latents
            wav = vae.decode(ref["tokens"].to(dev, dt).transpose(1, 2))
            print("decode wav   :", rel(wav[0, 0], ref["wav"][0]), "samples", wav.shape[-1], ref["wav"].shape[-1])

            # 5. VAE encode of the reference prompt clip (reference samples the posterior; we compare its mean)
            if prompt_latent is not None and args.ref_repo:
                import torchaudio

                audio, sr = sf_read(os.path.join(args.ref_repo, case["prompt_wav_path"]), dtype="float32", always_2d=True)
                w = torch.from_numpy(audio.T.copy()).mean(0, keepdim=True)
                w = torchaudio.functional.resample(w, sr, 44100)
                align = 882 * 4
                w = torch.nn.functional.pad(w, (-w.shape[-1] % align, 0))  # front pad, like the reference
                mean, std = vae.posterior(w.unsqueeze(0).to(dev, dt))
                print("encode mean vs ref sample:", rel(mean, prompt_latent), f" mean|std| {std.mean():.3e} mean|mu| {mean.abs().mean():.3e}")


if __name__ == "__main__":
    main()
