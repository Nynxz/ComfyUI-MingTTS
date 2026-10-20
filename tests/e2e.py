"""Headless end-to-end run through ComfyUI's own loaders and the pack's nodes.

    <comfy>/.venv/bin/python tests/e2e.py --comfy <ComfyUI> --model <ming tts .safetensors> --vae <ming vae .safetensors> \
        --out <dir> [--reference clip.wav --transcript "..."]

Loads the TTS with comfy.sd.load_diffusion_model (what Load Diffusion Model calls) and the VAE
with comfy.sd.VAE (Load VAE), runs Ming TTS for a few tasks and decodes with the stock
VAE Decode Audio code, writing one WAV per task.
"""

import argparse
import os
import sys
import time
import wave

import torch


def save_wav(path, audio):
    data = audio["waveform"][0].float().clamp(-1, 1).mean(0).cpu().numpy()
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(audio["sample_rate"])
        w.writeframes((data * 32767).astype("<i2").tobytes())


def load_wav(path):
    import numpy as np

    with wave.open(path) as w:
        rate, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        data = np.frombuffer(w.readframes(n), dtype=np.int16).reshape(-1, ch).astype(np.float32) / 32768
    return {"waveform": torch.from_numpy(data.T.copy()).unsqueeze(0), "sample_rate": rate}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference")
    ap.add_argument("--transcript", default="")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--script", action="store_true", help="run the Ming Script node instead (render, cached re-render, one line edited)")
    args = ap.parse_args()

    sys.path.insert(0, args.comfy)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.argv = sys.argv[:1]
    import comfy.options

    comfy.options.enable_args_parsing(False)
    import importlib

    import comfy.sd
    import comfy.utils
    pack = importlib.import_module(os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # registers
    from comfy_extras.nodes_audio import vae_decode_audio
    nodes = importlib.import_module(pack.__name__ + ".ming.nodes")

    model = comfy.sd.load_diffusion_model(args.model)
    print("model:", type(model.model).__name__, model.model.get_dtype_inference())
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(args.vae))
    print("vae:", type(vae.first_stage_model).__name__, "speaker encoder:", vae.ming_speaker_encoder is not None)

    voice = None
    if args.reference:
        voice = nodes.MingReferenceVoice.execute(vae, load_wav(args.reference), args.transcript, 20.0).result[0]
        print("voice: latent", None if voice.latent_front is None else tuple(voice.latent_front.shape), "speaker", tuple(voice.speaker.shape))

    text_en = "Keep your eyes forward and your shoulders relaxed. One more round [Laughter], you can do this!"
    cases = {
        "speak_en": ({"task": "speak"}, text_en),
        "instruction_en": ({"task": "instruction", "instruction": "Sound like an encouraging sports coach, energetic and firm without shouting."}, text_en),
        "builtin_US_James": ({"task": "built-in voice", "voice": "US_James"}, "The old radio crackled softly as the storm moved across the valley."),
        "emotion_sad": ({"task": "emotion", "emotion": "sad"}, "I waited by the window all night, but nobody came home."),
        "design": ({"task": "voice design", "description": "A very old woman with a thin, trembling voice, speaking slowly and kindly."},
                   "Come closer, child, and I will tell you how the river got its name."),
    }
    os.makedirs(args.out, exist_ok=True)
    if args.script:
        voices = {"voice_1": voice} if voice is not None else {}
        script = nodes.SCRIPT_EXAMPLE
        for label, text in (("first", script), ("cached", script), ("one line edited", script.replace("Stay here.", "Stay right here."))):
            t = time.time()
            audio, timeline = nodes.MingScript.execute(model, vae, text, "English", 0, 2.0, 0.25, 10, 0.0, **voices).result
            print(f"script ({label}): {audio['waveform'].shape[-1] / audio['sample_rate']:.2f}s audio in {time.time() - t:.1f}s")
        print(timeline)
        save_wav(os.path.join(args.out, "script.wav"), audio)
        return
    for name, (task, text) in cases.items():
        t = time.time()
        latent = nodes.MingTTS.execute(model, text, task, "English", args.seed, 2.0, 10, 0.0, 0.25, True, 50, voice=voice, vae=vae).result[0]
        audio = vae_decode_audio(vae, latent)
        seconds = audio["waveform"].shape[-1] / audio["sample_rate"]
        save_wav(os.path.join(args.out, f"{name}.wav"), audio)
        print(f"{name}: {seconds:.2f}s audio in {time.time() - t:.1f}s  latent {tuple(latent['samples'].shape)}")


if __name__ == "__main__":
    with torch.inference_mode():  # as ComfyUI executes nodes
        main()
