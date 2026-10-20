"""Ming-3.0-audio-tiny through ComfyUI's Load CLIP path: Ming Transcribe on clips with known
transcripts, and the stock TextGenerate node with an audio question.

    <comfy>/.venv/bin/python tests/e2e_thinker.py --comfy <ComfyUI> --clip <ming-3.0-audio-tiny .safetensors> \
        --clips "<wav>=<transcript>" ...
"""

import argparse
import importlib
import os
import sys
import time
import wave

import numpy as np
import torch


def load_wav(path):
    with wave.open(path) as w:
        rate, ch, n, width = w.getframerate(), w.getnchannels(), w.getnframes(), w.getsampwidth()
        raw = np.frombuffer(w.readframes(n), dtype=np.uint8).reshape(-1, width)
        if width == 3:  # 24-bit PCM: widen to int32 by putting the 3 bytes in the top of each word
            raw = np.concatenate([np.zeros((raw.shape[0], 1), np.uint8), raw], axis=1)
        data = raw.copy().view({1: np.int8, 2: np.int16, 3: np.int32, 4: np.int32}[width]).reshape(-1, ch)
        data = data.astype(np.float32) / float(2 ** (8 * (4 if width == 3 else width) - 1))
    return {"waveform": torch.from_numpy(data.T.copy()).unsqueeze(0), "sample_rate": rate}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--clips", nargs="+", required=True, help="wav=expected transcript")
    args = ap.parse_args()
    sys.path.insert(0, args.comfy)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.argv = sys.argv[:1]
    import comfy.options

    comfy.options.enable_args_parsing(False)
    import comfy.sd

    pack = importlib.import_module(os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    nodes = importlib.import_module(pack.__name__ + ".ming.nodes")
    from comfy_extras.nodes_textgen import TextGenerate

    t = time.time()
    clip = comfy.sd.load_clip(ckpt_paths=[args.clip])  # what Load CLIP calls
    print(f"loaded {type(clip.cond_stage_model).__name__} in {time.time() - t:.0f}s")
    with torch.inference_mode():
        for spec in args.clips:
            path, expected = spec.split("=", 1)
            t = time.time()
            transcript, language = nodes.MingTranscribe.execute(clip, load_wav(path), 256).result
            print(f"[{language}] {transcript!r}  ({time.time() - t:.1f}s)\n   expected {expected!r}")
        path = args.clips[0].split("=", 1)[0]
        t = time.time()
        text, _ = TextGenerate.execute(clip, "Describe the speaker's voice and how they sound in one sentence.", 128,
                                       {"sampling_mode": "off"}, audio=load_wav(path)).result
        print(f"TextGenerate: {text!r}  ({time.time() - t:.1f}s)")


if __name__ == "__main__":
    main()
