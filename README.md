# ComfyUI-Ming

Native ComfyUI support for **[Ming-3.0-tts](https://github.com/inclusionAI/Ming-3.0-tts)**, inclusionAI's
speech synthesis models: zero-shot voice cloning, 484 built-in voices, voice design from a text
description, free-form delivery direction, emotion / speed / pitch / volume control, Cantonese
and Sichuanese, seven languages, and inline sounds like `[Laughter]` and `[Sigh]`.

- **No requirements.** Pure ComfyUI and torch. The reference runtime's `flash-linear-attention`,
  `onnxruntime`, `x_transformers` and `diffusers` dependencies are reimplemented.
- **Stock loaders.** Load Diffusion Model loads the TTS model, Load VAE the audio VAE, and the
  generated latents decode with the stock **VAE Decode Audio**. Offloading, dtype options and
  LoRA work as for any model.
- **Four nodes:** **Ming Reference Voice** (clone a voice from a clip), **Ming TTS** (one task
  per node, the controls change with the task), **Ming Script** (multi-voice scripts, each
  line cached so editing one line re-renders only that line) and **Ming Transcribe** (speech
  recognition with Ming-3.0-audio-tiny, e.g. for a reference clip's transcript).
- **Audio understanding natively:** Ming-3.0-audio-tiny loads with the stock **Load CLIP** and
  runs in the stock **TextGenerate** node, which has an audio input: ask anything about a clip.

## Install

Clone into `ComfyUI/custom_nodes` and restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Nynxz/ComfyUI-MingTTS ComfyUI-Ming
```

## Models

Download from [nynxz/Ming-3.0](https://huggingface.co/nynxz/Ming-3.0) into the folder shown.

| File | Folder | Size | Speed (RTX 5090) | Notes |
|---|---|---|---|---|
| `ming-3.0-tts-0.5b_bf16.safetensors` | `models/diffusion_models` | 1.4 GB | 1.6x real time | Qwen2.5-0.5B backbone |
| `ming-3.0-tts-tiny_int8_convrot.safetensors` | `models/diffusion_models` | 8.7 GB | 1.0x real time | Ling V3 MoE, 7.9B total / 1.3B active; fits a 12 GB card |
| `ming-3.0-tts-tiny_bf16.safetensors` | `models/diffusion_models` | 15.9 GB | 0.26x when offloaded | the unquantized tiny |
| `ming-3.0_audio_vae.safetensors` | `models/vae` | 1.6 GB | | 44.1 kHz AudioVAE + speaker encoder, shared by all |
| `ming-3.0-audio-tiny_bf16.safetensors` | `models/text_encoders` | 18 GB | | speech recognition / audio understanding (optional) |

Real time = one second of audio per second of generation, measured per decode step (80 ms of
audio) with the model fully on the GPU. The bf16 tiny figure is with 9.7 of its 15.9 GB on the GPU.

The tiny's int8 file quantizes every MoE expert and the attention/MLP linears with int8 + ConvRot
(0.78% mean weight error). It keeps the router, the KDA decay gates, the flow head and the
embeddings in bf16/fp32. Its hidden states match bf16 at cos 0.998.

The audio VAE is byte-identical in every Ming-3.0 release, so you need it once. The released
`Ming-3.0-tts-0.5B/model.safetensors` also loads directly in both loaders (it contains the VAE),
but only the converted VAE file carries the speaker encoder that voice cloning needs.
The files on [nynxz/Ming-3.0](https://huggingface.co/nynxz/Ming-3.0) were made with [`scripts/convert.py`](#making-the-weights).

## Nodes

**Ming TTS**: model, text, task → LATENT (→ VAE Decode Audio with the Ming VAE)

| Task | Controls | Uses the reference voice |
|---|---|---|
| speak | (none) | yes: continues the clip if it has a transcript, else its speaker embedding |
| emotion | happy, sad, angry, fearful, disgusted, surprised, neutral | yes |
| speed / pitch / volume | slow/medium/fast, low/medium/high ×2 | yes |
| dialect | Cantonese, Sichuanese | yes |
| instruction | free-form direction ("whisper, slowly, like a secret") | optional |
| voice design | a description of the speaker to invent | no |
| built-in voice | 484 voices (5 English, 479 Chinese incl. emotion variants) | no |

**language** (auto, Chinese, English, Japanese, Korean, German, French, Spanish) is a separate
input on Ming TTS and Ming Script. It's needed for the last five. Its `language_text` socket
overrides it with a name or code, so Ming Transcribe's `language` output can be wired straight in.

Inline sound tags work in any task: `[Breathing] [Cough] [Sigh] [Laughter] [Uhm] [Surprise-ah]
[Surprise-oh] [Dissatisfaction-hnn] [Confirmation-en] [Question-ei] [Question-ah] [Question-en]
[Question-oh] [Question-huh]`.

Long text is split into ~50-unit fragments generated with the same voice (each call has a ~20-40 s
budget). Connect the Ming VAE to the TTS node's optional `vae` input to trim each fragment's
trailing silence before joining, as the reference does.

**Ming Reference Voice**: VAE + AUDIO + transcript → voice. 3-15 s of clean speech from one speaker.
With the exact transcript the model continues the clip acoustically (best similarity, `speak`
task). Without one it uses only the speaker embedding.

**Ming Script**: up to four voices and a script:

```
1: Did you hear that? Something is moving in the attic.
2 (fearful): Please tell me it's just the wind.
1 (whisper it, slowly): Stay here. I'll go and look.
[pause 0.8]
2: [Laughter] It was the cat. It was only the cat!
```

The parenthesis holds an emotion, `speed=fast pitch=low volume=high`, `voice=US_James`,
`design=<description>`, or any other text as a free-form direction. Each line is seeded from
its own text and cached, so changing one line re-renders only that line (the others come back
instantly), and inserting a line doesn't change the takes around it. Outputs AUDIO and a
timeline of line timings.

**Tips**

- `speak`, `emotion`, `speed/pitch/volume` and `dialect` were trained with a reference voice.
  Without one the model may ramble; a length cap stops it at ~0.6 s per word.
- `cfg` 2 is the reference default for both models. They apply it differently (the 0.5B
  effectively guides at 3), so the same value is not the same strength across models.
- Numbers, dates and units are read as written. The reference's optional FST text
  normalization needs `pynini` and isn't ported, so write them out if they come out wrong.

## Audio understanding (Ming-3.0-audio-tiny)

Load `ming-3.0-audio-tiny_bf16.safetensors` with **Load CLIP** (the type doesn't matter), then:

- **Ming Transcribe** (CLIP + AUDIO → transcript, language): the release's validated ASR prompt and
  decoding. Wire the transcript into Ming Reference Voice to clone a clip without typing it.
- **TextGenerate** (stock): connect the clip and the audio, and ask anything ("what is the
  speaker's mood?", "summarise this meeting"). Thinking mode works too.

audio-tiny's spoken replies come from a "talker" that is exactly Ming-3.0-tts-0.5B (all 537
tensors identical), so pipe TextGenerate's answer into Ming TTS with the 0.5B to get a voice.

## Making the weights

```bash
hf download inclusionAI/Ming-3.0-tts-0.5B --local-dir Ming-3.0-tts-0.5B
python scripts/convert.py Ming-3.0-tts-0.5B --name ming-3.0-tts-0.5b --out <models dir>     # + the VAE (needs `onnx`)
python scripts/convert.py Ming-3.0-tts-tiny --name ming-3.0-tts-tiny --out <models dir> --no-vae
python scripts/convert.py Ming-3.0-audio-tiny --name ming-3.0-audio-tiny --out <text_encoders dir> --no-vae

# int8 + ConvRot (uses comfy-model-tools/quant_int8_convrot.py's recipe; run with ComfyUI's python for comfy-kitchen)
python scripts/quantize_int8.py ming-3.0-tts-tiny_bf16.safetensors ming-3.0-tts-tiny_int8_convrot.safetensors --tool <comfy-model-tools>
```

The converter keeps the released key names. For the tiny it stacks each layer's 128 experts
into ComfyUI expert banks. It streams tensors to disk, so the 18 GB tiny converts with little
RAM. `onnx` is only needed to read the speaker encoder out of `campplus.onnx` once.

## How it works

Both models are autoregressive: an LLM reads the prompt (task instruction, text, speaker
embedding, optional reference audio), and at each step a small flow-matching transformer turns
the last hidden state into the next 64-d audio latent (80 ms) in 10 Euler steps with CFG. An
aggregator folds that latent back into the LLM's input and a stop head ends the utterance. The
44.1 kHz AudioVAE (Qwen2-style encoder/decoder with a sliding window, ISTFT head) turns the
latents into audio.

- **0.5B**: Qwen2.5-0.5B on ComfyUI's own Llama implementation (attention backends, KV cache).
- **tiny**: Ling V3, 24 layers in groups of three Kimi Delta Attention layers (linear attention
  with a per-channel decayed delta-rule state) and one Multi-head Latent Attention layer. 128-expert
  MoE runs on ComfyUI's expert banks. The reference needs `flash-linear-attention`'s Triton kernels.
  Here KDA is plain torch: a chunked form for the prompt and the recurrent form per step.
- The speaker encoder is CAM++ (3D-Speaker), reimplemented in torch from the shipped
  `campplus.onnx`.

## Verification

| Check | Result |
|---|---|
| 0.5B, fp32 vs the reference runtime, same noise, each step fed the reference's previous latent | LLM state ≤ 2.7e-6, flow head bit-identical, stop head identical |
| AudioVAE decode / encode vs reference | 6e-7 (fp32), cos 0.99997 |
| CAM++ torch port vs `campplus.onnx` (onnxruntime) | 2e-6 |
| tiny: prompt token ids vs reference, 4 tasks | identical |
| tiny: KDA vs `fla` `chunk_kda` Triton kernel (random inputs, 1-333 tokens) | 2e-3 (the kernel's TF32 matmuls) |
| tiny, fp32, layer by layer vs the reference, 4 tasks | ≤ 3e-7 per layer (MLA layers bit-identical), 7e-7 after all 24 + norm, flow head bit-identical |
| tiny, prefill + one-token decode steps (KDA state, conv cache, MLA KV cache) | ≤ 3.6e-7 |
| audio-tiny: log-mel vs openai-whisper / Whisper encoder vs the release | 3e-8 / bit-identical |
| audio-tiny ASR on the release's voice-preset clips (English, Chinese) | matches the official transcripts up to punctuation and one dropped/homophone word each |

Free-running decode is chaotic: a 1e-6 difference eventually changes the take, as it does between
two GPUs running the reference. So parity is checked per step, with each step fed the reference's
previous latent. One subtlety the tests caught: the 0.5B reference casts its whole model to bf16,
which rounds RoPE frequencies. The port reproduces that (it shifts hidden states by ~2e-3).

```bash
python tests/parity_05b.py --comfy <ComfyUI> --checkpoint <Ming-3.0-tts-0.5B>/model.safetensors --dumps <reference dumps>
python tests/parity_tiny.py --comfy <ComfyUI> --checkpoint <Ming-3.0-tts-tiny> --dumps <reference dumps>
python tests/e2e.py --comfy <ComfyUI> --model <tts .safetensors> --vae <vae .safetensors> --out out [--script]
```

## Credits & license

Ming-3.0-tts by [inclusionAI](https://github.com/inclusionAI/Ming-3.0-tts) (MIT). The text
frontend in `ming/frontend/` is vendored unchanged from it. CAM++ is from
[3D-Speaker](https://github.com/modelscope/3D-Speaker). This pack reimplements the model code on
ComfyUI's building blocks; Apache-2.0, see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
