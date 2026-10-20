"""Text + task + voice -> audio latents, following each reference runtime's request handling
(MingTTS.synthesize for the 0.5B, MingTinyTTS.generate for the tiny)."""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

import comfy.model_management

from . import prompts
from .speaker import SAMPLE_RATE as SPEAKER_RATE, fbank
from .vae import HOP, PATCH, SAMPLE_RATE

LATENT_SECONDS = HOP * PATCH / SAMPLE_RATE  # 0.08 s of audio per latent
ALIGN = HOP * PATCH


@dataclass
class Voice:
    """A reference voice: CAM++ speaker embedding, plus VAE latents of the clip and its
    transcript for acoustic cloning. The 0.5B pads the clip at the front and the tiny at the
    back before encoding, so both are kept."""
    speaker: torch.Tensor  # [1, 192]
    transcript: str = None
    latent_front: torch.Tensor = None  # [1, T, 64]
    latent_back: torch.Tensor = None


@dataclass
class Task:
    """A model-independent request; ``lower`` turns it into one model's prompt fields."""
    kind: str  # speak | emotion | basic | dialect | instruction | design | builtin
    controls: dict = field(default_factory=dict)
    language: str = None  # one of prompts.LANGUAGES' values, None = auto


@dataclass
class Lowered:
    instruction: str = None
    language: str = None  # the 'Target language' line, if any
    # voice: the reference's CAM++ embedding (the 0.5B drops the speaker slot when there is none)
    # free: the voice if given, else a zeroed slot (0.5B freeform) | mute: always a zeroed slot
    speaker: str = "voice"
    acoustic_prompt: bool = False  # continue the clip's latents (needs a transcript)


def lower(task, backbone, text):
    low = _lower(task, backbone, text)
    # the 0.5B runtime puts the 'Target language' line in the prompt for every task but dialect;
    # the tiny's task protocol only for the speak tasks (seed/multilingual), where it is required
    if backbone == "ling_v3":
        if task.kind == "speak":
            low.language = prompts.tiny_target_language(text, task.language)
    elif task.kind != "dialect":
        low.language = task.language
    return low


def _lower(task, backbone, text):
    k, c = task.kind, task.controls
    tiny = backbone == "ling_v3"
    if k == "speak":
        return Lowered(acoustic_prompt=True)
    if k == "emotion":
        emotion = prompts.EMOTIONS[c["emotion"]]
        if tiny:
            return Lowered(prompts.tiny_control_instruction(emotion_order=True, emotion=emotion))
        return Lowered(prompts.structured_instruction(emotion=emotion))
    if k == "basic":
        v = dict(speed=prompts.SPEEDS[c["speed"]], pitch=prompts.LEVELS[c["pitch"]], volume=prompts.LEVELS[c["volume"]])
        return Lowered(prompts.tiny_control_instruction(**v) if tiny else prompts.structured_instruction(**v))
    if k == "dialect":
        if tiny:
            return Lowered(prompts.tiny_control_instruction(dialect=prompts.TINY_DIALECTS[c["dialect"]]))
        return Lowered(prompts.structured_instruction(dialect=prompts.DIALECTS[c["dialect"]], schema="en"))
    if k == "instruction":
        text = c["instruction"].strip()
        if not text:
            raise ValueError("Ming TTS: the instruction is empty")
        return Lowered(prompts.tiny_freeform_instruction(text) if tiny else text, speaker="free")
    if k == "design":
        text = c["description"].strip()
        if not text:
            raise ValueError("Ming TTS: the voice description is empty")
        return Lowered(prompts.tiny_control_instruction(style=text) if tiny else prompts.structured_instruction(style=text), speaker="mute")
    if k == "builtin":
        ip = c["voice"]
        return Lowered(prompts.tiny_control_instruction(ip=ip) if tiny else prompts.structured_instruction(ip=ip), speaker="mute")
    raise ValueError(f"Ming TTS: unknown task {k!r}")


@torch.inference_mode()
def make_voice(vae, audio, transcript="", max_seconds=20.0):
    """Encode a reference clip with the Ming VAE (latents) and its CAM++ (speaker embedding)."""
    import torchaudio.functional as AF

    if getattr(vae, "ming_speaker_encoder", None) is None:
        raise ValueError("Ming Reference Voice: this VAE has no speaker encoder. Load the converted "
                         "ming-3.0_audio_vae.safetensors (scripts/convert.py adds CAM++ from campplus.onnx).")
    wave, rate = audio["waveform"], audio["sample_rate"]
    mono = wave[0].float().mean(0).cpu()
    mono = mono[: int(max_seconds * rate)] if max_seconds > 0 else mono
    if mono.numel() == 0:
        raise ValueError("Ming Reference Voice: the reference audio is empty")

    speaker = vae.ming_speaker_encoder(fbank(AF.resample(mono, rate, SPEAKER_RATE)).unsqueeze(0)).detach()
    voice = Voice(speaker=speaker, transcript=(transcript or "").strip() or None)
    if voice.transcript:
        wave44 = AF.resample(mono, rate, SAMPLE_RATE) if rate != SAMPLE_RATE else mono
        pad = -wave44.shape[-1] % ALIGN
        encode = lambda w: vae.encode(w.reshape(1, -1, 1)).movedim(1, -1).float().cpu()  # [1, T, 64]
        # 0.5B: front padding (the clip's tail seeds the first decode step); tiny: back padding
        voice.latent_front = encode(F.pad(wave44, (pad, 0)))
        voice.latent_back = encode(F.pad(wave44, (0, pad)))
    return voice


def length_cap(text):
    """Upper bound on latents for a fragment, so a missed stop cannot babble on: the tiny's own
    podcast bounds, 8 latents (0.64 s) per space-separated word of ASCII text, 4 per other character."""
    import re

    text = re.sub(r"\[[^\]]*\]", " ", text).strip()  # inline [tags] are short sounds
    units = len(text.split()) * 8 if text.isascii() else len(text) * 4
    return max(40, units + 25)


def trailing_silence_cut(wave, threshold=1e-3, tail_seconds=0.3):
    """Latent count to keep so the fragment ends ``tail_seconds`` after its last 0.1 s frame
    above ``threshold`` (the reference trims each fragment's waveform the same way)."""
    frame = int(SAMPLE_RATE * 0.1)
    n = wave.shape[-1] // frame
    if n == 0:
        return None
    scores = wave[: n * frame].abs().reshape(n, frame).mean(-1)
    loud = (scores > threshold).nonzero()
    end = (loud[-1].item() + 1) * frame if len(loud) else 0
    keep_seconds = end / SAMPLE_RATE + tail_seconds
    return max(1, int(-(-keep_seconds // LATENT_SECONDS)))


@torch.inference_mode()
def synthesize(model_patcher, text, task, voice=None, seed=0, cfg=2.0, flow_steps=10, temperature=0.0, sigma=0.25,
               split=True, max_fragment_length=50, trim_vae=None, max_steps=None, continuity=True):
    """Returns latents [1, 64, T]. Long text is split into fragments generated one after another
    with the same voice; with ``continuity`` (speak task) each continues the previous fragment's
    audio. Each fragment is trimmed of trailing silence before joining when ``trim_vae`` is given."""
    comfy.model_management.load_models_gpu([model_patcher])
    model = model_patcher.model.diffusion_model
    device = model_patcher.load_device
    dtype = model_patcher.model.get_dtype_inference()
    tiny = model.backbone == "ling_v3"
    tok = prompts.load_tokenizer("ming_tts_tiny" if tiny else "ming_tts_0.5b")

    if not text.strip():
        raise ValueError("Ming TTS: the text is empty")
    whole = task.kind in ("design", "instruction")  # the reference conditions these on the utterance as a whole
    fragments = prompts.prepare_text(text, task.language, split=split and not whole, max_length=max_fragment_length, raw=tiny)

    chunks, previous = [], None  # previous: (latents [1, T, 64], text) of the last fragment
    for index, fragment in enumerate(fragments):
        low = lower(task, model.backbone, fragment)
        use_voice = voice is not None and low.speaker != "mute"
        reference = None
        if use_voice and low.acoustic_prompt and voice.transcript:
            reference = (voice.latent_back if tiny else voice.latent_front, voice.transcript)
        # Acoustic prompts to try, best first. With ``continuity`` the previous fragment's audio and
        # text are appended to the reference clip, so the model continues its delivery instead of
        # starting cold; the 0.5B's 512-position context may force dropping the clip, then the carry-over.
        candidates = [reference]
        if continuity and previous is not None and low.acoustic_prompt:
            candidates = ([join_prompts(reference, previous)] if reference else []) + [previous] + candidates
        speaker = voice.speaker if use_voice else None  # None zeroes the projected row (the reference's zero/muted speaker)
        for prompt in candidates:
            latent, transcript = prompt if prompt else (None, None)
            steps = 0 if latent is None else latent.shape[1]
            if tiny:
                ids, spk_pos, lat_pos = prompts.build_input_ids_tiny(tok, fragment, low.instruction, transcript, steps, low.language)
            else:
                # the 0.5B runtime only adds a speaker slot for a reference voice or use_zero_spk_emb
                slots = 1 if use_voice or low.speaker != "voice" else 0
                ids, spk_pos, lat_pos = prompts.build_input_ids(tok, prompts.tts_prompt(low.language), fragment, num_speakers=slots,
                                                                instruction=low.instruction, prompt_text=transcript, prompt_latent_steps=steps)
            if not model.settings.max_context or len(ids) + length_cap(fragment) <= model.settings.max_context:
                break
        embeds = model.embed_prompt(ids, spk_pos, [speaker] * len(spk_pos), lat_pos, latent, device=device, dtype=dtype)
        latents = model.generate(embeds, prompt_latent=latent, cfg=cfg, flow_steps=flow_steps, sigma=sigma, temperature=temperature,
                                 seed=seed + index, max_steps=max_steps or length_cap(fragment)).float()
        if trim_vae is not None:
            wave = trim_vae.decode(latents.transpose(1, 2))[0].float().mean(-1)
            keep = trailing_silence_cut(wave)
            if keep is not None:
                latents = latents[:, :keep]
        chunks.append(latents.cpu())
        previous = (latents.cpu(), fragment)
    return torch.cat(chunks, dim=1).transpose(1, 2)


def join_prompts(first, second):
    """Concatenate two (latents, transcript) acoustic prompts."""
    (a, ta), (b, tb) = first, second
    space = "" if ta[-1:] and prompts._HAN.match(ta[-1]) or not ta.strip() else " "
    return torch.cat([a, b.to(a.dtype)], dim=1), ta + space + tb


# ---- scripts: one utterance per line, cached per line --------------------------------------

@dataclass
class ScriptLine:
    speaker: int = 1
    text: str = ""
    task: Task = None
    pause: float = 0.0  # a [pause N] line


def parse_script(script, language=None):
    """Lines of ``N: text`` or ``N (mods): text`` (N = voice slot, default 1), ``[pause 1.5]`` and
    ``# comments``. mods: an emotion (happy, sad, angry, fearful, disgusted, surprised, neutral),
    ``speed=fast pitch=low volume=high`` (unset ones are medium), ``voice=US_James`` (built-in),
    ``design=<description>``, or any other text as a free-form delivery instruction."""
    import re

    lines = []
    for raw in script.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.fullmatch(r"\[pause\s+([\d.]+)\s*s?\]", line, flags=re.I)
        if m:
            lines.append(ScriptLine(pause=float(m[1])))
            continue
        m = re.match(r"^(?:speaker_?)?(\d+)\s*(?:\(([^)]*)\))?\s*[:：]\s*(.*)$", line, flags=re.I)
        speaker, mods, text = (int(m[1]), (m[2] or "").strip(), m[3].strip()) if m else (1, "", line)
        if not text:
            continue
        task = script_task(mods)
        task.language = language
        lines.append(ScriptLine(speaker, text, task))
    return lines


def script_task(mods):
    if not mods:
        return Task("speak")
    low = mods.lower()
    if low in prompts.EMOTIONS:
        return Task("emotion", {"emotion": low})
    if low.startswith("voice="):
        return Task("builtin", {"voice": mods.split("=", 1)[1].strip()})
    if low.startswith("design="):
        return Task("design", {"description": mods.split("=", 1)[1].strip()})
    if "=" in low and all(k in ("speed", "pitch", "volume") for k in (p.split("=")[0].strip() for p in low.split())):
        controls = {"speed": "medium", "pitch": "medium", "volume": "medium"}
        controls.update(dict(p.split("=") for p in low.split()))
        return Task("basic", controls)
    return Task("instruction", {"instruction": mods})


_LINE_CACHE = {}
_LINE_CACHE_SIZE = 256


def line_key(*parts):
    import hashlib

    h = hashlib.sha1()
    for p in parts:
        h.update(p.numpy().tobytes() if torch.is_tensor(p) else repr(p).encode())
    return h.hexdigest()


@torch.inference_mode()
def render_script(model_patcher, vae, script, voices, seed=0, gap=0.25, language=None, **settings):
    """Renders each line separately (seeded from its own content, so editing or inserting a line
    leaves the others' audio unchanged) and joins them with ``gap`` seconds of silence.
    Returns (waveform [1, 1, samples], timeline text)."""
    model_id = id(model_patcher.model)
    pieces, timeline, t = [], [], 0.0
    for index, line in enumerate(parse_script(script, language)):
        if line.pause:
            pieces.append(torch.zeros(1, 1, int(line.pause * SAMPLE_RATE)))
            t += line.pause
            continue
        voice = voices.get(line.speaker)
        line_seed = (seed + int(line_key(line.text, line.task.kind, line.task.controls)[:8], 16)) % 2 ** 63
        key = line_key(model_id, line.speaker, line.text, line.task.kind, line.task.controls, line.task.language, line_seed, sorted(settings.items()),
                       None if voice is None else voice.speaker, None if voice is None else voice.transcript)
        audio = _LINE_CACHE.pop(key, None)
        if audio is None:
            latents = synthesize(model_patcher, line.text, line.task, voice=voice, seed=line_seed, trim_vae=vae, **settings)
            audio = vae.decode(latents)[0].float().mean(-1).cpu().reshape(1, 1, -1)
        _LINE_CACHE[key] = audio  # most recently used last
        while len(_LINE_CACHE) > _LINE_CACHE_SIZE:
            _LINE_CACHE.pop(next(iter(_LINE_CACHE)))
        if pieces and gap > 0:
            pieces.append(torch.zeros(1, 1, int(gap * SAMPLE_RATE)))
            t += gap
        seconds = audio.shape[-1] / SAMPLE_RATE
        timeline.append(f"{t:7.2f}-{t + seconds:7.2f}  [{line.speaker}] {line.text}")
        pieces.append(audio)
        t += seconds
    if not pieces:
        raise ValueError("Ming Script: the script has no lines to speak")
    return torch.cat(pieces, dim=-1), "\n".join(timeline)
