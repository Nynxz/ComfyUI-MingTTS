"""Ming-3.0-tts prompt protocol: token layout, task captions and text preparation.

Ports prompt_contract.py / tts_tasks.py of inclusionAI/Ming-3.0-tts. The released checkpoints
were trained on these exact strings (including the Chinese field names and JSON key order),
so they are reproduced verbatim rather than paraphrased.
"""

import json
import os
import re
from collections import OrderedDict
from functools import lru_cache

from .frontend.text_segment_cut import cut_text_by_semantic_length
from .frontend.toolkit import PARALINGUISTIC_TAGS, add_period, preprosess_text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_TTS_PROMPT = "Please generate speech based on the following description.\n"
TEXT_INPUT_PREFIX = " Text input:\n"
AUDIO_START = "<audio>"
AUDIO_PATCH = "<audioPatch>"

# Dense (Qwen2.5-0.5B) layout. The IDs are pinned: a stock Qwen2 tokenizer with the same
# extra tokens added in another order spells every name right and indexes the wrong rows.
DENSE = {
    "system": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
    "user": "<|im_start|>user\n",
    "user_end": "<|im_end|>\n",
    "assistant": "<|im_start|>assistant\n",
    "speaker_start": "<|vision_start|>",
    "speaker_pad": "<|vision_pad|>",
    "speaker_end": "<|vision_end|>\n",
    "instruction_eos": "<|im_end|>",
    "token_ids": {
        "<|endoftext|>": 151643, "<|im_start|>": 151644, "<|im_end|>": 151645, "<|vision_start|>": 151652,
        "<|vision_end|>": 151653, "<|vision_pad|>": 151654, "<audioPatch>": 151705, "<audio>": 151706,
    },
}

LANGUAGES = {"auto": None, "Chinese": "Chinese", "English": "English", "Japanese": "Japanese", "Korean": "Korean",
             "German": "German", "French": "French", "Spanish": "Spanish"}
_LANGUAGE_ALIASES = {"zh": "Chinese", "zh-cn": "Chinese", "cantonese": "Chinese", "yue": "Chinese", "mandarin": "Chinese", "en": "English",
                     "ja": "Japanese", "ko": "Korean", "de": "German", "fr": "French", "es": "Spanish"}


def resolve_language(value):
    """A language name or code (e.g. Ming Transcribe's 'English', or 'en') -> one of LANGUAGES, None for auto/empty."""
    key = (value or "").strip().lower()
    if key in ("", "auto"):
        return None
    for name in LANGUAGES:
        if name.lower() == key:
            return LANGUAGES[name]
    if key in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[key]
    raise ValueError(f"Ming TTS: unsupported language {value!r}; use one of {', '.join(n for n in LANGUAGES if n != 'auto')} (or auto)")


# Languages whose text bypasses the Chinese/English normalizer (it reads kanji as Chinese).
PASSTHROUGH_LANGUAGES = {"Japanese", "Korean", "German", "French", "Spanish"}

EMOTIONS = {"happy": "高兴", "sad": "悲伤", "angry": "愤怒", "fearful": "恐惧", "disgusted": "厌恶", "surprised": "惊讶", "neutral": "中性"}
SPEEDS = {"default": None, "slow": "慢速", "medium": "中速", "fast": "快速"}
LEVELS = {"default": None, "low": "低", "medium": "中", "high": "高"}
DIALECTS = {"Cantonese": "cantonese", "Sichuanese (chuanyu)": "chuanyu dialect"}

_ZH_FIELDS = ("序号", "说话人", "方言", "风格", "语速", "基频", "音量", "情感")
_EN_FIELDS = ("id", "speaker", "dialect", "style", "speed", "pitch", "volume", "emotion")
_BGM_FIELDS = ("Genre", "Mood", "Instrument", "Theme", "ENV", "SNR")
_BGM_MARKERS = ("Genre: ", "Mood: ", "Instrument: ", "Theme: ", "Duration: ")


@lru_cache(maxsize=4)
def load_tokenizer(name="ming_tts_0.5b"):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(ROOT, "tokenizers", name, "tokenizer.json"))
    pinned = TINY["token_ids"] if name == "ming_tts_tiny" else DENSE["token_ids"]
    wrong = {t: (i, tok.token_to_id(t)) for t, i in pinned.items() if tok.token_to_id(t) != i}
    if wrong:
        raise ValueError(f"Ming tokenizer does not match the trained token ids: {wrong}")
    return tok


@lru_cache(maxsize=1)
def builtin_voices():
    with open(os.path.join(ROOT, "ming", "voices.json"), encoding="utf-8") as f:
        return json.load(f)


def tts_prompt(language=None):
    return DEFAULT_TTS_PROMPT if not language else DEFAULT_TTS_PROMPT + f"Target language: {language}.\n"


def structured_instruction(*, dialect=None, style=None, speed=None, pitch=None, volume=None, emotion=None, ip=None, schema="zh"):
    """The ordered assistant-side JSON caption used by the structured tasks."""
    fields, ip_key = (_EN_FIELDS, "ip") if schema == "en" else (_ZH_FIELDS, "IP")
    item = OrderedDict(zip(fields, (1, "speaker_1", dialect, style, speed, pitch, volume, emotion)))
    item["BGM"] = OrderedDict((key, None) for key in _BGM_FIELDS)
    item[ip_key] = ip
    return json.dumps(OrderedDict((("audio_sequence", [item]),)), ensure_ascii=False)


def build_input_ids(tokenizer, prompt, text, num_speakers=0, instruction=None, prompt_text=None, prompt_latent_steps=0):
    """Dense token layout. Returns (ids, speaker pad positions, first prompt-latent row or -1)."""
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False).ids

    speakers = []
    for index in range(num_speakers):
        speakers += encode(f"  speaker_{index + 1}:") + encode(DENSE["speaker_start"]) + encode(DENSE["speaker_pad"]) + encode(DENSE["speaker_end"])
    instruction_block = encode(instruction) + encode(DENSE["instruction_eos"]) if instruction else []
    reference = encode(prompt_text) if prompt_latent_steps > 0 and prompt_text else []
    text_prefix = [] if all(m in text for m in _BGM_MARKERS) else encode(TEXT_INPUT_PREFIX)

    ids = (encode(DENSE["system"]) + encode(DENSE["user"]) + encode(prompt) + speakers + text_prefix + reference + encode(text)
           + encode(DENSE["user_end"]) + encode(DENSE["assistant"]) + instruction_block + encode(AUDIO_START))

    start = DENSE["token_ids"][DENSE["speaker_start"]]
    speaker_positions = [i + 1 for i, t in enumerate(ids) if t == start]
    latent_position = -1
    if reference:
        latent_position = len(ids)
        ids += encode(AUDIO_PATCH) * prompt_latent_steps
    return ids, speaker_positions, latent_position


# ---- Ming-3.0-tts-tiny (Ling V3) protocol: ming_tiny/prompt_contract.py + task_protocol.py ----

TINY = {
    "user": "<role>HUMAN</role>Please generate speech based on the following description.\n",
    "text_prefix": "\n Text input:\n",
    "assistant": "<role>ASSISTANT</role>",
    "token_ids": {"<audioPatch>": 157164, "<audio>": 157165, "</audio>": 157166, "<spk>": 157175, "</spk>": 157176, "<|role_end|>": 156895},
}
TINY_DIALECTS = {"Cantonese": "广粤话", "Sichuanese (chuanyu)": "川渝话"}
_TINY_FIELDS = ("序号", "说话人", "方言", "情感", "风格", "语速", "基频", "音量", "BGM", "IP")
_TINY_EMOTION_FIELDS = ("序号", "说话人", "方言", "风格", "语速", "基频", "音量", "情感", "BGM", "IP")
_HAN = re.compile(r"[㐀-䶿一-鿿]")
_INLINE_CONTROL = re.compile(r"\[[^\]\n]{1,128}\]|\([A-Za-z][A-Za-z _-]{0,127}\)")
_SPEAKER_LABEL = re.compile(r"(?im)^\s*speaker[_ ]?\d+\s*[:：]\s*")


def tiny_control_instruction(emotion_order=False, **controls):
    item = {"序号": 1, "说话人": "speaker_1", "方言": None, "情感": None, "风格": None, "语速": None, "基频": None, "音量": None,
            "BGM": {k: None for k in _BGM_FIELDS}, "IP": None}
    keys = {"dialect": "方言", "emotion": "情感", "style": "风格", "speed": "语速", "pitch": "基频", "volume": "音量", "ip": "IP"}
    for name, value in controls.items():
        item[keys[name]] = value
    order = _TINY_EMOTION_FIELDS if emotion_order else _TINY_FIELDS
    return json.dumps({"audio_sequence": [{k: item[k] for k in order}]}, ensure_ascii=False)


def tiny_freeform_instruction(text):
    text = text.strip()
    return text if text.startswith("Speaking instruction:") else f"Speaking instruction: Speak in this style: {text}"


def tiny_target_language(text, language=None):
    """Explicit language, else inferred like the tiny's SampleBuilder: any Han character -> Chinese."""
    if language:
        return language
    lexical = _INLINE_CONTROL.sub("", _SPEAKER_LABEL.sub("", text))
    return "Chinese" if _HAN.search(lexical) else "English"


def build_input_ids_tiny(tokenizer, text, instruction=None, prompt_text=None, prompt_latent_steps=0, target_language=None):
    """The tiny's base-format prompt with one speaker slot and the instruction on the assistant side.
    ``target_language`` adds the 'Target language' line (the speak tasks only).
    Returns (ids, [speaker position], first prompt-latent row or -1)."""
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False).ids
    t = TINY["token_ids"]
    user = TINY["user"] + (f"Target language: {target_language}.\n" if target_language else "") + "  speaker_1:"
    ids = encode(user) + [t["<spk>"]]
    speaker = len(ids)
    ids += [t["<audioPatch>"], t["</spk>"]] + encode(TINY["text_prefix"])
    if prompt_text and prompt_latent_steps > 0:
        ids += encode(prompt_text)
    ids += encode(text) + encode(TINY["assistant"])
    if instruction:
        ids += encode(instruction) + [t["<|role_end|>"]]
    ids.append(t["<audio>"])
    latent_position = -1
    if prompt_text and prompt_latent_steps > 0:
        latent_position = len(ids)
        ids += [t["<audioPatch>"]] * prompt_latent_steps
    return ids, [speaker], latent_position


def prepare_text(text, language=None, split=True, max_length=50, raw=False):
    """Split text into synthesis fragments the way the 0.5B reference frontend does with
    text_normalization=False. (Its optional FST normalization of numbers/dates needs pynini and
    is not ported, so write numbers out if they read wrongly.) ``raw`` keeps the text as given
    apart from splitting, as the tiny's runtime does."""
    if raw:
        fragments = cut_text_by_semantic_length(text.strip(), max_length)["fragments"] if split else [text.strip()]
        return [f for f in fragments if f.strip()]
    if language in PASSTHROUGH_LANGUAGES:
        text = preprosess_text(text).strip()
    else:
        text = add_period(preprosess_text(text))
    fragments = cut_text_by_semantic_length(text, max_length)["fragments"] if split else [text]
    return [f for f in fragments if f.strip()]


__all__ = ["PARALINGUISTIC_TAGS", "build_input_ids", "builtin_voices", "load_tokenizer", "prepare_text", "structured_instruction", "tts_prompt"]
