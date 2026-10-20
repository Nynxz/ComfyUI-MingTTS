"""Ming-3.0-audio-tiny as a ComfyUI text encoder, so the stock Load CLIP + TextGenerate nodes
(which take an optional AUDIO input) run speech recognition and audio question answering.

Follows the reference processor and chat template: '<role>SYSTEM</role>...<|role_end|>
<role>HUMAN</role><|audio_start|>{audio}<|audio_end|>{prompt}<|role_end|><role>ASSISTANT</role>'
then '\\n<think></think>' (or '\\n<think>' with thinking on), with the audio placeholder replaced by
the encoder's embeddings.
"""

import os

import torch

import comfy.model_management
import comfy.ops
import comfy.utils

from .thinker import SAMPLE_RATE, MingThinker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EOS = (156892, 156895)  # <|endoftext|>, <|role_end|>
ASR_PROMPT = ("Please recognize the language of this speech and transcribe it, placing a `\\t` "
              "between the identified language and the transcribe.")
CHAT_SYSTEM_PROMPT = (
    "You are a helpful assistant.\nListen to the user's audio and respond to their question or request.\n"
    "Answer the question or carry out the requested task. For lists, provide several concrete examples; for why/how "
    "questions, explain the main reason or method. Follow the user's explicit instructions, including any requirements "
    "for format, wording, and length. Answer directly and keep your response concise but complete. Do not repeat or "
    "transcribe the request unless asked to do so. Before responding, silently check that your answer satisfies the "
    "requested constraints."
)


def is_thinker(sd):
    return "model.audio.conv1.weight" in sd and "lm_head.weight" in sd and "model.layers.0.attention.A_log" in sd


class MingThinkerTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(os.path.join(ROOT, "tokenizers", "ming_audio_tiny", "tokenizer.json"))
        self.audio_start, self.audio_end = self.tok.token_to_id("<|audio_start|>"), self.tok.token_to_id("<|audio_end|>")

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def tokenize_with_weights(self, text, return_word_ids=False, audio=None, thinking=False, skip_template=False, system_prompt="", **kwargs):
        """Token list for TextGenerate: (id, 1.0) pairs, with the audio carried as one
        ({"type": "audio", ...}, 1.0) entry that the model expands into its embeddings."""
        items = []
        if audio is not None:
            wave = audio["waveform"][0].float().mean(0)
            if audio["sample_rate"] != SAMPLE_RATE:
                import torchaudio.functional as AF
                wave = AF.resample(wave, audio["sample_rate"], SAMPLE_RATE)
            items = [self.audio_start, {"type": "audio", "data": wave}, self.audio_end]
        if skip_template:
            ids = items + self.encode(text)
        else:
            system = system_prompt or ""
            head = "<role>SYSTEM</role>" + system
            if system and not system.endswith("\n"):
                head += "\n"
            if "detailed thinking" not in system:
                head += "detailed thinking on" if thinking else "detailed thinking off"
            ids = (self.encode(head + "<|role_end|><role>HUMAN</role>") + items + self.encode(text)
                   + self.encode("<|role_end|><role>ASSISTANT</role>" + ("\n<think>" if thinking else "\n<think></think>")))
        return {"ming_thinker": [[(t, 1.0) for t in ids]]}

    def decode(self, token_ids, skip_special_tokens=True):
        return self.tok.decode(list(token_ids), skip_special_tokens=skip_special_tokens)

    def state_dict(self):
        return {}


class MingThinkerTE(torch.nn.Module):
    def __init__(self, device="cpu", dtype=None, model_options={}, **kwargs):
        super().__init__()
        quant = model_options.get("quantization_metadata", None)
        ops = model_options.get("custom_operations", None) or (comfy.ops.mixed_precision_ops(quant, dtype, full_precision_mm=True) if quant else comfy.ops.manual_cast)
        self.thinker = MingThinker(device=device, dtype=dtype, operations=ops)
        self.dtypes = {dtype}
        # bf16 unless the text encoder runs in fp32 (fp16 overflows this LLM; fp8/int8 weights compute in bf16)
        self.compute_dtype = torch.float32 if dtype == torch.float32 else torch.bfloat16
        self.execution_device = None

    def set_clip_options(self, options):
        self.execution_device = options.get("execution_device", self.execution_device)

    def reset_clip_options(self):
        self.execution_device = None

    def load_sd(self, sd):
        from .ling import stack_experts

        sd = stack_experts({k: v for k, v in sd.items() if not k.startswith("model.layers.24.")})  # drop the MTP layer
        return self.thinker.load_state_dict(sd, strict=False, assign=getattr(self, "can_assign_sd", False))

    def memory_estimation_function(self, tokens, device=None):
        audio = sum(t[0]["data"].shape[-1] for t in next(iter(tokens.values()))[0] if isinstance(t[0], dict)) if tokens else 0
        return (audio / SAMPLE_RATE * 60_000_000) + 600_000_000  # whisper activations ~1 GB per 30 s chunk

    def embeds(self, items, device, dtype):
        out, run = [], []
        for t in items + [None]:
            if isinstance(t, int):
                run.append(t)
                continue
            if run:  # consecutive token ids in one lookup
                out.append(self.thinker.model.embed_tokens(torch.tensor([run], device=device), out_dtype=dtype)[0])
                run = []
            if isinstance(t, dict):
                out.append(self.thinker.audio_embeds(t["data"].to(device), dtype))
        return torch.cat(out, dim=0).unsqueeze(0)

    @torch.inference_mode()
    def generate(self, tokens, do_sample=False, max_length=256, temperature=1.0, top_k=50, top_p=0.95, min_p=0.0, repetition_penalty=1.0,
                 seed=None, presence_penalty=0.0, mtp=True, no_repeat_ngram_size=0):
        device = self.execution_device or comfy.model_management.get_torch_device()
        dtype = self.compute_dtype
        items = [t[0] for t in next(iter(tokens.values()))[0]]
        x = self.embeds(items, device, dtype)
        state = self.thinker.model.init_state(x.shape[1] + max_length + 1)
        gen = torch.Generator(device="cpu").manual_seed(seed if seed is not None else 0)
        out, pbar = [], comfy.utils.ProgressBar(max_length)
        for _ in range(max_length):
            h = self.thinker.model(x, state)[:, -1]
            logits = self.thinker.lm_head(h).float()[0]
            if repetition_penalty != 1.0 and out:
                seen = torch.tensor(sorted(set(out)), device=device)
                vals = logits[seen]
                logits[seen] = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
            if presence_penalty and out:
                logits[torch.tensor(sorted(set(out)), device=device)] -= presence_penalty
            if no_repeat_ngram_size and len(out) >= no_repeat_ngram_size:
                n = no_repeat_ngram_size
                prefix = tuple(out[-(n - 1):])
                banned = [out[i + n - 1] for i in range(len(out) - n + 1) if tuple(out[i:i + n - 1]) == prefix]
                if banned:
                    logits[torch.tensor(banned, device=device)] = float("-inf")
            token = sample(logits, do_sample, temperature, top_k, top_p, min_p, gen)
            if token in EOS:
                break
            out.append(token)
            pbar.update(1)
            comfy.model_management.throw_exception_if_processing_interrupted()
            x = self.thinker.model.embed_tokens(torch.tensor([[token]], device=device), out_dtype=dtype)
        return out


def sample(logits, do_sample, temperature, top_k, top_p, min_p, gen):
    if not do_sample or temperature <= 0:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.shape[-1]))[0][-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    if min_p and min_p > 0:
        probs = probs.masked_fill(probs < min_p * probs.max(), 0)
    if top_p and top_p < 1.0:
        sorted_p, idx = probs.sort(descending=True)
        drop = sorted_p.cumsum(-1) - sorted_p > top_p
        probs = probs.scatter(0, idx, sorted_p.masked_fill(drop, 0))
    probs = probs / probs.sum()
    return int(torch.multinomial(probs.cpu(), 1, generator=gen))


def clip_target():
    class Target:
        params = {}
        clip = MingThinkerTE
        tokenizer = MingThinkerTokenizer
    return Target()
