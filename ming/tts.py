"""Ming-3.0 TTS: an LLM that emits one 64-d audio latent (80 ms) per step through a flow head.

Two backbones share this decode loop:
  qwen2   Ming-3.0-tts-0.5B (Qwen2.5-0.5B), reimplementing modeling_bailingmm (also the talker of
          Ming-3.0-audio-tiny)
  ling_v3 Ming-3.0-tts-tiny (Ling V3 MoE, see ling.py), reimplementing ming_tiny/modeling_lingv3_tts
Parameter names follow the released checkpoints, so the 0.5B model.safetensors loads unchanged
(the ``audio.*`` VAE keys are left for Load VAE); the tiny's MoE experts are stacked at load.

One decode step: the last LLM hidden state conditions the flow head, which integrates 10
Euler steps from noise to the next latent (CFG against a zeroed hidden state); the latent is
folded back into one LLM input embedding by the aggregator; a 2-way stop head ends the
utterance.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

import comfy.model_management
import comfy.utils

from .flow import DiT, Aggregator, sample_patch
from .rope import Qwen2LM

LATENT_DIM = 64
SPEAKER_DIM = 192


@dataclass
class Qwen25_05BConfig:
    vocab_size: int = 151936
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    transformer_type: str = "llama"
    head_dim = 64
    rms_norm_add = False
    mlp_activation = "silu"
    qkv_bias = True
    rope_dims = None
    q_norm = None
    k_norm = None
    rope_scale = None
    final_norm: bool = True
    lm_head: bool = False


@dataclass
class DecodeSettings:
    """Where the two reference runtimes differ in their decode loop."""
    cfg_mode: str  # "ming05": cond + cfg*(cond - null) | "standard": null + cfg*(cond - null)
    min_latents: int  # the stop head is honoured from this many generated latents on
    max_context: int = 0  # prompt + steps limit (0.5B: 512-position static cache), 0 = none
    max_steps: int = 0  # decode-step limit (tiny: maximum_decode_steps), 0 = none
    state_dtype: torch.dtype = None  # flow state / history dtype (tiny: fp32, 0.5B: model dtype)
    noise_after_last: bool = True  # 0.5B adds SDE noise after the last Euler step too
    separate_noise_streams: bool = False  # tiny: initial noise from its own seeded CPU stream


SETTINGS = {
    "qwen2": DecodeSettings("ming05", min_latents=5, max_context=512),
    "ling_v3": DecodeSettings("standard", min_latents=13, max_steps=250, state_dtype=torch.float32,
                              noise_after_last=False, separate_noise_streams=True),
}


class MingTTS(nn.Module):
    def __init__(self, backbone="qwen2", flow_depth=8, aggregator_depth=8, flow_hidden=1024, history_patch_size=1, patch_size=1,
                 rope_theta=None, image_model=None, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.dtype = dtype
        self.backbone = backbone
        self.settings = SETTINGS[backbone]
        self.patch_size = patch_size
        self.history_patch_size = history_patch_size

        self.model = nn.Module()  # keeps the checkpoint's model.model.* names
        if backbone == "ling_v3":
            from .ling import LingV3, LingV3Config

            config = LingV3Config() if rope_theta is None else LingV3Config(rope_theta=rope_theta)
            self.model.model = LingV3(config, device=device, dtype=dtype, ops=operations)
        else:
            config = Qwen25_05BConfig()
            self.model.model = Qwen2LM(config, device=device, dtype=dtype, ops=operations)
        hidden = config.hidden_size
        self.flowloss = nn.Module()
        self.flowloss.cfm = nn.Module()
        self.flowloss.cfm.model = DiT(LATENT_DIM, flow_hidden, flow_depth, 16, 4.0, llm_cond_dim=hidden, device=device, dtype=dtype, operations=operations)
        self.linear_proj_audio = Aggregator(LATENT_DIM, flow_hidden, aggregator_depth, 16, 4.0, llm_input_dim=hidden, device=device, dtype=dtype, operations=operations)
        self.stop_head = operations.Linear(hidden, 2, device=device, dtype=dtype)
        self.spk_head = operations.Linear(SPEAKER_DIM, hidden, device=device, dtype=dtype)

    @property
    def llm(self):
        return self.model.model

    @property
    def dit(self):
        return self.flowloss.cfm.model

    def llm_state(self, capacity, device, dtype):
        if self.backbone == "ling_v3":
            return self.llm.init_state(capacity)
        return {"kv": self.llm.init_kv_cache(1, capacity, device, dtype)}

    def llm_forward(self, embeds, state):
        """Final-norm hidden states for ``embeds``, advancing the cache/recurrent ``state``."""
        if self.backbone == "ling_v3":
            return self.llm(embeds, state)
        hidden, _, state["kv"] = self.llm(None, embeds=embeds, past_key_values=state["kv"])
        return hidden

    def embed_prompt(self, input_ids, speaker_positions=(), speaker_embeds=(), latent_position=-1, prompt_latent=None, device=None, dtype=None):
        """Token embeddings with speaker vectors and prompt-audio latents spliced into their reserved rows.
        A speaker embedding of None zeroes its row (the reference's muted/zero speaker)."""
        ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        embeds = self.llm.embed_tokens(ids, out_dtype=dtype)
        for position, spk in zip(speaker_positions, speaker_embeds):
            embeds[0, position] = 0 if spk is None else self.spk_head(spk.to(device, embeds.dtype).reshape(1, -1))[0]
        if prompt_latent is not None and latent_position >= 0:
            patches = prompt_latent.to(device, embeds.dtype).reshape(-1, self.patch_size, LATENT_DIM)
            projected = self.linear_proj_audio(patches).reshape(1, -1, embeds.shape[-1])
            embeds[0, latent_position:latent_position + projected.shape[1]] = projected[0]
        return embeds

    def generate(self, embeds, prompt_latent=None, max_steps=None, cfg=2.0, flow_steps=10, sigma=0.25, temperature=0.0,
                 seed=0, noise_fn=None, stop_threshold=0.5, progress=True):
        """Autoregressive decode. Returns latents [1, T, 64] (the step the stop head fires on is included).

        Noise is drawn on the CPU from ``seed``, so a seed gives the same audio on any GPU (for the
        tiny, the initial noise matches the reference runtime's stream exactly).
        ``noise_fn(step) -> (noise, sde_noise)`` overrides it (parity tests).
        """
        s = self.settings
        device, dtype = embeds.device, embeds.dtype
        state_dtype = s.state_dtype or dtype
        budget = max_steps or 10 ** 9
        if s.max_steps:
            budget = min(budget, s.max_steps)
        if s.max_context:
            budget = min(budget, (s.max_context - embeds.shape[1]) // self.patch_size)
            if budget <= 0:
                raise ValueError(f"Ming TTS: the prompt is {embeds.shape[1]} tokens, which fills the {s.max_context}-position context; "
                                 "shorten the text or the reference clip")

        history = torch.zeros(1, self.history_patch_size, LATENT_DIM, device=device, dtype=state_dtype)
        if prompt_latent is not None:
            tail = prompt_latent.to(device, state_dtype)[:, -self.history_patch_size:]
            history[:, -tail.shape[1]:] = tail

        noise_gen = torch.Generator().manual_seed(seed)
        sde_gen = torch.Generator().manual_seed(seed + 1) if s.separate_noise_streams else noise_gen
        state = self.llm_state(embeds.shape[1] + budget * self.patch_size, device, dtype)
        pbar = comfy.utils.ProgressBar(budget) if progress else None
        latents = []
        for step in range(budget):
            cond = self.llm_forward(embeds, state)[:, -1:]
            if noise_fn is not None:
                noise, sde_noise = noise_fn(step)
            else:
                # (1, 64, patch) then transposed, as the tiny draws it; identical values for patch 1
                noise = torch.randn((1, LATENT_DIM, self.patch_size), generator=noise_gen).transpose(1, 2)
                sde_noise = torch.randn((flow_steps, 1, self.patch_size, LATENT_DIM), generator=sde_gen)
            latent = sample_patch(self.dit, cond, history, noise.to(device, state_dtype), sde_noise.to(device, state_dtype), cfg=cfg, sigma=sigma,
                                  temperature=temperature, patch_size=self.patch_size, cfg_mode=s.cfg_mode, noise_after_last=s.noise_after_last)
            latents.append(latent)
            if pbar is not None:
                pbar.update(1)
            if len(latents) >= s.min_latents and self.stop_head(cond)[0, 0].float().softmax(dim=-1)[1] > stop_threshold:
                break
            comfy.model_management.throw_exception_if_processing_interrupted()
            embeds = self.linear_proj_audio(latent.to(dtype))
            history = torch.cat([history[:, self.patch_size:], latent.to(state_dtype)], dim=1)
        return torch.cat(latents, dim=1)
