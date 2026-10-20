"""Half-split (Llama/Qwen2) rotary tables in ComfyUI's (cos, sin, -sin) format, with the
reference's bf16-rounded inverse frequencies (see flow.rope_inv_freq)."""

import torch

from comfy.text_encoders.llama import Llama2_

from .flow import rope_inv_freq


def qwen2_freqs_cis(head_dim, position_ids, theta, device=None):
    freqs = position_ids.float()[..., None] * rope_inv_freq(head_dim, theta, device)  # [B, seq, dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)
    half = sin.shape[-1] // 2
    return cos, sin[..., :half], -sin[..., half:]


class Qwen2LM(Llama2_):
    """ComfyUI's Llama2_ (attention backends, KV cache, quantized ops) with reference RoPE."""

    def compute_freqs_cis(self, position_ids, device):
        return qwen2_freqs_cis(self.config.head_dim, position_ids, self.config.rope_theta, device)
