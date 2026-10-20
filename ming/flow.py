"""Ming-3.0 flow-matching head (DiT) and latent aggregator, on ComfyUI ops.

Reimplements fm/dit.py + fm/modules.py + fm/CFM.py of inclusionAI/Ming-3.0-tts. Parameter
names follow the original (``blocks.N.attn.to_q``, ``mlp.ff.0.0`` ...) so released weights load
unchanged. Both modules attend over a handful of tokens (cond + history + current patch), so
plain SDPA is used rather than ComfyUI's attention dispatch.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rope_inv_freq(dim, base, device=None):
    """RoPE inverse frequencies rounded to bf16, as the reference runs them: its runtime casts the
    whole model to bf16, inv_freq buffers included (the flow head even stores its inv_freq in the
    checkpoint). Unrounded fp32 frequencies shift the LLM hidden states by ~2e-3 relative."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    return inv_freq.to(torch.bfloat16).float()


def rotary_freqs(seq_len, dim, device, base=10000.0):
    """x_transformers RotaryEmbedding: interleaved (GPT-J) pairs over the full head dim, fp32."""
    freqs = torch.outer(torch.arange(seq_len, device=device, dtype=torch.float32), rope_inv_freq(dim, base, device))
    return torch.stack((freqs, freqs), dim=-1).flatten(-2)  # [seq, dim]


def apply_rotary(t, freqs):
    x = t.unflatten(-1, (-1, 2))
    rotated = torch.stack((-x[..., 1], x[..., 0]), dim=-1).flatten(-2)
    return (t * freqs.cos() + rotated * freqs.sin()).to(t.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))

    def forward(self, x):
        w = self.weight.to(x.device)
        if w.dtype in (torch.float16, torch.bfloat16):
            x = x.to(w.dtype)
        return F.rms_norm(x, (x.shape[-1],), weight=w, eps=self.eps)


class Attention(nn.Module):
    def __init__(self, dim, heads, device=None, dtype=None, operations=None):
        super().__init__()
        self.heads = heads
        self.to_q = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.to_k = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.to_v = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.to_out = nn.ModuleList([operations.Linear(dim, dim, device=device, dtype=dtype), nn.Identity()])

    def forward(self, x, freqs):
        b, n, _ = x.shape
        q, k, v = (proj(x).view(b, n, self.heads, -1).transpose(1, 2) for proj in (self.to_q, self.to_k, self.to_v))
        q, k = apply_rotary(q, freqs), apply_rotary(k, freqs)
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, -1)
        return self.to_out[0](out)


class FeedForward(nn.Module):
    def __init__(self, dim, mult, device=None, dtype=None, operations=None):
        super().__init__()
        inner = int(dim * mult)
        self.ff = nn.Sequential(
            nn.Sequential(operations.Linear(dim, inner, device=device, dtype=dtype), nn.GELU(approximate="tanh")),
            nn.Identity(),
            operations.Linear(inner, dim, device=device, dtype=dtype),
        )

    def forward(self, x):
        return self.ff(x)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, device=None, dtype=None, operations=None):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, device=device, dtype=dtype)
        self.attn = Attention(hidden_size, num_heads, device=device, dtype=dtype, operations=operations)
        self.norm2 = RMSNorm(hidden_size, device=device, dtype=dtype)
        self.mlp = FeedForward(hidden_size, mlp_ratio, device=device, dtype=dtype, operations=operations)

    def forward(self, x, freqs):
        x = x + self.attn(self.norm1(x), freqs)
        return x + self.mlp(self.norm2(x))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, device=None, dtype=None, operations=None):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, device=device, dtype=dtype)
        self.linear = operations.Linear(hidden_size, out_channels, device=device, dtype=dtype)

    def forward(self, x):
        return self.linear(self.norm_final(x))


class TimestepEmbedder(nn.Module):
    def __init__(self, dim, freq_dim=256, device=None, dtype=None, operations=None):
        super().__init__()
        self.freq_dim = freq_dim
        self.time_mlp = nn.Sequential(
            operations.Linear(freq_dim, dim, device=device, dtype=dtype),
            nn.SiLU(),
            operations.Linear(dim, dim, device=device, dtype=dtype),
        )

    def forward(self, t, dtype):
        half = self.freq_dim // 2
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -(math.log(10000) / (half - 1)))
        emb = (1000 * t).float().unsqueeze(1) * emb.unsqueeze(0)  # 1000*t rounds in t's dtype, as in the reference
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1).to(t.dtype)
        return self.time_mlp(emb.to(dtype))


class CondEmbedder(nn.Module):
    def __init__(self, in_dim, hidden_size, device=None, dtype=None, operations=None):
        super().__init__()
        self.cond_embedder = operations.Linear(in_dim, hidden_size, device=device, dtype=dtype)

    def forward(self, c):
        return self.cond_embedder(c)


class DiT(nn.Module):
    """Predicts the flow for one latent patch from [t + cond, history patches, current patch]."""

    def __init__(self, in_channels=64, hidden_size=1024, depth=8, num_heads=16, mlp_ratio=4.0, llm_cond_dim=896,
                 device=None, dtype=None, operations=None, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.t_embedder = TimestepEmbedder(hidden_size, device=device, dtype=dtype, operations=operations)
        self.x_embedder = operations.Linear(in_channels, hidden_size, device=device, dtype=dtype)
        self.c_embedder = CondEmbedder(llm_cond_dim, hidden_size, device=device, dtype=dtype, operations=operations)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio, device=device, dtype=dtype, operations=operations) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, in_channels, device=device, dtype=dtype, operations=operations)

    def forward(self, x, t, c, latent_history):
        dtype = c.dtype  # the LLM's compute dtype; x/history may be fp32 flow state (tiny)
        y =self.t_embedder(t, dtype).unsqueeze(1) + self.c_embedder(c.to(dtype))
        x = torch.cat([y, self.x_embedder(latent_history.to(dtype)), self.x_embedder(x.to(dtype))], dim=1)
        freqs = rotary_freqs(x.shape[1], x.shape[-1] // self.num_heads, x.device)
        for block in self.blocks:
            x = block(x, freqs)
        return self.final_layer(x)

    def forward_with_cfg(self, x, t, c, latent_history, patch_size):
        """Conditional and null (zeroed LLM state) branches in one batch."""
        x = torch.cat([x, x], dim=0)
        latent_history = torch.cat([latent_history, latent_history], dim=0)
        c = torch.cat([c, torch.zeros_like(c)], dim=0)
        t = t.reshape(1).repeat(x.shape[0])
        return self.forward(x, t, c, latent_history)[:, -patch_size:, :]


class Aggregator(nn.Module):
    """Folds one generated latent patch into a single LLM input embedding via a CLS token."""

    def __init__(self, in_channels=64, hidden_size=1024, depth=8, num_heads=16, mlp_ratio=4.0, llm_input_dim=896,
                 device=None, dtype=None, operations=None, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.word_embedder = operations.Embedding(1, hidden_size, device=device, dtype=dtype)
        self.x_embedder = operations.Linear(in_channels, hidden_size, device=device, dtype=dtype)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio, device=device, dtype=dtype, operations=operations) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, llm_input_dim, device=device, dtype=dtype, operations=operations)

    def forward(self, x):
        x = self.x_embedder(x)
        cls = self.word_embedder(torch.zeros((x.shape[0], 1), dtype=torch.long, device=x.device), out_dtype=x.dtype)
        x = torch.cat([cls, x], dim=1)
        freqs = rotary_freqs(x.shape[1], x.shape[-1] // self.num_heads, x.device)
        for block in self.blocks:
            x = block(x, freqs)
        return self.final_layer(x)[:, :1, :]


# Empirically Pruned Step Sampling: timesteps in 1/32 units for low step counts.
EPSS_TIMESTEPS = {
    5: [0, 2, 4, 8, 16, 32],
    6: [0, 2, 4, 6, 8, 16, 32],
    7: [0, 2, 4, 6, 8, 16, 24, 32],
    10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
    12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
}


def flow_timesteps(steps, device, dtype, sway=-1.0):
    if steps in EPSS_TIMESTEPS:
        t = torch.tensor(EPSS_TIMESTEPS[steps], device=device, dtype=dtype) / 32
    else:
        t = torch.linspace(0, 1, steps + 1, device=device, dtype=dtype)
    return t + sway * (torch.cos(torch.pi / 2 * t) - 1 + t)


def sample_patch(dit, cond, latent_history, noise, sde_noise, cfg=2.0, sigma=0.25, temperature=0.0, patch_size=1,
                 cfg_mode="ming05", noise_after_last=True):
    """Euler(-Maruyama) integration of the flow from ``noise`` to one latent patch, in the
    dtype of ``noise``. ``sde_noise`` holds one noise tensor per step (its length is the step
    count); the stochastic term vanishes at temperature 0 (the reference default).

    cfg_mode "ming05" (0.5B): cond + cfg * (cond - null); "standard" (tiny): null + cfg * (cond - null).
    """
    steps = sde_noise.shape[0]
    t = flow_timesteps(steps, noise.device, noise.dtype)
    y = noise
    for i in range(steps):
        dt = t[i + 1] - t[i]
        pred, null = dit.forward_with_cfg(y, t[i], cond, latent_history, patch_size).to(y.dtype).chunk(2, dim=0)
        guided = pred + (pred - null) * cfg if cfg_mode == "ming05" else null + cfg * (pred - null)
        y = y + guided * dt
        if temperature > 0 and (noise_after_last or i < steps - 1):
            y = y + sigma * (temperature ** 0.5) * (dt.abs() ** 0.5) * sde_noise[i]
    return y
