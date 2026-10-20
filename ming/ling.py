"""Ling V3 (BailingMoeV3) backbone of Ming-3.0-tts-tiny and Ming-3.0-audio-tiny, on ComfyUI ops.

24 layers in groups of 4: three Kimi Delta Attention (KDA, linear attention with a per-channel
decayed delta-rule state) then one Multi-head Latent Attention (MLA) layer. Layer 0 has a dense
SwiGLU MLP, the rest a 128-expert top-8 MoE (+1 shared expert) whose experts run on ComfyUI's
expert banks (int8/fp8 capable).

The reference runs KDA through flash-linear-attention's Triton kernels (chunk_kda with the gate
and q/k L2 norm fused in). Here it is plain torch in fp32: a chunked form for prompts and the
recurrent form for decode steps (mathematically identical; fla.ops.kda.naive), so no extra
dependency is needed. Parameter names follow the released checkpoint except the MoE, whose
per-expert weights are stacked into banks at load (see stack_experts).
"""

import math
import re
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops
from comfy.text_encoders.llama import MLP, RMSNorm, moe_experts_forward


@dataclass
class LingV3Config:
    vocab_size: int = 157184
    hidden_size: int = 1536
    intermediate_size: int = 4608
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    head_dim: int = 128  # KDA head dim
    q_lora_rank: int = 256
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    num_experts: int = 128
    num_experts_per_tok: int = 8
    n_group: int = 8
    topk_group: int = 4
    routed_scaling_factor: float = 2.5
    moe_intermediate_size: int = 512
    first_k_dense_replace: int = 1
    layer_group_size: int = 4
    conv_size: int = 4
    kda_lower_bound: float = -5.0
    mlp_activation: str = "silu"


def is_mla_layer(config, index):
    g = config.layer_group_size
    return (index + 1) % g == 0 or index >= config.num_hidden_layers // g * g


def param(p, x, dtype=None):
    return comfy.ops.cast_to_input(p, x, copy=False) if dtype is None else p.to(device=x.device, dtype=dtype)


# ---- MLA -------------------------------------------------------------------------------------

def rope_cos_sin(positions, dim, theta, dtype):
    # fp32 inverse frequencies: unlike the 0.5B runtime, the tiny one moves the model to the GPU
    # without a dtype cast, so its (non-persistent) inv_freq buffer stays unrounded
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=positions.device, dtype=torch.float32) / dim))
    freqs = positions.float()[:, None] * inv_freq[None]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)  # the reference casts the tables to the activation dtype


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope_interleaved(x, cos, sin):
    """DeepSeek-style: de-interleave the pair layout, then half-rotate (keys stay de-interleaved)."""
    b, h, s, d = x.shape
    x = x.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    return x * cos + rotate_half(x) * sin


class MLAttention(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        c = config
        self.c = c
        self.qk_head_dim = c.qk_nope_head_dim + c.qk_rope_head_dim
        h = c.num_attention_heads
        self.q_a_proj = ops.Linear(c.hidden_size, c.q_lora_rank, bias=False, device=device, dtype=dtype)
        self.q_a_layernorm = RMSNorm(c.q_lora_rank, eps=c.rms_norm_eps, device=device, dtype=dtype)
        self.q_b_proj = ops.Linear(c.q_lora_rank, h * self.qk_head_dim, bias=False, device=device, dtype=dtype)
        self.kv_a_proj_with_mqa = ops.Linear(c.hidden_size, c.kv_lora_rank + c.qk_rope_head_dim, bias=False, device=device, dtype=dtype)
        self.kv_a_layernorm = RMSNorm(c.kv_lora_rank, eps=c.rms_norm_eps, device=device, dtype=dtype)
        self.kv_b_proj = ops.Linear(c.kv_lora_rank, h * (c.qk_nope_head_dim + c.v_head_dim), bias=False, device=device, dtype=dtype)
        self.g_proj = ops.Linear(c.hidden_size, h, bias=False, device=device, dtype=dtype)
        self.dense = ops.Linear(h * c.v_head_dim, c.hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, x, rope, cache):
        c = self.c
        b, s, _ = x.shape
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x))).view(b, s, -1, self.qk_head_dim).transpose(1, 2)
        q_pass, q_rot = q.split([c.qk_nope_head_dim, c.qk_rope_head_dim], dim=-1)
        k_pass, k_rot = self.kv_a_proj_with_mqa(x).split([c.kv_lora_rank, c.qk_rope_head_dim], dim=-1)
        kv = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(b, s, -1, c.qk_nope_head_dim + c.v_head_dim).transpose(1, 2)
        k_pass, v = kv.split([c.qk_nope_head_dim, c.v_head_dim], dim=-1)
        cos, sin = rope
        q_rot = apply_rope_interleaved(q_rot, cos, sin)
        k_rot = apply_rope_interleaved(k_rot.view(b, 1, s, c.qk_rope_head_dim), cos, sin).expand(-1, k_pass.shape[1], -1, -1)
        q = torch.cat((q_pass, q_rot), dim=-1)
        k = torch.cat((k_pass, k_rot), dim=-1)
        k, v = cache.append(k, v)
        # prefill: causal over the new rows; a single decode row sees everything cached
        out = F.scaled_dot_product_attention(q, k, v, is_causal=s > 1 and k.shape[2] == s, scale=self.qk_head_dim ** -0.5,
                                             attn_mask=None if s == 1 or k.shape[2] == s else cache.mask(s, q.dtype))
        out = out.transpose(1, 2)  # [b, s, h, dv]
        gate = torch.sigmoid(self.g_proj(x).float()).to(out.dtype)
        return self.dense((out * gate.unsqueeze(-1)).reshape(b, s, -1))


class KVCache:
    def __init__(self, capacity):
        self.capacity, self.k, self.v, self.n = capacity, None, None, 0

    def append(self, k, v):
        s = k.shape[2]
        if self.k is None:
            self.k = k.new_empty(*k.shape[:2], self.capacity, k.shape[-1])
            self.v = v.new_empty(*v.shape[:2], self.capacity, v.shape[-1])
        if self.n + s > self.capacity:
            raise ValueError(f"Ming: context exceeds {self.capacity} positions")
        self.k[:, :, self.n:self.n + s] = k
        self.v[:, :, self.n:self.n + s] = v
        self.n += s
        return self.k[:, :, :self.n], self.v[:, :, :self.n]

    def mask(self, s, dtype):
        n = self.n
        rows = torch.arange(n - s, n, device=self.k.device).view(-1, 1)
        visible = torch.arange(n, device=self.k.device).view(1, -1) <= rows
        return torch.zeros((s, n), device=self.k.device, dtype=dtype).masked_fill_(~visible, float("-inf"))


# ---- KDA -------------------------------------------------------------------------------------

def l2norm(x, eps=1e-6):
    return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + eps)


def kda_recurrent(q, k, v, g, beta, state):
    """Token-by-token delta rule. q/k/v/g [b, t, h, d] fp32 (q pre-scaled), beta [b, t, h]."""
    out = torch.empty_like(v)
    for i in range(q.shape[1]):
        state = state * g[:, i, ..., None].exp()
        k_i = k[:, i]
        delta = v[:, i] - torch.einsum("bhk,bhkv->bhv", k_i, state)
        state = state + torch.einsum("bhk,bhv->bhkv", beta[:, i, :, None] * k_i, delta)
        out[:, i] = torch.einsum("bhk,bhkv->bhv", q[:, i], state)
    return out, state


def kda_chunked(q, k, v, g, beta, state, chunk=64):
    """Chunked (WY) form of the same recurrence, as fla.ops.kda.naive.naive_chunk_kda.

    Zero padding to a whole chunk is exact: a padded step has k = v = 0, beta = 0 and g = 0
    (decay exp(0) = 1), so it leaves the state and the earlier outputs untouched."""
    b, t, h, dk = q.shape
    pad = -t % chunk
    if pad:
        q, k, v, g = (F.pad(x, (0, 0, 0, 0, 0, pad)) for x in (q, k, v, g))
        beta = F.pad(beta, (0, 0, 0, pad))
    n = q.shape[1] // chunk
    q, k, v, g = (x.view(b, n, chunk, h, -1).permute(0, 3, 1, 2, 4) for x in (q, k, v, g))  # [b, h, n, c, d]
    beta = beta.view(b, n, chunk, h).permute(0, 3, 1, 2)
    g = g.cumsum(-2)

    # A[i, j] = beta_i * <k_i * exp(g_i - g_j), k_j> for j < i, then (I + A)^-1 by forward substitution
    decay = (g.unsqueeze(-2) - g.unsqueeze(-3))  # [b, h, n, c_i, c_j, d]
    strict = torch.tril(torch.ones(chunk, chunk, dtype=torch.bool, device=q.device), diagonal=-1)
    A = torch.einsum("bhnid,bhnijd,bhnjd->bhnij", k, decay.masked_fill(~strict[..., None], float("-inf")).exp(), k)
    A = -(A * beta.unsqueeze(-1))
    for i in range(1, chunk):
        A[..., i, :i] = A[..., i, :i] + (A[..., i, :, None] * A[..., :, :i]).sum(-2)
    A = (A + torch.eye(chunk, device=q.device)) * beta.unsqueeze(-2)
    w = A @ (g.exp() * k)
    u = A @ v

    causal = torch.tril(torch.ones(chunk, chunk, dtype=torch.bool, device=q.device))
    qk = torch.einsum("bhnid,bhnijd,bhnjd->bhnij", q, decay.masked_fill(~causal[..., None], float("-inf")).exp(), k)
    out = torch.empty_like(v)
    for i in range(n):
        g_i, k_i = g[:, :, i], k[:, :, i]
        v_i = u[:, :, i] - w[:, :, i] @ state
        out[:, :, i] = (q[:, :, i] * g_i.exp()) @ state + qk[:, :, i] @ v_i
        state = state * g_i[:, :, -1, :, None].exp() + ((g_i[:, :, -1:] - g_i).exp() * k_i).transpose(-1, -2) @ v_i
    out = out.permute(0, 2, 3, 1, 4).reshape(b, n * chunk, h, -1)[:, :t]
    return out, state


class ShortConv(nn.Module):
    """Causal depthwise conv + SiLU (fla ShortConvolution); the cache holds the last k-1 inputs."""

    def __init__(self, channels, kernel, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel, device=device, dtype=dtype))

    def forward(self, x, cache):
        w = param(self.weight, x)
        kernel = w.shape[-1]
        x = x.transpose(1, 2)  # [b, c, t]
        history = cache if cache is not None else x.new_zeros(x.shape[0], x.shape[1], kernel - 1)
        full = torch.cat([history, x], dim=-1)
        y = F.conv1d(full, w, groups=w.shape[0])
        return F.silu(y).transpose(1, 2), full[..., -(kernel - 1):]


class KDAttention(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        c = config
        self.c = c
        size = c.num_attention_heads * c.head_dim
        for name in ("q_proj", "k_proj", "v_proj", "f_proj", "g_proj"):
            setattr(self, name, ops.Linear(c.hidden_size, size, bias=False, device=device, dtype=dtype))
        self.b_proj = ops.Linear(c.hidden_size, c.num_attention_heads, bias=False, device=device, dtype=dtype)
        self.q_conv1d = ShortConv(size, c.conv_size, device=device, dtype=dtype)
        self.k_conv1d = ShortConv(size, c.conv_size, device=device, dtype=dtype)
        self.v_conv1d = ShortConv(size, c.conv_size, device=device, dtype=dtype)
        self.A_log = nn.Parameter(torch.empty(c.num_attention_heads, device=device, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(size, device=device, dtype=torch.float32))
        self.o_norm = nn.Module()
        self.o_norm.weight = nn.Parameter(torch.empty(c.head_dim, device=device, dtype=dtype))
        self.o_proj = ops.Linear(size, c.hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, x, rope, cache):
        c = self.c
        b, t, _ = x.shape
        heads = lambda y: y.view(b, t, c.num_attention_heads, c.head_dim)
        q, cache.q = self.q_conv1d(self.q_proj(x), cache.q)
        k, cache.k = self.k_conv1d(self.k_proj(x), cache.k)
        v, cache.v = self.v_conv1d(self.v_proj(x), cache.v)
        q, k, v = (heads(y).float() for y in (q, k, v))
        q, k = l2norm(q) * c.head_dim ** -0.5, l2norm(k)
        A = param(self.A_log, x, torch.float32).exp().view(-1, 1)
        g = c.kda_lower_bound * torch.sigmoid(A * (heads(self.f_proj(x)).float() + param(self.dt_bias, x, torch.float32).view(c.num_attention_heads, -1)))
        beta = torch.sigmoid(self.b_proj(x).float())
        state = cache.state if cache.state is not None else x.new_zeros(b, c.num_attention_heads, c.head_dim, c.head_dim, dtype=torch.float32)
        fn = kda_recurrent if t <= 16 else kda_chunked
        out, cache.state = fn(q, k, v, g, beta, state)
        # gated RMSNorm (fla FusedRMSNormGated, sigmoid), in fp32
        out = out.to(x.dtype).float()
        out = out * torch.rsqrt(out.square().mean(-1, keepdim=True) + c.rms_norm_eps) * param(self.o_norm.weight, x, torch.float32)
        out = out * torch.sigmoid(heads(self.g_proj(x)).float())
        return self.o_proj(out.to(x.dtype).reshape(b, t, -1))


class KDAState:
    def __init__(self):
        self.q = self.k = self.v = self.state = None


# ---- MoE + layers ------------------------------------------------------------------------------

class LingGate(nn.Module):
    """Sigmoid router with a load-balancing bias that only affects selection, group-limited
    top-k (8 of 128 experts from the best 4 of 8 groups), all in fp32 as the reference does."""

    def __init__(self, config, device=None, dtype=None):
        super().__init__()
        self.c = config
        self.weight = nn.Parameter(torch.empty(config.num_experts, config.hidden_size, device=device, dtype=dtype))
        self.expert_bias = nn.Parameter(torch.zeros(config.num_experts, device=device, dtype=torch.float32))

    def forward(self, x):
        c = self.c
        scores = torch.sigmoid(F.linear(x.float(), param(self.weight, x, torch.float32)))
        routing = (scores + param(self.expert_bias, x, torch.float32)).view(x.shape[0], c.n_group, -1)
        group_idx = routing.topk(2, dim=-1)[0].sum(-1).topk(c.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros(routing.shape[:2], dtype=torch.bool, device=x.device).scatter_(1, group_idx, True)
        idx = routing.masked_fill(~group_mask.unsqueeze(-1), float("-inf")).view(x.shape[0], -1).topk(c.num_experts_per_tok, dim=-1)[1]
        weight = scores.gather(1, idx)
        return idx, weight / (weight.sum(-1, keepdim=True) + 1e-20) * c.routed_scaling_factor


class LingMoE(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None, expert_ops=None):
        super().__init__()
        self.c = config
        self.gate = LingGate(config, device=device, dtype=dtype)
        self.experts = nn.Module()
        self.experts.gate_up_proj = expert_ops.MoEExperts(num_experts=config.num_experts, in_features=config.hidden_size,
                                                          out_features=2 * config.moe_intermediate_size, bias=False, device=device, dtype=dtype)
        self.experts.down_proj = expert_ops.MoEExperts(num_experts=config.num_experts, in_features=config.moe_intermediate_size,
                                                       out_features=config.hidden_size, bias=False, device=device, dtype=dtype)
        self.shared_experts = MLP(config, device=device, dtype=dtype, ops=ops, intermediate_size=config.moe_intermediate_size)

    def forward(self, x):
        b, s, h = x.shape
        flat = x.reshape(-1, h)
        idx, weight = self.gate(flat)
        out = moe_experts_forward(flat, idx, weight, self.c.num_experts, self.experts.gate_up_proj, self.experts.down_proj, comfy.ops._swiglu_eager)
        return out.view(b, s, h).to(x.dtype) + self.shared_experts(x)


class LingLayer(nn.Module):
    def __init__(self, config, index, device=None, dtype=None, ops=None, expert_ops=None):
        super().__init__()
        self.mla = is_mla_layer(config, index)
        self.attention = (MLAttention if self.mla else KDAttention)(config, device=device, dtype=dtype, ops=ops)
        if index < config.first_k_dense_replace:
            self.mlp = MLP(config, device=device, dtype=dtype, ops=ops)
        else:
            self.mlp = LingMoE(config, device=device, dtype=dtype, ops=ops, expert_ops=expert_ops)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)

    def forward(self, x, rope, cache):
        x = x + self.attention(self.input_layernorm(x), rope, cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class LingV3(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.config = config
        expert_ops = ops if hasattr(ops, "MoEExperts") else comfy.ops.mixed_precision_ops({}, dtype)
        self.word_embeddings = ops.Embedding(config.vocab_size, config.hidden_size, device=device, dtype=dtype)
        self.layers = nn.ModuleList([LingLayer(config, i, device=device, dtype=dtype, ops=ops, expert_ops=expert_ops)
                                     for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)

    def embed_tokens(self, ids, out_dtype=None):
        return self.word_embeddings(ids, out_dtype=out_dtype)

    def init_state(self, capacity):
        return {"pos": 0, "layers": [KVCache(capacity) if layer.mla else KDAState() for layer in self.layers]}

    def forward(self, embeds, state):
        """Returns the final-norm hidden states [b, t, h] and advances ``state``."""
        t = embeds.shape[1]
        positions = torch.arange(state["pos"], state["pos"] + t, device=embeds.device)
        rope = rope_cos_sin(positions, self.config.qk_rope_head_dim, self.config.rope_theta, embeds.dtype)
        x = embeds
        for layer, cache in zip(self.layers, state["layers"]):
            x = layer(x, rope, cache)
        state["pos"] += t
        return self.norm(x)


# ---- checkpoint layout ---------------------------------------------------------------------------

EXPERT_KEY = re.compile(r"^((?:.*\.)?mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


def stack_experts(state_dict):
    """Released per-expert weights -> ComfyUI expert banks: experts.gate_up_proj.weight [E, 2I, H]
    (gate rows then up rows, as _swiglu_eager splits them) and experts.down_proj.weight [E, H, I].
    """
    banks = {}
    for key in list(state_dict.keys()):
        m = EXPERT_KEY.match(key)
        if m:
            banks.setdefault(m[1], {}).setdefault(int(m[2]), {})[m[3]] = state_dict.pop(key)
    for prefix, experts in banks.items():
        order = sorted(experts)
        state_dict[f"{prefix}.gate_up_proj.weight"] = torch.stack([torch.cat([experts[i]["gate_proj"], experts[i]["up_proj"]]) for i in order])
        state_dict[f"{prefix}.down_proj.weight"] = torch.stack([experts[i]["down_proj"] for i in order])
    return state_dict
