"""Ming-3.0 AudioVAE (44.1 kHz mono, 64-channel latents at 12.5 Hz) on ComfyUI ops.

Reimplements audio_tokenizer/{modeling_audio_vae,vae_modules,istft}.py of
inclusionAI/Ming-3.0-tts. Encoder: 882-sample frames (50 Hz) -> 24 causal Qwen2 layers ->
every 4 frames pooled into a CLS token by 4 more layers -> 64-d Gaussian latent. Decoder:
latent -> x4 linear upsample -> 24 causal Qwen2 layers -> ISTFT head. All Qwen2 stacks attend
over a 64-frame causal sliding window (the reference applies it through the 4D mask).

The same file can also carry the CAM++ speaker encoder (``speaker_encoder.*``): it is shared
by every Ming TTS model, so voice cloning needs only this one VAE file.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops
from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy.text_encoders.llama import RMSNorm, TransformerBlock

from .rope import qwen2_freqs_cis

SAMPLE_RATE = 44100
HOP = 882  # samples per encoder frame (50 Hz)
PATCH = 4  # encoder frames per latent (12.5 Hz)
LATENT_DIM = 64
WINDOW = 64


@dataclass
class Qwen2VAEConfig:
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    head_dim: int = 64
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    qkv_bias = True
    rms_norm_add = False
    mlp_activation = "silu"
    q_norm = None
    k_norm = None
    rope_dims = None
    rope_scale = None


def sliding_causal_mask(seq, window, device, dtype):
    i = torch.arange(seq, device=device)
    rel = i.view(-1, 1) - i.view(1, -1)
    visible = (rel >= 0) & (rel < window)
    return torch.zeros((seq, seq), device=device, dtype=dtype).masked_fill_(~visible, torch.finfo(dtype).min / 4)


class Qwen2Stack(nn.Module):
    """``transformers.Qwen2Model`` driven by inputs_embeds (its 1-row vocab is never used)."""

    def __init__(self, num_layers, config, device=None, dtype=None, operations=None):
        super().__init__()
        self.config = config
        self.embed_tokens = operations.Embedding(1, config.hidden_size, device=device, dtype=dtype)
        self.layers = nn.ModuleList([TransformerBlock(config, i, device=device, dtype=dtype, ops=operations) for i in range(num_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)

    def forward(self, x, window=WINDOW):
        seq = x.shape[1]
        freqs = qwen2_freqs_cis(self.config.head_dim, torch.arange(seq, device=x.device).unsqueeze(0), self.config.rope_theta, x.device)
        mask = sliding_causal_mask(seq, window, x.device, x.dtype)
        attention = optimized_attention_for_device(x.device, mask=True, small_input=True)
        for layer in self.layers:
            x, _ = layer(x, attention_mask=mask, freqs_cis=freqs, optimized_attention=attention)
        return self.norm(x)


class Encoder(nn.Module):
    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        h = config.hidden_size
        self.encoder = Qwen2Stack(24, config, device=device, dtype=dtype, operations=operations)
        self.fc1 = operations.Linear(HOP, h, bias=False, device=device, dtype=dtype)
        self.fc2 = operations.Linear(h, h, device=device, dtype=dtype)
        self.fc3 = operations.Linear(h, LATENT_DIM * 2, device=device, dtype=dtype)
        self.norm = operations.LayerNorm(h, device=device, dtype=dtype)  # present in the checkpoint, unused by the reference
        self.aggregator = Qwen2Stack(4, config, device=device, dtype=dtype, operations=operations)
        self.cls_embed = nn.Parameter(torch.empty(1, 1, h, device=device, dtype=dtype))

    def forward(self, waveform):
        """[B, samples] -> Gaussian parameters [B, T_latent, 128]."""
        frames = -(-waveform.shape[-1] // HOP)
        waveform = F.pad(waveform, (0, frames * HOP - waveform.shape[-1]))
        x = self.encoder(self.fc2(self.fc1(waveform.unfold(-1, HOP, HOP))))
        b, t, d = x.shape
        x = F.pad(x, (0, 0, 0, -t % PATCH)).reshape(-1, PATCH, d)
        cls = comfy.ops.cast_to_input(self.cls_embed, x).expand(x.shape[0], -1, -1)
        x = self.aggregator(torch.cat((x, cls), dim=1).reshape(b, -1, d))
        return self.fc3(x.reshape(-1, PATCH + 1, d)[:, -1].reshape(b, -1, d))


class ISTFTHead(nn.Module):
    """Predicts log-magnitude + phase and inverts with a "same"-padded ISTFT, in fp32."""

    def __init__(self, dim, n_fft, hop, device=None, dtype=None, operations=None):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.out = operations.Linear(dim, n_fft + 2, device=device, dtype=dtype)
        self.istft = nn.Module()
        self.istft.register_buffer("window", torch.hann_window(n_fft, device=device), persistent=False)

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            pred = F.linear(x.float(), self.out.weight.to(x.device, torch.float32), self.out.bias.to(x.device, torch.float32)).transpose(1, 2)
            mag, phase = pred.chunk(2, dim=1)
            spec = torch.clip(torch.exp(mag), max=1e2) * torch.polar(torch.ones_like(phase), phase)
            window = self.istft.window.to(x.device, torch.float32)
            frames = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward") * window[None, :, None]
            t = frames.shape[-1]
            size = (t - 1) * self.hop + self.n_fft
            fold = lambda v: F.fold(v, output_size=(1, size), kernel_size=(1, self.n_fft), stride=(1, self.hop))[:, 0, 0]
            pad = (self.n_fft - self.hop) // 2
            y = fold(frames)[:, pad:-pad]
            envelope = fold(window.square().expand(1, t, -1).transpose(1, 2))[0, pad:-pad]
            return (y / envelope).unsqueeze(1)


class Decoder(nn.Module):
    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        self.decoder = Qwen2Stack(24, config, device=device, dtype=dtype, operations=operations)
        self.fc1 = operations.Linear(LATENT_DIM, config.hidden_size, device=device, dtype=dtype)
        self.head = ISTFTHead(config.hidden_size, HOP * 4, HOP, device=device, dtype=dtype, operations=operations)

    def forward(self, latent):
        """[B, T_latent, 64] -> [B, 1, samples]."""
        x = self.fc1(latent).transpose(1, 2)
        x = F.interpolate(x, scale_factor=PATCH, mode="linear", align_corners=False).transpose(1, 2)
        return self.head(self.decoder(x))


class MingAudioVAE(nn.Module):
    """ComfyUI first_stage_model: encode [B, C, samples] -> [B, 64, T], decode the reverse (mono)."""

    def __init__(self, speaker_encoder=False, device=None, dtype=None, operations=comfy.ops.disable_weight_init):
        super().__init__()
        config = Qwen2VAEConfig()
        self.encoder = Encoder(config, device=device, dtype=dtype, operations=operations)
        self.decoder = Decoder(config, device=device, dtype=dtype, operations=operations)
        if speaker_encoder:
            from .speaker import CAMPPlus
            self.speaker_encoder = CAMPPlus()
        else:
            self.speaker_encoder = None

    def posterior(self, audio):
        mean, scale = self.encoder(audio.mean(dim=1)).float().chunk(2, dim=-1)
        return mean, F.softplus(scale) + 1e-4

    def encode(self, audio):
        # deterministic: the posterior mean (the reference samples it, but its std is ~4e-4 of |mean|)
        return self.posterior(audio)[0].transpose(1, 2)

    def decode(self, latent):
        return self.decoder(latent.transpose(1, 2))
