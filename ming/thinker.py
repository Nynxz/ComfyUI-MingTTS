"""Ming-3.0-audio-tiny "thinker": speech recognition and audio understanding on ComfyUI ops.

Reimplements modeling_bailing_moe_v3_audio + processing_bailing_moe_v3_audio of
inclusionAI/Ming-3.0-audio-tiny: a Whisper-large-style encoder over 128-bin log-mels (16 kHz,
no 30 s padding, 30 s chunks), three stride-2 convs + a linear projecting every 8 encoder
frames into one L2-normalised LLM input embedding, and the Ling V3 MoE (ling.py, RoPE theta 6e6)
with an LM head. Its spoken replies come from a "talker" that is exactly Ming-3.0-tts-0.5B.
Parameter names follow the release (the MTP layer is dropped).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops

from .ling import LingV3, LingV3Config

SAMPLE_RATE = 16000
N_FFT, HOP, N_MELS = 400, 160, 128
CHUNK = 3000  # mel frames per encoder pass (30 s)


def mel_filters(device, n_mels=N_MELS, sr=SAMPLE_RATE, n_fft=N_FFT):
    """librosa.filters.mel(sr, n_fft, n_mels) (Slaney scale and norm), which openai-whisper ships as mel_filters.npz."""
    def hz_to_mel(f):
        f = torch.as_tensor(f, dtype=torch.float64)
        lin = f / (200.0 / 3)
        log = 15.0 + torch.log(f / 1000.0) / (math.log(6.4) / 27.0)
        return torch.where(f >= 1000.0, log, lin)

    def mel_to_hz(m):
        lin = m * (200.0 / 3)
        log = 1000.0 * torch.exp((math.log(6.4) / 27.0) * (m - 15.0))
        return torch.where(m >= 15.0, log, lin)

    fft_freqs = torch.linspace(0, sr / 2, n_fft // 2 + 1, dtype=torch.float64)
    mel_f = mel_to_hz(torch.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2), n_mels + 2, dtype=torch.float64))
    fdiff = mel_f[1:] - mel_f[:-1]
    ramps = mel_f[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = torch.clamp(torch.minimum(lower, upper), min=0)
    weights *= (2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels]))[:, None]
    return weights.float().to(device)


def log_mel(waveform):
    """openai-whisper log_mel_spectrogram of a [samples] 16 kHz waveform -> [frames, 128]."""
    window = torch.hann_window(N_FFT, device=waveform.device)
    stft = torch.stft(waveform.float(), N_FFT, HOP, window=window, return_complex=True)
    power = stft[..., :-1].abs() ** 2
    log_spec = torch.clamp(mel_filters(waveform.device) @ power, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).transpose(0, 1)


class WhisperAttention(nn.Module):
    def __init__(self, n_state, n_head, device=None, dtype=None, ops=None):
        super().__init__()
        self.n_head = n_head
        self.query = ops.Linear(n_state, n_state, device=device, dtype=dtype)
        self.key = ops.Linear(n_state, n_state, bias=False, device=device, dtype=dtype)
        self.value = ops.Linear(n_state, n_state, device=device, dtype=dtype)
        self.out = ops.Linear(n_state, n_state, device=device, dtype=dtype)

    def forward(self, x, mask=None):
        b, t, _ = x.shape
        q, k, v = (proj(x).view(b, t, self.n_head, -1).transpose(1, 2) for proj in (self.query, self.key, self.value))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(out.transpose(1, 2).reshape(b, t, -1))


class WhisperBlock(nn.Module):
    def __init__(self, n_state, n_head, device=None, dtype=None, ops=None):
        super().__init__()
        self.attn = WhisperAttention(n_state, n_head, device=device, dtype=dtype, ops=ops)
        self.attn_ln = ops.LayerNorm(n_state, device=device, dtype=dtype)
        self.mlp = nn.Sequential(ops.Linear(n_state, 4 * n_state, device=device, dtype=dtype), nn.GELU(),
                                 ops.Linear(4 * n_state, n_state, device=device, dtype=dtype))
        self.mlp_ln = ops.LayerNorm(n_state, device=device, dtype=dtype)

    def forward(self, x, mask=None):
        x = x + self.attn(self.attn_ln(x), mask)
        return x + self.mlp(self.mlp_ln(x))


class WhisperEncoder(nn.Module):
    """openai-whisper AudioEncoder without the fixed 30 s length (positions are sliced)."""

    def __init__(self, n_mels=128, n_ctx=3000, n_state=1280, n_head=20, n_layer=32, device=None, dtype=None, ops=None):
        super().__init__()
        self.conv1 = ops.Conv1d(n_mels, n_state, 3, padding=1, device=device, dtype=dtype)
        self.conv2 = ops.Conv1d(n_state, n_state, 3, stride=2, padding=1, device=device, dtype=dtype)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state, device=device, dtype=dtype))
        self.blocks = nn.ModuleList([WhisperBlock(n_state, n_head, device=device, dtype=dtype, ops=ops) for _ in range(n_layer)])
        self.ln_post = ops.LayerNorm(n_state, device=device, dtype=dtype)

    def forward(self, mel, lengths):
        """mel [b, frames, 128] (zero padded), lengths [b] -> [b, frames/2, n_state], output lengths."""
        x = F.gelu(self.conv1(mel.transpose(1, 2)))
        x = F.gelu(self.conv2(x)).transpose(1, 2)
        x = x + comfy.ops.cast_to_input(self.positional_embedding[: x.shape[1]], x)
        # the reference batches zero-padded 30 s chunks without an attention mask; so does this
        for block in self.blocks:
            x = block(x)
        return self.ln_post(x), (lengths - 1) // 2 + 1


class MingThinker(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None, rope_theta=6000000.0, **kwargs):
        super().__init__()
        ops = operations
        config = LingV3Config(rope_theta=rope_theta)
        self.config = config
        self.model = LingV3(config, device=device, dtype=dtype, ops=ops)
        self.model.audio = WhisperEncoder(device=device, dtype=dtype, ops=ops)
        h = config.hidden_size
        self.linear_proj_audio = nn.Sequential(
            ops.Conv1d(1280, h, 3, stride=2, padding=1, device=device, dtype=dtype), nn.GELU(),
            ops.Conv1d(h, h, 3, stride=2, padding=1, device=device, dtype=dtype), nn.GELU(),
            ops.Conv1d(h, h, 3, stride=2, padding=1, device=device, dtype=dtype), nn.GELU(),
            nn.Identity(),  # the reference's Transpose(-1, -2)
            ops.Linear(h, h, device=device, dtype=dtype),
        )
        self.lm_head = ops.Linear(h, config.vocab_size, bias=False, device=device, dtype=dtype)

    def audio_embeds(self, waveform_16k, dtype):
        """[samples] -> L2-normalised audio embeddings [n, hidden], one per 8 encoder frames (~80 ms of audio)."""
        mel = log_mel(waveform_16k)
        chunks = list(mel.split(CHUNK))
        lengths = torch.tensor([c.shape[0] for c in chunks], device=mel.device)
        batch = nn.utils.rnn.pad_sequence(chunks, batch_first=True).to(dtype)
        enc, enc_len = self.model.audio(batch, lengths)
        x = enc.transpose(1, 2)
        lens = enc_len
        for i in (0, 2, 4):
            x = F.gelu(self.linear_proj_audio[i](x))
            lens = (lens - 1) // 2 + 1
        x = self.linear_proj_audio[7](x.transpose(1, 2))
        out = torch.cat([x[i, : lens[i]] for i in range(x.shape[0])], dim=0)
        return F.normalize(out.float(), dim=-1).to(dtype)

    @staticmethod
    def num_audio_tokens(samples):
        frames = samples // HOP  # stft (center=True) gives samples//hop + 1 frames; whisper drops the last
        total = 0
        for start in range(0, frames, CHUNK):
            n = min(CHUNK, frames - start)
            n = (n - 1) // 2 + 1
            for _ in range(3):
                n = (n - 1) // 2 + 1
            total += n
        return total
