"""CAM++ speaker encoder (192-d x-vector) used by every Ming-3.0 TTS model for voice cloning.

The checkpoints ship it as campplus.onnx: 3D-Speaker's CAMPPlus (speakerlab/models/campplus)
exported with each BatchNorm that directly follows a conv folded into that conv. This is the
same network in that folded form, so scripts/convert_campplus.py can copy the ONNX tensors
over one-to-one; the BatchNorms that precede a conv stay explicit.

Input: 16 kHz mono audio -> 80-bin Kaldi fbank (dither 0), mean-normalized over time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

SAMPLE_RATE = 16000


def conv1d(cin, cout, k=1, stride=1, dilation=1, bias=True):
    return nn.Conv1d(cin, cout, k, stride=stride, padding=(k - 1) // 2 * dilation, dilation=dilation, bias=bias)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=(stride, 1), padding=1)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.shortcut = nn.Conv2d(cin, cout, 1, stride=(stride, 1)) if stride != 1 or cin != cout else None

    def forward(self, x):
        out = self.conv2(F.relu(self.conv1(x)))
        return F.relu(out + (x if self.shortcut is None else self.shortcut(x)))


class FCM(nn.Module):
    """2-D conv front end over (freq, time); halves the frequency axis three times."""

    def __init__(self, channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, 3, padding=1)
        self.layer1 = nn.Sequential(ResBlock(channels, channels, 2), ResBlock(channels, channels, 1))
        self.layer2 = nn.Sequential(ResBlock(channels, channels, 2), ResBlock(channels, channels, 1))
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=(2, 1), padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x.unsqueeze(1)))
        x = F.relu(self.conv2(self.layer2(self.layer1(x))))
        return x.flatten(1, 2)


class CAMLayer(nn.Module):
    """Context-aware masking: a local conv gated by utterance- and 100-frame-segment context."""

    def __init__(self, cin, cout, dilation, seg_len=100):
        super().__init__()
        self.seg_len = seg_len
        self.linear_local = conv1d(cin, cout, 3, dilation=dilation, bias=False)
        self.linear1 = conv1d(cin, cin // 2)
        self.linear2 = conv1d(cin // 2, cout)

    def forward(self, x):
        seg = F.avg_pool1d(x, self.seg_len, self.seg_len, ceil_mode=True)
        seg = seg.repeat_interleave(self.seg_len, dim=-1)[..., :x.shape[-1]]
        context = F.relu(self.linear1(x.mean(-1, keepdim=True) + seg))
        return self.linear_local(x) * torch.sigmoid(self.linear2(context))


class DenseLayer(nn.Module):
    def __init__(self, cin, growth, bottleneck, dilation):
        super().__init__()
        self.bn = nn.BatchNorm1d(cin)
        self.linear1 = conv1d(cin, bottleneck)  # + folded BatchNorm
        self.cam_layer = CAMLayer(bottleneck, growth, dilation)

    def forward(self, x):
        return self.cam_layer(F.relu(self.linear1(F.relu(self.bn(x)))))


class DenseBlock(nn.Module):
    def __init__(self, layers, cin, growth, bottleneck, dilation):
        super().__init__()
        self.layers = nn.ModuleList([DenseLayer(cin + i * growth, growth, bottleneck, dilation) for i in range(layers)])

    def forward(self, x):
        for layer in self.layers:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class Transit(nn.Module):
    def __init__(self, cin, cout, folded_bias=False):
        super().__init__()
        self.bn = nn.BatchNorm1d(cin)
        self.linear = conv1d(cin, cout, bias=folded_bias)

    def forward(self, x):
        return self.linear(F.relu(self.bn(x)))


class CAMPPlus(nn.Module):
    def __init__(self, feat_dim=80, embedding_size=192, growth=32, bottleneck=128, init_channels=128):
        super().__init__()
        self.head = FCM()
        self.tdnn = conv1d(32 * (feat_dim // 8), init_channels, 5, stride=2)  # + folded BatchNorm
        channels = init_channels
        blocks, transits = [], []
        for i, (layers, dilation) in enumerate(((12, 1), (24, 2), (16, 2))):
            blocks.append(DenseBlock(layers, channels, growth, bottleneck, dilation))
            channels += layers * growth
            # the last transit's conv absorbed the BatchNorm of the output nonlinearity
            transits.append(Transit(channels, channels // 2, folded_bias=i == 2))
            channels //= 2
        self.blocks = nn.ModuleList(blocks)
        self.transits = nn.ModuleList(transits)
        self.dense = conv1d(channels * 2, embedding_size, bias=False)
        self.dense_bn = nn.BatchNorm1d(embedding_size, affine=False)

    def forward(self, feats):
        """[B, frames, 80] fbank -> [B, 192]."""
        x = F.relu(self.tdnn(self.head(feats.transpose(1, 2))))
        for block, transit in zip(self.blocks, self.transits):
            x = transit(block(x))
        x = F.relu(x)
        stats = torch.cat([x.mean(-1), x.std(-1, unbiased=True)], dim=-1)
        return self.dense_bn(self.dense(stats.unsqueeze(-1))).squeeze(-1)


def fbank(waveform_16k):
    """[samples] float waveform at 16 kHz -> mean-normalized [frames, 80] Kaldi fbank."""
    import torchaudio.compliance.kaldi as kaldi

    feats = kaldi.fbank(waveform_16k.float().reshape(1, -1).cpu(), num_mel_bins=80, dither=0, sample_frequency=SAMPLE_RATE)
    return feats - feats.mean(dim=0, keepdim=True)


def conv_modules_in_onnx_order(model):
    """Every conv in the order the ONNX export executes them (used by the converter)."""
    head = model.head
    convs = [head.conv1]
    for layer in (head.layer1, head.layer2):
        for block in layer:
            convs += [block.conv1, block.conv2] + ([block.shortcut] if block.shortcut is not None else [])
    convs += [head.conv2, model.tdnn]
    for block, transit in zip(model.blocks, model.transits):
        for layer in block.layers:
            convs += [layer.linear1, layer.cam_layer.linear_local, layer.cam_layer.linear1, layer.cam_layer.linear2]
        convs.append(transit.linear)
    convs.append(model.dense)
    return convs
