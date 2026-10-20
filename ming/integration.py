"""Registers Ming-3.0 TTS with ComfyUI's native loaders.

After ``register()``:
  Load Diffusion Model (UNETLoader) -> detects a Ming TTS checkpoint (converted or the original
                                       model.safetensors) and returns a MODEL
  Load VAE                          -> detects the Ming AudioVAE; VAE Encode/Decode Audio work on it
                                       (44.1 kHz mono, latents [B, 64, T] at 12.5 Hz)
Nothing in ComfyUI is replaced: detection is chained in front of the built-in function, the
model config is appended to ``comfy.supported_models.models``, and the VAE constructor is
wrapped to build the Ming VAE when (and only when) it sees one.
"""

import logging

import torch

import comfy.latent_formats
import comfy.model_base
import comfy.model_detection
import comfy.model_management
import comfy.sd
import comfy.supported_models
import comfy.supported_models_base
import comfy.utils

from . import vae as ming_vae
from .tts import MingTTS


class MingAudioLatent(comfy.latent_formats.LatentFormat):
    latent_channels = 64
    latent_dimensions = 1


class MingTTSModel(comfy.model_base.BaseModel):
    """Holds the TTS network for ComfyUI's model management (offload, dtypes, LoRA); sampling is
    autoregressive, so it is driven by the Ming TTS node rather than KSampler."""

    def __init__(self, model_config, model_type=comfy.model_base.ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=MingTTS)


class MingTTSConfig(comfy.supported_models_base.BASE):
    unet_config = {"image_model": "ming_tts"}
    unet_extra_config = {}
    sampling_settings = {}
    latent_format = MingAudioLatent
    memory_usage_factor = 0.01
    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    def get_model(self, state_dict, prefix="", device=None):
        return MingTTSModel(self, device=device)

    def process_unet_state_dict(self, state_dict):
        # the original checkpoints also carry the VAE (audio.*), an unused text head and buffers we recompute
        for k in list(state_dict.keys()):
            if k.startswith("audio.") or k.endswith("rotary_embed.inv_freq") or k == "model.lm_head.weight":
                state_dict.pop(k)
        if self.unet_config.get("backbone") == "ling_v3":
            from .ling import stack_experts
            state_dict = stack_experts(state_dict)  # no-op for converted files
        return state_dict


def detect_ming_config(state_dict, key_prefix="", metadata=None):
    keys = state_dict.keys()
    p = key_prefix
    if f"{p}linear_proj_audio.word_embedder.weight" not in keys or f"{p}flowloss.cfm.model.x_embedder.weight" not in keys:
        return None
    if f"{p}model.model.layers.0.attention.A_log" in keys:
        backbone = {"backbone": "ling_v3", "rope_theta": float((metadata or {}).get("ming_rope_theta", 10000.0))}
    elif f"{p}model.model.layers.0.self_attn.q_proj.weight" in keys:
        backbone = {"backbone": "qwen2"}
    else:
        return None
    return {
        "image_model": "ming_tts",
        **backbone,
        "flow_depth": comfy.model_detection.count_blocks(keys, f"{p}flowloss.cfm.model.blocks." + "{}."),
        "aggregator_depth": comfy.model_detection.count_blocks(keys, f"{p}linear_proj_audio.blocks." + "{}."),
        "flow_hidden": state_dict[f"{p}flowloss.cfm.model.x_embedder.weight"].shape[0],
    }


# ---- VAE -------------------------------------------------------------------------------------

def is_ming_vae(sd):
    return sd is not None and ("encoder.cls_embed" in sd or "audio.encoder.cls_embed" in sd) and \
        ("decoder.head.out.weight" in sd or "audio.decoder.head.out.weight" in sd)


def setup_ming_vae(vae, speaker_sd):
    vae.latent_channels = ming_vae.LATENT_DIM
    vae.latent_dim = 1
    vae.output_channels = 2  # stereo in is accepted (mono is replicated) and downmixed by the encoder
    vae.pad_channel_value = "replicate"
    vae.crop_input = False  # the encoder pads, and the TTS prompt path front-pads itself
    vae.audio_sample_rate = ming_vae.SAMPLE_RATE
    vae.upscale_ratio = vae.downscale_ratio = ming_vae.HOP * ming_vae.PATCH
    vae.process_input = lambda audio: audio
    vae.process_output = lambda audio: audio
    size = comfy.model_management.dtype_size
    vae.memory_used_encode = lambda shape, dtype: (shape[-1] / ming_vae.HOP * 400_000 + 100_000_000) * size(dtype)
    vae.memory_used_decode = lambda shape, dtype: (shape[-1] * 1_500_000 + 100_000_000) * size(dtype)

    vae.ming_speaker_encoder = None
    if speaker_sd:
        from .speaker import CAMPPlus

        encoder = CAMPPlus().eval()
        encoder.load_state_dict({k: v.float() for k, v in speaker_sd.items()})
        vae.ming_speaker_encoder = encoder  # fp32 on CPU: tiny, and BatchNorm statistics need fp32


def register():
    detect = comfy.model_detection.detect_unet_config
    if not getattr(detect, "_ming", False):
        def detect_unet_config(state_dict, key_prefix, metadata=None):
            config = detect_ming_config(state_dict, key_prefix, metadata)
            if config is not None:
                return config
            return detect(state_dict, key_prefix, metadata=metadata)
        detect_unet_config._ming = True
        comfy.model_detection.detect_unet_config = detect_unet_config

    prefix = comfy.model_detection.unet_prefix_from_state_dict
    if not getattr(prefix, "_ming", False):
        def unet_prefix_from_state_dict(state_dict):
            # "model.model." is a known audio-model prefix to ComfyUI, but in Ming it is only the LLM
            if detect_ming_config(state_dict) is not None:
                return ""
            return prefix(state_dict)
        unet_prefix_from_state_dict._ming = True
        comfy.model_detection.unet_prefix_from_state_dict = unet_prefix_from_state_dict

    if not any(getattr(m, "unet_config", {}).get("image_model") == "ming_tts" for m in comfy.supported_models.models):
        comfy.supported_models.models.append(MingTTSConfig)

    init = comfy.sd.VAE.__init__
    if not getattr(init, "_ming", False):
        def vae_init(self, sd=None, device=None, config=None, dtype=None, metadata=None):
            if config is not None or not is_ming_vae(sd):
                return init(self, sd=sd, device=device, config=config, dtype=dtype, metadata=metadata)
            sd = {k[len("audio."):] if k.startswith("audio.") else k: v for k, v in sd.items()}
            speaker_sd = {k[len("speaker_encoder."):]: sd.pop(k) for k in list(sd) if k.startswith("speaker_encoder.")}
            sd.pop("decoder.head.istft.window", None)
            # The stock config path builds AutoencoderKL(**config["params"]); for this one call it
            # builds the Ming VAE instead, and ComfyUI's own device/dtype/patcher setup does the rest.
            autoencoder = comfy.sd.AutoencoderKL
            comfy.sd.AutoencoderKL = lambda **params: ming_vae.MingAudioVAE()
            try:
                init(self, sd=sd, device=device, config={"params": {}}, dtype=dtype, metadata=metadata)
            finally:
                comfy.sd.AutoencoderKL = autoencoder
            setup_ming_vae(self, speaker_sd)
        vae_init._ming = True
        comfy.sd.VAE.__init__ = vae_init

    load_te = comfy.sd.load_text_encoder_state_dicts
    if not getattr(load_te, "_ming", False):
        def load_text_encoder_state_dicts(state_dicts=[], embedding_directory=None, clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION, model_options={}, disable_dynamic=False):
            from . import textgen
            if len(state_dicts) != 1 or not textgen.is_thinker(state_dicts[0]):
                return load_te(state_dicts, embedding_directory=embedding_directory, clip_type=clip_type, model_options=model_options, disable_dynamic=disable_dynamic)
            quant = comfy.utils.detect_layer_quantization(state_dicts[0], "")
            # build on the offload device: at 9-17 GB, the generation-time load decides what fits on the GPU
            model_options = {"initial_device": comfy.model_management.text_encoder_offload_device(), **model_options}
            if quant is not None:
                model_options["quantization_metadata"] = quant
            return comfy.sd.CLIP(textgen.clip_target(), embedding_directory=embedding_directory, parameters=comfy.utils.calculate_parameters(state_dicts[0]),
                                 state_dict=state_dicts, model_options=model_options, disable_dynamic=disable_dynamic)
        load_text_encoder_state_dicts._ming = True
        comfy.sd.load_text_encoder_state_dicts = load_text_encoder_state_dicts

    logging.info("Ming-3.0: registered TTS with Load Diffusion Model, the AudioVAE with Load VAE and audio-tiny with Load CLIP")
