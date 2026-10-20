"""ComfyUI-Ming: native Ming-3.0 speech synthesis (inclusionAI Ming-3.0-tts).

Importing the pack registers the TTS model with Load Diffusion Model and the AudioVAE with
Load VAE, so stock loaders and VAE Decode Audio work; two nodes drive generation:
Ming Reference Voice (voice cloning) and Ming TTS.
"""

from typing_extensions import override

from comfy_api.latest import ComfyExtension

from .ming.integration import register

register()

from .ming.nodes import NODES  # noqa: E402


class MingExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return NODES


async def comfy_entrypoint():
    return MingExtension()
