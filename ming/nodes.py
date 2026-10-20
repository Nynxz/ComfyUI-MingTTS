"""Ming-3.0 TTS nodes. Models load through the stock loaders (see integration.py):
Load Diffusion Model -> MODEL, Load VAE -> VAE; audio comes out through VAE Decode Audio."""

from comfy_api.latest import IO

from . import prompts
from .pipeline import Task, make_voice, render_script, synthesize

MingVoice = IO.Custom("MING_VOICE")

PARALINGUISTIC_HELP = "Inline sound tags (Chinese or English text): " + " ".join(prompts.PARALINGUISTIC_TAGS)


def voice_options():
    voices = prompts.builtin_voices()
    return [v["ip"] for v in voices if v["lang"] == "en"] + [v["ip"] for v in voices if v["lang"] == "zh"]


def task_input():
    D = IO.DynamicCombo
    return D.Input("task", tooltip="What the model is asked to do. Each task was trained with its own prompt; controls cannot be mixed.", options=[
        D.Option("speak", []),
        D.Option("emotion", [IO.Combo.Input("emotion", options=list(prompts.EMOTIONS), default="happy")]),
        D.Option("speed / pitch / volume", [
            IO.Combo.Input("speed", options=["slow", "medium", "fast"], default="medium"),
            IO.Combo.Input("pitch", options=["low", "medium", "high"], default="medium"),
            IO.Combo.Input("volume", options=["low", "medium", "high"], default="medium"),
        ]),
        D.Option("dialect", [IO.Combo.Input("dialect", options=list(prompts.DIALECTS), default="Cantonese")]),
        D.Option("instruction", [
            IO.String.Input("instruction", multiline=True, default="Speak in a warm, intimate storytelling voice, unhurried.",
                            tooltip="Free-form delivery direction (tone, pace, emphasis). Works without a reference voice; with one it keeps that voice."),
        ]),
        D.Option("voice design", [
            IO.String.Input("description", multiline=True, default="A calm, deep male voice, middle-aged, speaking slowly with a slight rasp.",
                            tooltip="Describe the speaker to invent. Ignores the reference voice."),
        ]),
        D.Option("built-in voice", [
            IO.Combo.Input("voice", options=voice_options(), default="US_Emma",
                           tooltip="484 built-in voices (5 English, the rest Chinese, many with _emotion variants). Ignores the reference voice."),
        ]),
    ])


KINDS = {"speak": "speak", "emotion": "emotion", "speed / pitch / volume": "basic", "dialect": "dialect",
         "instruction": "instruction", "voice design": "design", "built-in voice": "builtin"}


LANGUAGE_HELP = ("The language of the text, as the prompt's 'Target language' line. Needed for Japanese, Korean, German, French "
                 "and Spanish; auto is fine for Chinese and English. The tiny only uses it for the speak task.")


def language_inputs():
    return [
        IO.Combo.Input("language", options=list(prompts.LANGUAGES), default="auto", tooltip=LANGUAGE_HELP),
        IO.String.Input("language_text", force_input=True, optional=True,
                        tooltip="Overrides language when connected: a name or code (English, en, Japanese, ja...), "
                                "e.g. Ming Transcribe's language output."),
    ]


def resolve_language(language, language_text=None):
    return prompts.resolve_language(language_text if language_text and language_text.strip() else language)


def resolve_task(task, language=None):
    controls = {k: v for k, v in task.items() if k != "task"}
    return Task(KINDS[task["task"]], controls, language)


class MingReferenceVoice(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MingReferenceVoice",
            display_name="Ming Reference Voice",
            category="audio/ming",
            description="Turns a short clip into a voice for Ming TTS. With its transcript the model continues the clip "
                        "acoustically (best similarity, 'speak' task); without one only the speaker embedding is used.",
            inputs=[
                IO.Vae.Input("vae", tooltip="The Ming AudioVAE (ming-3.0_audio_vae.safetensors), which also holds the speaker encoder."),
                IO.Audio.Input("audio", tooltip="3-15 s of clean speech from one speaker."),
                IO.String.Input("transcript", multiline=True, default="", tooltip="Exactly what is said in the clip. Leave empty to use the speaker embedding only."),
                IO.Float.Input("max_seconds", default=20.0, min=0.0, max=60.0, step=0.5, advanced=True,
                               tooltip="Use only the start of the clip (0 = all). The clip counts against the 512-step context."),
            ],
            outputs=[MingVoice.Output("voice")],
        )

    @classmethod
    def execute(cls, vae, audio, transcript, max_seconds) -> IO.NodeOutput:
        return IO.NodeOutput(make_voice(vae, audio, transcript, max_seconds))


class MingTTS(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MingTTS",
            display_name="Ming TTS",
            category="audio/ming",
            description="Generates speech latents with a Ming-3.0 TTS model; decode them with VAE Decode Audio and the Ming VAE.",
            inputs=[
                IO.Model.Input("model"),
                IO.String.Input("text", multiline=True, default="The old radio crackled softly as the storm moved across the valley.",
                                tooltip=PARALINGUISTIC_HELP),
                task_input(),
                *language_inputs(),
                MingVoice.Input("voice", optional=True, tooltip="From Ming Reference Voice. Used by speak, emotion, speed/pitch/volume, dialect and instruction."),
                IO.Vae.Input("vae", optional=True, tooltip="Optional Ming VAE: trims each text fragment's trailing silence before joining, like the reference."),
                IO.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
                IO.Float.Input("cfg", default=2.0, min=0.0, max=10.0, step=0.1, tooltip="Guidance of the flow head against a null LLM state (reference: 2)."),
                IO.Int.Input("flow_steps", default=10, min=1, max=64, advanced=True, tooltip="Flow integration steps per 80 ms latent (reference: 10)."),
                IO.Float.Input("temperature", default=0.0, min=0.0, max=2.0, step=0.05, advanced=True,
                               tooltip="Extra noise injected while integrating each latent (0 = the reference's deterministic flow)."),
                IO.Float.Input("sigma", default=0.25, min=0.0, max=2.0, step=0.01, advanced=True, tooltip="Scale of the temperature noise."),
                IO.Boolean.Input("split_text", default=True, advanced=True,
                                 tooltip="Split long text into fragments generated separately (the model's context is ~40 s per call)."),
                IO.Int.Input("max_fragment_length", default=50, min=10, max=400, advanced=True,
                             tooltip="Fragment size in 'semantic length' units (a Chinese character or English word each)."),
                IO.Boolean.Input("continuity", default=True, advanced=True,
                                 tooltip="speak task: each fragment continues the previous one's audio (fed back as the acoustic "
                                         "prompt with its text), so pace and tone carry across long text instead of resetting."),
            ],
            outputs=[IO.Latent.Output()],
        )

    @classmethod
    def execute(cls, model, text, task, language, seed, cfg, flow_steps, temperature, sigma, split_text, max_fragment_length, continuity=True,
                voice=None, vae=None, language_text=None) -> IO.NodeOutput:
        latents = synthesize(model, text, resolve_task(task, resolve_language(language, language_text)), voice=voice, seed=seed, cfg=cfg, flow_steps=flow_steps, temperature=temperature,
                             sigma=sigma, split=split_text, max_fragment_length=max_fragment_length, trim_vae=vae, continuity=continuity)
        return IO.NodeOutput({"samples": latents, "type": "audio"})


SCRIPT_EXAMPLE = """# One line per utterance: <voice slot>: text   or   <slot> (mods): text
1: Did you hear that? Something is moving in the attic.
2 (fearful): Please tell me it's just the wind.
1 (whisper it, slowly): Stay here. I'll go and look.
[pause 0.8]
2: [Laughter] It was the cat. It was only the cat!"""


class MingScript(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MingScript",
            display_name="Ming Script",
            category="audio/ming",
            description="Speaks a multi-line script with up to four voices. Each line is generated on its own and cached, "
                        "so editing one line re-renders only that line. Line syntax: '2 (sad): text'. mods are an emotion, "
                        "'speed=fast pitch=low volume=high', 'voice=US_James', 'design=<description>', or free-form "
                        "direction; '[pause 1.5]' inserts silence.",
            inputs=[
                IO.Model.Input("model"),
                IO.Vae.Input("vae", tooltip="The Ming VAE: lines are decoded, trimmed and joined as audio."),
                IO.String.Input("script", multiline=True, default=SCRIPT_EXAMPLE, tooltip=PARALINGUISTIC_HELP),
                *language_inputs(),
                MingVoice.Input("voice_1", optional=True),
                MingVoice.Input("voice_2", optional=True),
                MingVoice.Input("voice_3", optional=True),
                MingVoice.Input("voice_4", optional=True),
                IO.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=False,
                             tooltip="Base seed; each line adds a hash of its own text, so lines keep their take when others change."),
                IO.Float.Input("cfg", default=2.0, min=0.0, max=10.0, step=0.1),
                IO.Float.Input("gap", default=0.25, min=0.0, max=5.0, step=0.05, tooltip="Silence between lines, in seconds."),
                IO.Int.Input("flow_steps", default=10, min=1, max=64, advanced=True),
                IO.Float.Input("temperature", default=0.0, min=0.0, max=2.0, step=0.05, advanced=True),
            ],
            outputs=[IO.Audio.Output(), IO.String.Output(display_name="timeline")],
        )

    @classmethod
    def execute(cls, model, vae, script, language, seed, cfg, gap, flow_steps, temperature, voice_1=None, voice_2=None, voice_3=None, voice_4=None,
                language_text=None) -> IO.NodeOutput:
        voices = {i: v for i, v in enumerate((voice_1, voice_2, voice_3, voice_4), start=1) if v is not None}
        wave, timeline = render_script(model, vae, script, voices, seed=seed, gap=gap, language=resolve_language(language, language_text),
                                       cfg=cfg, flow_steps=flow_steps, temperature=temperature)
        return IO.NodeOutput({"waveform": wave, "sample_rate": vae.audio_sample_rate}, timeline)


class MingTranscribe(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MingTranscribe",
            display_name="Ming Transcribe",
            category="audio/ming",
            description="Speech recognition with Ming-3.0-audio-tiny (loaded with Load CLIP). Gives the transcript a "
                        "Ming Reference Voice needs. For other questions about audio, use the stock TextGenerate node.",
            inputs=[
                IO.Clip.Input("clip", tooltip="ming-3.0-audio-tiny, loaded with Load CLIP (any type)."),
                IO.Audio.Input("audio", tooltip="Up to a few minutes of speech; it is encoded in 30 s chunks."),
                IO.Int.Input("max_tokens", default=512, min=16, max=8192, advanced=True),
            ],
            outputs=[IO.String.Output(display_name="transcript"), IO.String.Output(display_name="language")],
        )

    @classmethod
    def execute(cls, clip, audio, max_tokens) -> IO.NodeOutput:
        from comfy import model_management
        from .textgen import ASR_PROMPT

        tokens = clip.tokenize(ASR_PROMPT, audio=audio)
        model = clip.cond_stage_model
        if not hasattr(model, "thinker"):
            raise ValueError("Ming Transcribe needs ming-3.0-audio-tiny loaded with Load CLIP")
        model.reset_clip_options()
        clip.load_model(tokens)
        model.set_clip_options({"execution_device": clip.patcher.load_device})
        with model_management.cuda_device_context(clip.patcher.load_device):
            ids = model.generate(tokens, do_sample=False, max_length=max_tokens, no_repeat_ngram_size=10)  # the validated ASR setting
        text = clip.decode(ids).strip()
        language, sep, transcript = text.partition("\t")
        if not sep:  # the prompt spells the separator as a literal `\t`; accept that too
            language, sep, transcript = text.partition("\\t")
        return IO.NodeOutput((transcript if sep else text).strip(), language.strip() if sep else "")


NODES = [MingReferenceVoice, MingTTS, MingScript, MingTranscribe]
