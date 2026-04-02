"""
tts.py
------
Text-to-Speech — three free local models, no API keys needed.

  speecht5  — fast, clean, default
  bark      — expressive, supports [laughs] [sighs] etc., slower
  mms       — lightest and fastest

Each model runs in a subprocess — GPU is fully free afterward.
"""

from __future__ import annotations
import os
from _proc import run_task


def generate_voiceover(
    text: str,
    output_file: str = "voiceover.wav",
    model: str = "speecht5",
    bark_voice_preset: str = "v2/en_speaker_6",
    speecht5_speaker_index: int = 7306,
    elevenlabs_api_key: str | None = None,
    azure_key: str | None = None,
    azure_region: str | None = None,
    emotion_style: str = "cheerful",
) -> str:
    if not output_file.endswith(".wav"):
        output_file = output_file.rsplit(".", 1)[0] + ".wav"

    print(f"[TTS] model={model!r}  output={output_file}")

    result = run_task("tts_gen", {
        "model":       model,
        "text":        text,
        "output_file": os.path.abspath(output_file),
    })

    print(f"[TTS] ✓ Voiceover ready → {result['path']}")
    return result["path"]
