"""
music_generator.py
------------------
Generates background music using facebook/musicgen-small.
Runs on CPU inside a subprocess — no VRAM used, GPU stays free for video.
"""

from __future__ import annotations
import os
from _proc import run_task


def generate_music_from_text(
    prompt: str,
    duration_seconds: int = 15,
    output_filename: str = "generated_music.wav",
    model_id: str = "facebook/musicgen-medium",  # NEW: change default to "medium"
) -> str:
    print(f"[MusicGen] prompt='{prompt[:60]}…'  duration={duration_seconds}s")

    result = run_task("music_gen", {
        "prompt":          prompt,
        "duration_seconds": duration_seconds,
        "output_filename":  os.path.abspath(output_filename),
        "model_id":        model_id,  # NEW: pass model_id to subprocess
    })

    print(f"[MusicGen] ✓ Music ready → {result['path']}")
    return result["path"]
