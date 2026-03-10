"""
backend.py
----------
Orchestrates the full ad-generation pipeline:

  1. scene_generator  → LLaMA produces CogVideoX scene prompts
  2. video_generator  → CogVideoX renders each scene as a clip
  3. tts              → edge-tts (or ElevenLabs / Azure) generates a voiceover
  4. music_generator  → MusicGen creates background music
  5. video_creator    → MoviePy stitches clips + mixes audio tracks → final .mp4

Optional API keys can be passed in; if omitted the pipeline degrades gracefully
to free local models (edge-tts for voice, musicgen-small for music).
"""

from __future__ import annotations

import os
from pathlib import Path

from scene_generator import generate_prompts
from video_generator import render_scenes
from music_generator import generate_music_from_text
from tts import generate_voiceover
from video_creator import create_video_with_audio


# ── Audio mixing helper ────────────────────────────────────────────────────────

def _mix_audio(voice_path: str, music_path: str, output_path: str = "mixed_audio.wav") -> str:
    """
    Blends voiceover (full volume) with background music (lowered to 25%).
    Returns path to the mixed audio file.
    Falls back to voice_path alone if mixing fails.
    """
    try:
        from moviepy.editor import AudioFileClip, CompositeAudioClip

        voice = AudioFileClip(voice_path)
        music = AudioFileClip(music_path).volumex(0.25)

        # Trim music to voice length
        if music.duration > voice.duration:
            music = music.subclip(0, voice.duration)

        mixed = CompositeAudioClip([voice, music])
        mixed.write_audiofile(output_path)
        voice.close()
        music.close()
        return os.path.abspath(output_path)

    except Exception as e:
        print(f"  ⚠ Audio mixing failed ({e}), using voiceover only.")
        return voice_path


# ── Main entry point ────────────────────────────────────────────────────────────

def generate_ad(
    product: str,
    num_scenes: int = 2,
    voiceover_text: str | None = None,
    music_prompt: str | None = None,
    output_filename: str = "final_ad.mp4",
    work_dir: str | Path = "ad_workspace",
    tts_model: str = "speecht5",   # "speecht5" | "bark" | "mms"
) -> str:
    """
    Full pipeline: product description → final advertisement .mp4

    Args:
        product:              Short description / name of the product.
        num_scenes:           How many video clips to render (default 2).
        voiceover_text:       The script to speak. Auto-generated from product if None.
        music_prompt:         MusicGen text prompt. Auto-generated from product if None.
        output_filename:      Name of the final output file.
        work_dir:             Folder where intermediate files are saved.
        elevenlabs_api_key:   Optional ElevenLabs key for high-quality TTS.
        azure_tts_key:        Optional Azure Speech key.
        azure_tts_region:     Required if azure_tts_key is supplied.
        emotion_style:        Azure emotion style (ignored for other TTS providers).

    Returns:
        Absolute path to the rendered .mp4 advertisement.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # ── Defaults ──────────────────────────────────────────────────────────────
    if voiceover_text is None:
        voiceover_text = (
            f"Introducing {product}. Crafted for those who demand the very best. "
            f"Experience the difference — {product}."
        )

    if music_prompt is None:
        music_prompt = (
            f"Elegant, cinematic background music for a luxury {product} advertisement. "
            "Slow build, orchestral strings, warm and aspirational."
        )

    # ── Step 1: Generate scene prompts ────────────────────────────────────────
    print("\n── Step 1 / 5 : Generating scene prompts with LLaMA ──────────────")
    prompts = generate_prompts(product, num_scenes)

    # ── Step 2: Render video scenes ───────────────────────────────────────────
    print("\n── Step 2 / 5 : Rendering scenes with CogVideoX ──────────────────")
    scene_paths = render_scenes(prompts, output_dir=work_dir / "scenes")

    # ── Step 3: Generate voiceover ────────────────────────────────────────────
    print("\n── Step 3 / 5 : Generating voiceover ─────────────────────────────")
    voice_path = generate_voiceover(
        text=voiceover_text,
        output_file=str(work_dir / "voiceover.wav"),
        model=tts_model,
    )

    # ── Step 4: Generate background music ─────────────────────────────────────
    print("\n── Step 4 / 5 : Generating background music with MusicGen ────────")
    music_path = generate_music_from_text(
        prompt=music_prompt,
        duration_seconds=num_scenes * 7,   # ~7 s per scene
        output_filename=str(work_dir / "background_music.wav"),
    )

    # ── Mix voice + music ─────────────────────────────────────────────────────
    print("\n  Mixing voiceover and background music…")
    mixed_audio = _mix_audio(
        voice_path,
        music_path,
        output_path=str(work_dir / "mixed_audio.wav"),
    )

    # ── Step 5: Stitch everything together ────────────────────────────────────
    print("\n── Step 5 / 5 : Compositing final video ──────────────────────────")
    final_path = create_video_with_audio(
        video_files=scene_paths,
        audio_file=mixed_audio,
        output_filename=str(work_dir / output_filename),
    )

    print(f"\n🎬  Advertisement ready → {final_path}")
    return final_path
