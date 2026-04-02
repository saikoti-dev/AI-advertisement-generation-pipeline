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
from moviepy.editor import AudioFileClip, CompositeAudioClip, VideoFileClip
from moviepy.editor import concatenate_audioclips

# ── Audio mixing helper ────────────────────────────────────────────────────────


def _align_audio(file_path: str, target_duration: float, output_path: str) -> str:
    """
    Load audio from file_path, trim or pad with silence to exactly target_duration.
    Returns path to the aligned audio file.
    """
    import numpy as np
    from moviepy.audio.AudioClip import AudioArrayClip

    audio = AudioFileClip(file_path)
    if audio.duration > target_duration:
        audio = audio.subclip(0, target_duration)
    elif audio.duration < target_duration:
        silence_duration = target_duration - audio.duration
        nchannels = audio.nchannels
        fps = audio.fps
        # Build a silent numpy array: shape (n_samples, nchannels)
        n_samples = int(silence_duration * fps)
        silent_array = np.zeros((n_samples, nchannels), dtype=np.float32)
        silence = AudioArrayClip(silent_array, fps=fps)
        # Use concatenate_audioclips — CompositeAudioClip doesn't carry .fps,
        # which causes write_audiofile() to fail with an AttributeError.
        audio = concatenate_audioclips([audio, silence])
    audio.write_audiofile(output_path)
    audio.close()
    return output_path
    
    
    
def _mix_audio(voice_path: str, music_path: str, output_path: str, target_duration: float) -> str:
    """
    Align both audio clips to target_duration, mix voice (full volume) with music (0.25),
    and write the result.
    """
    # Align both to target duration
    aligned_voice = _align_audio(voice_path, target_duration, output_path.replace(".wav", "_voice.wav"))
    aligned_music = _align_audio(music_path, target_duration, output_path.replace(".wav", "_music.wav"))

    voice = AudioFileClip(aligned_voice)
    music = AudioFileClip(aligned_music).volumex(0.25)
    mixed = CompositeAudioClip([voice, music])
    mixed.write_audiofile(output_path)
    voice.close()
    music.close()
    # Clean up temporary aligned files if needed
    os.remove(aligned_voice)
    os.remove(aligned_music)
    return output_path


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

    # ── Step 1: Generate scene prompts ────────────────────────────────────────
    print("\n── Step 1 / 5 : Generating scene prompts with LLaMA ──────────────")
    prompts = generate_prompts(product, num_scenes)

    # ── Defaults (after prompts exist so voiceover can reference them) ────────
    if voiceover_text is None:
        voiceover_text = generate_voiceover_script(product, scene_prompts=prompts)

    if music_prompt is None:
        music_prompt = generate_music_prompt(product, duration_secs=num_scenes * 7, mood="cinematic epic")

     # Step 2: Render video scenes
    print("\n── Step 2 / 5 : Rendering scenes with CogVideoX ──────────────────")
    scene_paths = render_scenes(prompts, output_dir=work_dir / "scenes")

    # ── NEW: Get total video duration ───────────────────────────────────────
    total_duration = 0.0
    for sp in scene_paths:
        clip = VideoFileClip(str(sp))
        total_duration += clip.duration
        clip.close()
    print(f"Total video duration: {total_duration:.2f} seconds")

    # Step 3: Generate voiceover
    print("\n── Step 3 / 5 : Generating voiceover ─────────────────────────────")
    voice_path = generate_voiceover(
        text=voiceover_text,
        output_file=str(work_dir / "voiceover.wav"),
        model=tts_model,
    )

    # Step 4: Generate background music
    print("\n── Step 4 / 5 : Generating background music with MusicGen ────────")
    music_path = generate_music_from_text(
        prompt=music_prompt,
        duration_seconds=total_duration,      # Use exact video duration
        output_filename=str(work_dir / "background_music.wav"),
    )

    # Mix audio using the aligned clips
    print("\n  Mixing voiceover and background music…")
    mixed_audio = _mix_audio(
        voice_path,
        music_path,
        output_path=str(work_dir / "mixed_audio.wav"),
        target_duration=total_duration,
    )

    # Step 5: Stitch everything together
    print("\n── Step 5 / 5 : Compositing final video ──────────────────────────")
    final_path = create_video_with_audio(
        video_files=scene_paths,
        audio_file=mixed_audio,
        output_filename=str(work_dir / output_filename),
        target_duration=total_duration,   # Pass target to video_creator
    )

    print(f"\n🎬  Advertisement ready → {final_path}")
    return final_path
