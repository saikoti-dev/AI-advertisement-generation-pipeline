import os

import numpy as np
import scipy.io.wavfile
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration


def generate_music_from_text(
    prompt: str,
    duration_seconds: int = 15,
    output_filename: str = "generated_music.wav",
) -> str:
    """
    Generates music from a text prompt using Facebook's MusicGen model.
    Downloads the model automatically on first run.

    Args:
        prompt:           Text description of the desired music.
        duration_seconds: How many seconds of audio to generate.
        output_filename:  Path to save the output .wav file.

    Returns:
        Absolute path to the saved audio file.
    """
    print(f"Loading MusicGen for prompt: '{prompt}'…")

    model_id = "facebook/musicgen-small"
    model = MusicgenForConditionalGeneration.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)

    inputs = processor(text=[prompt], padding=True, return_tensors="pt")

    # MusicGen generates at 50 tokens/second of audio
    max_new_tokens = int(duration_seconds * 50)

    print(f"Generating {duration_seconds}s of music… (this may take a moment)")
    audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)

    sampling_rate = model.config.audio_encoder.sampling_rate
    audio_data = audio_values[0, 0].cpu().numpy()
    audio_data = (audio_data * 32767).astype(np.int16)

    scipy.io.wavfile.write(output_filename, rate=sampling_rate, data=audio_data)
    print(f"✓ Music saved to {output_filename}")

    return os.path.abspath(output_filename)
