"""
Emotional Text-to-Speech — 100% free, fully local, no API keys needed.
All models download automatically from Hugging Face on first run.

Three options ranked by quality vs. speed:

  1. SpeechT5   (microsoft/speecht5_tts)  — DEFAULT
                Fast, lightweight (~500 MB), runs on CPU fine.
                Good clean quality, neutral delivery.

  2. Bark       (suno/bark)               — EXPRESSIVE
                Slow but very natural. Supports emotion cues in text:
                  [laughs], [sighs], [whispers], [gasps], [clears throat]
                  ♪ for singing, CAPS for emphasis, ... for pause
                Needs ~6 GB VRAM for full speed; falls back to CPU.

  3. MMS-TTS   (facebook/mms-tts-eng)    — LIGHTWEIGHT
                Smallest and fastest. Decent quality. Good for testing.

Install:
    pip install transformers torch scipy datasets soundfile
"""

from __future__ import annotations

import os
import numpy as np
import scipy.io.wavfile
import torch


# ── Option 1: SpeechT5 (default — fast, reliable) ────────────────────────────

def speecht5_generate(
    text: str,
    output_file: str = "voiceover.wav",
    speaker_embedding_index: int = 7306,
) -> str:
    """
    Generate speech with microsoft/speecht5_tts.

    speaker_embedding_index controls the voice character.
    Values 0–7499 all give slightly different voices from the CMU Arctic dataset.
    Some good ones to try: 7306 (warm male), 1000 (female), 4500 (neutral).

    Returns absolute path to the saved .wav file.
    """
    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
    from datasets import load_dataset

    print("Loading SpeechT5 model…")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model     = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
    vocoder   = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

    print("Loading speaker embedding…")
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(
        embeddings_dataset[speaker_embedding_index]["xvector"]
    ).unsqueeze(0).to(device)

    inputs = processor(text=text, return_tensors="pt").to(device)

    print(f"Generating speech with SpeechT5… (device: {device})")
    with torch.no_grad():
        speech = model.generate_speech(
            inputs["input_ids"],
            speaker_embeddings,
            vocoder=vocoder,
        )

    speech_np = speech.cpu().numpy()
    speech_int16 = (speech_np * 32767).astype(np.int16)
    scipy.io.wavfile.write(output_file, rate=16000, data=speech_int16)

    print(f"✓ SpeechT5 audio saved → {output_file}")
    return os.path.abspath(output_file)


# ── Option 2: Bark (most expressive, emotion-aware) ───────────────────────────

def bark_generate(
    text: str,
    output_file: str = "voiceover.wav",
    voice_preset: str = "v2/en_speaker_6",
) -> str:
    """
    Generate expressive emotional speech with suno/bark.

    Emotion cues you can embed directly in text:
        [laughs]          [sighs]         [whispers]
        [gasps]           [clears throat] [music]
        CAPS for emphasis  ... for pause

    Example:
        "Introducing our new product. [sighs] It's... truly remarkable."

    voice_preset options (en):
        v2/en_speaker_0  through  v2/en_speaker_9

    Returns absolute path to the saved .wav file.
    """
    from transformers import AutoProcessor, BarkModel

    print("Loading Bark model… (first run downloads ~5 GB)")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained("suno/bark")
    model     = BarkModel.from_pretrained(
        "suno/bark",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    if device == "cuda":
        model.enable_cpu_offload()

    inputs = processor(text, voice_preset=voice_preset)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print(f"Generating speech with Bark… (device: {device}, this may take a minute)")
    with torch.no_grad():
        audio_array = model.generate(**inputs)

    audio_np = audio_array.cpu().numpy().squeeze()
    audio_int16 = (audio_np * 32767).astype(np.int16)
    scipy.io.wavfile.write(output_file, rate=24000, data=audio_int16)

    print(f"✓ Bark audio saved → {output_file}")
    return os.path.abspath(output_file)


# ── Option 3: MMS-TTS (lightest, fastest) ─────────────────────────────────────

def mms_generate(
    text: str,
    output_file: str = "voiceover.wav",
) -> str:
    """
    Generate speech with facebook/mms-tts-eng — the smallest option.
    Great for quick tests or CPU-only machines.

    Returns absolute path to the saved .wav file.
    """
    from transformers import VitsModel, AutoTokenizer

    print("Loading MMS-TTS model…")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    model     = VitsModel.from_pretrained("facebook/mms-tts-eng").to(device)

    inputs = tokenizer(text, return_tensors="pt").to(device)

    print(f"Generating speech with MMS-TTS… (device: {device})")
    with torch.no_grad():
        output = model(**inputs)

    audio_np = output.waveform[0].cpu().float().numpy()
    audio_int16 = (audio_np * 32767).astype(np.int16)

    sampling_rate = model.config.sampling_rate
    scipy.io.wavfile.write(output_file, rate=sampling_rate, data=audio_int16)

    print(f"✓ MMS-TTS audio saved → {output_file}")
    return os.path.abspath(output_file)


# ── Smart wrapper ─────────────────────────────────────────────────────────────

def generate_voiceover(
    text: str,
    output_file: str = "voiceover.wav",
    model: str = "speecht5",
    bark_voice_preset: str = "v2/en_speaker_6",
    speecht5_speaker_index: int = 7306,
    # Legacy params kept for compatibility — ignored
    elevenlabs_api_key: str | None = None,
    azure_key: str | None = None,
    azure_region: str | None = None,
    emotion_style: str = "cheerful",
) -> str:
    """
    Unified voiceover generator using free HuggingFace models only.

    Args:
        text:                    The script to speak.
        output_file:             Where to save the .wav.
        model:                   Which HF model to use:
                                   "speecht5" — fast, clean (default)
                                   "bark"     — slow, expressive, emotion cues in text
                                   "mms"      — fastest, lightest
        bark_voice_preset:       Voice character for Bark.
        speecht5_speaker_index:  Speaker embedding index for SpeechT5 (0–7499).

    Tip — adding emotion with Bark:
        Pass text like:
          "Introducing {product}. [sighs] Crafted for those who demand the very best."
    """
    if not output_file.endswith(".wav"):
        output_file = output_file.rsplit(".", 1)[0] + ".wav"

    if model == "bark":
        return bark_generate(text, output_file, voice_preset=bark_voice_preset)
    elif model == "mms":
        return mms_generate(text, output_file)
    else:
        return speecht5_generate(text, output_file, speaker_embedding_index=speecht5_speaker_index)
