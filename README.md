# Ad Generator

AI-powered advertisement generator. Give it a product description, get a full video ad with AI-generated scenes, voiceover, and background music.

## Pipeline

```
Product description
       │
       ▼
 scene_generator.py   ← LLaMA 3.2 generates cinematic scene prompts
       │
       ▼
 video_generator.py   ← CogVideoX-5b renders each scene as a video clip
       │
       ▼
    tts.py            ← edge-tts (free) / ElevenLabs / Azure generates voiceover
       │
       ▼
 music_generator.py   ← MusicGen generates background music
       │
       ▼
 video_creator.py     ← MoviePy stitches clips, mixes audio → final .mp4
       │
       ▼
   app.py (Streamlit) ← serves the UI
```

## Setup

```bash
pip install -r requirements.txt
```

> **GPU requirement**: CogVideoX-5b needs ~16 GB VRAM. LLaMA 3.2-1B and MusicGen are much lighter.

## Run

```bash
streamlit run app.py
```

## Optional: Better TTS

By default the app uses **edge-tts** (free, no key needed).

For higher quality:
- **ElevenLabs**: get a free key at https://elevenlabs.io and paste it in Advanced options
- **Azure TTS**: get a key at https://azure.microsoft.com/en-us/services/cognitive-services/speech-services/ and enter key + region in Advanced options

## File structure

```
ad_generator/
├── app.py               # Streamlit UI
├── backend.py           # Pipeline orchestrator
├── scene_generator.py   # LLaMA scene prompt generation
├── video_generator.py   # CogVideoX video rendering
├── music_generator.py   # MusicGen background music
├── tts.py               # Emotional TTS (edge-tts / ElevenLabs / Azure)
├── video_creator.py     # MoviePy stitching & audio mixing
└── requirements.txt
```
