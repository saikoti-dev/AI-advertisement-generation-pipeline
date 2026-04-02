# Ad Generator

To execute the stack run the runner.ipynb file

AI-powered video advertisement generator. Give it a product description and it produces a fully rendered `.mp4` ad — cinematic scenes, voiceover, and background music — entirely from local models, no paid API keys required.

---

## How it works

```
Product description
       │
       ▼
 scene_generator.py   ─── Qwen2.5-1.5B generates cinematic scene prompts
       │
       ▼
 video_generator.py   ─── CogVideoX-2b/5b renders each scene as a video clip
       │
       ▼
      tts.py          ─── SpeechT5 / Bark / MMS generates voiceover audio
       │
       ▼
 music_generator.py   ─── MusicGen creates background music
       │
       ▼
 video_creator.py     ─── MoviePy stitches clips, mixes audio → final .mp4
       │
       ▼
     app.py           ─── Streamlit UI serves the whole thing
```

Each model runs in its **own subprocess** — the GPU is fully freed between steps, so even a 16 GB card can run the full pipeline sequentially without OOM errors.

---

## Requirements

### Hardware
| Component | Minimum |
|-----------|---------|
| GPU VRAM | 16 GB (for CogVideoX-5b) — 8 GB works with the `-2b` model |
| RAM | 16 GB |
| Disk | ~30 GB free (model weights) |

### Software
- Python 3.10+
- CUDA 11.8+ (for GPU inference)
- `ffmpeg` accessible on your `PATH`

---

## Installation

```bash
git clone https://github.com/your-username/ad-generator
cd ad-generator
pip install -r requirements.txt
```

> On first run, model weights are downloaded automatically from Hugging Face and cached in `~/.cache/huggingface/`.

---

## Running

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

1. Type a product description in the text box
2. Expand **Advanced options** to tweak scene count, TTS model, or enter optional API keys
3. Click **✦ Generate Advertisement**
4. Watch the live step tracker and log panel — the pipeline prints progress for every stage
5. Download the finished `.mp4` when the green result panel appears

You can also click **↻ Continue from last step** to resume if a run was interrupted mid-pipeline.

---

## Python API

You can drive the pipeline directly from code without the UI:

```python
from backend import generate_ad

path = generate_ad(
    product="A noise-cancelling travel headphone with 40-hour battery life",
    num_scenes=3,
    tts_model="speecht5",          # or "bark" / "mms"
    output_filename="headphone_ad.mp4",
    work_dir="my_workspace",
)
print(f"Done → {path}")
```

### `generate_ad` parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `product` | `str` | — | Product name / description fed to every model |
| `num_scenes` | `int` | `2` | Number of video clips to render |
| `voiceover_text` | `str \| None` | `None` | Script to speak; auto-generated if omitted |
| `music_prompt` | `str \| None` | `None` | MusicGen prompt; auto-generated if omitted |
| `output_filename` | `str` | `"final_ad.mp4"` | Output filename inside `work_dir` |
| `work_dir` | `str \| Path` | `"ad_workspace"` | Folder for all intermediate and final files |
| `tts_model` | `str` | `"speecht5"` | TTS engine — `"speecht5"`, `"bark"`, or `"mms"` |

---

## TTS models

All three TTS options are free and run locally:

| Model | Speed | Quality | Notes |
|-------|-------|---------|-------|
| `speecht5` | Fast | Clean, neutral | Good default |
| `bark` | Slow | Expressive | Supports `[laughs]`, `[sighs]`, emotions |
| `mms` | Fastest | Lightweight | Best for low-resource environments |

---

## Models used

| Step | Model | Source |
|------|-------|--------|
| Scene prompts | `Qwen/Qwen2.5-1.5B-Instruct` | Hugging Face |
| Voiceover script | `Qwen/Qwen2.5-1.5B-Instruct` | Hugging Face |
| Video generation | `THUDM/CogVideoX-2b` (default) | Hugging Face |
| TTS | `microsoft/speecht5_tts` / `suno/bark` / `facebook/mms-tts-eng` | Hugging Face |
| Background music | `facebook/musicgen-medium` | Hugging Face |

The video model can be swapped to `THUDM/CogVideoX-5b` for higher quality if you have ≥16 GB VRAM — pass `model_id="THUDM/CogVideoX-5b"` to `render_scenes()` or set it in `video_generator.py`.

---

## File structure

```
ad-generator/
├── app.py                  # Streamlit UI — step tracker, live log panel, download
├── backend.py              # Pipeline orchestrator — calls all stages in order
├── scene_generator.py      # LLM scene prompt + voiceover script generation
├── video_generator.py      # CogVideoX rendering, one subprocess per scene
├── tts.py                  # SpeechT5 / Bark / MMS voiceover generation
├── music_generator.py      # MusicGen background music
├── video_creator.py        # MoviePy clip stitching + audio mixing
├── pipeline_logger.py      # Captures stdout/stderr + tqdm into a live log list
├── _proc.py                # Subprocess launcher — streams output, returns JSON result
├── _subprocess_runner.py   # Entry point run inside each model subprocess
└── requirements.txt
```

### Intermediate files (inside `ad_workspace/`)

```
ad_workspace/
├── scenes/
│   ├── scene_01.mp4
│   └── scene_02.mp4
├── voiceover.wav
├── background_music.wav
├── mixed_audio.wav
└── final_ad.mp4            ← final output
```

---

## Troubleshooting

**Exit code -9 / pipeline crashes mid-way**
The Linux OOM killer terminated the subprocess — you've run out of system RAM. Restart your runtime to reclaim memory, then rerun. Using `CogVideoX-2b` instead of `5b` reduces peak VRAM significantly.

**`CompositeAudioClip has no attribute 'fps'`**
Upgrade to the latest version of this repo — this was a MoviePy padding bug that has been fixed in `backend.py`.

**Slow first run**
Model weights are being downloaded. Subsequent runs use the local cache and are much faster.

**`ffmpeg` not found**
Install ffmpeg and ensure it's on your PATH:
```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

---

## License

MIT
