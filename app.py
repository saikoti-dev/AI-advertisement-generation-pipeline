import os
import threading
import time
import json
from pathlib import Path

import streamlit as st

# ── Import pipeline modules ───────────────────────────────────────────────────
from pipeline_logger import PipelineLogger
from scene_generator import generate_prompts
from video_generator import render_scenes
from tts import generate_voiceover
from music_generator import generate_music_from_text
from video_creator import create_video_with_audio
from moviepy.editor import VideoFileClip


# ── Audio alignment helpers (copied from backend.py) ─────────────────────────
from moviepy.editor import AudioFileClip, CompositeAudioClip
import numpy as np

def _align_audio(file_path: str, target_duration: float, output_path: str) -> str:
    from moviepy.audio.AudioClip import AudioArrayClip
    audio = AudioFileClip(file_path)
    if audio.duration > target_duration:
        audio = audio.subclip(0, target_duration)
    elif audio.duration < target_duration:
        silence_duration = target_duration - audio.duration
        nchannels = audio.nchannels
        fps = audio.fps
        n_samples = int(silence_duration * fps)
        silent_array = np.zeros((n_samples, nchannels), dtype=np.float32)
        silence = AudioArrayClip(silent_array, fps=fps)
        audio = CompositeAudioClip([audio, silence.set_start(audio.duration)])
        audio.fps = fps
    audio.write_audiofile(output_path)
    audio.close()
    return output_path

def _mix_audio(voice_path: str, music_path: str, output_path: str, target_duration: float) -> str:
    aligned_voice = _align_audio(voice_path, target_duration, output_path.replace(".wav", "_voice.wav"))
    aligned_music = _align_audio(music_path, target_duration, output_path.replace(".wav", "_music.wav"))
    voice = AudioFileClip(aligned_voice)
    music = AudioFileClip(aligned_music).volumex(0.25)
    mixed = CompositeAudioClip([voice, music])
    mixed.write_audiofile(output_path)
    voice.close()
    music.close()
    os.remove(aligned_voice)
    os.remove(aligned_music)
    return output_path



# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Ad Generator", layout="centered")

# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    "gradient_index": 0,
    "generating": False,
    "done": False,
    "video_path": None,
    "error": None,
    "progress_msg": "Starting…",
    "current_step": 0,
    "log_lines": [],
    "elevenlabs_key": "",
    "azure_key": "",
    "azure_region": "",
    "num_scenes": 2,
    "voiceover_text": "",
    "music_prompt": "",
    "_done_event": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Gradient palette ──────────────────────────────────────────────────────────
gradients = [
    ("#5a189a", "#9b5de5", "#d8b4fe"),
    ("#0077b6", "#00b4d8", "#90e0ef"),
    ("#b5451b", "#f4845f", "#ffd1ba"),
    ("#1a7a4a", "#2dc653", "#a8f0c6"),
    ("#8b0000", "#e63946", "#ffb3b8"),
]
gi = st.session_state.gradient_index
c1, c2, c3 = gradients[gi]
gradients_js = str([[a, b, c] for a, b, c in gradients])

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Syne:wght@700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

header[data-testid="stHeader"] {{
    display: none !important; height: 0 !important; min-height: 0 !important;
    padding: 0 !important; margin: 0 !important; overflow: hidden !important;
}}
[data-testid="stToolbar"],footer,#MainMenu,
[data-testid="stDecoration"],[data-testid="stStatusWidget"] {{ display: none !important; }}
[data-testid="stAppViewContainer"] > .main > .block-container {{ padding-top: 2rem !important; }}
:root {{ --c1: {c1}; --c2: {c2}; --c3: {c3}; }}
html, body, [data-testid="stAppViewContainer"], .stApp {{
    background: #0a0a12 !important; font-family: 'DM Sans', sans-serif !important;
}}
.blob-wrap {{ position: fixed; inset: 0; z-index: 0; overflow: hidden; pointer-events: none; }}
.blob {{ position: absolute; border-radius: 50%; filter: blur(110px); opacity: 0.55; animation: drift 8s ease-in-out infinite alternate; }}
.blob-a {{ width: 680px; height: 680px; top: -180px; left: -180px; background: radial-gradient(circle, {c1}, transparent 70%); animation-delay: 0s; }}
.blob-b {{ width: 600px; height: 600px; bottom: -160px; right: -160px; background: radial-gradient(circle, {c2}, transparent 70%); animation-delay: -3s; }}
.blob-c {{ width: 500px; height: 500px; top: 40%; left: 40%; background: radial-gradient(circle, {c3}, transparent 70%); opacity: 0.35; animation-delay: -5s; }}
@keyframes drift {{
    0%   {{ transform: translate(0,0)        scale(1);    }}
    33%  {{ transform: translate(30px,-30px)  scale(1.05); }}
    66%  {{ transform: translate(-20px,20px)  scale(0.97); }}
    100% {{ transform: translate(10px,10px)   scale(1.03); }}
}}
[data-testid="stAppViewContainer"] > .main {{ position: relative; z-index: 1; }}
.page-title {{
    font-family: 'Syne', sans-serif !important; font-size: 2.5rem !important; font-weight: 800 !important;
    line-height: 1.05 !important; margin: 0 0 8px !important;
    background: linear-gradient(90deg, #fff 20%, {c2} 60%, {c3} 100%) !important;
    -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important;
    background-clip: text !important; background-size: 200% auto !important; animation: shimmer 4s linear infinite !important;
}}
@keyframes shimmer {{ 0% {{ background-position: 0% center; }} 100% {{ background-position: 200% center; }} }}
.page-sub {{ color: rgba(255,255,255,0.45); font-size: 0.92rem; font-weight: 300; letter-spacing: 0.02em; margin-bottom: 32px; }}
[data-testid="stTextArea"] label, .stTextArea label {{
    color: rgba(255,255,255,0.7) !important; font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important; font-weight: 500 !important; letter-spacing: 0.04em !important; text-transform: uppercase !important;
}}
[data-testid="stTextArea"] textarea {{
    background: rgba(255,255,255,0.07) !important; backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255,255,255,0.14) !important; border-radius: 16px !important;
    color: rgba(255,255,255,0.92) !important; font-family: 'DM Sans', sans-serif !important;
    font-size: 0.97rem !important; font-weight: 300 !important; padding: 14px 16px !important;
    resize: none !important; caret-color: {c2} !important;
    transition: border 0.25s ease, box-shadow 0.25s ease !important;
}}
[data-testid="stTextArea"] textarea:focus {{
    border: 1px solid rgba(255,255,255,0.35) !important;
    box-shadow: 0 0 0 3px rgba(255,255,255,0.06) !important; outline: none !important;
}}
[data-testid="stTextArea"] textarea::placeholder {{ color: rgba(255,255,255,0.28) !important; }}
[data-testid="stTextInput"] input {{
    background: rgba(255,255,255,0.07) !important; border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 12px !important; color: rgba(255,255,255,0.88) !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.9rem !important;
}}
[data-testid="stTextInput"] label {{
    color: rgba(255,255,255,0.6) !important; font-size: 0.82rem !important; font-weight: 500 !important;
    letter-spacing: 0.04em !important; text-transform: uppercase !important;
}}
[data-testid="stNumberInput"] input {{
    background: rgba(255,255,255,0.07) !important; border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 12px !important; color: rgba(255,255,255,0.88) !important; font-family: 'DM Sans', sans-serif !important;
}}
[data-testid="stButton"] > button {{
    width: 100% !important; background: #0e0e16 !important;
    border: 1px solid rgba(255,255,255,0.15) !important; border-radius: 50px !important;
    padding: 14px 48px !important; color: rgba(255,255,255,0.88) !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 1rem !important; font-weight: 500 !important;
    letter-spacing: 0.06em !important; cursor: pointer !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border 0.2s ease !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.06) !important; margin-top: 12px !important;
}}
[data-testid="stButton"] > button:hover {{
    transform: scale(1.02) translateY(-1px) !important; background: #16161f !important;
    border: 1px solid rgba(255,255,255,0.28) !important;
    box-shadow: 0 6px 28px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.09) !important;
}}
[data-testid="stButton"] > button:active {{ transform: scale(0.98) !important; }}
hr {{ border: none !important; border-top: 1px solid rgba(255,255,255,0.08) !important; margin: 24px 0 !important; }}
[data-testid="stAlert"] {{
    background: rgba(255,255,255,0.07) !important; border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 14px !important; color: rgba(255,255,255,0.85) !important; font-family: 'DM Sans', sans-serif !important;
}}
.result-label {{ text-align: center; color: rgba(255,255,255,0.5); font-size: 0.85rem; letter-spacing: 0.05em; text-transform: uppercase; margin-top: 10px; }}
.section-label {{ color: rgba(255,255,255,0.35); font-size: 0.75rem; letter-spacing: 0.1em; text-transform: uppercase; margin: 24px 0 8px; font-weight: 500; }}
details summary {{ color: rgba(255,255,255,0.45) !important; font-size: 0.85rem !important; cursor: pointer !important; letter-spacing: 0.03em !important; margin-bottom: 12px !important; }}

/* ── Log panel ── */
.log-panel-header {{ display: flex; align-items: center; gap: 10px; margin: 20px 0 10px; }}
.log-panel-title {{ font-family: 'DM Sans', sans-serif; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: rgba(255,255,255,0.35); }}
.log-dot {{ width: 7px; height: 7px; border-radius: 50%; background: {c2}; flex-shrink: 0; animation: pulse-dot 1.4s ease-in-out infinite; }}
.log-dot.done {{ animation: none; background: #2dc653; }}
.log-dot.error {{ animation: none; background: #e63946; }}
@keyframes pulse-dot {{ 0%, 100% {{ opacity: 1; transform: scale(1); }} 50% {{ opacity: 0.4; transform: scale(0.7); }} }}
.log-box {{
    background: rgba(0,0,0,0.55); border: 1px solid rgba(255,255,255,0.09); border-radius: 14px;
    padding: 14px 16px; max-height: 420px; overflow-y: auto;
    font-family: 'JetBrains Mono', 'Courier New', monospace; font-size: 0.76rem;
    line-height: 1.65; color: rgba(255,255,255,0.75); white-space: pre-wrap; word-break: break-all;
}}
.log-line-step  {{ color: {c2}; font-weight: 600; display: block; }}
.log-line-ok    {{ color: #2dc653; display: block; }}
.log-line-warn  {{ color: #f4845f; display: block; }}
.log-line-err   {{ color: #e63946; font-weight: 600; display: block; }}
.log-line-tqdm  {{ color: rgba(255,255,255,0.4); font-size: 0.72rem; display: block; }}
.log-line-plain {{ color: rgba(255,255,255,0.72); display: block; }}

/* ── Step tracker ── */
.step-tracker {{ display: flex; gap: 0; margin: 20px 0; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; overflow: hidden; }}
.step-item {{ flex: 1; padding: 10px 4px; text-align: center; font-family: 'DM Sans', sans-serif; font-size: 0.72rem; font-weight: 500; letter-spacing: 0.03em; color: rgba(255,255,255,0.28); border-right: 1px solid rgba(255,255,255,0.07); transition: background 0.3s, color 0.3s; line-height: 1.4; }}
.step-item:last-child {{ border-right: none; }}
.step-item.active {{ background: rgba(255,255,255,0.08); color: rgba(255,255,255,0.9); }}
.step-item.done-step {{ background: rgba(45,198,83,0.12); color: #2dc653; }}
</style>

<div class="blob-wrap">
    <div class="blob blob-a"></div>
    <div class="blob blob-b"></div>
    <div class="blob blob-c"></div>
</div>

<script>
(function() {{
    const gradients = {gradients_js};
    const currentIndex = {gi};
    function hexToRgb(hex) {{ return {{ r: parseInt(hex.slice(1,3),16), g: parseInt(hex.slice(3,5),16), b: parseInt(hex.slice(5,7),16) }}; }}
    function lerpColor(a, b, t) {{
        const ca = hexToRgb(a), cb = hexToRgb(b);
        return '#' + ['r','g','b'].map(ch => Math.round(ca[ch] + (cb[ch]-ca[ch])*t).toString(16).padStart(2,'0')).join('');
    }}
    function applyColors(nc1, nc2, nc3) {{
        const a = document.querySelector('.blob-a'), b = document.querySelector('.blob-b'), c = document.querySelector('.blob-c');
        if(a) a.style.background = `radial-gradient(circle,${{nc1}},transparent 70%)`;
        if(b) b.style.background = `radial-gradient(circle,${{nc2}},transparent 70%)`;
        if(c) c.style.background = `radial-gradient(circle,${{nc3}},transparent 70%)`;
    }}
    function animate(from, to, duration) {{
        const start = performance.now();
        (function frame(now) {{
            const t = Math.min((now-start)/duration,1);
            const e = t<0.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2;
            applyColors(lerpColor(from[0],to[0],e), lerpColor(from[1],to[1],e), lerpColor(from[2],to[2],e));
            if(t<1) requestAnimationFrame(frame);
        }})(start);
    }}
    const prev = parseInt(sessionStorage.getItem('gi')||'0');
    function run() {{
        if(prev !== currentIndex) {{ animate(gradients[prev], gradients[currentIndex], 2200); sessionStorage.setItem('gi', String(currentIndex)); }}
        else {{ applyColors(...gradients[currentIndex]); }}
    }}
    let tries=0;
    (function wait() {{
        if(document.querySelector('.blob-a')||tries>20) run(); else {{ tries++; setTimeout(wait,100); }}
    }})();
}})();
</script>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="page-title">Ad Generator</p>', unsafe_allow_html=True)
st.markdown('<p class="page-sub">Describe your product — get a complete AI-generated advertisement.</p>', unsafe_allow_html=True)

# ── Main input ────────────────────────────────────────────────────────────────
user_text = st.text_area(
    "Product description",
    height=120,
    placeholder="e.g. A minimal leather wallet with RFID blocking, built for everyday carry…",
)


# ── Advanced options ──────────────────────────────────────────────────────────
with st.expander("⚙ Advanced options"):
    st.markdown('<p class="section-label">Generation settings</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        num_scenes = st.number_input("Number of scenes", min_value=1, max_value=6, value=2, step=1)
    with col2:
        tts_model = st.selectbox(
            "TTS model (all free, local, no API key)",
            options=["speecht5", "bark", "mms"],
            format_func=lambda x: {
                "speecht5": "SpeechT5 — fast, clean (default)",
                "bark":     "Bark — slow but expressive, supports [emotion cues]",
                "mms":      "MMS-TTS — lightest, fastest, CPU-friendly",
            }[x],
        )
    voiceover_text = st.text_area(
        "Custom voiceover script (leave blank to auto-generate)",
        height=80,
        placeholder="Leave blank to auto-generate from your product description…",
    )
    if tts_model == "bark":
        st.info(
            "**Bark emotion cues** — embed these directly in your voiceover script:\n\n"
            "`[laughs]`  `[sighs]`  `[whispers]`  `[gasps]`  `[clears throat]`  `...` for pause  `CAPS` for emphasis\n\n"
            "Example: *\"Introducing our wallet. [sighs] It's... truly remarkable.\"*"
        )
    music_prompt = st.text_area(
        "Background music style (leave blank to auto-generate)",
        height=60,
        placeholder="e.g. Elegant cinematic strings, slow build, luxury mood…",
    )
    elevenlabs_key = azure_key = azure_region = emotion_style = ""


# ── Helpers ───────────────────────────────────────────────────────────────────
STEPS = [
    ("1", "Scene prompts"),
    ("2", "Video render"),
    ("3", "Voiceover"),
    ("4", "Music"),
    ("5", "Composite"),
]

def _render_step_tracker(current_step, done, error):
    items_html = ""
    for idx, (num, label) in enumerate(STEPS, 1):
        if done and not error:
            cls, icon = "done-step", "✓ "
        elif idx < current_step:
            cls, icon = "done-step", "✓ "
        elif idx == current_step:
            cls, icon = "active", ("" if (done or error) else "⟳ ")
        else:
            cls, icon = "", ""
        items_html += f'<div class="step-item {cls}">{icon}{num}. {label}</div>'
    st.markdown(f'<div class="step-tracker">{items_html}</div>', unsafe_allow_html=True)


def _classify_line(line):
    l = line.lower()
    if line.startswith("──") or ("step" in l and "/" in l and "──" in line):
        return "log-line-step"
    if (
        line.startswith("[progress]")
        or line.startswith("[tqdm]")
        or line.startswith("[VideoGen] step ")
        or line.startswith("[mem:")
        or line.startswith("[device_map]")
        or line.startswith("[proc]")
        or line.startswith("Loading weights")
        or line.startswith("Loading pipeline")
        or line.startswith("Fetching")
    ):
        return "log-line-tqdm"
    if any(kw in line for kw in ["✓", "✔", "🎬"]) or any(kw in l for kw in ["done", "saved", "ready", "complete"]):
        return "log-line-ok"
    if any(kw in line for kw in ["⚠", "WARNING"]) or any(kw in l for kw in ["warning", "warn", "skipping"]):
        return "log-line-warn"
    if any(kw in line for kw in ["ERROR", "Error", "Traceback", "Exception"]) or any(kw in l for kw in ["error", "exception", "failed"]):
        return "log-line-err"
    return "log-line-plain"


def _render_log_panel(lines, generating, error, done):
    dot_class = "error" if error else ("done" if done else "")
    status_text = "Error" if error else ("Done" if done else "Running…")
    st.markdown(
        f'<div class="log-panel-header">'
        f'<div class="log-dot {dot_class}"></div>'
        f'<span class="log-panel-title">Pipeline log — {status_text} &nbsp;({len(lines)} lines)</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if not lines:
        st.markdown('<div class="log-box"><span class="log-line-plain">Waiting for pipeline to start…</span></div>', unsafe_allow_html=True)
        return

    _OVERWRITE_STARTS = (
        "[VideoGen] step ",
        "[progress]",
        "[mem:",
        "[device_map]",
        "Loading weights",
        "Loading pipeline",
    )
    def _is_overwrite(ln):
        return any(ln.startswith(p) for p in _OVERWRITE_STARTS)

    collapsed = []
    for line in lines:
        if _is_overwrite(line) and collapsed and _is_overwrite(collapsed[-1]):
            collapsed[-1] = line
        else:
            collapsed.append(line)

    rows = ""
    for line in collapsed:
        cls = _classify_line(line)
        safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        rows += f'<span class="{cls}">{safe}</span>\n'

    scroll_js = """<script>(function(){var b=document.querySelectorAll('.log-box');if(b.length){b[b.length-1].scrollTop=b[b.length-1].scrollHeight;}})();</script>"""
    st.markdown(f'<div class="log-box">{rows}</div>{scroll_js}', unsafe_allow_html=True)


# ── Background pipeline thread ────────────────────────────────────────────────
def _run_pipeline(product, scenes, vo_text, mus_prompt, tts_model_choice, log_list, done_event, resume=False):
    """
    Runs the full ad-generation pipeline in a background thread.

    done_event is a plain threading.Event — it is set() in the finally block
    so the main-thread polling loop wakes up immediately and reliably.
    We do NOT rely on st.session_state.generating as the stop signal because
    Streamlit can serve a stale cached value of session state to the polling
    loop even after the background thread writes False into it.
    """
    def _step(n, msg):
        st.session_state.current_step = n
        logger.add(f"── Step {n} / 5 : {msg} " + "─" * max(0, 48 - len(msg)))

    logger = PipelineLogger(log_list)
    work_dir = Path("ad_workspace")
    work_dir.mkdir(parents=True, exist_ok=True)
    state_file = work_dir / "pipeline_state.json"

    try:
        # ── Load / save pipeline state ────────────────────────────────────
        if resume and state_file.exists():
            with open(state_file, "r") as f:
                saved = json.load(f)
            if saved["product"] != product:
                logger.add("⚠ Warning: Product description changed. Resume may produce inconsistent results.")
            scenes = saved.get("num_scenes", scenes)
            vo_text = saved.get("voiceover_text", vo_text) if not vo_text else vo_text
            mus_prompt = saved.get("music_prompt", mus_prompt) if not mus_prompt else mus_prompt
            tts_model_choice = saved.get("tts_model", tts_model_choice)
        else:
            with open(state_file, "w") as f:
                json.dump({
                    "product": product,
                    "num_scenes": scenes,
                    "voiceover_text": vo_text,
                    "music_prompt": mus_prompt,
                    "tts_model": tts_model_choice,
                }, f)

        # ── Step 1: Scene prompts ─────────────────────────────────────────
        prompts_file = work_dir / "scene_prompts.json"
        if resume and prompts_file.exists():
            with open(prompts_file, "r") as f:
                prompts = json.load(f)
            logger.add("── Step 1 / 5 : Using cached scene prompts")
        else:
            _step(1, "Generating scene prompts with Qwen 2.5-7B")
            llm_model = "Qwen/Qwen2.5-7B-Instruct"
            prompts = generate_prompts(product, scenes, llm_model=llm_model)
            with open(prompts_file, "w") as f:
                json.dump(prompts, f)

        # ── Step 2: Render scenes ─────────────────────────────────────────
        scene_dir = work_dir / "scenes"
        scene_dir.mkdir(exist_ok=True)
        expected_scene_paths = [scene_dir / f"scene_{i:02d}.mp4" for i in range(1, scenes + 1)]
        if resume and all(p.exists() for p in expected_scene_paths):
            scene_paths = expected_scene_paths
            logger.add("── Step 2 / 5 : Using existing rendered scenes")
        else:
            _step(2, "Rendering scenes with CogVideoX")
            scene_paths = render_scenes(prompts, output_dir=scene_dir)

        # ── NEW: Get total video duration ─────────────────────────────────
        total_duration = 0.0
        for sp in scene_paths:
            clip = VideoFileClip(str(sp))
            total_duration += clip.duration
            clip.close()
        logger.add(f"Total video duration: {total_duration:.2f} seconds")

        # ── Step 3: Voiceover ─────────────────────────────────────────────
        voice_path = work_dir / "voiceover.wav"
        if resume and voice_path.exists():
            logger.add("── Step 3 / 5 : Using existing voiceover")
        else:
            _step(3, "Generating voiceover")
            if not vo_text:
                vo_text = f"Introducing {product}. Crafted for those who demand the very best. Experience the difference — {product}."
            voice_path = Path(generate_voiceover(
                text=vo_text,
                output_file=str(voice_path),
                model=tts_model_choice,
            ))

        # ── Step 4: Background music ──────────────────────────────────────
        music_path = work_dir / "background_music.wav"
        if resume and music_path.exists():
            logger.add("── Step 4 / 5 : Using existing background music")
        else:
            _step(4, "Generating background music with MusicGen")
            if not mus_prompt:
                mus_prompt = f"Elegant, cinematic background music for a luxury {product} advertisement. Slow build, orchestral strings, warm and aspirational."
            music_path = Path(generate_music_from_text(
                prompt=mus_prompt,
                duration_seconds=total_duration,          # ← use total_duration
                output_filename=str(music_path),
            ))

        # ── Mix audio ─────────────────────────────────────────────────────
        mixed_path = work_dir / "mixed_audio.wav"
        if resume and mixed_path.exists() and voice_path.exists() and music_path.exists():
            logger.add("  Using existing mixed audio")
            mixed_audio = str(mixed_path)
        else:
            logger.add("  Mixing voiceover and background music…")
            try:
                mixed_audio = _mix_audio(
                    voice_path=str(voice_path),
                    music_path=str(music_path),
                    output_path=str(mixed_path),
                    target_duration=total_duration,   # ← use computed duration
                )
                logger.add("  ✓ Audio mixed successfully.")
            except Exception as e:
                logger.add(f"  ⚠ Audio mixing failed ({e}), using voiceover only.")
                # Align the voiceover to the video duration
                aligned_voice = _align_audio(
                    str(voice_path), total_duration,
                    str(work_dir / "voice_aligned.wav")
                )
                mixed_audio = aligned_voice

        # ── Step 5: Composite final video ─────────────────────────────────
        final_path = work_dir / "final_ad.mp4"
        if resume and final_path.exists():
            logger.add("── Step 5 / 5 : Using existing final video")
        else:
            _step(5, "Compositing final video")
            final_path = Path(create_video_with_audio(
                video_files=scene_paths,
                audio_file=mixed_audio,
                output_filename=str(final_path),
                target_duration=total_duration,   # ← add this line
            ))

        logger.add(f"🎬  Advertisement ready → {final_path}")
        st.session_state.video_path = str(final_path)
        st.session_state.error = None
        st.session_state.done = True

    except Exception as exc:
        import traceback
        logger.add(f"ERROR: {exc}")
        logger.add(traceback.format_exc())
        st.session_state.error = str(exc)
        st.session_state.done = False

    finally:
        # This block always runs — success, failure, or any uncaught exception.
        # Order matters:
        #   1. generating = False  — mark pipeline as no longer running
        #   2. done_event.set()    — wake the polling loop on the main thread
        #   3. logger.stop()       — restore sys.stdout / sys.stderr
        st.session_state.generating = False
        done_event.set()
        logger.stop()


# ── Buttons: Generate & Continue ─────────────────────────────────────────────
col_gen, col_resume = st.columns(2)
with col_gen:
    gen_clicked = st.button("✦ Generate Advertisement", disabled=st.session_state.generating, use_container_width=True)
with col_resume:
    resume_clicked = st.button("↻ Continue from last step", disabled=st.session_state.generating, use_container_width=True)

if gen_clicked or resume_clicked:
    if not user_text.strip():
        st.warning("Please enter a product description first.")
    else:
        st.session_state.gradient_index = (st.session_state.gradient_index + 1) % len(gradients)

        st.session_state.done = False
        st.session_state.video_path = None
        st.session_state.error = None
        st.session_state.generating = True
        st.session_state.current_step = 0
        st.session_state.log_lines = []

        # Create a fresh threading.Event and store it in session state so the
        # polling loop below can retrieve it after st.rerun() re-executes the script.
        # threading.Event is a plain Python object — is_set() always reflects the
        # live value regardless of Streamlit's session-state caching behaviour.
        done_event = threading.Event()
        st.session_state._done_event = done_event

        thread = threading.Thread(
            target=_run_pipeline,
            args=(
                user_text.strip(), num_scenes, voiceover_text, music_prompt,
                tts_model, st.session_state.log_lines, done_event, resume_clicked,
            ),
            daemon=True,
        )
        thread.start()
        st.rerun()

# ── Live progress & log display ───────────────────────────────────────────────
# IMPORTANT: do NOT gate solely on st.session_state.done — it is written by the
# background thread and Streamlit's session-state snapshot can miss that write.
# Use the threading.Event and disk presence as reliable signals.
_done_event_check = st.session_state.get("_done_event")
_pipeline_ever_ran = (
    st.session_state.generating
    or st.session_state.done
    or (_done_event_check is not None and _done_event_check.is_set())
    or bool(st.session_state.get("video_path"))
    or bool(st.session_state.get("error"))
    or Path("ad_workspace/final_ad.mp4").exists()
)
if _pipeline_ever_ran:
    st.markdown("<hr>", unsafe_allow_html=True)

    tracker_slot = st.empty()
    log_slot     = st.empty()
    result_slot  = st.empty()

    def _refresh(tracker_slot, log_slot):
        with tracker_slot.container():
            _render_step_tracker(
                st.session_state.current_step,
                done=st.session_state.done and not st.session_state.error,
                error=bool(st.session_state.error),
            )
        with log_slot.container():
            _render_log_panel(
                lines=st.session_state.log_lines,
                generating=st.session_state.generating,
                error=bool(st.session_state.error),
                done=st.session_state.done,
            )

    # Initial render
    _refresh(tracker_slot, log_slot)

    # ── Polling loop ──────────────────────────────────────────────────────────
    # We poll on done_event (threading.Event), NOT on st.session_state.generating,
    # because threading.Event.is_set() always reflects the live value by reference
    # and bypasses Streamlit's per-execution session-state caching.
    done_event = st.session_state.get("_done_event")

    # Track whether we entered this execution while the pipeline was still active.
    # If yes, we'll call st.rerun() once the while-loop exits so Streamlit starts
    # a completely fresh script execution and reads the background-thread's session
    # state writes reliably (session state written from a bg thread can appear stale
    # when read later in the *same* script execution that started the polling loop).
    was_generating = bool(st.session_state.generating)

    while done_event is not None and not done_event.is_set():
        time.sleep(1)
        _refresh(tracker_slot, log_slot)

    # Pipeline finished — sync the session flag and do one final log refresh.
    st.session_state.generating = False
    _refresh(tracker_slot, log_slot)

    # ── KEY FIX ───────────────────────────────────────────────────────────────
    # If we were actively polling (was_generating=True), the pipeline just
    # finished in this execution. Trigger a rerun so the next execution reads
    # st.session_state.done / video_path fresh from the shared dict — no stale
    # values. In the rerun, was_generating will be False (generating is now False),
    # so this branch is skipped and we fall straight through to the result display.
    if was_generating:
        st.rerun()

    # ── Result ────────────────────────────────────────────────────────────────
    # IMPORTANT: do NOT gate this on st.session_state.done or st.session_state.error.
    # Those flags are written by the background thread, and Streamlit can snapshot
    # session state before those writes land — so they may still read as False/None
    # even after the pipeline has fully completed.
    #
    # Instead we use two sources that are always reliable after the while-loop exits:
    #   1. done_event.is_set()  — a plain Python threading.Event shared by reference
    #   2. disk presence        — if final_ad.mp4 exists, the pipeline succeeded
    #
    # st.session_state.error is used only as a hint (it may or may not be set).
    video_placeholder = st.empty()

    pipeline_done = done_event is not None and done_event.is_set()

    if pipeline_done:
        # ── Find the video file ───────────────────────────────────────────────
        # Try session state first (works when bg-thread writes did land), then
        # fall back to known filename, then scan the whole workspace.
        workspace = Path("ad_workspace")
        video_path = st.session_state.get("video_path") or ""

        if not video_path or not os.path.exists(str(video_path)):
            # Try the well-known output path first
            candidate = workspace / "final_ad.mp4"
            if candidate.exists():
                video_path = str(candidate)
            else:
                # Last resort: newest .mp4 in workspace root (not in scenes/)
                top_mp4s = [
                    p for p in workspace.glob("*.mp4")
                    if "scene" not in p.name
                ] if workspace.exists() else []
                if top_mp4s:
                    video_path = str(sorted(top_mp4s, key=os.path.getmtime)[-1])

        # ── Display ───────────────────────────────────────────────────────────
        if video_path and os.path.exists(str(video_path)):
            # Sync back into session state so the download button is stable
            st.session_state.video_path = video_path
            with video_placeholder.container():
                st.video(video_path)
                st.markdown('<p class="result-label">✦ Advertisement ready</p>', unsafe_allow_html=True)
                with open(video_path, "rb") as f:
                    st.download_button(
                        label="⬇ Download Advertisement",
                        data=f,
                        file_name="advertisement.mp4",
                        mime="video/mp4",
                    )
        else:
            # Pipeline finished but no video on disk → something went wrong
            err = st.session_state.get("error") or "No video file found on disk."
            video_placeholder.error(f"Generation failed: {err}")

    elif st.session_state.get("error"):
        video_placeholder.error(f"Generation failed: {st.session_state.error}")
