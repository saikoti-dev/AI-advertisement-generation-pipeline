import os
import threading
import time
from pathlib import Path

import streamlit as st

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
    # Optional API keys (persisted across reruns)
    "elevenlabs_key": "",
    "azure_key": "",
    "azure_region": "",
    "num_scenes": 2,
    "voiceover_text": "",
    "music_prompt": "",
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

header[data-testid="stHeader"] {{
    display: none !important; height: 0 !important; min-height: 0 !important;
    padding: 0 !important; margin: 0 !important; overflow: hidden !important;
}}
[data-testid="stToolbar"],footer,#MainMenu,
[data-testid="stDecoration"],[data-testid="stStatusWidget"] {{ display: none !important; }}

[data-testid="stAppViewContainer"] > .main > .block-container {{
    padding-top: 2rem !important;
}}

:root {{ --c1: {c1}; --c2: {c2}; --c3: {c3}; }}

html, body, [data-testid="stAppViewContainer"], .stApp {{
    background: #0a0a12 !important;
    font-family: 'DM Sans', sans-serif !important;
}}

.blob-wrap {{
    position: fixed; inset: 0; z-index: 0;
    overflow: hidden; pointer-events: none;
}}
.blob {{
    position: absolute; border-radius: 50%;
    filter: blur(110px); opacity: 0.55;
    animation: drift 8s ease-in-out infinite alternate;
}}
.blob-a {{
    width: 680px; height: 680px; top: -180px; left: -180px;
    background: radial-gradient(circle, {c1}, transparent 70%);
    animation-delay: 0s;
}}
.blob-b {{
    width: 600px; height: 600px; bottom: -160px; right: -160px;
    background: radial-gradient(circle, {c2}, transparent 70%);
    animation-delay: -3s;
}}
.blob-c {{
    width: 500px; height: 500px; top: 40%; left: 40%;
    background: radial-gradient(circle, {c3}, transparent 70%);
    opacity: 0.35; animation-delay: -5s;
}}
@keyframes drift {{
    0%   {{ transform: translate(0,0)        scale(1);    }}
    33%  {{ transform: translate(30px,-30px)  scale(1.05); }}
    66%  {{ transform: translate(-20px,20px)  scale(0.97); }}
    100% {{ transform: translate(10px,10px)   scale(1.03); }}
}}

[data-testid="stAppViewContainer"] > .main {{ position: relative; z-index: 1; }}

.page-title {{
    font-family: 'Syne', sans-serif !important;
    font-size: 2.5rem !important; font-weight: 800 !important;
    line-height: 1.05 !important; margin: 0 0 8px !important;
    background: linear-gradient(90deg, #fff 20%, {c2} 60%, {c3} 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    background-size: 200% auto !important;
    animation: shimmer 4s linear infinite !important;
}}
@keyframes shimmer {{
    0%   {{ background-position: 0% center; }}
    100% {{ background-position: 200% center; }}
}}

.page-sub {{
    color: rgba(255,255,255,0.45); font-size: 0.92rem; font-weight: 300;
    letter-spacing: 0.02em; margin-bottom: 32px;
}}

[data-testid="stTextArea"] label, .stTextArea label {{
    color: rgba(255,255,255,0.7) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important; font-weight: 500 !important;
    letter-spacing: 0.04em !important; text-transform: uppercase !important;
}}
[data-testid="stTextArea"] textarea {{
    background: rgba(255,255,255,0.07) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 16px !important;
    color: rgba(255,255,255,0.92) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.97rem !important; font-weight: 300 !important;
    padding: 14px 16px !important; resize: none !important;
    caret-color: {c2} !important;
    transition: border 0.25s ease, box-shadow 0.25s ease !important;
}}
[data-testid="stTextArea"] textarea:focus {{
    border: 1px solid rgba(255,255,255,0.35) !important;
    box-shadow: 0 0 0 3px rgba(255,255,255,0.06) !important;
    outline: none !important;
}}
[data-testid="stTextArea"] textarea::placeholder {{ color: rgba(255,255,255,0.28) !important; }}

[data-testid="stTextInput"] input {{
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 12px !important;
    color: rgba(255,255,255,0.88) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
}}
[data-testid="stTextInput"] label {{
    color: rgba(255,255,255,0.6) !important;
    font-size: 0.82rem !important; font-weight: 500 !important;
    letter-spacing: 0.04em !important; text-transform: uppercase !important;
}}

[data-testid="stNumberInput"] input {{
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 12px !important;
    color: rgba(255,255,255,0.88) !important;
    font-family: 'DM Sans', sans-serif !important;
}}

[data-testid="stButton"] > button {{
    width: 100% !important;
    background: #0e0e16 !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 50px !important;
    padding: 14px 48px !important; color: rgba(255,255,255,0.88) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important; font-weight: 500 !important;
    letter-spacing: 0.06em !important; cursor: pointer !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border 0.2s ease !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.06) !important;
    margin-top: 12px !important;
}}
[data-testid="stButton"] > button:hover {{
    transform: scale(1.02) translateY(-1px) !important;
    background: #16161f !important;
    border: 1px solid rgba(255,255,255,0.28) !important;
    box-shadow: 0 6px 28px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.09) !important;
}}
[data-testid="stButton"] > button:active {{ transform: scale(0.98) !important; }}

hr {{
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.08) !important;
    margin: 24px 0 !important;
}}

[data-testid="stAlert"] {{
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 14px !important;
    color: rgba(255,255,255,0.85) !important;
    font-family: 'DM Sans', sans-serif !important;
}}

.result-label {{
    text-align: center; color: rgba(255,255,255,0.5);
    font-size: 0.85rem; letter-spacing: 0.05em;
    text-transform: uppercase; margin-top: 10px;
}}

.section-label {{
    color: rgba(255,255,255,0.35); font-size: 0.75rem;
    letter-spacing: 0.1em; text-transform: uppercase;
    margin: 24px 0 8px; font-weight: 500;
}}

details summary {{
    color: rgba(255,255,255,0.45) !important;
    font-size: 0.85rem !important; cursor: pointer !important;
    letter-spacing: 0.03em !important; margin-bottom: 12px !important;
}}
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

    function hexToRgb(hex) {{
        return {{ r: parseInt(hex.slice(1,3),16), g: parseInt(hex.slice(3,5),16), b: parseInt(hex.slice(5,7),16) }};
    }}
    function lerpColor(a, b, t) {{
        const ca = hexToRgb(a), cb = hexToRgb(b);
        return '#' + ['r','g','b'].map(ch => Math.round(ca[ch] + (cb[ch]-ca[ch])*t).toString(16).padStart(2,'0')).join('');
    }}
    function applyColors(nc1, nc2, nc3) {{
        const a = document.querySelector('.blob-a');
        const b = document.querySelector('.blob-b');
        const c = document.querySelector('.blob-c');
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
        if(document.querySelector('.blob-a')||tries>20) run();
        else {{ tries++; setTimeout(wait,100); }}
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

# ── Advanced options (collapsed by default) ───────────────────────────────────
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

    # Hidden — no longer used, kept so backend.py call signature stays compatible
    elevenlabs_key = ""
    azure_key = ""
    azure_region = ""
    emotion_style = "cheerful"


# ── Background generation thread ──────────────────────────────────────────────
def _run_pipeline(product, scenes, vo_text, mus_prompt, tts_model_choice):
    try:
        from backend import generate_ad

        st.session_state.progress_msg = "Generating scene prompts with LLaMA…"
        path = generate_ad(
            product=product,
            num_scenes=int(scenes),
            voiceover_text=vo_text or None,
            music_prompt=mus_prompt or None,
            output_filename="final_ad.mp4",
            work_dir="ad_workspace",
            tts_model=tts_model_choice,
        )
        st.session_state.video_path = path
        st.session_state.error = None
    except Exception as exc:
        st.session_state.error = str(exc)
    finally:
        st.session_state.generating = False
        st.session_state.done = True


# ── Generate button ───────────────────────────────────────────────────────────
if st.button("✦ Generate Advertisement", disabled=st.session_state.generating):
    if not user_text.strip():
        st.warning("Please enter a product description first.")
    else:
        st.session_state.gradient_index = (st.session_state.gradient_index + 1) % len(gradients)
        st.session_state.done = False
        st.session_state.video_path = None
        st.session_state.error = None
        st.session_state.generating = True
        st.session_state.progress_msg = "Starting pipeline…"

        thread = threading.Thread(
            target=_run_pipeline,
            args=(
                user_text.strip(),
                num_scenes,
                voiceover_text,
                music_prompt,
                tts_model,
            ),
            daemon=True,
        )
        thread.start()
        st.rerun()


# ── Progress display ──────────────────────────────────────────────────────────
if st.session_state.generating:
    with st.spinner(st.session_state.get("progress_msg", "Generating your advertisement… (this takes a few minutes)")):
        time.sleep(2)
    st.rerun()


# ── Result display ────────────────────────────────────────────────────────────
if st.session_state.done:
    st.markdown("<hr>", unsafe_allow_html=True)

    if st.session_state.error:
        st.error(f"Something went wrong:\n\n{st.session_state.error}")
        st.info("Check that all required models are downloaded and your GPU has enough VRAM.")

    elif st.session_state.video_path and os.path.exists(st.session_state.video_path):
        st.video(st.session_state.video_path)
        st.markdown('<p class="result-label">✦ Advertisement ready</p>', unsafe_allow_html=True)

        with open(st.session_state.video_path, "rb") as f:
            st.download_button(
                label="⬇ Download Advertisement",
                data=f,
                file_name="advertisement.mp4",
                mime="video/mp4",
            )
    else:
        st.warning("Generation finished but no video file was found. Check the console for errors.")
