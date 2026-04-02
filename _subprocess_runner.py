"""
_subprocess_runner.py
---------------------
Single entry-point for every model subprocess.

Usage (called internally by _proc.py):
    python _subprocess_runner.py <task_name> <json_payload>

Tasks:
    scene_gen  — Qwen2.5-1.5B  → scene prompts
    video_gen  — CogVideoX-2b  → one rendered scene
    tts_gen    — SpeechT5 / Bark / MMS  → voiceover .wav
    music_gen  — MusicGen-small → background music .wav

Each task prints a RESULT_JSON:<json> sentinel line when done.
Everything else streams straight to Colab as live log output.
"""

from __future__ import annotations
import json
import sys
from scene_generator import generate_voiceover_script, generate_music_prompt

# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _send(result: dict):
    print("RESULT_JSON:" + json.dumps(result), flush=True)


def _ram_info():
    """Returns (free_gb, total_gb) of system RAM."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.available / 1024**3, vm.total / 1024**3
    except ImportError:
        # fallback: read /proc/meminfo
        info = {}
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        info[parts[0].rstrip(":")] = int(parts[1]) * 1024
            free = (info.get("MemAvailable", 0)) / 1024**3
            total = info.get("MemTotal", 0) / 1024**3
            return free, total
        except Exception:
            return -1.0, -1.0


def _vram_info():
    """Returns (free_mb, total_mb) of GPU VRAM, or (inf, inf) if no GPU."""
    import torch
    if not torch.cuda.is_available():
        return float("inf"), float("inf")
    free, total = torch.cuda.mem_get_info()
    return free / 1024**2, total / 1024**2


def _flush_gpu():
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _print_mem(tag: str):
    ram_free, ram_total = _ram_info()
    vram_free, vram_total = _vram_info()
    print(
        f"[mem:{tag}] "
        f"RAM {ram_free:.1f}/{ram_total:.1f}GB free  "
        f"VRAM {vram_free:.0f}/{vram_total:.0f}MB free",
        flush=True,
    )


def _build_max_memory(vram_headroom_mb: float = 512, ram_headroom_gb: float = 1.5) -> dict:
    """
    Build a max_memory dict for from_pretrained(device_map='auto').

    This tells accelerate exactly how much of each device it may use:
      - GPU: all free VRAM minus a safety headroom (for activations during forward pass)
      - CPU: all free RAM minus a safety headroom (for the OS and other processes)

    accelerate then greedily packs as many layers as possible onto the GPU,
    spilling the rest to CPU RAM — much faster than sequential offload because
    the GPU-resident layers never move during inference.

    Note: vram_headroom_mb already includes the extra 2 GB buffer added to every
    call site to keep the Linux OOM killer away.
    """
    import torch

    ram_free, _ = _ram_info()
    ram_budget_gb = max(0.5, ram_free - ram_headroom_gb)

    if torch.cuda.is_available():
        vram_free_mb, vram_total_mb = _vram_info()
        # Hard cap: never use more than 85 % of total VRAM regardless of what
        # is currently free — fragmentation and driver overhead mean "free"
        # overstates what diffusion activations can actually use.
        hard_cap_mb = vram_total_mb * 0.85
        vram_budget_mb = max(256, min(vram_free_mb - vram_headroom_mb, hard_cap_mb))
        mem = {0: f"{int(vram_budget_mb)}MiB", "cpu": f"{ram_budget_gb:.1f}GiB"}
        print(
            f"[device_map] GPU budget={int(vram_budget_mb)}MB  "
            f"(free={int(vram_free_mb)}MB  total={int(vram_total_mb)}MB  "
            f"headroom={int(vram_headroom_mb)}MB)  CPU budget={ram_budget_gb:.1f}GB",
            flush=True,
        )
    else:
        mem = {"cpu": f"{ram_budget_gb:.1f}GiB"}
        print(f"[device_map] CPU-only budget={ram_budget_gb:.1f}GB", flush=True)

    return mem


# ═══════════════════════════════════════════════════════════════════
# Task: scene_gen  (Qwen2.5-1.5B)
# ═══════════════════════════════════════════════════════════════════

def _task_scene_gen(args: dict):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    product    = args["product"]
    num_scenes = args["num_scenes"]
    model_id   = args.get("llm_model", "Qwen/Qwen2.5-1.5B-Instruct")   # configurable

    _print_mem("before_scene_gen")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=_build_max_memory(vram_headroom_mb=512, ram_headroom_gb=1.0),
    )
    model.eval()
    _print_mem("after_scene_gen_load")

    # ------------------------------------------------------------------
    # NEW: detailed system prompt with examples for CogVideoX
    # ------------------------------------------------------------------
    system_prompt = (
        "You are an expert prompt engineer for CogVideoX-2b, a compact text-to-video diffusion model.\n"
        "CogVideoX-2b produces its best results with SIMPLE, STATIC or SLOWLY-MOVING scenes.\n"
        "Each prompt describes one 6-second continuous shot.\n\n"

        "── WHAT CogVideoX-2b RENDERS WELL ──────────────────────────────────────\n"
        "• A single hero object resting on a surface with soft, even lighting\n"
        "• Very slow camera movements: gentle push-in, slight pan, slow tilt\n"
        "• Close-up or medium-close framing (fills the frame — less background to hallucinate)\n"
        "• Simple, uncluttered backgrounds: black velvet, marble slab, white studio, bokeh blur\n"
        "• Subtle, real-world motion: liquid ripple, fabric drape, smoke curl, slow rotation\n"
        "• High-contrast lighting that makes the product 'pop': rim light, spotlight, soft-box\n\n"

        "── WHAT TO STRICTLY AVOID ───────────────────────────────────────────────\n"
        "✗ Human faces or hands (2b distorts them badly)\n"
        "✗ Text, logos, or readable words in the scene\n"
        "✗ Fast action, sports, or rapid camera cuts\n"
        "✗ Wide outdoor landscapes (too much detail → muddy result)\n"
        "✗ Multiple subjects interacting\n"
        "✗ Abstract or metaphorical descriptions ('the essence of speed')\n"
        "✗ Anything requiring 3-D physics simulation (explosions, water splashes, crumbling)\n\n"

        "── MANDATORY PROMPT STRUCTURE ───────────────────────────────────────────\n"
        "Every prompt MUST contain these five elements in order:\n"
        "1. SUBJECT  — the product + its material/finish (e.g., 'a matte-black aluminium bottle')\n"
        "2. SETTING  — surface + background (e.g., 'resting on a brushed-concrete slab, dark studio')\n"
        "3. LIGHTING — specific light quality (e.g., 'soft overhead key light, warm amber rim light')\n"
        "4. MOTION   — what physically moves (e.g., 'camera slowly pushes in 10 %')\n"
        "5. STYLE    — quality tags at the end: 'cinematic, photorealistic, shallow depth of field, 4K'\n\n"

        "Write as ONE dense paragraph of 55–75 words. No lists. No scene numbers. No extra commentary.\n\n"

        "── EXAMPLES (luxury watch) ──────────────────────────────────────────────\n"
        "GOOD: A polished rose-gold chronograph watch rests on a smooth black marble surface. "
        "Soft diffused studio lighting with a warm amber rim light catches the sapphire crystal bezel. "
        "The camera executes a slow, gentle push-in, the dial filling the frame. "
        "No motion except the imperceptible sweep of the second hand. "
        "Cinematic, photorealistic, shallow depth of field, 4K.\n\n"
        "GOOD: Extreme close-up of a stainless-steel watch crown against a dark velvet background. "
        "A single cool-white spotlight casts a crisp shadow. "
        "The camera tilts up 5 degrees to reveal the engraved case-back. "
        "Absolute stillness apart from a faint specular shimmer on the metal. "
        "Luxury product advertisement, photorealistic, 4K.\n\n"
        "BAD (avoid): A man runs through a neon-lit city holding the watch above his head — "
        "too many subjects, faces, fast motion, complex background."
    )

    # Shot-type library — gives the LLM concrete framing options to rotate through
    shot_types = [
        "medium-close shot on a clean studio surface, camera slowly pushing in",
        "extreme close-up of a key detail (texture, logo, material finish), camera holding still with very subtle breathing motion",
        "three-quarter angle on a contrasting background, gentle 5-degree camera tilt upward",
        "overhead flat-lay with a slow clockwise orbit of the camera",
        "low-angle hero shot looking up at the product, soft backlight creating a rim-light halo",
    ]
    # Cycle through shot types so every scene is a different framing
    scene_descriptions = "\n".join(
        f"  Scene {i+1}: {shot_types[i % len(shot_types)]}"
        for i in range(num_scenes)
    )

    user_prompt = (
        f"Write exactly {num_scenes} CogVideoX-2b prompts for a luxury advertisement for: {product}.\n\n"
        "CRITICAL RULES:\n"
        "- No human faces, hands, or bodies\n"
        "- No fast motion — only slow camera drifts or imperceptible object motion\n"
        "- Each scene must be a DIFFERENT framing/angle (do not repeat the same shot)\n"
        "- Each prompt must include all 5 elements: subject, setting, lighting, motion, style tags\n\n"
        f"Use these specific shot types for each scene:\n{scene_descriptions}\n\n"
        f"Output exactly {num_scenes} prompt paragraphs separated by a blank line. "
        "Nothing else — no labels, no numbering, no explanation."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    primary_device = next(model.parameters()).device
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(primary_device)

    print("[SceneGen] Running inference…", flush=True)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=900, do_sample=False, temperature=0.7)

    generated = [out_ids[i][len(inputs.input_ids[i]):] for i in range(len(out_ids))]
    raw = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

    # Prompts are separated by blank lines; filter out any short/empty chunks
    import re as _re
    chunks = [c.strip() for c in _re.split(r'\n\s*\n', raw)]
    prompts = [c for c in chunks if len(c) > 40][:num_scenes]
    while len(prompts) < num_scenes:
        prompts.append(
            f"{product} elegantly presented on a clean surface with soft golden studio lighting. "
            "Camera slowly pushes in revealing fine details. Shallow depth of field. "
            "Luxury product advertisement. Photorealistic. 4K."
        )

    print(f"[SceneGen] ✓ {len(prompts)} prompts ready", flush=True)
    _send({"prompts": prompts})


# ═══════════════════════════════════════════════════════════════════
# Task: video_gen  (CogVideoX-2b)
# ═══════════════════════════════════════════════════════════════════

def _task_video_gen(args: dict):
    import torch
    import gc
    from diffusers import CogVideoXPipeline
    from diffusers.utils import export_to_video

    prompt   = args["prompt"]
    path     = args["path"]
    seed     = args["seed"]
    # Allow caller to override model; default to the lighter 2b variant
    MODEL_ID = args.get("model_id", "THUDM/CogVideoX-2b")

    _print_mem("before_video_gen")

    # CogVideoX-2b fits comfortably in ~8 GB VRAM at float16.
    # 49 frames @ 8 fps = ~6 s clip, which is the sweet-spot for the 2b model.
    # More frames → OOM on smaller GPUs; fewer → too short for crossfades.
    safe_config = {
        "num_steps":  50,   # 2b is much lighter — full 50 steps fits easily
        "num_frames": 49,   # 49 is the native "full" clip length for CogVideoX
        "fps":         8,   # native fps for CogVideoX
        "guidance":    6.0,
    }
    print(
        f"[VideoGen] model={MODEL_ID}  "
        f"steps={safe_config['num_steps']}  frames={safe_config['num_frames']}",
        flush=True,
    )

    pipe = CogVideoXPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    )

    pipe.enable_sequential_cpu_offload()
    print("[VideoGen] Sequential CPU offload enabled", flush=True)

    # Memory-efficient VAE decoding (works on both GPU and CPU paths)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    _print_mem("after_pipe_load")

    total_steps  = safe_config["num_steps"]
    step_counter = {"n": 0}

    def _cb(pipeline, step, timestep, kwargs):
        step_counter["n"] += 1
        print(f"[VideoGen] step {step_counter['n']}/{total_steps}", flush=True)
        return {}

    gen_device = "cuda" if torch.cuda.is_available() else "cpu"
    frames = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=total_steps,
        num_frames=safe_config["num_frames"],
        guidance_scale=safe_config["guidance"],
        generator=torch.Generator(device=gen_device).manual_seed(seed),
        callback_on_step_end=_cb,
    ).frames[0]

    export_to_video(frames, path, fps=safe_config["fps"])

    del frames, pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print(f"[VideoGen] ✓ saved → {path}", flush=True)
    _send({"path": path})
# ═══════════════════════════════════════════════════════════════════
# Task: tts_gen
# ═══════════════════════════════════════════════════════════════════

def _task_tts_gen(args: dict):
    import numpy as np
    import scipy.io.wavfile
    import torch

    model_name  = args["model"]
    text        = args["text"]
    output_file = args["output_file"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _print_mem(f"before_tts_{model_name}")

    if model_name == "speecht5":
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
        from datasets import load_dataset

        processor   = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        model       = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
        vocoder     = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
        ds          = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        speaker_emb = torch.tensor(ds[7306]["xvector"]).unsqueeze(0).to(device)
        inputs      = processor(text=text, return_tensors="pt").to(device)

        with torch.no_grad():
            speech = model.generate_speech(inputs["input_ids"], speaker_emb, vocoder=vocoder)

        audio_np = (speech.cpu().numpy() * 32767).astype("int16")
        scipy.io.wavfile.write(output_file, rate=16000, data=audio_np)
        del processor, model, vocoder, speaker_emb, inputs, speech

    elif model_name == "bark":
        from transformers import AutoProcessor, BarkModel, GenerationConfig

        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained("suno/bark")
        model = BarkModel.from_pretrained(
            "suno/bark",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)

        inputs = processor(text, voice_preset="v2/en_speaker_6")
        inputs = {k: v.to(device) for k, v in inputs.items()}   # move to GPU

        gen_config = GenerationConfig(
            max_new_tokens=768,
            do_sample=True,
            temperature=0.7,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

        with torch.no_grad():
            audio = model.generate(**inputs, generation_config=gen_config)

        audio_np = (audio.cpu().numpy().squeeze() * 32767).astype("int16")
        scipy.io.wavfile.write(output_file, rate=24000, data=audio_np)
        del processor, model, inputs, audio

    elif model_name == "mms":
        from transformers import VitsModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
        model     = VitsModel.from_pretrained("facebook/mms-tts-eng").to(device)
        inputs    = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)

        audio_np = (out.waveform[0].cpu().float().numpy() * 32767).astype("int16")
        scipy.io.wavfile.write(output_file, rate=model.config.sampling_rate, data=audio_np)
        del tokenizer, model, inputs, out

    else:
        raise ValueError(f"Unknown TTS model: {model_name!r}")

    _flush_gpu()
    print(f"[TTS/{model_name}] ✓ saved → {output_file}", flush=True)
    _send({"path": output_file})


# ═══════════════════════════════════════════════════════════════════
# Task: music_gen  (MusicGen-small, CPU)
# ═══════════════════════════════════════════════════════════════════

def _task_music_gen(args: dict):
    import gc
    import numpy as np
    import scipy.io.wavfile
    import torch
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    prompt          = args["prompt"]
    duration_secs   = args["duration_seconds"]
    output_filename = args["output_filename"]
    # NEW: Accept model_id from arguments, default to "medium"
    model_id = args.get("model_id", "facebook/musicgen-medium")

    _print_mem("before_music_gen")

    processor = AutoProcessor.from_pretrained(model_id)
    # NEW: Load model and move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[MusicGen] Loading model on {device}...", flush=True)
    model = MusicgenForConditionalGeneration.from_pretrained(model_id).to(device)

    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)

    # MusicGen generates 50 tokens per second of audio
    max_new_tokens = int(duration_secs * 50)

    print(f"[MusicGen] Generating {max_new_tokens} tokens on {device}...", flush=True)
    with torch.no_grad():
        audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.8)

    sr = model.config.audio_encoder.sampling_rate
    # Move audio to CPU for saving
    audio_np = (audio_values[0, 0].cpu().numpy() * 32767).astype("int16")
    scipy.io.wavfile.write(output_filename, rate=sr, data=audio_np)

    # Clean up GPU memory
    del model, processor, audio_values
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print(f"[MusicGen] ✓ saved → {output_filename}", flush=True)
    _send({"path": output_filename})

# ═══════════════════════════════════════════════════════════════════
# LLM addition
# ═══════════════════════════════════════════════════════════════════




def _task_llm_generate(args: dict):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    system_prompt = args["system_prompt"]
    user_prompt   = args["user_prompt"]
    model_id      = args.get("llm_model", "Qwen/Qwen2.5-1.5B-Instruct")
    max_tokens    = args.get("max_tokens", 512)
    temperature   = args.get("temperature", 0.7)

    _print_mem("before_llm_gen")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=_build_max_memory(vram_headroom_mb=512, ram_headroom_gb=1.0),
    )
    model.eval()
    _print_mem("after_llm_load")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(next(model.parameters()).device)

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=temperature)

    generated = out_ids[0][len(inputs.input_ids[0]):]
    raw_output = tokenizer.decode(generated, skip_special_tokens=True)

    _send({"output": raw_output})
# ═══════════════════════════════════════════════════════════════════
# Dispatch
# ═══════════════════════════════════════════════════════════════════

_TASKS = {
    "scene_gen": _task_scene_gen,
    "video_gen": _task_video_gen,
    "tts_gen":   _task_tts_gen,
    "music_gen": _task_music_gen,
    "llm_gen":   _task_llm_generate,   # new
}

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: _subprocess_runner.py <task> <json_args>", file=sys.stderr)
        sys.exit(1)

    task_name = sys.argv[1]
    task_args = json.loads(sys.argv[2])

    if task_name not in _TASKS:
        print(f"Unknown task: {task_name!r}. Valid: {list(_TASKS)}", file=sys.stderr)
        sys.exit(1)

    try:
        _TASKS[task_name](task_args)
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
        
        
        
"""HEllo mateo"""



