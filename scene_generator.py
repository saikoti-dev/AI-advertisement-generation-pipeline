"""
scene_generator.py
------------------
Generates CogVideoX scene prompts using Qwen2.5-1.5B-Instruct.
Runs the model in a subprocess — when it exits the GPU is completely free.
"""

from __future__ import annotations
from _proc import run_task


def generate_prompts(product: str, num_scenes: int = 2, llm_model: str = None) -> list[str]:
    """
    Generate scene prompts using a local LLM.
    Args:
        product:      Product description.
        num_scenes:   Number of prompts to generate.
        llm_model:    Optional Hugging Face model ID (e.g., "Qwen/Qwen2.5-7B-Instruct").
                      Defaults to "Qwen/Qwen2.5-1.5B-Instruct" if None.
    """
    print(f"[SceneGen] product={product!r}  num_scenes={num_scenes}")
    if llm_model is None:
        llm_model = "Qwen/Qwen2.5-1.5B-Instruct"

    result = run_task("scene_gen", {
        "product":    product,
        "num_scenes": num_scenes,
        "llm_model":  llm_model,
    })
    prompts = result["prompts"]
    print(f"[SceneGen] ✓ {len(prompts)} prompts received:")
    for i, p in enumerate(prompts, 1):
        print(f"  [{i}] {p[:90]}…")
    return prompts
    
    
    
    
    
    
from _proc import run_task

def generate_voiceover_script(product: str, scene_prompts: list[str] = None) -> str:
    system = (
        "You write short, persuasive voiceover scripts for luxury product ads. "
        "The script should be 2-3 sentences, spoken by a warm, confident male or female voice. "
        "Use emotional language and end with a call to action."
    )
    if scene_prompts:
        scenes_summary = "\n".join(f"- {p[:50]}..." for p in scene_prompts[:2])
        user = f"Product: {product}\nScenes:\n{scenes_summary}\n\nWrite a 2-sentence voiceover script."
    else:
        user = f"Write a 2-sentence voiceover script for a luxury advertisement for {product}."

    result = run_task("llm_gen", {
        "system_prompt": system,
        "user_prompt": user,
        "max_tokens": 150,
        "temperature": 0.8,
    })
    return result["output"].strip()

def generate_music_prompt(product: str, duration_secs: int = 30, mood: str = "cinematic epic") -> str:
    system = (
        "You generate detailed music prompts for MusicGen (text-to-music). "
        "Describe instruments, tempo, structure (intro/build/climax), and mood. "
        "Never mention vocals. Aim for 30-50 words."
    )
    user = (
        f"Product: {product}\nDuration: {duration_secs} seconds\nMood: {mood}\n"
        "Write a music prompt that would produce a rich, professional background track for a luxury ad."
    )
    result = run_task("llm_gen", {
        "system_prompt": system,
        "user_prompt": user,
        "max_tokens": 200,
        "temperature": 0.7,
    })
    return result["output"].strip()
