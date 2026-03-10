import gc
from pathlib import Path

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video


SCENE_DIR = Path("scenes")


def render_scenes(prompts: list[str], output_dir: Path = SCENE_DIR) -> list[Path]:
    """
    Renders each prompt as a short video clip using CogVideoX-5b.
    Returns a list of Path objects pointing to the rendered .mp4 files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CogVideoX-5b…")
    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

    clip_paths: list[Path] = []

    for i, prompt in enumerate(prompts, 1):
        path = output_dir / f"scene_{i:02d}.mp4"
        print(f"\nRendering scene {i}/{len(prompts)} → {path}")

        frames = pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=50,
            num_frames=49,
            guidance_scale=6,
            generator=torch.Generator(device="cuda").manual_seed(42 + i),
        ).frames[0]

        export_to_video(frames, str(path), fps=8)
        clip_paths.append(path)
        print(f"  ✓ Scene {i} saved.")

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return clip_paths
