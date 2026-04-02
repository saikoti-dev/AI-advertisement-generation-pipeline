"""
video_generator.py
------------------
Renders each scene prompt as a video clip using CogVideoX-5b.
Each scene spawns its own subprocess — the GPU is completely free between scenes.
OOM recovery (3 attempts with reduced settings) is handled inside the subprocess.
"""

from __future__ import annotations
from pathlib import Path
from tqdm import tqdm as _tqdm

from _proc import run_task

SCENE_DIR = Path("scenes")


def render_scenes(
    prompts: list[str],
    output_dir: Path = SCENE_DIR,
    model_id: str = "THUDM/CogVideoX-2b",
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_paths: list[Path] = []

    for i, prompt in enumerate(_tqdm(prompts, desc="Rendering scenes", unit="scene"), 1):
        path = output_dir / f"scene_{i:02d}.mp4"
        print(f"\n[VideoGen] Scene {i}/{len(prompts)} → {path.name}")
        print(f"  Prompt: {prompt[:80]}…")

        result = run_task("video_gen", {
            "prompt":    prompt,
            "path":      str(path),
            "seed":      42 + i,
            "offload":   "disk",
            "model_id":  model_id,
        })
        clip_paths.append(Path(result["path"]))
        print(f"[VideoGen] ✓ Scene {i} done")

    print(f"\n[VideoGen] ✓ All {len(clip_paths)} scenes rendered.")
    return clip_paths
