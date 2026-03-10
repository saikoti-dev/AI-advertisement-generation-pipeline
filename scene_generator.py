import gc
import torch
from transformers import pipeline as hf_pipeline


def generate_prompts(product: str, num_scenes: int = 2) -> list[str]:
    """
    Uses LLaMA to generate CogVideoX-optimised scene prompts for the product.
    Returns a list of `num_scenes` prompt strings.
    """
    print("Loading LLaMA for scene prompt generation…")
    llm = hf_pipeline(
        "text-generation",
        model="meta-llama/Llama-3.2-1B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You generate video prompts for CogVideoX, a text-to-video diffusion model. "
                "CogVideoX works best with prompts that describe ONE continuous 5-second shot: "
                "subject, action, environment, camera movement, lighting, and mood — all in a "
                "single dense paragraph of 60-80 words. "
                "Never use lists, scene numbers, or labels. Only output the raw prompt paragraphs, "
                "one per line, nothing else."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Write exactly {num_scenes} CogVideoX video prompts for a luxury ad for {product}. "
                f"All shots must feature {product} as the hero subject. "
                "The shots should flow as a narrative: "
                f"1) {product} beautifully presented in an elegant environment, soft studio lighting, camera slowly pushing in. "
                f"2) Close-up detail shot of {product} highlighting its texture, colour, and quality. "
                "Each prompt must be one paragraph, 60-80 words, cinematic, photorealistic, 4K."
            ),
        },
    ]

    out = llm(messages, max_new_tokens=900)
    raw = out[0]["generated_text"][-1]["content"]

    # One prompt per non-empty line
    prompts = [ln.strip() for ln in raw.splitlines() if len(ln.strip()) > 40][:num_scenes]

    # Fallback prompts if LLaMA output is short
    while len(prompts) < num_scenes:
        prompts.append(
            f"{product} elegantly presented on a clean surface with soft golden studio lighting. "
            f"Camera slowly pushes in revealing fine details. Shallow depth of field. "
            f"Luxury product advertisement. Photorealistic. 4K."
        )

    print(f"Generated {len(prompts)} scene prompts:")
    for i, p in enumerate(prompts, 1):
        print(f"  [{i}] {p[:90]}…")

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return prompts
