import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Style prompts mapped to artist names
STYLE_PROMPTS = {
    "Picasso (Cubism)": "in the style of Pablo Picasso, cubism, fragmented geometric shapes, bold outlines, multiple viewpoints, oil painting",
    "Van Gogh":         "in the style of Vincent van Gogh, swirling brushstrokes, post-impressionism, vivid colors, thick impasto",
    "Monet":            "in the style of Claude Monet, impressionism, soft light, water reflections, loose brushwork, pastel colors",
    "Kandinsky":        "in the style of Wassily Kandinsky, abstract expressionism, bold colors, geometric shapes",
    "Munch (Scream)":   "in the style of Edvard Munch, expressionism, swirling lines, dramatic emotion, dark tones",
}

_pipe = None

def get_pipeline():
    global _pipe
    if _pipe is None:
        _pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            safety_checker=None,
        ).to(DEVICE)
        _pipe.enable_attention_slicing()   # saves memory on CPU
    return _pipe

def run_style_transfer(
    content_img: Image.Image,
    style_name:  str   = "Picasso (Cubism)",
    strength:    float = 0.75,   # 0 = no change, 1 = full transformation
    guidance:    float = 12.0,   # how closely to follow the style prompt
    steps:       int   = 50,
) -> Image.Image:

    pipe = get_pipeline()

    content_img = content_img.convert("RGB").resize((512, 512))
    prompt = f"A portrait {STYLE_PROMPTS[style_name]}, masterpiece, highly detailed"
    negative = "blurry, ugly, deformed, watermark, text, low quality, photo, realistic"

    result = pipe(
        prompt          = prompt,
        negative_prompt = negative,
        image           = content_img,
        strength        = strength,
        guidance_scale  = guidance,
        num_inference_steps = steps,
    ).images[0]

    return result