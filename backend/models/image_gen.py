"""
backend/models/image_gen.py
============================
Local GPU Image Generation — SD 1.5 + LCM LoRA

Model     : stable-diffusion-v1-5/stable-diffusion-v1-5
Adapter   : latent-consistency-models/lcm-sd15-weights  (LoRA, fused at load)
Device    : cuda:0, torch.float16
VRAM      : ~2.5 GB with attention_slicing + vae_slicing

Why LCM LoRA?
  Standard SD 1.5 needs 20-25 DDIM steps (~8 s/image).
  LCM LoRA distills this to 4-8 steps (~1.5 s/image, same quality).
  Guidance scale must be LOW (1.0-2.0) — higher values break LCM conditioning.

Usage:
  gen = ImageGenerator()
  uri = await gen.generate("A brave knight climbing a dark mountain")
  # → "data:image/png;base64,..."
"""

import asyncio
import base64
import logging
from io import BytesIO
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# ── Optional deps ─────────────────────────────────────────────────────────────
try:
    from diffusers import LCMScheduler, StableDiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logger.warning("diffusers not installed — ImageGenerator will use SVG placeholder")

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("google-genai not installed — Gemini API integration disabled")

# ── Constants ─────────────────────────────────────────────────────────────────
SD15_ID   = "stable-diffusion-v1-5/stable-diffusion-v1-5"
LCM_ID    = "latent-consistency/lcm-lora-sdv1-5"

NEG_PROMPT = (
    "blurry, low quality, deformed, ugly, bad anatomy, "
    "extra limbs, watermark, signature, text, nsfw"
)

# LCM optimal: 4-8 steps; guidance 1.0-2.0
DEFAULT_STEPS    = 6
DEFAULT_GUIDANCE = 1.5
IMG_SIZE         = 512


class ImageGenerator:
    """
    Stable Diffusion 1.5 + LCM LoRA image generator.

    VRAM breakdown (float16, cuda:0):
      UNet       ≈ 1.7 GB
      VAE        ≈ 0.4 GB
      CLIP text  ≈ 0.2 GB
      Activations≈ 0.2 GB
      Total      ≈ 2.5 GB  ✅ well under 3 GB target
    """

    def __init__(self):
        self.pipe = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.ready = False
        import threading
        self._lock = threading.Lock()

        if not DIFFUSERS_AVAILABLE:
            logger.error("diffusers library not installed — ImageGenerator disabled")
            return
        if not torch.cuda.is_available():
            logger.warning("No CUDA GPU — ImageGenerator uses SVG placeholders")
            return

        self._load()

    # ── Model Loading ─────────────────────────────────────────────────────────
    def _load(self):
        try:
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Loading {SD15_ID} on {gpu_name} [float16] …")

            # Base SD 1.5 pipeline — float16 on cuda:0
            self.pipe = StableDiffusionPipeline.from_pretrained(
                SD15_ID,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False,
            )

            # Swap scheduler to LCMScheduler — enables 4-8 step inference
            self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)

            # Load + fuse LCM LoRA (merges adapter weights, zero overhead at runtime)
            self.pipe.load_lora_weights(LCM_ID)
            self.pipe.fuse_lora()

            # Move to GPU
            self.pipe = self.pipe.to(self.device)

            # VRAM optimizations — keeps usage under 3 GB
            self.pipe.enable_attention_slicing(slice_size=1)
            self.pipe.enable_vae_slicing()
            self.pipe.set_progress_bar_config(disable=True)

            self.ready = True
            vram_gb = torch.cuda.memory_allocated() / 1e9
            logger.info(f"✅ SD 1.5 + LCM ready | VRAM: {vram_gb:.2f} GB")

        except Exception as exc:
            logger.error(f"❌ ImageGenerator load failed: {exc}", exc_info=True)
            self.ready = False

    # ── Synchronous Inference ─────────────────────────────────────────────────
    def _run(
        self,
        prompt: str,
        negative_prompt: str,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> str:
        """Return a base64 PNG data-URI. Called inside run_in_executor."""
        with self._lock:
            gen = torch.Generator(device=self.device).manual_seed(seed)

            with torch.no_grad():
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    width=IMG_SIZE,
                    height=IMG_SIZE,
                    generator=gen,
                )

            buf = BytesIO()
            result.images[0].save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            return f"data:image/png;base64,{b64}"

    # ── Gemini Inference ──────────────────────────────────────────────────────
    def _run_gemini(self, prompt: str, api_key: str) -> str:
        """Return a base64 JPEG data-URI from Gemini API. Called inside run_in_executor."""
        client = genai.Client(api_key=api_key)
        
        # Use Nano Banana 2 preview version
        model = "gemini-3.1-flash-image-preview"
        
        result = client.models.generate_images(
            model=model,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                output_mime_type="image/jpeg",
                aspect_ratio="1:1"
            )
        )
        
        if result.generated_images and len(result.generated_images) > 0:
            generated_image = result.generated_images[0]
            # Depending on exactly how the SDK returns byte payloads, 
            # usually it's in image.image_bytes
            b64 = base64.b64encode(generated_image.image.image_bytes).decode('utf-8')
            return f"data:image/jpeg;base64,{b64}"
            
        raise RuntimeError("No image returned from Gemini API")

    # ── Public Async API ──────────────────────────────────────────────────────
    async def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE,
        seed: int = 42,
        gemini_api_key: Optional[str] = None,
    ) -> str:
        """
        Generate an image and return a base64 PNG/JPEG data-URI.

        Args:
            prompt          : SD prompt (ideally from build_visual_prompt()).
            negative_prompt : What to avoid (defaults to quality preset).
            num_steps       : LCM steps — 4 to 8 recommended.
            guidance_scale  : CFG — keep between 1.0 and 2.0 for LCM.
            seed            : RNG seed for reproducibility.
            gemini_api_key  : Optional. If present, use Gemini API.

        Returns:
            "data:image/...;base64,..."  or SVG placeholder on failure.
        """
        # Dynamic Routing to Gemini API
        if gemini_api_key and GENAI_AVAILABLE:
            try:
                loop = asyncio.get_event_loop()
                data = await loop.run_in_executor(
                    None,
                    lambda: self._run_gemini(prompt, gemini_api_key),
                )
                logger.info("✅ Image generated via Gemini API")
                return data
            except Exception as exc:
                logger.error(f"❌ Gemini Image generation error: {exc}", exc_info=True)
                logger.info("Falling back to local SD 1.5 pipeline...")

        # Local SD 1.5 Pipeline Fallback
        if not self.ready:
            return self._placeholder(prompt)

        neg = negative_prompt or NEG_PROMPT

        try:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None,
                lambda: self._run(prompt, neg, num_steps, guidance_scale, seed),
            )
            logger.info("✅ Image generated")
            return data
        except Exception as exc:
            logger.error(f"❌ Image generation error: {exc}", exc_info=True)
            return self._placeholder(prompt)

    # ── SVG Placeholder ───────────────────────────────────────────────────────
    @staticmethod
    def _placeholder(prompt: str = "") -> str:
        short = (prompt[:60] + "...") if len(prompt) > 60 else prompt
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512">'
            '<rect width="512" height="512" fill="#1e1e2e"/>'
            '<text x="256" y="220" font-family="Arial" font-size="72" '
            'fill="#cdd6f4" text-anchor="middle">🎨</text>'
            '<text x="256" y="295" font-family="Arial" font-size="20" '
            'fill="#a6adc8" text-anchor="middle">Image unavailable</text>'
            f'<text x="256" y="330" font-family="Arial" font-size="12" '
            f'fill="#585b70" text-anchor="middle">{short}</text>'
            "</svg>"
        )
        b64 = base64.b64encode(svg.encode()).decode()
        return f"data:image/svg+xml;base64,{b64}"

    def is_ready(self) -> bool:
        return self.ready
