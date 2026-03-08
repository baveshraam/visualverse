"""
Comic Generator Module - GPU-Only Image Generation
Generates comic strips using Stable Diffusion 1.5 on local GPU
"""

import base64
import logging
from typing import Dict, Any, List
from io import BytesIO
import torch

logger = logging.getLogger(__name__)


class ComicGenerator:
    """
    Comic Strip Generator using LOCAL GPU ONLY
    
    Pipeline:
    1. Segment story into scenes/beats
    2. Extract characters and settings for each scene
    3. Generate optimized image prompts
    4. Generate images using Stable Diffusion 1.5 on GPU
    5. Return base64-encoded images or SVG placeholders
    """
    
    def __init__(self):
        """Initialize comic generator with Stable Diffusion 1.5"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.use_local_gpu = False
        
        # Initialize Stable Diffusion (use already-downloaded Tiny-SD)
        if torch.cuda.is_available():
            try:
                from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
                
                logger.info(f"🚀 Loading Tiny-SD to GPU: {torch.cuda.get_device_name(0)}")
                
                # Use Tiny-SD - already downloaded, fast and reliable
                model_id = "segmind/tiny-sd"
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    requires_safety_checker=False
                )
                
                # Move to GPU
                self.pipe = self.pipe.to(self.device)
                
                # Critical optimizations to fix tensor indexing errors
                self.pipe.enable_attention_slicing(slice_size=1)
                self.pipe.enable_vae_slicing()
                
                # Disable progress bars
                self.pipe.set_progress_bar_config(disable=True)
                
                self.use_local_gpu = True
                logger.info("✅ Tiny-SD ready on GPU!")
                
            except Exception as e:
                logger.error(f"❌ Failed to load Tiny-SD: {e}")
                logger.info("Will use SVG placeholders instead")
                self.use_local_gpu = False
                self.pipe = None
        else:
            logger.warning("⚠️ No CUDA GPU detected - using SVG placeholders")
        
    async def generate(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comic strip from preprocessed text
        
        Returns:
            Dict with:
            - title: Story title
            - summary: Brief summary
            - panels: List of panel dicts with caption, image_url, characters, setting
        """
        text = preprocessed.get("original_text", "")
        sentences = preprocessed.get("sentences", [])
        characters = preprocessed.get("characters", [])
        locations = preprocessed.get("locations", [])
        
        # Generate title
        title = self._generate_title(text, preprocessed)
        
        # Generate summary
        summary = self._generate_summary(sentences)
        
        # Segment into panels (4-6 panels typically)
        panels = self._segment_into_panels(sentences, characters, locations)
        
        # Generate images for each panel
        for panel in panels:
            panel["image_url"] = await self._generate_panel_image(panel)
        
        return {
            "title": title,
            "summary": summary,
            "panels": panels
        }
    
    def _generate_title(self, text: str, preprocessed: Dict[str, Any]) -> str:
        """Generate a title for the comic"""
        characters = preprocessed.get("characters", [])
        
        # Use first character's name if available
        if characters:
            return f"The Story of {characters[0]}"
        
        # Extract first significant noun phrase
        noun_phrases = preprocessed.get("noun_phrases", [])
        if noun_phrases:
            return f"A Tale of {noun_phrases[0].title()}"
        
        # Default title
        first_words = text.split()[:5]
        return " ".join(first_words) + "..."
    
    def _generate_summary(self, sentences: List[str]) -> str:
        """Generate a brief summary"""
        if not sentences:
            return "A visual story unfolds..."
        
        # Use first sentence as summary
        summary = sentences[0]
        if len(summary) > 150:
            summary = summary[:147] + "..."
        
        return summary
    
    def _segment_into_panels(self, sentences: List[str], 
                            characters: List[str], 
                            locations: List[str]) -> List[Dict[str, Any]]:
        """
        Segment story into comic panels.
        
        Panel count scales dynamically with input size:
        - 1-2 sentences  → 1-2 panels (one per sentence)
        - 3-4 sentences  → 3-4 panels
        - 5-7 sentences  → 4-5 panels
        - 8-12 sentences → 5-7 panels
        - 13-20 sentences → 7-10 panels
        - 21-30 sentences → 10-12 panels
        - 31+ sentences  → 12-16 panels (capped)
        
        Roughly: ~1 panel per 2 sentences, min 1, max 16
        """
        n = len(sentences)
        
        if n <= 2:
            num_panels = max(1, n)
        elif n <= 4:
            num_panels = n  # 1 panel per sentence for short stories
        elif n <= 7:
            # ~1 panel per 1.5 sentences
            num_panels = max(4, round(n / 1.5))
        elif n <= 12:
            # ~1 panel per 1.7 sentences
            num_panels = max(5, round(n / 1.7))
        elif n <= 20:
            # ~1 panel per 2 sentences
            num_panels = max(7, round(n / 2))
        elif n <= 30:
            # ~1 panel per 2.5 sentences
            num_panels = max(10, round(n / 2.5))
        else:
            # ~1 panel per 3 sentences, capped at 16
            num_panels = min(16, max(12, round(n / 3)))
        
        if len(sentences) < num_panels:
            # If too few sentences, use each sentence as a panel
            panels = []
            for i, sent in enumerate(sentences):
                panels.append(self._create_panel(i + 1, sent, characters, locations))
            return panels
        
        # Distribute sentences across panels
        sentences_per_panel = len(sentences) // num_panels
        panels = []
        
        for i in range(num_panels):
            start_idx = i * sentences_per_panel
            end_idx = start_idx + sentences_per_panel if i < num_panels - 1 else len(sentences)
            
            panel_text = " ".join(sentences[start_idx:end_idx])
            panels.append(self._create_panel(i + 1, panel_text, characters, locations))
        
        return panels
    
    def _create_panel(self, panel_num: int, text: str, 
                     characters: List[str], locations: List[str]) -> Dict[str, Any]:
        """Create a single panel structure"""
        # Generate image prompt from text
        prompt = self._generate_image_prompt(text, characters, locations)
        
        # Use full text as caption (no truncation)
        caption = text
        
        return {
            "id": f"panel_{panel_num}",
            "panel_number": panel_num,
            "caption": caption,
            "full_text": text,
            "prompt": prompt,
            "characters": characters,
            "setting": locations[0] if locations else "Unknown location",
            "image_url": None  # Will be filled by image generation
        }
    
    def _generate_image_prompt(self, text: str, 
                               characters: List[str], 
                               locations: List[str]) -> str:
        """
        Generate a DreamShaper-optimized prompt for the panel.
        
        DreamShaper excels with detailed, descriptive prompts including
        style keywords, quality boosters, and negative prompt hints.
        """
        # DreamShaper quality boosters
        quality = "masterpiece, best quality, highly detailed"
        style = "comic book art style, vibrant colors, dynamic composition, cel shading, bold outlines"
        
        # Scene description (simplified from text)
        scene = text[:180] if len(text) > 180 else text
        
        # Characters
        char_desc = ""
        if characters:
            char_desc = f"featuring {', '.join(characters[:2])}, "
        
        # Location
        loc_desc = ""
        if locations:
            loc_desc = f"set in {locations[0]}, "
        
        # Mood/atmosphere
        mood = self._detect_mood(text)
        mood_desc = f"{mood} atmosphere, cinematic lighting"
        
        prompt = f"{quality}, {style}, {char_desc}{loc_desc}{scene}, {mood_desc}"
        
        return prompt
    
    def _detect_mood(self, text: str) -> str:
        """Detect the mood/atmosphere of the text"""
        text_lower = text.lower()
        
        # Check for mood indicators
        if any(word in text_lower for word in ["happy", "joy", "laugh", "smile", "excited"]):
            return "cheerful and bright"
        elif any(word in text_lower for word in ["sad", "cry", "tears", "lonely", "grief"]):
            return "melancholic and somber"
        elif any(word in text_lower for word in ["angry", "rage", "fight", "battle", "war"]):
            return "intense and dramatic"
        elif any(word in text_lower for word in ["fear", "dark", "scary", "horror", "terror"]):
            return "dark and mysterious"
        elif any(word in text_lower for word in ["love", "romance", "heart", "kiss"]):
            return "romantic and warm"
        elif any(word in text_lower for word in ["adventure", "journey", "discover", "explore"]):
            return "adventurous and exciting"
        else:
            return "neutral and balanced"
    
    async def _generate_panel_image(self, panel: Dict[str, Any]) -> str:
        """
        Generate image for a panel using LOCAL GPU ONLY.
        No external APIs - simple and reliable.
        """
        panel_num = panel.get('panel_number', 1)
        prompt = panel.get('prompt', '')
        caption = panel.get('caption', '')
        
        # Try GPU generation if available
        if self.use_local_gpu and self.pipe is not None:
            try:
                logger.info(f"🎨 Generating panel {panel_num} on GPU...")
                
                # Build optimized prompt
                comic_prompt = f"comic book style, vibrant colors, professional illustration, {prompt}"
                negative_prompt = "blurry, low quality, deformed, ugly, bad anatomy, watermark, signature, text"
                
                # Generate image with proper error handling
                with torch.no_grad():
                    try:
                        result = self.pipe(
                            prompt=comic_prompt[:77],  # Truncate to avoid CLIP errors
                            negative_prompt=negative_prompt,
                            num_inference_steps=15,  # Faster for small model
                            guidance_scale=7.0,
                            width=512,
                            height=512,
                            generator=torch.Generator(device=self.device).manual_seed(panel_num)
                        )
                        image = result.images[0]
                    except RuntimeError as e:
                        if "index" in str(e) and "out of bounds" in str(e):
                            # Retry with even more conservative settings
                            logger.warning(f"Retrying with conservative settings...")
                            result = self.pipe(
                                prompt=comic_prompt[:50],
                                num_inference_steps=10,
                                guidance_scale=6.0,
                                width=512,
                                height=512,
                            )
                            image = result.images[0]
                        else:
                            raise
                
                # Convert to base64
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                img_bytes = buffer.getvalue()
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                
                logger.info(f"✅ Panel {panel_num} generated successfully!")
                return f"data:image/png;base64,{b64}"
                
            except Exception as e:
                logger.error(f"❌ GPU generation failed for panel {panel_num}: {str(e)}")
                logger.info(f"Using SVG placeholder for panel {panel_num}")
        
        # Fallback: SVG placeholder
        return self._get_placeholder_image(panel_num, caption)
    
    def _get_placeholder_image(self, panel_number: int, caption: str = "") -> str:
        """
        Generate a beautiful SVG placeholder image
        Fast and works without external API
        """
        colors = [
            ("#FF6B6B", "#C44D4D"),  # Red
            ("#4ECDC4", "#36A89F"),  # Teal
            ("#45B7D1", "#2E8DA8"),  # Blue
            ("#96CEB4", "#6BAF8F"),  # Green
            ("#FFEAA7", "#D4C680"),  # Yellow
            ("#DDA0DD", "#B87AB8"),  # Purple
        ]
        
        color, dark_color = colors[(panel_number - 1) % len(colors)]
        
        # Shorten caption for display
        short_caption = caption[:50] + "..." if len(caption) > 50 else caption
        
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
            <defs>
                <linearGradient id="bg{panel_number}" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:{color}"/>
                    <stop offset="100%" style="stop-color:{dark_color}"/>
                </linearGradient>
            </defs>
            <rect width="512" height="512" fill="url(#bg{panel_number})"/>
            <rect x="10" y="10" width="492" height="492" fill="none" stroke="white" stroke-width="4" rx="10"/>
            <text x="256" y="180" font-family="Comic Sans MS, cursive, sans-serif" font-size="72" fill="white" text-anchor="middle" opacity="0.9">🎬</text>
            <text x="256" y="280" font-family="Arial, sans-serif" font-size="36" fill="white" text-anchor="middle" font-weight="bold">PANEL {panel_number}</text>
            <text x="256" y="380" font-family="Arial, sans-serif" font-size="16" fill="white" text-anchor="middle" opacity="0.8">{short_caption}</text>
            <text x="256" y="470" font-family="Arial, sans-serif" font-size="12" fill="white" text-anchor="middle" opacity="0.5">VisualVerse Comic Generator</text>
        </svg>'''
        
        # Return as base64 encoded SVG
        svg_bytes = svg.encode('utf-8')
        b64 = base64.b64encode(svg_bytes).decode('utf-8')
        return f"data:image/svg+xml;base64,{b64}"
    
    def generate_comic_layout(self, panels: List[Dict[str, Any]], 
                             layout: str = "grid") -> Dict[str, Any]:
        """
        Generate final comic layout configuration
        
        Layouts:
        - grid: 2x2 or 2x3 grid
        - vertical: Single column
        - manga: Right-to-left reading
        """
        num_panels = len(panels)
        
        if layout == "grid":
            if num_panels <= 2:
                rows, cols = 1, num_panels
            elif num_panels <= 4:
                rows, cols = 2, 2
            elif num_panels <= 6:
                rows, cols = 2, 3
            elif num_panels <= 9:
                rows, cols = 3, 3
            else:
                rows, cols = 4, 3
        elif layout == "vertical":
            rows, cols = num_panels, 1
        elif layout == "manga":
            rows, cols = 2, 2  # Will be reversed in frontend
        else:
            rows, cols = 2, 2
        
        return {
            "layout": layout,
            "rows": rows,
            "cols": cols,
            "panel_order": list(range(len(panels))),
            "reading_direction": "rtl" if layout == "manga" else "ltr"
        }
