"""
VisualVerse Backend — FastAPI v2.0
====================================
Production NLP Pipeline:
  • Llama 3.2 3B Instruct (unsloth, 4-bit BnB)  → story generation
  • SpaCy NER + langdetect                        → NLP pre-pipeline
  • SVO dependency parsing (processor.py)         → structural image prompts
  • SD 1.5 + LCM LoRA (cuda:0, float16)          → image generation

Heavy models are loaded once in @app.on_event("startup") — never at import time.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Existing NLP modules ───────────────────────────────────────────────────────
from nlp.preprocessing.preprocessor import TextPreprocessor
from nlp.classification.classifier import TextClassifier
from nlp.keyphrase.extractor import KeyphraseExtractor
from nlp.topic_model.topic_modeler import TopicModeler
from nlp.relation.relation_extractor import RelationExtractor

# SVO structural image prompting (dependency parsing)
from nlp.processor import build_visual_prompt

from comic_gen.comic_generator import ComicGenerator
from mindmap_gen.mindmap_generator import MindMapGenerator
from story_gen.story_generator import StoryGenerator   # GPT-2 fallback

# ── New production models (loaded at startup) ─────────────────────────────────
from models.story_gen import LlamaStoryEngine
from models.image_gen import ImageGenerator

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="VisualVerse API",
    description="Production NLP System — Llama 3.2 3B (unsloth) + SD 1.5 LCM",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lightweight components (safe to init at module level) ─────────────────────
preprocessor       = TextPreprocessor()
classifier         = TextClassifier()
keyphrase_extractor = KeyphraseExtractor()
topic_modeler      = TopicModeler()
relation_extractor  = RelationExtractor()
comic_generator    = ComicGenerator()
mindmap_generator  = MindMapGenerator()
legacy_story_gen   = StoryGenerator()   # GPT-2 fallback

# Heavy GPU models — populated by startup event
llama_engine:     Optional[LlamaStoryEngine] = None
image_generator:  Optional[ImageGenerator]   = None


# ── Startup Event ─────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """
    Load heavy GPU models exactly once at server start.
    Both engines degrade gracefully if GPU / libs are unavailable.
    """
    global llama_engine, image_generator

    logger.info("=" * 60)
    logger.info(" VisualVerse — initialising production models …")
    logger.info("=" * 60)

    # 1. Llama 3.2 3B (unsloth, 4-bit NF4) — ~2 GB VRAM
    try:
        llama_engine = LlamaStoryEngine()
        logger.info(
            f"LlamaStoryEngine: {'✅ ready' if llama_engine.is_ready() else '⚠️  fallback mode'}"
        )
    except Exception as exc:
        logger.error(f"LlamaStoryEngine init failed: {exc}")
        llama_engine = None

    # 2. SD 1.5 + LCM LoRA — ~2.5 GB VRAM
    try:
        image_generator = ImageGenerator()
        logger.info(
            f"ImageGenerator: {'✅ ready' if image_generator.is_ready() else '⚠️  SVG placeholder mode'}"
        )
    except Exception as exc:
        logger.error(f"ImageGenerator init failed: {exc}")
        image_generator = None

    logger.info("=" * 60)
    logger.info(" Startup complete — serving at http://localhost:8000")
    logger.info("=" * 60)


# ── Request / Response Models ─────────────────────────────────────────────────
class TextInput(BaseModel):
    text: str
    mode: Optional[str] = "auto"      # "auto" | "comic" | "mindmap"
    language: Optional[str] = "auto"  # "auto" | "en" | "hi" | "ta"


class ProcessingResult(BaseModel):
    mode: str
    title: str
    summary: str
    language: Optional[str] = "en"
    comic_data: Optional[List[Dict[str, Any]]] = None
    mindmap_data: Optional[Dict[str, Any]] = None


class TrainingRequest(BaseModel):
    training_model: str
    dataset_name: Optional[str] = None

    class Config:
        protected_namespaces = ()


class ImageRequest(BaseModel):
    prompt: str
    gemini_api_key: Optional[str] = None


class StoryRequest(BaseModel):
    keywords: str
    language: Optional[str] = "en"   # "en" | "hi" | "ta"


class StoryResponse(BaseModel):
    story: str
    word_count: int
    language: str
    keywords: str
    characters: Optional[List[str]] = []
    locations: Optional[List[str]] = []
    model: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "message": "VisualVerse API v2.0",
        "version": "2.0.0",
        "engines": {
            "story":  "Llama-3.2-3B (unsloth, 4-bit NF4)",
            "image":  "SD 1.5 + LCM LoRA (cuda:0, float16)",
            "nlp":    "SpaCy en_core_web_sm + SVO dependency parsing",
        },
        "endpoints": {
            "process":        "/api/process",
            "generate_story": "/api/generate-story",
            "generate_image": "/api/generate-image",
            "health":         "/api/health",
        },
    }


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "supported_languages": ["en", "hi", "ta"],
        "models": {
            "preprocessor":        preprocessor.is_ready(),
            "classifier":          classifier.is_ready(),
            "keyphrase_extractor": keyphrase_extractor.is_ready(),
            "topic_modeler":       topic_modeler.is_ready(),
            "relation_extractor":  relation_extractor.is_ready(),
            "llama_engine":        llama_engine.is_ready() if llama_engine else False,
            "image_generator":     image_generator.is_ready() if image_generator else False,
        },
    }


@app.post("/api/classify")
async def classify_text(input_data: TextInput):
    try:
        lang = input_data.language if input_data.language != "auto" else None
        preprocessed = preprocessor.process(input_data.text, language=lang)
        classification = classifier.classify(preprocessed)
        return {
            "text_type":  classification["type"],
            "confidence": classification["confidence"],
            "features":   classification["features"],
            "language":   preprocessed.get("language", "en"),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/process", response_model=ProcessingResult)
async def process_text(input_data: TextInput):
    """Main pipeline — routes to comic or mindmap based on text type."""
    try:
        lang = input_data.language if input_data.language != "auto" else None
        preprocessed = preprocessor.process(input_data.text, language=lang)
        detected_language = preprocessed.get("language", "en")

        mode = input_data.mode
        if mode == "auto":
            classification = classifier.classify(preprocessed)
            mode = "comic" if classification["type"] == "narrative" else "mindmap"

        if mode == "comic":
            # Comic panels: use SVO-derived visual prompts for each panel
            result = await _generate_comic_with_svo(preprocessed)
            return ProcessingResult(
                mode="comic",
                title=result["title"],
                summary=result["summary"],
                language=detected_language,
                comic_data=result["panels"],
            )
        else:
            keyphrases = keyphrase_extractor.extract(preprocessed, top_k=20)
            topics = topic_modeler.model_topics(preprocessed, keyphrases)
            topics["original_text"] = preprocessed.get("original_text", "")
            relations = relation_extractor.extract(preprocessed, keyphrases)
            mindmap = mindmap_generator.generate(keyphrases, topics, relations)
            return ProcessingResult(
                mode="mindmap",
                title=mindmap["title"],
                summary=mindmap["summary"],
                language=detected_language,
                mindmap_data=mindmap["graph"],
            )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


async def _generate_comic_with_svo(preprocessed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate comic panels using build_visual_prompt (SVO dep-parse) for each panel.
    Replaces ComicGenerator's heuristic _generate_image_prompt with NLP-derived prompts.
    """
    result = await comic_generator.generate(preprocessed)

    if image_generator and image_generator.is_ready():
        original_text = preprocessed.get("original_text", "")
        for panel in result.get("panels", []):
            # Derive a structurally-grounded prompt from the panel caption
            panel_text = panel.get("full_text") or panel.get("caption") or original_text
            structured_prompt = build_visual_prompt(panel_text)
            panel["image_url"] = await image_generator.generate(
                prompt=structured_prompt,
                seed=panel.get("panel_number", 1),
            )

    return result


@app.post("/api/generate-story", response_model=StoryResponse)
async def generate_story(request: StoryRequest):
    """
    Generate a narrative story from keywords.

    NLP Pipeline (Llama engine):
      1. Language Detection — langdetect + Unicode script heuristics
      2. Named Entity Recognition — SpaCy PERSON + GPE → context injection
      3. Llama 3.2 3B Instruct (unsloth, 4-bit NF4 quantization)
      4. Post-processing — artefact removal, punctuation

    Falls back to GPT-2 models when Llama engine is unavailable.
    """
    if request.language not in ("en", "hi", "ta"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{request.language}'. Choose: en, hi, ta",
        )

    try:
        if llama_engine and llama_engine.is_ready():
            result = await llama_engine.generate(
                keywords=request.keywords,
                language=request.language,
            )
        else:
            logger.warning("Llama engine unavailable — using GPT-2 fallback")
            result = await legacy_story_gen.generate(
                keywords=request.keywords,
                language=request.language,
            )

        return StoryResponse(
            story=result["story"],
            word_count=result["word_count"],
            language=result["language"],
            keywords=result["keywords"],
            characters=result.get("characters", []),
            locations=result.get("locations", []),
            model=result.get("model"),
        )

    except Exception as exc:
        logger.error(f"Story generation error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/generate-image")
async def generate_image_endpoint(request: ImageRequest):
    """
    Generate a comic-style image from a text prompt.

    The raw prompt is first processed through build_visual_prompt()
    (SVO dependency parsing) to produce a structurally-grounded SD prompt.
    """
    try:
        # NLP step: convert raw text to SVO-structured visual prompt
        structured_prompt = build_visual_prompt(request.prompt)
        logger.info(f"Structural prompt: {structured_prompt[:100]}…")

        if image_generator and image_generator.is_ready():
            image_data = await image_generator.generate(
                prompt=structured_prompt,
                gemini_api_key=request.gemini_api_key
            )
        else:
            logger.warning("ImageGenerator unavailable — using ComicGenerator fallback")
            panel = {
                "prompt": structured_prompt,
                "panel_number": 1,
                "caption": request.prompt,
            }
            image_data = await comic_generator._generate_panel_image(panel)

        return {"image_url": image_data, "prompt_used": structured_prompt}

    except Exception as exc:
        logger.error(f"Image generation failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/train/{model_type}")
async def train_model(model_type: str, request: TrainingRequest):
    try:
        if model_type == "keyphrase":
            result = await keyphrase_extractor.train(request.dataset_name)
        elif model_type == "topic":
            result = await topic_modeler.train(request.dataset_name)
        elif model_type == "relation":
            result = await relation_extractor.train(request.dataset_name)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")
        return {"status": "success", "model_type": model_type, "result": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/models/status")
async def get_model_status():
    return {
        "keyphrase_model": {
            "trained": keyphrase_extractor.is_trained(),
            "metrics": keyphrase_extractor.get_metrics(),
        },
        "topic_model": {
            "trained": topic_modeler.is_trained(),
            "metrics": topic_modeler.get_metrics(),
        },
        "relation_model": {
            "trained": relation_extractor.is_trained(),
            "metrics": relation_extractor.get_metrics(),
        },
    }


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print(" VisualVerse v2.0  |  Llama 3.2 3B + SD 1.5 LCM")
    print(" Docs: http://localhost:8000/docs")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
