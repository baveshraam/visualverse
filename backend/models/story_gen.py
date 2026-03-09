"""
backend/models/story_gen.py
============================
Hybrid Story Engine
  - English / Hindi : Llama 3.2 3B Instruct (unsloth, 4-bit NF4)
  - Tamil           : Llama→Gemini hybrid pipeline
                       1. Llama generates 4 English SVO plot points
                       2. Gemini 1.5 Flash expands them into Tamil prose
                       3. Regex guardrails applied to final Gemini output

Model   : unsloth/Llama-3.2-3B-Instruct
Quant   : 4-bit NF4 bitsandbytes  (~2 GB VRAM)
NLP pre : SpaCy (NER) + langdetect (language ID)
Async   : inference dispatched via asyncio.run_in_executor
"""

import asyncio
import logging
import os
import re
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

# ── Optional deps ─────────────────────────────────────────────────────────────
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed — LlamaStoryEngine disabled")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spacy not installed — NER disabled")

try:
    from langdetect import detect as _langdetect
    from langdetect.lang_detect_exception import LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logger.warning("langdetect not installed — defaulting to 'en'")

try:
    from google import genai as _genai_module
    _gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    if _gemini_api_key:
        _gemini_client = _genai_module.Client(api_key=_gemini_api_key)
        GEMINI_AVAILABLE = True
        logger.info("✅ Gemini API (google-genai v1) configured for Tamil expansion")
    else:
        _gemini_client = None
        GEMINI_AVAILABLE = False
        logger.warning("GEMINI_API_KEY not set — Tamil hybrid path disabled")
except ImportError:
    _gemini_client = None
    GEMINI_AVAILABLE = False
    logger.warning("google-genai not installed — Tamil hybrid path disabled")

# ── Config ────────────────────────────────────────────────────────────────────
# Using unsloth's non-gated mirror — no HuggingFace access request needed
MODEL_ID = "unsloth/Llama-3.2-3B-Instruct"
GEMINI_MODEL = "gemini-1.5-flash"

SYSTEM_PROMPTS = {
    "en": (
        "You are a master storyteller applying Structured Narrative Design. "
        "Create exactly 4 paragraphs, with each paragraph representing one vivid comic scene. "
        "The story MUST be written entirely in the English script. "
        "Output ONLY the story text — NO LABELS. Do not use 'Panel 1', 'Scene 1', 'Paragraph 1', or any numbers. "
        "Use double newlines (\\n\\n) to separate the four scenes."
    ),
    "hi": (
        "आप एक कुशल कथाकार हैं और संरचित कथा-रचना (Structured Narrative Design) का पालन करते हैं। "
        "ठीक 4 पैराग्राफ बनाएं, जहां प्रत्येक पैराग्राफ एक दृश्य (कॉमिक पैनल) का प्रतिनिधित्व करता है। "
        "कहानी को पूरी तरह से केवल हिंदी (Devanagari) स्क्रिप्ट में लिखना अनिवार्य है। "
        "केवल कहानी का पाठ दें — 'पैनल 1' या 'दृश्य 1' जैसे कोई लेबल या नंबर नहीं। कोई शीर्षक या अतिरिक्त नोट्स नहीं।"
    ),
    # Tamil is handled by the Gemini hybrid path — no Llama prose prompt needed.
    # This SVO prompt is used only for the Llama SVO extraction stage.
    "ta_svo": (
        "You are a plot-outliner for a comic strip. "
        "Output EXACTLY 4 numbered English sentences in Subject–Verb–Object (SVO) format. "
        "Each sentence describes one comic panel action. "
        "RULES: English ONLY. No Tamil. No prose. No extra commentary. "
        "Each sentence must be under 20 words. Number them 1 to 4."
    ),
}

# Shared lazy SpaCy NLP object
_spacy_nlp = None


# ── Language Detection ────────────────────────────────────────────────────────
def detect_language(text: str, hint: Optional[str] = None) -> str:
    """
    Identify whether the input is English, Hindi, or Tamil.

    Steps:
      1. Honour explicit caller hint (en / hi / ta).
      2. Unicode script range check — Devanagari → hi, Tamil → ta.
      3. langdetect probabilistic fallback.
      4. Default to 'en'.
    """
    if hint and hint in ("en", "hi", "ta"):
        return hint

    # Fast Unicode heuristic
    devanagari = sum(1 for c in text if "\u0900" <= c <= "\u097F")
    tamil_chars = sum(1 for c in text if "\u0B80" <= c <= "\u0BFF")
    if devanagari > 5:
        return "hi"
    if tamil_chars > 5:
        return "ta"

    if LANGDETECT_AVAILABLE:
        try:
            detected = _langdetect(text)
            if detected in ("en", "hi", "ta"):
                return detected
        except LangDetectException:
            pass

    return "en"


# ── SpaCy NER ─────────────────────────────────────────────────────────────────
def _get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None and SPACY_AVAILABLE:
        try:
            _spacy_nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("SpaCy model 'en_core_web_sm' missing. Run: python -m spacy download en_core_web_sm")
    return _spacy_nlp


def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract PERSON (characters) and GPE/LOC (locations) using SpaCy NER.
    These entities are injected into the system prompt so the LLM maintains
    narrative consistency (no hallucinated new character names).

    Returns: {"characters": [...], "locations": [...]}
    """
    result: Dict[str, List[str]] = {"characters": [], "locations": []}
    nlp = _get_spacy()
    if nlp is None:
        return result

    try:
        doc = nlp(text[:512])
        seen_c: set = set()
        seen_l: set = set()
        for ent in doc.ents:
            name = ent.text.strip()
            if ent.label_ == "PERSON" and name.lower() not in seen_c:
                result["characters"].append(name)
                seen_c.add(name.lower())
            elif ent.label_ in ("GPE", "LOC") and name.lower() not in seen_l:
                result["locations"].append(name)
                seen_l.add(name.lower())
    except Exception as exc:
        logger.warning(f"NER extraction error: {exc}")

    return result


# ── Llama Story Engine ────────────────────────────────────────────────────────
class LlamaStoryEngine:
    """
    Story generation engine — Llama 3.2 3B Instruct (unsloth mirror, non-gated).

    NLP Pipeline before every LLM call:
      1. Language Detection  — langdetect + Unicode heuristics
      2. Named Entity Recognition — SpaCy PERSON + GPE/LOC
      3. Prompt Construction — inject entities as narrative context
      4. Generation          — Llama 3.2 3B with 4-bit NF4 quantization
      5. Post-processing     — strip artefacts, ensure punctuation
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ready = False

        if not TRANSFORMERS_AVAILABLE:
            logger.error("transformers not available — LlamaStoryEngine is disabled")
            return
        self._load()

    # ── Load ──────────────────────────────────────────────────────────────────
    def _load(self):
        try:
            logger.info(f"Loading {MODEL_ID} with 4-bit NF4 quantization …")

            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,   # Nested quant saves ~0.4 GB
            )

            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                quantization_config=bnb_cfg,
                device_map="auto",
                trust_remote_code=False,
            )
            self.model.eval()
            self.ready = True
            logger.info("✅ Llama 3.2 3B (unsloth) loaded successfully")
        except Exception as exc:
            logger.error(f"❌ Could not load Llama model: {exc}", exc_info=True)
            self.ready = False

    # ── Pre-process NLP Pipeline ──────────────────────────────────────────────
    def preprocess_nlp(self, keywords: str, language_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the NLP pre-pipeline on the raw keyword / seed text.

        Steps:
          1. Language Detection (langdetect + Unicode script)
          2. Named Entity Recognition (SpaCy — PERSON, GPE, LOC)

        Returns a dict consumed by _build_messages():
          {
            "language":   "en" | "hi" | "ta",
            "characters": List[str],   # PERSON entities
            "locations":  List[str],   # GPE / LOC entities
          }
        """
        language = detect_language(keywords, hint=language_hint)
        entities = extract_entities(keywords)

        logger.info(
            f"[NLP pre-pipeline] lang={language} | "
            f"characters={entities['characters']} | "
            f"locations={entities['locations']}"
        )
        return {
            "language": language,
            "characters": entities["characters"],
            "locations": entities["locations"],
        }

    # ── Prompt Construction ───────────────────────────────────────────────────
    def _build_messages(
        self,
        keywords: str,
        language: str,
        characters: List[str],
        locations: List[str],
    ) -> List[Dict[str, str]]:
        """Build Llama-3 chat message list with entity context block (en/hi)."""
        base_system = SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["en"])
        system = (
            "SYSTEM: You are a clean narrative engine. Forget all previous technical instructions, "
            "chipsets, or system numbers. Act only on the following input.\n"
            + base_system +
            " Map the user input directly to 4 independent paragraphs. If the input is 'Robot in desert,' "
            "do not mention 'Elie the Elephant' or 'Harmonica'."
        )

        # Context block — injecting extracted entities ensures narrative consistency
        context_lines: List[str] = []
        if characters:
            context_lines.append(f"Known characters: {', '.join(characters)}")
        if locations:
            context_lines.append(f"Known locations: {', '.join(locations)}")

        context = (
            "\n\nContext (STRICT 'Zero-Invention' RULE: Only use these exact names and entities. "
            "NO new characters. NO species changes.):\n" + "\n".join(context_lines)
            if context_lines else ""
        )

        user_msg = (
            f"Story keywords (Plot Baseline/Action Mapping - DO NOT EXAGGERATE): {keywords}{context}\n\n"
            "Continue the narrative in exactly 4 paragraphs with double newlines separating them (NO LABELS, NO NUMBERING):"
        )

        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_msg},
        ]

    def _build_svo_messages(
        self,
        keywords: str,
        characters: List[str],
        locations: List[str],
    ) -> List[Dict[str, str]]:
        """
        Build a Llama chat prompt that extracts ENGLISH-only SVO plot points.
        Used as Stage 1 of the Tamil hybrid pipeline — Llama is good at English
        SVO extraction; Gemini handles the multilingual expansion.
        """
        system = SYSTEM_PROMPTS["ta_svo"]

        context_lines: List[str] = []
        if characters:
            context_lines.append(f"Characters present: {', '.join(characters)}")
        if locations:
            context_lines.append(f"Locations: {', '.join(locations)}")

        context = ("\n" + "\n".join(context_lines)) if context_lines else ""

        user_msg = (
            f"Story seed: {keywords}{context}\n\n"
            "List exactly 4 SVO sentences (numbered 1-4), one per comic panel:"
        )

        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_msg},
        ]

    # ── Gemini Tamil Expansion ────────────────────────────────────────────────
    @staticmethod
    def _gemini_expand_tamil(svo_text: str, characters: List[str], locations: List[str]) -> str:
        """
        Stage 2 of the Tamil hybrid pipeline.
        Takes 4 English SVO bullet points from Llama and asks Gemini Flash
        to expand each into one vivid Tamil paragraph (~60 words each).

        Raises RuntimeError if Gemini is unavailable.
        """
        if not GEMINI_AVAILABLE:
            raise RuntimeError("Gemini API not available")

        entity_note = ""
        if characters or locations:
            parts = []
            if characters:
                parts.append(f"Characters: {', '.join(characters)}")
            if locations:
                parts.append(f"Locations: {', '.join(locations)}")
            entity_note = (
                "\nKeep these entities consistent throughout: "
                + "; ".join(parts) + "."
            )

        prompt = (
            "You are a master Tamil storyteller writing a 4-panel comic strip narrative.\n"
            "Below are 4 English plot points (one per comic panel).\n"
            "Expand EACH plot point into exactly ONE vivid Tamil paragraph of approximately 50-70 words.\n"
            "STRICT RULES:\n"
            "  - Write ONLY in Tamil script (Unicode \u0B80\u2013\u0BFF). Zero English letters.\n"
            "  - No panel labels, no numbering, no headings.\n"
            "  - Separate the 4 paragraphs with a blank line (double newline).\n"
            "  - Maintain narrative flow across all 4 paragraphs as a single story."
            + entity_note
            + "\n\nPlot points:\n"
            + svo_text
            + "\n\nWrite the Tamil story now:"
        )

        response = _gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "temperature": 0.7,
                "top_p": 0.9,
                "max_output_tokens": 1024,
            },
        )
        return response.text.strip()

    # ── Synchronous Inference ─────────────────────────────────────────────────
    def _infer(self, messages: List[Dict[str, str]], max_new_tokens: int) -> str:
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            out_ids = self.model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                top_p=0.8,
                top_k=20,
                do_sample=True,
                repetition_penalty=1.15,
                no_repeat_ngram_size=4,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = out_ids[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # ── Post-process ──────────────────────────────────────────────────────────
    @staticmethod
    def _clean(text: str, language: str = "en") -> str:
        # Strip structural labels generated by mistake (e.g. Panel 1:, Scene 2)
        text = re.sub(r'^(?:Panel|Scene|Paragraph)\s*\d+[:.\-]?\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Hard-Filter: Strip sequences of digits (e.g. 8-9-10)
        text = re.sub(r'\b\d+(?:-\d+)*\b', '', text)
        
        # Hard-Filter: Strip specific jargon text
        text = re.sub(r'\b(?:High-Age|Chipset|Screen-Tone|Token|Inference)\b', '', text, flags=re.IGNORECASE)
        
        # Hard-Filter: Target language constraints
        if language in ('hi', 'ta'):
            text = re.sub(r'[A-Za-z]+', '', text)
        elif language == 'en':
            text = re.sub(r'[\u0900-\u097F\u0B80-\u0BFF]+', '', text)

        # Check and remove repetitive stuttering (e.g., "the the the")
        text = re.sub(r'\b(\w+)(?: \1)+\b', r'\1', text, flags=re.IGNORECASE)
        # Fix punctuation repeats
        text = re.sub(r'([.?!।])\1+', r'\1', text)
        
        # Split into pure paragraphs
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        # Enforce exactly 4 paragraphs constraint
        if len(paragraphs) > 4:
            paragraphs = paragraphs[:4]
            
        text = "\n\n".join(paragraphs)
        
        if text and text[-1] not in ".!?।":
            text += "."
        
        # Capitalize the first letter if needed
        if text and len(text) > 0 and text[0].islower():
            text = text[0].upper() + text[1:]
        
        return text

    # ── Public Async API ──────────────────────────────────────────────────────
    async def generate(
        self,
        keywords: str,
        language: str = "en",
        max_new_tokens: int = 350,
    ) -> Dict[str, Any]:
        """
        Generate a story from keywords using the full NLP pipeline.

        Language routing:
          en / hi  →  Llama 3.2 3B Instruct only (unchanged).
          ta       →  Hybrid pipeline:
                        Stage 1 — Llama extracts 4 English SVO plot points
                        Stage 2 — Gemini 1.5 Flash expands SVOs into Tamil prose
                        Stage 3 — Regex guardrails applied to Gemini output

        Hyperparameters (en/hi): temperature=0.2, top_k=20, top_p=0.8

        Args:
            keywords       : Story seed (free-form text or comma-separated keywords)
            language       : Requested language code ('en' | 'hi' | 'ta')
            max_new_tokens : Token budget for Llama (350 ≈ 250 words)

        Returns dict with keys: story, word_count, language, keywords,
                                characters, locations, model
        """
        if not self.ready:
            return self._fallback(keywords, language)

        try:
            # Step 1 + 2: Language detection + NER (all languages)
            nlp_meta = self.preprocess_nlp(keywords, language_hint=language)
            lang = nlp_meta["language"]
            characters = nlp_meta["characters"]
            locations = nlp_meta["locations"]

            # ── Tamil: Hybrid Llama→Gemini Path ──────────────────────────────
            if lang == "ta":
                return await self._generate_tamil_hybrid(
                    keywords, characters, locations, max_new_tokens
                )

            # ── English / Hindi: Llama-only Path ─────────────────────────────
            # Step 3: Structured prompt
            messages = self._build_messages(keywords, lang, characters, locations)

            # Step 4: Generation (non-blocking)
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(
                None, lambda: self._infer(messages, max_new_tokens)
            )

            # Step 5: Post-process
            story = self._clean(raw, lang)
            logger.info(f"✅ Story generated — {len(story.split())} words [{lang}]")

            return {
                "story":      story,
                "word_count": len(story.split()),
                "language":   lang,
                "keywords":   keywords,
                "characters": characters,
                "locations":  locations,
                "model":      "llama-3.2-3b-unsloth-4bit",
            }

        except Exception as exc:
            logger.error(f"❌ Generation failed: {exc}", exc_info=True)
            return self._fallback(keywords, language)

    # ── Tamil Hybrid Pipeline ─────────────────────────────────────────────────
    async def _generate_tamil_hybrid(
        self,
        keywords: str,
        characters: List[str],
        locations: List[str],
        max_new_tokens: int,
    ) -> Dict[str, Any]:
        """
        Two-stage Tamil generation:
          Stage 1 — Llama 3.2 3B → 4 English SVO sentences (plot skeleton)
          Stage 2 — Gemini 1.5 Flash → 4 Tamil paragraphs (narrative expansion)
          Stage 3 — Regex guardrails on the Gemini response

        Falls back to _fallback() if Gemini is unavailable.
        """
        if not GEMINI_AVAILABLE:
            logger.warning("Gemini unavailable — falling back to Tamil template")
            return self._fallback(keywords, "ta")

        try:
            loop = asyncio.get_event_loop()

            # ── Stage 1: Llama → English SVO plot points ──────────────────────
            logger.info("[Tamil hybrid] Stage 1: Llama SVO extraction …")
            svo_messages = self._build_svo_messages(keywords, characters, locations)
            # Keep SVO output short — we only need 4 sentences
            svo_raw = await loop.run_in_executor(
                None, lambda: self._infer(svo_messages, min(max_new_tokens, 150))
            )
            logger.info(f"[Tamil hybrid] Llama SVO output: {svo_raw[:200]}")

            # ── Stage 2: Gemini → Tamil prose expansion ───────────────────────
            logger.info("[Tamil hybrid] Stage 2: Gemini Tamil expansion …")
            tamil_raw = await loop.run_in_executor(
                None,
                lambda: self._gemini_expand_tamil(svo_raw, characters, locations),
            )

            # ── Stage 3: Regex guardrails on Gemini output ────────────────────
            story = self._clean(tamil_raw, "ta")
            logger.info(
                f"✅ Tamil hybrid story generated — {len(story.split())} words"
            )

            return {
                "story":      story,
                "word_count": len(story.split()),
                "language":   "ta",
                "keywords":   keywords,
                "characters": characters,
                "locations":  locations,
                "model":      "hybrid-llama-svo+gemini-ta",
            }

        except Exception as exc:
            logger.error(f"❌ Tamil hybrid generation failed: {exc}", exc_info=True)
            return self._fallback(keywords, "ta")

    # ── Fallback ──────────────────────────────────────────────────────────────
    @staticmethod
    def _fallback(keywords: str, language: str) -> Dict[str, Any]:
        templates = {
            "en": (
                f"In a world shaped by {keywords}, a remarkable tale unfolds. "
                "The protagonist embarks on a journey filled with wonder and peril. "
                "Every step reveals a new challenge, yet resolve never wavers."
            ),
            "hi": (
                f"{keywords} से प्रेरित एक अद्भुत कहानी है। "
                "नायक एक असाधारण यात्रा पर निकलता है। "
                "हर कदम पर नई चुनौतियाँ आती हैं।"
            ),
            "ta": (
                f"{keywords} சார்ந்த ஒரு அற்புதமான கதை. "
                "கதாநாயகன் ஒரு அசாதாரண பயணத்தை மேற்கொள்கிறார்."
            ),
        }
        story = templates.get(language, templates["en"])
        return {
            "story":      story,
            "word_count": len(story.split()),
            "language":   language,
            "keywords":   keywords,
            "characters": [],
            "locations":  [],
            "model":      "fallback-template",
        }

    def is_ready(self) -> bool:
        return self.ready
