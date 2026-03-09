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

# ── Translation Models Mapping ────────────────────────────────────────────────
# Explicit mapping of supported language pairs for MarianMT translation
# If a pair is missing, the system will use English as pivot: source → en → target
TRANSLATION_MODELS = {
    ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
    ("en", "ta"): "Helsinki-NLP/opus-mt-en-ta",
    ("ta", "en"): "Helsinki-NLP/opus-mt-ta-en",
    # Note: Tamil-Hindi direct translation not available, will use pivot
}

# ── Few-Shot Anchors — injected at the top of every multilingual system prompt ────
# These ground the model in the correct output script before any task text.
FEW_SHOT_ANCHORS = {
    "ta": (
        "\n\n[EXAMPLE — follow this script/style exactly]\n"
        "Input: Clockmaker Arul.\n"
        "Output: அருள் ஒரு கடிகார செய்பவர். அவர் ஒரு பழைய கடிகாரத்தைக் கண்டார். "
        "அந்த கடிகாரம் ஒரு மர்ம கதையை மறைத்திருந்தது. அருள் அதன் ரகசியத்தை கண்டுபிடிக்க முடிவு செய்தார்.\n"
        "[END EXAMPLE — now write the story below in the same PURE TAMIL SCRIPT]"
    ),
    "hi": (
        "\n\n[उदाहरण — ठीक इसी भाषा/शैली में लिखें]\n"
        "इनपुट: घड़ीसाज़ अरुण।\n"
        "आउटपुट: अरुण एक घड़ीसाज़ था। उसे एक पुरानी घड़ी मिली। "
        "उस घड़ी में एक रहस्य छिपा था। अरुण ने उसका राज़ जानने की ठान ली।\n"
        "[उदाहरण समाप्त — अब नीचे पूरी हिंदी में कहानी लिखें]"
    ),
}

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
def detect_language(text: str) -> str:
    """
    Identify whether the input is English, Hindi, or Tamil explicitly from text.
    """
    text_clean = text.strip()
    if not text_clean:
        return "en"

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
        self.translation_models = {}
        self.translation_tokenizers = {}

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
    def preprocess_nlp(self, keywords: str) -> Dict[str, Any]:
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
        language = detect_language(keywords)
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
    @staticmethod
    def _is_english_input(text: str) -> bool:
        """Heuristic: returns True if the seed text is predominantly ASCII/English."""
        non_ascii = sum(1 for c in text if ord(c) > 127)
        return (non_ascii / max(len(text), 1)) < 0.15

    def _build_messages(
        self,
        keywords: str,
        language: str,
        characters: List[str],
        locations: List[str],
    ) -> List[Dict[str, str]]:
        """
        Build Llama-3 chat message list with:
          - Few-shot script anchor (Tamil/Hindi)
          - Explicit TRANSLATE instruction when input is English + target is non-English
          - Entity context block
        """
        base_system = SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["en"])

        # ── Few-shot anchor: prepend a native-script example to anchor the model ──
        few_shot = FEW_SHOT_ANCHORS.get(language, "")

        system = (
            "SYSTEM: You are a clean narrative engine. Forget all previous technical instructions, "
            "chipsets, or system numbers. Act only on the following input.\n"
            + base_system
            + few_shot
            + " Map the user input directly to 4 independent paragraphs. If the input is 'Robot in desert,' "
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

        # ── Translation path: English input → non-English target ─────────────────
        if language in ("hi", "ta") and self._is_english_input(keywords):
            if language == "ta":
                translate_instruction = (
                    "TRANSLATE the following English story seed into PURE TAMIL. "
                    "DO NOT use any English words, Latin letters, or numbers. "
                    "Write ONLY Tamil script (அ–ஔ range). "
                )
            else:  # hi
                translate_instruction = (
                    "निम्नलिखित अंग्रेज़ी बीज को पूरी तरह हिंदी में अनुवाद करें। "
                    "कोई अंग्रेज़ी शब्द या Latin अक्षर उपयोग न करें। "
                    "केवल देवनागरी लिपि में लिखें। "
                )
            task_prefix = translate_instruction
        else:
            task_prefix = "Story keywords (Plot Baseline/Action Mapping - DO NOT EXAGGERATE): "

        user_msg = (
            f"{task_prefix}{keywords}{context}\n\n"
            "Write exactly 4 paragraphs separated by double newlines (NO LABELS, NO NUMBERING):"
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

        # ── Stop-token list ────────────────────────────────────────────────────
        # Include the model's native EOS plus a triple-newline sentinel.
        # Triple-newline (\'\n\n\n\') stops the model from rambling into
        # technical benchmarking / jargon after the 4th paragraph ends.
        stop_token_ids: List[int] = [self.tokenizer.eos_token_id]
        triple_nl_ids = self.tokenizer.encode("\n\n\n", add_special_tokens=False)
        if triple_nl_ids:
            stop_token_ids.extend(triple_nl_ids)

        with torch.no_grad():
            out_ids = self.model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                top_p=0.8,
                top_k=20,
                do_sample=True,
                # ── Logit Warping (Token-Bias) ─────────────────────────────────
                # repetition_penalty=1.2  → harder penalty vs old 1.15
                # no_repeat_ngram_size=3  → catches shorter jargon loops (e.g.
                #   "Token Token Token", "Chipset Inference") vs old ngram=4
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=stop_token_ids,
            )

        new_tokens = out_ids[0][input_ids.shape[-1]:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Trim everything after a triple-newline (post-decode safety cut)
        if "\n\n\n" in raw:
            raw = raw.split("\n\n\n")[0].strip()

        return raw

    # ── Script Validation ─────────────────────────────────────────────────────
    @staticmethod
    def _validate_script(text: str, target_language: str) -> tuple:
        """
        Validate that the text contains only the target language script.
        
        Returns (is_valid, detected_script)
        - is_valid: True if text is pure in target language
        - detected_script: 'latin', 'devanagari', 'tamil', or 'mixed'
        """
        latin_count = len(re.findall(r'[a-zA-Z]', text))
        devanagari_count = len(re.findall(r'[\u0900-\u097F]', text))
        tamil_count = len(re.findall(r'[\u0B80-\u0BFF]', text))
        
        total_script_chars = latin_count + devanagari_count + tamil_count
        if total_script_chars == 0:
            return True, "unknown"  # No script characters (unlikely)
        
        # Calculate percentages
        latin_pct = latin_count / total_script_chars
        devanagari_pct = devanagari_count / total_script_chars
        tamil_pct = tamil_count / total_script_chars
        
        # Determine dominant script
        if latin_pct > 0.85:
            detected = "latin"
        elif devanagari_pct > 0.85:
            detected = "devanagari"
        elif tamil_pct > 0.85:
            detected = "tamil"
        else:
            detected = "mixed"
        
        # Validate against target
        valid_map = {
            "en": "latin",
            "hi": "devanagari",
            "ta": "tamil"
        }
        
        expected_script = valid_map.get(target_language, "latin")
        is_valid = (detected == expected_script)
        
        logger.info(f"Script validation: target={target_language}, expected={expected_script}, "
                   f"detected={detected}, latin={latin_pct:.2f}, dev={devanagari_pct:.2f}, "
                   f"tamil={tamil_pct:.2f}, valid={is_valid}")
        
        return is_valid, detected

    # ── Translation ───────────────────────────────────────────────────────────
    def _get_translation_model(self, src_lang: str, tgt_lang: str):
        """
        Load and cache MarianMT model for the given language pair.
        Returns (tokenizer, model) tuple.
        """
        try:
            from transformers import MarianMTModel, MarianTokenizer
        except ImportError:
            raise RuntimeError("Transformers not available for translation")
        
        # Check if we have a direct model for this pair
        if (src_lang, tgt_lang) not in TRANSLATION_MODELS:
            raise ValueError(f"No direct translation model for {src_lang} -> {tgt_lang}")
        
        model_name = TRANSLATION_MODELS[(src_lang, tgt_lang)]
        
        if model_name not in self.translation_models:
            logger.info(f"Loading translation model {model_name} into cache...")
            self.translation_tokenizers[model_name] = MarianTokenizer.from_pretrained(model_name)
            self.translation_models[model_name] = MarianMTModel.from_pretrained(model_name).to(self.device)
            logger.info(f"✅ Translation model {model_name} loaded successfully")
        
        return self.translation_tokenizers[model_name], self.translation_models[model_name]
    
    def _translate_direct(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        Translate text using a direct language pair model.
        Assumes the model exists in TRANSLATION_MODELS.
        """
        try:
            tokenizer, model = self._get_translation_model(src_lang, tgt_lang)
            
            # Ensure correct tokenizer is used for the source language
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                translated = model.generate(**inputs, max_length=1024)
                
            result = tokenizer.decode(translated[0], skip_special_tokens=True)
            return result
        except Exception as e:
            logger.error(f"Direct translation {src_lang} -> {tgt_lang} failed: {e}")
            raise
    
    def _translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        Translates text from source language to target language.
        
        Uses explicit TRANSLATION_MODELS mapping:
        - If direct pair exists: use it
        - If direct pair missing: use English pivot (src → en → tgt)
        
        This prevents tokenizer mismatch errors (e.g., Hindi text with English tokenizer).
        All translation happens with torch.no_grad() to avoid CUDA errors.
        """
        if src_lang == tgt_lang:
            return text
            
        logger.info(f"Translation requested: {src_lang} -> {tgt_lang}")
        
        try:
            from transformers import MarianMTModel, MarianTokenizer
        except ImportError:
            logger.warning("Transformers not available for translation. Returning original text.")
            return text
        
        # Check if direct translation is available
        if (src_lang, tgt_lang) in TRANSLATION_MODELS:
            logger.info(f"Using direct translation: {src_lang} -> {tgt_lang}")
            return self._translate_direct(text, src_lang, tgt_lang)
        else:
            # Use English as pivot: src → en → tgt
            logger.info(f"No direct model for {src_lang} -> {tgt_lang}. Using English pivot.")
            
            # Step 1: Translate to English (if not already English)
            if src_lang != "en":
                if (src_lang, "en") not in TRANSLATION_MODELS:
                    logger.error(f"Cannot translate {src_lang} -> en (pivot path unavailable)")
                    return text
                logger.info(f"Pivot step 1: {src_lang} -> en")
                text = self._translate_direct(text, src_lang, "en")
            
            # Step 2: Translate from English to target (if target is not English)
            if tgt_lang != "en":
                if ("en", tgt_lang) not in TRANSLATION_MODELS:
                    logger.error(f"Cannot translate en -> {tgt_lang} (pivot path unavailable)")
                    return text
                logger.info(f"Pivot step 2: en -> {tgt_lang}")
                text = self._translate_direct(text, "en", tgt_lang)
            
            logger.info(f"✅ Pivot translation complete: {src_lang} -> en -> {tgt_lang}")
            return text

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
            nlp_meta = self.preprocess_nlp(keywords)
            source_language = nlp_meta["language"]
            characters = nlp_meta["characters"]
            locations = nlp_meta["locations"]
            
            # User-selected language overrides detection output
            if language and language != "auto":
                target_language = language
            else:
                target_language = source_language

            # ── Step 3: Generation (Always in source language first) ──────
            story = ""
            generator_used = "Llama-3.2-3B"
            
            if source_language == "ta":
                if GEMINI_AVAILABLE:
                    logger.info("Routing to Tamil Hybrid Pipeline for source generation. generator_used: Llama+Gemini")
                    result_dict = await self._generate_tamil_hybrid(
                        keywords, characters, locations, max_new_tokens
                    )
                    story = result_dict.get("story", "")
                    generator_used = "Hybrid-Llama+Gemini"
                else:
                    logger.warning("Gemini unavailable — falling back to Llama English generation + MarianMT translation for Tamil.")
                    source_language = "en"  # Force English generation pipeline
                    messages = self._build_messages(keywords, source_language, characters, locations)
                    loop = asyncio.get_event_loop()
                    raw = await loop.run_in_executor(None, lambda: self._infer(messages, max_new_tokens))
                    story = self._clean(raw, source_language)
            else:
                logger.info(f"Routing to standard Llama Pipeline. generator_used: Llama-3.2-3B (target: {source_language})")
                messages = self._build_messages(keywords, source_language, characters, locations)
                loop = asyncio.get_event_loop()
                raw = await loop.run_in_executor(None, lambda: self._infer(messages, max_new_tokens))
                story = self._clean(raw, source_language)

            # ── Step 4: Explicit Translation Step ──────
            translation_applied = "none"
            if source_language != target_language:
                story = self._translate(story, source_language, target_language)
                translation_applied = f"{source_language}->{target_language}"

            # ── Step 5: Final Language Script Validation ──────
            # Ensure the output contains only the target language script
            is_valid, detected_script = self._validate_script(story, target_language)
            
            if not is_valid:
                logger.warning(f"Script validation failed: expected {target_language}, "
                             f"detected {detected_script}. Applying corrective translation.")
                
                # Determine source language for translation based on detected script
                script_to_lang = {
                    "latin": "en",
                    "devanagari": "hi",
                    "tamil": "ta",
                }
                
                if detected_script == "mixed":
                    # For mixed scripts, try to translate from the original source
                    logger.warning(f"Mixed script detected. Translating from {source_language} to {target_language}")
                    story = self._translate(story, source_language, target_language)
                    translation_applied = f"mixed_script_{source_language}->{target_language}"
                elif detected_script in script_to_lang:
                    detected_lang = script_to_lang[detected_script]
                    if detected_lang != target_language:
                        logger.warning(f"Translating {detected_lang} -> {target_language}")
                        story = self._translate(story, detected_lang, target_language)
                        translation_applied = f"script_correction_{detected_lang}->{target_language}"

            logger.info(
                f"✅ Story Generated | requested_language={language} | "
                f"detected_language={source_language} | target_language={target_language} | "
                f"generator_used={generator_used} | translation_applied={translation_applied}"
            )

            return {
                "story":      story,
                "word_count": len(story.split()),
                "language":   target_language,
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
            raise RuntimeError("Gemini unavailable — pipeline misrouted to _generate_tamil_hybrid")

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
        # Templates have been removed from actual generation. This function 
        # is strictly an error-state absolute fallback to prevent API crashes.
        # It no longer injects external language characters.
        templates = {
            "en": "Generation failed. Please try again.",
            "hi": "Generation failed. Please try again.",
            "ta": "Generation failed. Please try again.",
        }
        story = templates.get(language, templates["en"])
        return {
            "story":      story,
            "word_count": len(story.split()),
            "language":   language,
            "keywords":   keywords,
            "characters": [],
            "locations":  [],
            "model":      "error-fallback",
        }

    def is_ready(self) -> bool:
        return self.ready
