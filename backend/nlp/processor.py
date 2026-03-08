"""
backend/nlp/processor.py
=========================
Structural Image Prompting via SpaCy Dependency Parsing

NLP Technique: Universal Dependency Parsing
  nsubj  → nominal subject of the verb
  ROOT   → main predicate verb
  dobj   → direct object
  pobj   → prepositional object (fallback for intransitives)
  amod   → adjectival modifier (makes descriptions richer)
  compound → compound noun modifier

Public API:
  extract_svo_triples(text)         → List[{subject, verb, obj}]
  build_visual_prompt(story_text)   → str  (SD-ready prompt)

Example:
  Input : "The brave knight climbed the dark mountain."
  Triple: subject="brave knight", verb="climb", obj="dark mountain"
  Prompt: "A brave knight, climbing a dark mountain, cinematic lighting, high quality"
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── SpaCy (lazy-loaded) ───────────────────────────────────────────────────────
_nlp = None


def _get_nlp():
    """Lazy-load SpaCy en_core_web_sm (shared singleton)."""
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
        logger.info("processor.py: SpaCy en_core_web_sm loaded")
    except Exception as exc:
        logger.warning(f"processor.py: SpaCy unavailable — {exc}")
        _nlp = None
    return _nlp


# ── Internal helpers ──────────────────────────────────────────────────────────
def _noun_phrase(token) -> str:
    """
    Build a rich noun phrase for a head token by collecting:
      - adjectival modifiers (amod)
      - compound noun modifiers (compound)

    E.g. token='knight', children=[amod:'brave'] → "brave knight"
    """
    mods: List[str] = []
    for child in token.children:
        if child.dep_ in ("amod", "compound") and child.i < token.i:
            sub = [c.text for c in child.children if c.dep_ == "compound" and c.i < child.i]
            mods.append(" ".join(sub + [child.text]))
    return " ".join(mods + [token.text]).strip()


def _find_object(verb) -> Optional[str]:
    """
    Walk verb children to find the primary object.
    Priority: dobj → attr → pobj (via prep child)
    """
    for child in verb.children:
        if child.dep_ in ("dobj", "attr"):
            return _noun_phrase(child)
    # Prepositional object: verb → prep → pobj
    for child in verb.children:
        if child.dep_ == "prep":
            for gc in child.children:
                if gc.dep_ == "pobj":
                    return f"{child.text} {_noun_phrase(gc)}"
    return None


def _to_participle(lemma: str) -> str:
    """
    Convert a verb lemma to a present-participle form.
    Simple heuristic — good enough for cinematic prompt generation.
      climb  → climbing
      chase  → chasing
      run    → running
    """
    if lemma.endswith("ie"):
        return lemma[:-2] + "ying"          # die → dying
    if lemma.endswith("e") and not lemma.endswith("ee"):
        return lemma[:-1] + "ing"           # climb + e removed  → climbing
    # CVC doubling for short words: run → running
    if re.search(r"[^aeiou][aeiou][^aeiou]$", lemma) and len(lemma) <= 5:
        return lemma + lemma[-1] + "ing"
    return lemma + "ing"


# ── Public API ────────────────────────────────────────────────────────────────
def extract_svo_triples(text: str) -> List[Dict[str, str]]:
    """
    Extract Subject-Verb-Object triples using SpaCy dependency parsing.

    For each sentence:
      1. Locate ROOT verb.
      2. Find nsubj / nsubjpass child → subject.
      3. Find dobj / attr / pobj → object.
      4. Enrich subject and object with adjective + compound modifiers.

    Args:
        text : English story text (sentences processed up to 2 000 chars).

    Returns:
        List of dicts, each: {"subject": str, "verb": str (lemma), "obj": str}

    Example:
        [
          {"subject": "brave knight", "verb": "climb", "obj": "dark mountain"},
          {"subject": "dragon",       "verb": "guard", "obj": "golden treasure"},
        ]
    """
    nlp = _get_nlp()
    if nlp is None:
        return []

    triples: List[Dict[str, str]] = []
    doc = nlp(text[:2000])

    for sent in doc.sents:
        # Find ROOT verb
        root = next(
            (t for t in sent if t.dep_ == "ROOT" and t.pos_ in ("VERB", "AUX")),
            None,
        )
        if root is None:
            continue

        # Find subject
        subj_tok = next(
            (c for c in root.children if c.dep_ in ("nsubj", "nsubjpass")),
            None,
        )
        if subj_tok is None:
            continue

        triples.append({
            "subject": _noun_phrase(subj_tok),
            "verb":    root.lemma_,
            "obj":     _find_object(root) or "",
        })

    logger.debug(f"extract_svo_triples: {len(triples)} triple(s) found")
    return triples


def build_visual_prompt(story_text: str, style: Optional[str] = None) -> str:
    """
    Convert story text into a Stable Diffusion image prompt via SVO parsing.

    Pipeline:
      1. extract_svo_triples(story_text)
      2. Format each triple:  "A {subject}, {verb}ing {obj}"
      3. Join up to 4 triples and append cinematic quality tags.

    Args:
        story_text : Generated story paragraph(s).
        style      : Optional comma-separated style override.
                     Default: "cinematic lighting, high quality, detailed, 8k, masterpiece"

    Returns:
        Prompt string ready for Stable Diffusion (≤ 400 chars).

    Example:
        Input : "The brave knight climbed the dark mountain. A dragon breathed fire."
        Output: "A brave knight, climbing the dark mountain, a dragon, breathing fire,
                 cinematic lighting, high quality, detailed, 8k, masterpiece"
    """
    if style is None:
        style = "cinematic lighting, high quality, detailed, 8k, masterpiece"

    triples = extract_svo_triples(story_text)
    parts: List[str] = []

    for triple in triples[:4]:   # cap at 4 to stay within CLIP token limit
        subj = triple["subject"].strip()
        verb = triple["verb"].strip()
        obj  = triple["obj"].strip()

        if not subj:
            continue

        participle = _to_participle(verb)
        phrase = f"A {subj}, {participle} {obj}".strip() if obj else f"A {subj}, {participle}"
        parts.append(phrase)

    prompt = (", ".join(parts) + ", " + style) if parts else (story_text[:150] + ", " + style)

    # Hard-cap at 400 chars to avoid CLIP truncation
    if len(prompt) > 400:
        prompt = prompt[:397] + "..."

    logger.info(f"build_visual_prompt: {len(parts)} SVO part(s), {len(prompt)} chars")
    return prompt
