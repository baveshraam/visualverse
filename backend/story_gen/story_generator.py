"""
Story Generator Module
Generates narrative stories from keywords using trained GPT-2 models

Usage:
    generator = StoryGenerator()
    story = await generator.generate(
        keywords="genre: sci-fi, hero: astronaut, plot: Mars mission",
        language="en"
    )
"""

import os
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Try to import transformers
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available - story generation will use fallback")


class StoryGenerator:
    """
    Story Generator using trained distilgpt2 models
    
    Pipeline:
    1. Load trained model for specified language
    2. Format keywords with special tokens
    3. Generate story using trained model
    4. Post-process and return clean story text
    """
    
    # Special tokens (must match training)
    KEYWORDS_TOKEN = "<|keywords|>"
    STORY_TOKEN = "<|story|>"
    EOS_TOKEN = "<|endoftext|>"
    
    def __init__(self, models_dir: str = "backend/models"):
        """Initialize the story generator"""
        self.models_dir = Path(models_dir)
        self.models = {}
        self.tokenizers = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available - using fallback mode")
            return
        
        # Pre-load models for all languages
        for lang in ['en', 'hi', 'ta']:
            try:
                self._load_model(lang)
                logger.info(f"Loaded story model for {lang}")
            except Exception as e:
                logger.warning(f"Could not load model for {lang}: {e}")
    
    def _load_model(self, language: str):
        """Load trained model and tokenizer for a language"""
        # Load the default multilingual story models
        model_path = self.models_dir / f"story_model_{language}"
        
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            return False
        
        try:
            # Load tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained(str(model_path))
            
            # Load model
            model = GPT2LMHeadModel.from_pretrained(str(model_path))
            model.to(self.device)
            model.eval()
            
            # Cache
            self.tokenizers[language] = tokenizer
            self.models[language] = model
            
            return True
        except Exception as e:
            logger.error(f"Error loading model for {language}: {e}")
            return False
    
    async def generate(self, keywords: str, language: str = "en", 
                      max_length: int = 400, temperature: float = 0.8) -> Dict[str, Any]:
        """
        Generate a story from keywords
        
        Args:
            keywords: Comma-separated keywords or free-form keyword text
            language: 'en', 'hi', or 'ta'
            max_length: Maximum tokens to generate (default 400 ≈ 300 words)
            temperature: Sampling temperature (0.7-0.9 recommended)
        
        Returns:
            Dict with story text, word count, and metadata
        """
        
        # Check if model is available
        if not TRANSFORMERS_AVAILABLE or language not in self.models:
            logger.warning(f"Model not available for {language}, using fallback")
            return self._generate_fallback_story(keywords, language)
        
        try:
            model = self.models[language]
            tokenizer = self.tokenizers[language]
            
            # Format prompt with special tokens
            prompt = f"{self.KEYWORDS_TOKEN} {keywords} {self.STORY_TOKEN}"
            
            # Tokenize
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=100  # Limit keyword length
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=max_length,
                    min_length=200,  # Ensure minimum story length
                    temperature=temperature,
                    top_p=0.9,
                    top_k=50,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    repetition_penalty=1.2,  # Reduce repetition
                    no_repeat_ngram_size=3,  # Prevent repeating 3-grams
                )
            
            # Decode - skip special tokens to avoid decoding errors
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract story (remove prompt keywords if present)
            story = self._extract_story(generated_text)
            
            # Post-process
            story = self._post_process_story(story, language)
            
            # Calculate stats
            word_count = len(story.split())
            
            logger.info(f"Generated {language} story: {word_count} words")
            
            return {
                "story": story,
                "word_count": word_count,
                "language": language,
                "keywords": keywords,
                "model": "distilgpt2-finetuned"
            }
            
        except Exception as e:
            logger.error(f"Error generating story: {e}")
            return self._generate_fallback_story(keywords, language)
    
    def _extract_story(self, generated_text: str) -> str:
        """Extract clean story from generated text
        
        Note: Since we decode with skip_special_tokens=True,
        special tokens are already removed from the text.
        """
        # Remove any remaining special tokens (in case they appear as text)
        story = generated_text.replace(self.KEYWORDS_TOKEN, '')
        story = story.replace(self.STORY_TOKEN, '')
        story = story.replace(self.EOS_TOKEN, '')
        story = story.replace('<|pad|>', '')
        
        # Clean up extra whitespace
        story = ' '.join(story.split())
        
        return story.strip()
    
    def _post_process_story(self, story: str, language: str) -> str:
        """Clean up generated story"""
        
        # Remove leading/trailing whitespace
        story = story.strip()
        
        # Ensure story ends with punctuation
        if story and story[-1] not in '.!?।':
            story += '.'
        
        # Remove multiple spaces
        while '  ' in story:
            story = story.replace('  ', ' ')
        
        # Remove multiple newlines
        while '\n\n\n' in story:
            story = story.replace('\n\n\n', '\n\n')
        
        # Capitalize first letter
        if story and len(story) > 0:
            story = story[0].upper() + story[1:]
        
        return story
    
    def _generate_fallback_story(self, keywords: str, language: str) -> Dict[str, Any]:
        """
        Templates have been removed from actual generation. This function 
        is strictly an error-state absolute fallback to prevent API crashes.
        It no longer injects external language characters.
        """
        templates = {
            "en": "Generation failed. Please try again.",
            "hi": "Generation failed. Please try again.",
            "ta": "Generation failed. Please try again.",
        }
        
        story = templates.get(language, templates["en"])
        
        return {
            "story": story,
            "word_count": len(story.split()),
            "language": language,
            "keywords": keywords,
            "model": "error-fallback"
        }
    
    def is_ready(self, language: str = "en") -> bool:
        """Check if generator is ready for a specific language"""
        return TRANSFORMERS_AVAILABLE and language in self.models
