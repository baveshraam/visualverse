# Quick Reference: Multilingual Story Generation

## Usage

### API Endpoint
```bash
POST http://localhost:8000/api/generate-story
Content-Type: application/json

{
  "keywords": "Your story seed text here",
  "language": "hi"  # or "en", "ta", "auto"
}
```

### Python Usage
```python
from models.story_gen import LlamaStoryEngine
import asyncio

async def generate():
    engine = LlamaStoryEngine()
    
    # English to Hindi
    result = await engine.generate(
        keywords="A clockmaker named Arul found a golden watch",
        language="hi"
    )
    print(result['story'])

asyncio.run(generate())
```

## Translation Paths

### Supported Direct Translations
- EN → HI ✅
- HI → EN ✅
- EN → TA ✅
- TA → EN ✅

### Pivot Translations (via English)
- HI → TA ✅ (hi → en → ta)
- TA → HI ✅ (ta → en → hi)

### Same Language (No Translation)
- EN → EN
- HI → HI
- TA → TA

## Testing

### Run Basic Tests
```bash
python backend/test_translation_simple.py
```

### Run Logic Tests
```bash
python backend/test_translation_logic.py
```

### Run Full Translation Tests
```bash
python backend/test_story_translation.py
```

## Troubleshooting

### Issue: CUDA Out of Memory
**Solution:** Models use 4-bit quantization (~2GB). Ensure GPU has ≥6GB VRAM.

### Issue: Tamil generation fails
**Cause:** GEMINI_API_KEY not set  
**Solution:** Set environment variable or fallback to MarianMT (automatic)

### Issue: Mixed language in output
**Status:** Auto-corrected by script validation system

### Issue: Tokenizer mismatch error
**Status:** Fixed by explicit TRANSLATION_MODELS mapping

## Model Downloads

On first run, the following models will be auto-downloaded:
1. unsloth/Llama-3.2-3B-Instruct (~6GB)
2. Helsinki-NLP/opus-mt-en-hi (~300MB)
3. Helsinki-NLP/opus-mt-hi-en (~300MB)
4. Helsinki-NLP/opus-mt-en-ta (~300MB)
5. Helsinki-NLP/opus-mt-ta-en (~300MB)

**Total:** ~7.2GB disk space

## Architecture

```
User Input
    ↓
Detect Source Language
    ↓
Generate Story (in source language)
    ↓
[If source ≠ target]
    ↓
Translate (direct or pivot)
    ↓
Validate Script Purity
    ↓
[If script mismatch]
    ↓
Corrective Translation
    ↓
Return Final Story
```

## Key Features

✅ Deterministic translation pipeline  
✅ No template contamination  
✅ Automatic script validation  
✅ Tokenizer mismatch prevention  
✅ Model caching for performance  
✅ CUDA-safe inference  
✅ Graceful fallbacks  

## Performance

- **Cold Start:** 15-20 seconds (model loading)
- **Story Generation:** 5-10 seconds
- **Translation:** 1-4 seconds
- **Total (EN→HI):** ~10-15 seconds

## Memory Requirements

- **GPU VRAM:** 4-6 GB recommended
- **RAM:** 8 GB minimum
- **Disk:** 10 GB for models

## Files Changed

1. `backend/models/story_gen.py` - Main implementation
2. `backend/main.py` - FastAPI endpoint (no changes needed)
3. Test files created for verification

## Success Criteria

All tests from basic functionality should pass:
- ✅ TRANSLATION_MODELS mapping present
- ✅ Engine initialization successful
- ✅ Language detection accurate
- ✅ Script validation working
- ✅ No template text in outputs
- ✅ All 9 language pairs supported
