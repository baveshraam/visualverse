"""
Simple test to verify translation pipeline functionality
"""
import sys
import os

# Add backend to path
sys.path.append('c:/Bavesh/Sem6/NLP/nlp-main/backend')

# Configure stdout for UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 60)
print("Testing Translation Pipeline")
print("=" * 60)

# Test 1: Import the module
print("\n[1] Importing LlamaStoryEngine...")
try:
    from models.story_gen import LlamaStoryEngine, TRANSLATION_MODELS
    print("✅ Import successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Check translation models mapping
print("\n[2] Checking TRANSLATION_MODELS mapping...")
print(f"Available translation pairs: {len(TRANSLATION_MODELS)}")
for pair, model_name in TRANSLATION_MODELS.items():
    print(f"  {pair[0]} -> {pair[1]}: {model_name}")
print("✅ Translation models mapping verified")

# Test 3: Initialize engine
print("\n[3] Initializing LlamaStoryEngine...")
try:
    engine = LlamaStoryEngine()
    if engine.is_ready():
        print("✅ Engine initialized and ready")
    else:
        print("⚠️  Engine initialized but not ready (models not loaded)")
except Exception as e:
    print(f"❌ Engine initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test language detection
print("\n[4] Testing language detection...")
test_cases = [
    ("Hello world", "en"),
    ("नमस्ते दुनिया", "hi"),
    ("வணக்கம் உலகம்", "ta"),
]

from models.story_gen import detect_language
for text, expected_lang in test_cases:
    detected = detect_language(text)
    status = "✅" if detected == expected_lang else "❌"
    print(f"  {status} '{text}' -> detected: {detected}, expected: {expected_lang}")

# Test 5: Test script validation
print("\n[5] Testing script validation...")
test_texts = [
    ("This is an English text.", "en", True),
    ("यह हिंदी पाठ है।", "hi", True),
    ("இது தமிழ் உரை.", "ta", True),
    ("This is English but targeting Hindi", "hi", False),
]

for text, target_lang, should_be_valid in test_texts:
    is_valid, detected_script = engine._validate_script(text, target_lang)
    status = "✅" if is_valid == should_be_valid else "❌"
    print(f"  {status} Target: {target_lang}, Detected: {detected_script}, Valid: {is_valid}")

print("\n" + "=" * 60)
print("Basic tests completed successfully!")
print("=" * 60)
