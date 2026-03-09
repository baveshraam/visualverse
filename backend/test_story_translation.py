import asyncio
import os
import sys

# Configure stdout for UTF-8 to prevent charmap errors in Windows
sys.stdout.reconfigure(encoding='utf-8')

# Add backend to path
sys.path.append('c:/Bavesh/Sem6/NLP/nlp-main/backend')

from models.story_gen import LlamaStoryEngine

async def test_generation():
    engine = LlamaStoryEngine()
    
    print("\n=======================================================")
    print("[1] Testing EN -> HI Translation pipeline...")
    print("=======================================================")
    result_hi = await engine.generate(
        keywords="A clockmaker named Arul found a golden pocket watch that could stop time. He pressed the button and the rain froze in mid-air.",
        language="hi",
        max_new_tokens=150
    )
    print(f"Detected Source: {result_hi.get('keywords')} (Processed as {engine.preprocess_nlp(result_hi['keywords'])['language']})")
    print(f"Target Language: {result_hi['language']}")
    print(f"Final Output:\n{result_hi['story']}\n")
    
    print("\n=======================================================")
    print("[2] Testing EN -> TA Translation pipeline...")
    print("=======================================================")
    result_ta = await engine.generate(
        keywords="A clockmaker named Arul found a golden pocket watch that could stop time. He pressed the button and the rain froze in mid-air.",
        language="ta",
        max_new_tokens=150
    )
    print(f"Detected Source: {result_ta.get('keywords')} (Processed as {engine.preprocess_nlp(result_ta['keywords'])['language']})")
    print(f"Target Language: {result_ta['language']}")
    print(f"Final Output:\n{result_ta['story']}\n")
    
    print("\n=======================================================")
    print("[3] Testing HI -> EN Translation pipeline...")
    print("=======================================================")
    result_en = await engine.generate(
        keywords="अरुल नाम के एक घड़ीसाज़ को एक सुनहरी जेब घड़ी मिली जो समय को रोक सकती थी। उसने बटन दबाया और बारिश बीच हवा में जम गई।",
        language="en",
        max_new_tokens=150
    )
    print(f"Detected Source: {result_en.get('keywords')} (Processed as {engine.preprocess_nlp(result_en['keywords'])['language']})")
    print(f"Target Language: {result_en['language']}")
    print(f"Final Output:\n{result_en['story']}\n")
    
    print("\n--- All Tests Done ---")

if __name__ == '__main__':
    asyncio.run(test_generation())
