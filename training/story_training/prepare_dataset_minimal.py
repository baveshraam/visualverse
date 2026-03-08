"""
Minimal Dataset Preparation - Quick version for testing
Creates small synthetic dataset for story generation training
"""

import json
import os
from pathlib import Path

def create_minimal_dataset():
    """Create minimal synthetic dataset for quick training"""
    
    # Output directory
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("  MINIMAL DATASET PREPARATION")
    print("="*60 + "\n")
    
    # English samples (100 samples)
    en_samples = []
    templates_en = [
        ("genre: sci-fi, hero: astronaut, plot: space mission", 
         "The astronaut floated through the vastness of space. Her mission was clear - explore the unknown planet ahead. As she landed, strange lights appeared in the distance."),
        ("genre: fantasy, hero: wizard, plot: magic quest",
         "The old wizard clutched his staff tightly. A magical stone had been stolen from the kingdom. He must embark on a dangerous quest to retrieve it before dark forces use its power."),
        ("genre: mystery, detective, plot: murder investigation",
         "Detective Sarah examined the crime scene carefully. Clues were scattered everywhere, but something didn't add up. She knew the murderer was close, hiding in plain sight."),
        ("genre: romance, plot: love story, setting: Paris",
         "In the heart of Paris, two strangers met by chance. Their eyes locked across the café. Little did they know this moment would change their lives forever."),
        ("genre: horror, plot: haunted house, setting: mansion",
         "The old mansion creaked in the wind. Strange noises echoed through empty hallways. As darkness fell, the true horrors of the house began to reveal themselves."),
        ("genre: adventure, hero: explorer, plot: treasure hunt",
         "The explorer deciphered the ancient map. Hidden treasure awaited in a forgotten temple. With courage and determination, he ventured into the jungle."),
        ("genre: thriller, plot: conspiracy, setting: city",
         "She discovered files that shouldn't exist. A conspiracy that reached the highest levels of power. Now they were hunting her, and time was running out."),
        ("genre: comedy, plot: misunderstanding, characters: friends",
         "The mix-up was hilarious. Nobody could believe what just happened. As friends gathered to sort it out, more chaos ensued, making everything even funnier."),
        ("genre: drama, plot: family conflict, setting: home",
         "Years of silence broke that evening. Family secrets spilled across the dinner table. Healing would take time, but honesty was the first step forward."),
        ("genre: war, hero: soldier, plot: survival",
         "Behind enemy lines, the soldier fought to survive. Ammunition was running low. He had to make it back home, no matter the cost."),
    ]
    
    # Expand templates
    for keywords, story in templates_en * 10:  # 100 samples
        en_samples.append({"keywords": keywords, "story": story})
    
    # Hindi samples (50 samples)
    hi_samples = []
    templates_hi = [
        ("शैली: विज्ञान कथा, पात्र: वैज्ञानिक, कथानक: आविष्कार",
         "वैज्ञानिक ने एक नया आविष्कार किया। यह दुनिया को बदल सकता था। लेकिन खतरे भी थे।"),
        ("शैली: रोमांच, नायक: खोजकर्ता, कथानक: खजाने की खोज",
         "खोजकर्ता ने प्राचीन नक्शा खोजा। छिपा हुआ खजाना उसका इंतजार कर रहा था। यात्रा शुरू हुई।"),
        ("शैली: रहस्य, पात्र: जासूस, कथानक: जांच",
         "जासूस ने सुराग इकट्ठा किए। अपराधी कौन था? सच्चाई जल्द ही सामने आएगी।"),
        ("शैली: प्रेम, कथानक: प्रेम कहानी",
         "दो दिल मिले। प्यार खिल उठा। जीवन बदल गया।"),
        ("शैली: डरावनी, स्थान: हवेली",
         "पुरानी हवेली डरावनी थी। रात में अजीब आवाजें आती थीं। कोई अंदर गया तो वापस नहीं आया।"),
    ]
    
    for keywords, story in templates_hi * 10:  # 50 samples
        hi_samples.append({"keywords": keywords, "story": story})
    
    # Tamil samples (50 samples)
    ta_samples = []
    templates_ta = [
        ("வகை: அறிவியல், கதாநாயகன்: விஞ்ஞானி, கதைக்களம்: எதிர்காலம்",
         "விஞ்ஞானி ஒரு புதிய கண்டுபிடிப்பு செய்தார். இது உலகை மாற்றும். ஆபத்துகள் இருந்தன."),
        ("வகை: சாகசம், கதாநாயகன்: ஆராய்ச்சியாளர், கதை: பொக்கிஷம்",
         "பழைய வரைபடத்தை கண்டுபிடித்தார். மறைக்கப்பட்ட பொக்கிஷம் காத்திருந்தது. பயணம் தொடங்கியது."),
        ("வகை: மர்மம், கதாபாத்திரம்: துப்பறியும், கதை: விசாரணை",
         "துப்பறியும் ஆதாரங்களை சேகரித்தார். குற்றவாளி யார்? உண்மை விரைவில் வெளிப்படும்."),
        ("வகை: காதல், கதை: காதல் கதை",
         "இரண்டு இதயங்கள் சந்தித்தன. காதல் மலர்ந்தது. வாழ்க்கை மாறியது."),
        ("வகை: திகில், இடம்: பழைய வீடு",
         "பழைய வீடு பயமுறுத்தியது. இரவில் விசித்திர ஒலிகள். உள்ளே சென்றவர் திரும்பவில்லை."),
    ]
    
    for keywords, story in templates_ta * 10:  # 50 samples
        ta_samples.append({"keywords": keywords, "story": story})
    
    # Save datasets
    datasets = {
        'en': en_samples,
        'hi': hi_samples,
        'ta': ta_samples
    }
    
    for lang, samples in datasets.items():
        output_file = output_dir / f"story_dataset_{lang}_minimal.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"✅ {lang.upper()}: {len(samples)} samples → {output_file}")
    
    print(f"\n✅ Total samples: {sum(len(s) for s in datasets.values())}")
    print(f"📁 Saved to: {output_dir}")
    print("\nReady for training!")

if __name__ == "__main__":
    create_minimal_dataset()
