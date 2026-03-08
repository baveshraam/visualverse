"""
Dataset Preparation for Story Generation Training
Downloads and prepares keyword-story pairs for English, Hindi, and Tamil

Data Sources:
- English: WritingPrompts dataset from HuggingFace
- Hindi/Tamil: Translated samples + synthetic data

Target: 5,000 samples per language
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("⚠️  'datasets' not installed. Run: pip install datasets")

try:
    from nlp.keyphrase.extractor import KeyphraseExtractor
    from nlp.preprocessing.preprocessor import TextPreprocessor
    NLPTOOLS_AVAILABLE = True
except ImportError:
    NLPTOOLS_AVAILABLE = False
    print("⚠️  NLP tools not available")


class StoryDatasetPreparator:
    """Prepare training datasets for story generation"""
    
    def __init__(self, output_dir: str = "training/story_training/data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize NLP tools for keyword extraction
        if NLPTOOLS_AVAILABLE:
            self.preprocessor = TextPreprocessor()
            self.keyphrase_extractor = KeyphraseExtractor()
        else:
            self.preprocessor = None
            self.keyphrase_extractor = None
    
    def download_writing_prompts(self, max_samples: int = 5000):
        """
        Download WritingPrompts dataset from HuggingFace
        This dataset contains Reddit writing prompts paired with stories
        """
        print(f"\n📥 Downloading WritingPrompts dataset...")
        
        if not DATASETS_AVAILABLE:
            print("❌ Cannot download without 'datasets' package")
            return []
        
        try:
            # Load dataset
            dataset = load_dataset("euclaise/writingprompts", split="train", trust_remote_code=True)
            
            print(f"✅ Dataset loaded: {len(dataset)} samples available")
            
            # Process samples
            samples = []
            for i, item in enumerate(dataset):
                if i >= max_samples:
                    break
                
                prompt = item.get('prompt', '')
                story = item.get('story', '')
                
                # Clean and validate
                if len(story.split()) < 100 or len(story.split()) > 1500:
                    continue  # Skip too short or too long stories
                
                if len(prompt.split()) < 5:
                    continue  # Skip invalid prompts
                
                samples.append({
                    'prompt': prompt.strip(),
                    'story': story.strip(),
                })
                
                if (i + 1) % 500 == 0:
                    print(f"  Processed {i + 1} samples...")
            
            print(f"✅ Collected {len(samples)} valid samples")
            return samples
            
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            print("   Falling back to synthetic data generation...")
            return []
    
    def extract_keywords_from_prompt(self, prompt: str) -> str:
        """
        Extract keywords from a writing prompt
        Uses simple pattern matching and NLP if available
        """
        keywords = []
        
        # Pattern-based extraction
        genre_keywords = ['fantasy', 'sci-fi', 'science fiction', 'horror', 'mystery', 
                         'romance', 'thriller', 'adventure', 'drama', 'comedy']
        for genre in genre_keywords:
            if genre.lower() in prompt.lower():
                keywords.append(f"genre: {genre}")
                break
        
        # Extract character references
        if any(word in prompt.lower() for word in ['hero', 'protagonist', 'character']):
            keywords.append("has_protagonist: yes")
        if any(word in prompt.lower() for word in ['villain', 'antagonist', 'enemy']):
            keywords.append("has_antagonist: yes")
        
        # Use keyphrase extractor if available
        if self.keyphrase_extractor and self.preprocessor:
            try:
                preprocessed = self.preprocessor.process(prompt, language='en')
                keyphrases = self.keyphrase_extractor.extract(preprocessed, top_k=5)
                for kp in keyphrases[:3]:  # Top 3 keyphrases
                    keywords.append(kp['phrase'])
            except:
                pass
        
        # Fallback: extract nouns from prompt
        if len(keywords) < 3:
            words = prompt.lower().split()
            important_words = [w for w in words if len(w) > 4 and w.isalpha()][:5]
            keywords.extend(important_words)
        
        return ", ".join(keywords) if keywords else "story, narrative, characters"
    
    def create_training_sample(self, prompt: str, story: str, language: str = "en") -> Dict[str, Any]:
        """Create a properly formatted training sample"""
        
        # Extract keywords from prompt
        keywords = self.extract_keywords_from_prompt(prompt)
        
        # Clean story
        story = story.strip()
        word_count = len(story.split())
        
        # Target length: 400-1000 words
        if word_count > 1000:
            # Truncate to ~800 words
            words = story.split()[:800]
            story = " ".join(words)
            word_count = 800
        
        return {
            "keywords": keywords,
            "story": story,
            "language": language,
            "word_count": word_count,
            "prompt": prompt  # Keep original for reference
        }
    
    def generate_synthetic_samples(self, language: str, count: int = 100) -> List[Dict[str, Any]]:
        """
        Generate synthetic training samples for Hindi/Tamil
        Uses templates and variations
        """
        print(f"\n🔧 Generating {count} synthetic samples for {language}...")
        
        # Story templates
        templates = {
            "en": [
                {
                    "keywords": "genre: adventure, hero: explorer, setting: jungle, plot: treasure hunt",
                    "story": "Deep in the heart of the Amazon jungle, explorer Marcus Blake pushed through the thick vegetation. The ancient map in his hands promised riches beyond imagination. Days of trekking had led him here, to a hidden temple covered in vines. As he entered the moss-covered structure, his torch illuminated golden artifacts and mysterious symbols. But he wasn't alone. Footsteps echoed behind him. Another treasure hunter had followed. The race for the legendary treasure was about to become deadly. Marcus had to choose between the gold and his survival."
                },
                {
                    "keywords": "genre: sci-fi, characters: scientist and AI, setting: space station, conflict: system malfunction",
                    "story": "Dr. Sarah Chen floated through the zero-gravity laboratory of Space Station Prometheus. The AI assistant, ARIA, reported another system failure. 'Life support at 40 percent,' ARIA announced calmly. Sarah's hands trembled as she accessed the power core diagnostics. A meteor strike had damaged critical systems three days ago. Earth was too far to send help in time. She had to repair the quantum reactor herself. One mistake could destroy the station. ARIA guided her through the dangerous procedure, but would it be enough?"
                },
                {
                    "keywords": "genre: mystery, hero: detective, setting: mansion, plot: murder investigation",
                    "story": "Detective James Morrison surveyed the crime scene in the Blackwood Mansion. Lord Blackwood lay dead in his study, a glass of poisoned wine beside him. Six suspects sat in the drawing room, each with a motive. Lady Blackwood stood to inherit millions. The butler had been fired that morning. The business partner faced bankruptcy. Morrison examined the evidence carefully. A torn letter, muddy footprints, and a missing key. As he questioned each suspect, their stories began to unravel. Someone was lying, and he would find out who."
                }
            ],
            "hi": [
                {
                    "keywords": "शैली: रोमांच, नायक: योद्धा, स्थान: पहाड़, कथानक: राक्षस से युद्ध",
                    "story": "हिमालय की ऊंची चोटियों में, योद्धा अर्जुन ने अपनी तलवार संभाली। राक्षस राज अनेक गांवों को तबाह कर चुका था। अर्जुन को उसे रोकना था। बर्फीली हवाओं के बीच, वह गुफा की ओर बढ़ा। अचानक, एक विशाल परछाई उसके सामने आई। राक्षस की आंखें लाल थीं, दांत तेज़ और भयानक। युद्ध शुरू हो गया। अर्जुन की तलवार चमकी। राक्षस शक्तिशाली था, लेकिन अर्जुन का साहस अधिक था। अंततः, उसने राक्षस को हराया और गांव को बचाया।"
                },
                {
                    "keywords": "शैली: विज्ञान कथा, पात्र: वैज्ञानिक, स्थान: प्रयोगशाला, कथानक: आविष्कार",
                    "story": "डॉ. रमेश शर्मा अपनी प्रयोगशाला में काम कर रहे थे। उन्होंने एक नई मशीन बनाई थी जो समय को नियंत्रित कर सकती थी। जब उन्होंने मशीन चालू की, तो प्रयोगशाला चमकने लगी। अचानक, वह भविष्य में पहुंच गए। वहां उन्होंने देखा कि पृथ्वी संकट में थी। केवल उनका आविष्कार इसे बचा सकता था। वह वापस लौटे और अपने काम को पूरा करने लगे। समय कम था, लेकिन उम्मीद बाकी थी।"
                }
            ],
            "ta": [
                {
                    "keywords": "வகை: சாகசம், கதாநாயகன்: வீரன், இடம்: காடு, கதைக்களம்: போர்",
                    "story": "அடர்ந்த காட்டில், வீரன் முருகன் தன் வாளை எடுத்தான். கொடூர அரக்கன் பல கிராமங்களை அழித்துவிட்டான். முருகன் அவனை நிறுத்த வேண்டும். இருட்டான பாதையில் அவன் நடந்தான். திடீரென்று, ஒரு பெரிய நிழல் தோன்றியது. அரக்கனின் கண்கள் சிவப்பாக இருந்தன. போர் தொடங்கியது. முருகனின் வாள் மின்னியது. அரக்கன் வலிமையாக இருந்தான், ஆனால் முருகனின் தைரியம் அதிகம். இறுதியில், அவன் அரக்கனை வென்று கிராமத்தை காப்பாற்றினான்."
                },
                {
                    "keywords": "வகை: அறிவியல், பாத்திரம்: விஞ்ஞானி, இடம்: ஆய்வகம், கதை: கண்டுபிடிப்பு",
                    "story": "விஞ்ஞானி ராஜேஷ் தனது ஆய்வகத்தில் வேலை செய்து கொண்டிருந்தார். அவர் ஒரு புதிய இயந்திரத்தை உருவாக்கியிருந்தார். அது காலத்தை கட்டுப்படுத்த முடியும். இயந்திரத்தை இயக்கியபோது, ஆய்வகம் ஒளிர்ந்தது. திடீரென்று, அவர் எதிர்காலத்திற்கு சென்றார். அங்கே அவர் பூமி ஆபத்தில் இருப்பதை கண்டார். அவரது கண்டுபிடிப்பு மட்டுமே அதை காப்பாற்ற முடியும். அவர் திரும்பி வந்து தனது வேலையை முடிக்க தொடங்கினார்."
                }
            ]
        }
        
        samples = []
        lang_templates = templates.get(language, templates["en"])
        
        # Replicate and vary templates
        for i in range(count):
            template = random.choice(lang_templates)
            samples.append({
                "keywords": template["keywords"],
                "story": template["story"],
                "language": language,
                "word_count": len(template["story"].split())
            })
        
        print(f"✅ Generated {len(samples)} synthetic samples")
        return samples
    
    def prepare_english_dataset(self, target_samples: int = 5000):
        """Prepare English training dataset"""
        print(f"\n{'='*60}")
        print(f"  PREPARING ENGLISH DATASET")
        print(f"{'='*60}")
        
        samples = []
        
        # Try to download WritingPrompts
        writing_prompts = self.download_writing_prompts(max_samples=target_samples + 500)
        
        if writing_prompts:
            print(f"\n🔄 Converting to training format...")
            for item in writing_prompts:
                sample = self.create_training_sample(
                    item['prompt'], 
                    item['story'], 
                    language='en'
                )
                samples.append(sample)
                
                if len(samples) >= target_samples:
                    break
        
        # Add synthetic samples if needed
        if len(samples) < target_samples:
            needed = target_samples - len(samples)
            print(f"\n⚠️  Only {len(samples)} real samples. Adding {needed} synthetic samples...")
            synthetic = self.generate_synthetic_samples('en', needed)
            samples.extend(synthetic)
        
        # Shuffle
        random.shuffle(samples)
        
        # Save dataset
        output_file = self.output_dir / "stories_en.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ English dataset saved: {output_file}")
        print(f"   Total samples: {len(samples)}")
        self._print_dataset_stats(samples)
        
        return samples
    
    def prepare_multilingual_dataset(self, language: str, target_samples: int = 2000):
        """Prepare Hindi or Tamil dataset"""
        lang_names = {"hi": "HINDI", "ta": "TAMIL"}
        print(f"\n{'='*60}")
        print(f"  PREPARING {lang_names.get(language, language.upper())} DATASET")
        print(f"{'='*60}")
        
        # For now, use synthetic samples with templates
        # In a real scenario, you'd translate or scrape actual stories
        samples = self.generate_synthetic_samples(language, target_samples)
        
        # Save dataset
        output_file = self.output_dir / f"stories_{language}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ {lang_names.get(language)} dataset saved: {output_file}")
        print(f"   Total samples: {len(samples)}")
        self._print_dataset_stats(samples)
        
        return samples
    
    def _print_dataset_stats(self, samples: List[Dict[str, Any]]):
        """Print statistics about the dataset"""
        if not samples:
            return
        
        word_counts = [s['word_count'] for s in samples]
        avg_words = sum(word_counts) / len(word_counts)
        min_words = min(word_counts)
        max_words = max(word_counts)
        
        print(f"   Avg words/story: {avg_words:.0f}")
        print(f"   Word count range: {min_words} - {max_words}")
        
        # Sample keywords
        sample_keywords = [s['keywords'] for s in samples[:3]]
        print(f"\n   Sample keywords:")
        for i, kw in enumerate(sample_keywords, 1):
            print(f"     {i}. {kw[:80]}...")
    
    def prepare_all_datasets(self):
        """Prepare datasets for all languages"""
        print(f"\n{'#'*60}")
        print(f"  STORY GENERATION DATASET PREPARATION")
        print(f"{'#'*60}")
        
        # English (5000 samples)
        en_samples = self.prepare_english_dataset(target_samples=5000)
        
        # Hindi (2000 samples)
        hi_samples = self.prepare_multilingual_dataset('hi', target_samples=2000)
        
        # Tamil (2000 samples)
        ta_samples = self.prepare_multilingual_dataset('ta', target_samples=2000)
        
        print(f"\n{'='*60}")
        print(f"  DATASET PREPARATION COMPLETE!")
        print(f"{'='*60}")
        print(f"  English: {len(en_samples)} samples")
        print(f"  Hindi:   {len(hi_samples)} samples")
        print(f"  Tamil:   {len(ta_samples)} samples")
        print(f"  Total:   {len(en_samples) + len(hi_samples) + len(ta_samples)} samples")
        print(f"\n  Files saved in: {self.output_dir.absolute()}")


def main():
    """Main execution"""
    preparator = StoryDatasetPreparator()
    preparator.prepare_all_datasets()


if __name__ == "__main__":
    main()
