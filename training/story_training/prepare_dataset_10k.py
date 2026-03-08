"""
Dataset Preparation for Story Generation Training (10K+ Version)
Downloads and prepares keyword-story pairs for English, Hindi, and Tamil

Data Sources:
- English: WritingPrompts dataset from HuggingFace (15,000 samples)
- Hindi: Synthetic data (5,000 samples)
- Tamil: Synthetic data (5,000 samples)

Total: 25,000 samples
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


class StoryDatasetPreparator10K:
    """Prepare large training datasets for story generation (10K+ samples)"""
    
    def __init__(self, output_dir: str = "training/story_training/data_10k"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize NLP tools for keyword extraction
        if NLPTOOLS_AVAILABLE:
            self.preprocessor = TextPreprocessor()
            self.keyphrase_extractor = KeyphraseExtractor()
        else:
            self.preprocessor = None
            self.keyphrase_extractor = None
    
    def download_writing_prompts(self, max_samples: int = 15000) -> List[Dict[str, str]]:
        """Download WritingPrompts dataset from HuggingFace"""
        if not DATASETS_AVAILABLE:
            print("❌ Cannot download without 'datasets' package")
            return []
        
        print(f"\n📥 Downloading WritingPrompts dataset...")
        try:
            dataset = load_dataset("euclaise/writingprompts", split="train")
            print(f"✅ Dataset loaded: {len(dataset)} samples available")
            
            samples = []
            for i, item in enumerate(dataset):
                if len(samples) >= max_samples * 2:  # Get extra for filtering
                    break
                
                prompt = item.get('prompt', '').strip()
                story = item.get('story', '').strip()
                
                if not prompt or not story:
                    continue
                
                # Filter by word count (100-1000 words)
                word_count = len(story.split())
                if word_count < 100 or word_count > 1000:
                    continue
                
                samples.append({'prompt': prompt, 'story': story})
                
                if (i + 1) % 1000 == 0:
                    print(f"  Processed {i + 1} samples...")
            
            print(f"✅ Collected {len(samples)} valid samples")
            return samples[:max_samples]
            
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            return []
    
    def extract_keywords_from_prompt(self, prompt: str) -> str:
        """Extract keywords from a writing prompt"""
        if self.keyphrase_extractor:
            try:
                keyphrases = self.keyphrase_extractor.extract(prompt, top_n=8)
                keywords = [kp['phrase'] for kp in keyphrases]
                return ", ".join(keywords[:6]) if keywords else prompt[:100]
            except:
                pass
        
        # Fallback: use first 100 chars
        keywords = []
        
        # Check for genre indicators
        genres = ['fantasy', 'sci-fi', 'mystery', 'romance', 'horror', 'thriller', 
                 'adventure', 'comedy', 'drama', 'historical']
        for genre in genres:
            if genre in prompt.lower():
                keywords.append(f"genre: {genre}")
                break
        
        # Check for protagonist
        if 'you' in prompt.lower() or 'you are' in prompt.lower():
            keywords.append("has_protagonist: yes")
        
        # Extract key words (simple approach)
        words = prompt.lower().split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has',
                     'that', 'this', 'it', 'you', 'your'}
        important_words = [w for w in words if w not in stop_words and len(w) > 3][:4]
        keywords.extend(important_words)
        
        return ", ".join(keywords) if keywords else prompt[:100]
    
    def prepare_english_dataset(self, target_samples: int = 15000):
        """Prepare English dataset from WritingPrompts"""
        print(f"\n{'='*60}")
        print(f"  PREPARING ENGLISH DATASET (10K+ VERSION)")
        print(f"{'='*60}\n")
        
        # Download real samples
        writing_prompts = self.download_writing_prompts(max_samples=target_samples)
        
        if len(writing_prompts) < target_samples:
            print(f"\n⚠️  Only {len(writing_prompts)} real samples. Adding synthetic...")
            num_synthetic = target_samples - len(writing_prompts)
            synthetic = self.generate_synthetic_samples('en', num_synthetic)
            writing_prompts.extend(synthetic)
        
        # Convert to training format
        print(f"\n🔄 Converting to training format...")
        training_data = []
        for item in writing_prompts:
            if 'prompt' in item:
                keywords = self.extract_keywords_from_prompt(item['prompt'])
                story = item['story']
            else:
                keywords = item['keywords']
                story = item['story']
            
            training_data.append({
                'keywords': keywords,
                'story': story
            })
        
        # Shuffle
        random.shuffle(training_data)
        
        # Save
        output_file = self.output_dir / "stories_en_10k.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        # Stats
        word_counts = [len(item['story'].split()) for item in training_data]
        avg_words = sum(word_counts) / len(word_counts)
        
        print(f"\n✅ English dataset saved: {output_file}")
        print(f"   Total samples: {len(training_data)}")
        print(f"   Avg words/story: {int(avg_words)}")
        print(f"   Word count range: {min(word_counts)} - {max(word_counts)}")
        print(f"\n   Sample keywords:")
        for i, item in enumerate(training_data[:3], 1):
            keywords = item['keywords'][:80] + "..." if len(item['keywords']) > 80 else item['keywords']
            print(f"     {i}. {keywords}")
    
    def generate_synthetic_samples(self, language: str, num_samples: int) -> List[Dict[str, str]]:
        """Generate synthetic story samples"""
        samples = []
        
        if language == 'en':
            templates = self._get_english_templates()
        elif language == 'hi':
            templates = self._get_hindi_templates()
        elif language == 'ta':
            templates = self._get_tamil_templates()
        else:
            return samples
        
        for _ in range(num_samples):
            template = random.choice(templates)
            samples.append({
                'keywords': template['keywords'],
                'story': template['story']
            })
        
        return samples
    
    def _get_english_templates(self):
        """Get English story templates"""
        return [
            {
                'keywords': 'genre: mystery, hero: detective, setting: mansion, plot: murder investigation',
                'story': 'The old mansion stood silent on the hill. Detective James arrived at midnight, called by the frantic butler. A murder had occurred in the library. The victim was the wealthy owner. James examined the scene carefully. Blood stains led to the study. Secret passages revealed hidden rooms. Each family member had a motive. The detective questioned everyone thoroughly. Clues pointed to the nephew. But something felt wrong. James discovered a hidden diary. It revealed the truth. The butler was guilty. He confessed under pressure. Justice was served.'
            },
            {
                'keywords': 'genre: sci-fi, hero: astronaut, setting: Mars, plot: survival mission',
                'story': 'Commander Sarah Chen landed on Mars alone. Her crew had perished in the crash. The base was destroyed beyond repair. She had limited oxygen supplies. Food would last only three weeks. Communication systems were damaged beyond hope. Sarah assessed her situation calmly. She gathered all available resources carefully. A backup shelter offered some protection. Solar panels could generate minimal power. Water could be extracted from ice. Sarah worked tirelessly each day. She rationed her supplies carefully. A rescue mission was months away. But Sarah refused to give up. She survived against all odds.'
            },
            {
                'keywords': 'genre: adventure, hero: explorer, setting: jungle, plot: treasure hunt',
                'story': 'Dr. Maria Rodriguez entered the dense jungle. Ancient maps promised hidden treasure deep within. Her team consisted of five brave members. They hacked through thick vegetation daily. Dangerous animals lurked in the shadows. The heat was almost unbearable always. They discovered ancient temple ruins eventually. Traps protected the sacred grounds everywhere. One by one they solved puzzles. The treasure chamber lay ahead finally. Golden artifacts filled every corner completely. But greed divided the team quickly. Maria chose wisdom over wealth. She preserved the historical site. The true treasure was knowledge gained.'
            },
            {
                'keywords': 'genre: fantasy, hero: wizard, setting: kingdom, plot: dragon battle',
                'story': 'The kingdom of Aldoria faced destruction. A mighty dragon terrorized the lands. Young wizard Elric volunteered to help. He studied ancient spells carefully. The dragon had scales of iron. Fire breath destroyed everything it touched. Elric prepared for three full moons. He gathered magical ingredients from far. The final battle began at dawn. Elric cast his most powerful spells. The dragon fought back viciously fierce. Lightning struck the beast repeatedly hard. Magic shields protected the wizard barely. Finally the dragon weakened significantly. Elric sealed it away forever. Peace returned to the kingdom.'
            },
            {
                'keywords': 'genre: romance, setting: Paris, characters: artists, plot: rekindled love',
                'story': 'Sophie returned to Paris after ten years away. She was now a famous sculptor internationally. The old art studio still stood there. Inside she found her former love Pierre. He had become a renowned painter since. Their eyes met across the room. Memories flooded back immediately rushing. They had parted due to misunderstandings. Both had regretted it deeply forever. They began talking about old times. Their passion for art remained unchanged. Days turned into romantic weeks together. They created art side by side. Love rekindled stronger than before. Together they opened a gallery beautiful. Their second chance had finally arrived.'
            }
        ]
    
    def _get_hindi_templates(self):
        """Get Hindi story templates"""
        return [
            {
                'keywords': 'शैली: विज्ञान कथा, पात्र: वैज्ञानिक, स्थान: प्रयोगशाला, कथानक: आविष्कार',
                'story': 'डॉ. राज ने अपनी प्रयोगशाला में कई वर्ष बिताए। वह एक नई मशीन बना रहे थे। यह समय यात्रा संभव बना सकती थी। कई प्रयोग विफल हो चुके थे। लेकिन राज ने हार नहीं मानी। एक दिन अंततः सफलता मिली। मशीन ने काम करना शुरू किया। राज ने भविष्य की यात्रा की। वहाँ उन्होंने अद्भुत चीजें देखीं। तकनीक बहुत उन्नत हो चुकी थी। लोग सुखी और स्वस्थ थे। राज वापस लौट आए खुशी से। उन्होंने अपनी खोज साझा की। विज्ञान ने नई दिशा पकड़ी।'
            },
            {
                'keywords': 'शैली: रोमांच, नायक: योद्धा, कथानक: राक्षस से युद्ध',
                'story': 'राजा विक्रम एक वीर योद्धा थे। उनके राज्य पर राक्षस ने हमला किया। गाँव के लोग डर से काँप रहे थे। विक्रम ने युद्ध की तैयारी शुरू की। उन्होंने अपनी तलवार तेज की। ध्यान और योग से शक्ति बढ़ाई। युद्ध का दिन आ गया अंततः। राक्षस बहुत शक्तिशाली था निश्चित रूप से। लेकिन विक्रम का साहस अटूट था। भयंकर युद्ध घंटों तक चला लगातार। अंत में विक्रम ने जीत हासिल की। राज्य में फिर से शांति छा गई। लोगों ने विक्रम का सम्मान किया। उनकी वीरता की कहानियाँ फैल गईं।'
            },
            {
                'keywords': 'शैली: रहस्य, पात्र: जासूस, स्थान: पुरानी हवेली, कथानक: छुपा खजाना',
                'story': 'जासूस रमेश को एक पुराने मामले की सूचना मिली। एक हवेली में खजाना छुपा था। कई लोग उसे खोज चुके थे असफल। रमेश ने जाँच शुरू की सावधानी से। पुरानी हवेली भूतिया लग रही थी। दीवारों पर रहस्यमय निशान थे। रमेश ने सुराग जोड़ना शुरू किया। गुप्त कमरे मिले एक के बाद एक। पहेलियाँ सुलझाते गए धीरे धीरे। अंत में खजाने का कमरा मिला। सोने के सिक्के चमक रहे थे। रमेश ने सरकार को सूचित किया। खजाना संग्रहालय में रखा गया। रमेश का नाम इतिहास में दर्ज हुआ।'
            }
        ]
    
    def _get_tamil_templates(self):
        """Get Tamil story templates"""
        return [
            {
                'keywords': 'வகை: சாகசம், கதாநாயகன்: வீரன், இடம்: காடு, கதைக்களம்: போர்',
                'story': 'வீரன் ஒரு துணிச்சலான போர்வீரன். அவன் காட்டில் வாழ்ந்தான். ஒரு நாள் எதிரிகள் தாக்கினர். கிராமம் ஆபத்தில் இருந்தது. வீரன் தன் வாளை எடுத்தான். அவன் காட்டில் இரகசிய பாதை அறிந்தான். எதிரிகளை திடுக்கிடச் செய்தான். கடும் போர் நடந்தது நீண்ட நேரம். வீரன் தன் திறமையால் வென்றான். கிராம மக்கள் மகிழ்ச்சியடைந்தனர். வீரனின் பெயர் பரவியது. அவன் ஒரு புராணக்கதையானான்.'
            },
            {
                'keywords': 'வகை: அறிவியல், கதாநாயகன்: விஞ்ஞானி, கதைக்களம்: எதிர்காலம்',
                'story': 'டாக்டர் கார்த்திக் ஒரு சிறந்த விஞ்ஞானி. அவர் ஒரு புதிய கண்டுபிடிப்பில் பணியாற்றினார். காலப்பயணம் சாத்தியமாக்க முயன்றார். பல ஆண்டுகள் உழைத்தார் கடுமையாக. ஒரு நாள் வெற்றி கிடைத்தது. இயந்திரம் இயங்கத் தொடங்கியது. கார்த்திக் எதிர்காலம் சென்றார். அங்கு அற்புதமான விஷயங்கள் கண்டார். தொழில்நுட்பம் மிகவும் முன்னேறியிருந்தது. மக்கள் மகிழ்ச்சியாக வாழ்ந்தனர். அவர் திரும்பி வந்தார் மகிழ்வுடன். தன் கண்டுபிடிப்பை பகிர்ந்தார். உலகம் மாற்றம் பெற்றது.'
            },
            {
                'keywords': 'வகை: காதல், இடம்: கடற்கரை, கதைக்களம்: மீண்டும் சந்திப்பு',
                'story': 'அருண் மற்றும் காவ்யா பழைய காதலர்கள். அவர்கள் பல ஆண்டுகளுக்கு முன் பிரிந்தனர். விதி அவர்களை மீண்டும் இணைத்தது. கடற்கரையில் சந்தித்தனர் தற்செயலாக. பழைய நினைவுகள் வெள்ளமெடுத்தது. இருவரும் ம silence maintained காக பேசினர். அவர்கள் இன்னும் ஒருவரை ஒருவர் நேசித்தனர். தவறான புரிதல்கள் தீர்க்கப்பட்டன். அன்பு மீண்டும் மலர்ந்தது வலுவாக. ஒன்றாக புதிய வாழ்க்கை தொடங்கினர். அவர்களின் காதல் கதை தொடர்ந்தது.'
            }
        ]
    
    def prepare_hindi_dataset(self, num_samples: int = 5000):
        """Prepare Hindi dataset with synthetic samples"""
        print(f"\n{'='*60}")
        print(f"  PREPARING HINDI DATASET (10K+ VERSION)")
        print(f"{'='*60}\n")
        
        print(f"🔧 Generating {num_samples} synthetic samples for hi...")
        samples = self.generate_synthetic_samples('hi', num_samples)
        print(f"✅ Generated {len(samples)} synthetic samples")
        
        # Save
        output_file = self.output_dir / "stories_hi_10k.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        
        # Stats
        word_counts = [len(item['story'].split()) for item in samples]
        avg_words = sum(word_counts) / len(word_counts)
        
        print(f"\n✅ HINDI dataset saved: {output_file}")
        print(f"   Total samples: {len(samples)}")
        print(f"   Avg words/story: {int(avg_words)}")
        print(f"   Word count range: {min(word_counts)} - {max(word_counts)}")
        print(f"\n   Sample keywords:")
        for i, item in enumerate(samples[:3], 1):
            keywords = item['keywords'][:80] + "..." if len(item['keywords']) > 80 else item['keywords']
            print(f"     {i}. {keywords}")
    
    def prepare_tamil_dataset(self, num_samples: int = 5000):
        """Prepare Tamil dataset with synthetic samples"""
        print(f"\n{'='*60}")
        print(f"  PREPARING TAMIL DATASET (10K+ VERSION)")
        print(f"{'='*60}\n")
        
        print(f"🔧 Generating {num_samples} synthetic samples for ta...")
        samples = self.generate_synthetic_samples('ta', num_samples)
        print(f"✅ Generated {len(samples)} synthetic samples")
        
        # Save
        output_file = self.output_dir / "stories_ta_10k.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        
        # Stats
        word_counts = [len(item['story'].split()) for item in samples]
        avg_words = sum(word_counts) / len(word_counts)
        
        print(f"\n✅ TAMIL dataset saved: {output_file}")
        print(f"   Total samples: {len(samples)}")
        print(f"   Avg words/story: {int(avg_words)}")
        print(f"   Word count range: {min(word_counts)} - {max(word_counts)}")
        print(f"\n   Sample keywords:")
        for i, item in enumerate(samples[:3], 1):
            keywords = item['keywords'][:80] + "..." if len(item['keywords']) > 80 else item['keywords']
            print(f"     {i}. {keywords}")
    
    def prepare_all_datasets(self):
        """Prepare all language datasets"""
        print(f"\n{'#'*60}")
        print(f"  STORY GENERATION DATASET PREPARATION (10K+ VERSION)")
        print(f"{'#'*60}\n")
        
        self.prepare_english_dataset(target_samples=15000)
        self.prepare_hindi_dataset(num_samples=5000)
        self.prepare_tamil_dataset(num_samples=5000)
        
        print(f"\n{'='*60}")
        print(f"  DATASET PREPARATION COMPLETE!")
        print(f"{'='*60}")
        print(f"  English: 15000 samples")
        print(f"  Hindi:   5000 samples")
        print(f"  Tamil:   5000 samples")
        print(f"  Total:   25000 samples")
        print(f"\n  Files saved in: {self.output_dir.absolute()}")


if __name__ == "__main__":
    preparator = StoryDatasetPreparator10K()
    preparator.prepare_all_datasets()
