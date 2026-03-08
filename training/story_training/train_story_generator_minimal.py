"""
Minimal Story Generator Training - Quick version
Trains distilgpt2 models on minimal dataset for fast completion
"""

import json
import os
import sys
import torch
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import time

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

class MinimalStoryTrainer:
    def __init__(self):
        self.data_dir = Path(__file__).parent / "data"
        self.models_dir = Path(__file__).parent.parent.parent / "backend" / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Check GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n🖥️  Device: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA: {torch.version.cuda}")
        
    def load_dataset(self, language):
        """Load minimal dataset"""
        data_file = self.data_dir / f"story_dataset_{language}_minimal.json"
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Format for training
        formatted = []
        for item in data:
            text = f"<|keywords|>{item['keywords']}<|story|>{item['story']}<|endoftext|>"
            formatted.append({"text": text})
        
        return Dataset.from_list(formatted)
    
    def train_model(self, language, epochs=2):
        """Train a single language model"""
        print(f"\n{'='*60}")
        print(f"  TRAINING {language.upper()} MODEL")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # Load dataset
        print(f"📚 Loading {language} dataset...")
        dataset = self.load_dataset(language)
        print(f"   Samples: {len(dataset)}")
        
        # Initialize tokenizer (use existing or create new)
        print(f"🔧 Initializing tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        
        # Add special tokens
        special_tokens = {
            "additional_special_tokens": ["<|keywords|>", "<|story|>"],
            "pad_token": "<|pad|>"
        }
        tokenizer.add_special_tokens(special_tokens)
        
        # Initialize model
        print(f"🤖 Initializing model...")
        model = GPT2LMHeadModel.from_pretrained("distilgpt2")
        model.resize_token_embeddings(len(tokenizer))
        
        # Tokenize dataset
        print(f"🔤 Tokenizing...")
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=256,
                padding="max_length"
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Training arguments - optimized for speed
        output_dir = self.models_dir / f"story_model_{language}"
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=2 if self.device == "cuda" else 1,
            gradient_accumulation_steps=4,
            save_strategy="epoch",
            save_total_limit=1,
            logging_steps=10,
            learning_rate=5e-5,
            warmup_steps=10,
            weight_decay=0.01,
            fp16=False,  # Disabled for stability
            report_to="none",
            dataloader_num_workers=0,
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # Train
        print(f"🚀 Training started...")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {training_args.per_device_train_batch_size}")
        print(f"   Device: {self.device}\n")
        
        trainer.train()
        
        # Save model and tokenizer
        print(f"\n💾 Saving model...")
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        
        # Save training info
        training_info = {
            "language": language,
            "model": "distilgpt2",
            "training_samples": len(dataset),
            "epochs": epochs,
            "training_time_seconds": time.time() - start_time,
            "device": self.device
        }
        
        with open(output_dir / "training_info.json", 'w') as f:
            json.dump(training_info, f, indent=2)
        
        elapsed = time.time() - start_time
        print(f"✅ Training complete!")
        print(f"   Time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        print(f"   Model saved: {output_dir}")
        
        # Cleanup
        del model, trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return training_info
    
    def test_model(self, language):
        """Test a trained model"""
        print(f"\n🧪 Testing {language.upper()} model...")
        
        model_path = self.models_dir / f"story_model_{language}"
        
        tokenizer = GPT2Tokenizer.from_pretrained(str(model_path))
        model = GPT2LMHeadModel.from_pretrained(str(model_path))
        model.to(self.device)
        model.eval()
        
        # Test prompts
        test_prompts = {
            "en": "genre: adventure, hero: explorer, plot: treasure hunt",
            "hi": "शैली: रोमांच, पात्र: वीर, कथानक: खोज",
            "ta": "வகை: சாகசம், கதாநாயகன்: வீரன், கதை: தேடல்"
        }
        
        prompt = test_prompts.get(language, test_prompts["en"])
        input_text = f"<|keywords|>{prompt}<|story|>"
        
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=150,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"   Input: {prompt}")
        print(f"   Output: {generated[:200]}...")
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

def main():
    print("\n" + "="*70)
    print("  VISUALVERSE MINIMAL STORY GENERATOR TRAINING")
    print("  Quick training for demonstration purposes")
    print("="*70)
    
    trainer = MinimalStoryTrainer()
    
    languages = ['en', 'hi', 'ta']
    results = {}
    
    total_start = time.time()
    
    for lang in languages:
        try:
            info = trainer.train_model(lang, epochs=2)
            results[lang] = info
            
            # Test the model
            trainer.test_model(lang)
            
        except Exception as e:
            print(f"❌ Error training {lang}: {e}")
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - total_start
    
    print(f"\n{'='*70}")
    print("  TRAINING SUMMARY")
    print(f"{'='*70}\n")
    
    for lang, info in results.items():
        print(f"{lang.upper():4s}: ✅ {info['training_samples']} samples, "
              f"{info['training_time_seconds']:.1f}s")
    
    print(f"\n⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"📁 Models saved to: {trainer.models_dir}")
    print(f"\n✅ All models trained successfully!")
    print(f"\nNext steps:")
    print(f"  1. Start backend: cd backend && python main.py")
    print(f"  2. Start frontend: npm run dev")
    print(f"  3. Open http://localhost:5173")

if __name__ == "__main__":
    main()
