"""
Story Generator Model Training Script
Trains distilgpt2 models for keyword-to-story generation

Training Pipeline:
1. Load prepared datasets (stories_en.json, stories_hi.json, stories_ta.json)
2. Fine-tune distilgpt2 with keyword-story pairs
3. Evaluate on validation set
4. Save trained models

Model Format:
  Input:  "<|keywords|> {keywords} <|story|>"
  Output: "{generated_story}  "
"""

import os
import sys
import json
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

try:
    from transformers import (
        GPT2LMHeadModel,
        GPT2Tokenizer,
        GPT2Config,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
    )
    from datasets import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("❌ transformers not installed. Run: pip install transformers datasets torch")


class StoryGeneratorTrainer:
    """Train keyword-to-story generation models"""
    
    def __init__(self, data_dir: str = "training/story_training/data_10k",
                 models_dir: str = "backend/models"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Special tokens
        self.special_tokens = {
            "keywords_token": "<|keywords|>",
            "story_token": "<|story|>",
            "pad_token": "<|pad|>",
            "eos_token": " "
        }
    
    def load_dataset(self, language: str) -> List[Dict[str, Any]]:
        """Load prepared dataset for a language"""
        dataset_file = self.data_dir / f"stories_{language}_10k.json"
        
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_file}")
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ Loaded {len(data)} samples for {language}")
        return data
    
    def format_training_sample(self, keywords: str, story: str) -> str:
        """
        Format a training sample with special tokens
        Format: <|keywords|> {keywords} <|story|> {story}  "
        """
        formatted = (
            f"{self.special_tokens['keywords_token']} {keywords} "
            f"{self.special_tokens['story_token']} {story} "
            f"{self.special_tokens['eos_token']}"
        )
        return formatted
    
    def prepare_tokenizer(self, base_model: str = "distilgpt2") -> GPT2Tokenizer:
        """Prepare tokenizer with special tokens"""
        tokenizer = GPT2Tokenizer.from_pretrained(base_model)
        
        # Add special tokens
        special_tokens_dict = {
            'additional_special_tokens': [
                self.special_tokens['keywords_token'],
                self.special_tokens['story_token']
            ],
            'pad_token': self.special_tokens['pad_token'],
            'eos_token': self.special_tokens['eos_token']
        }
        
        tokenizer.add_special_tokens(special_tokens_dict)
        
        print(f"✅ Tokenizer prepared. Vocab size: {len(tokenizer)}")
        return tokenizer
    
    def prepare_training_data(self, samples: List[Dict[str, Any]], tokenizer: GPT2Tokenizer):
        """Prepare data for training"""
        print(f"\n🔄 Formatting {len(samples)} training samples...")
        
        # Format all samples
        formatted_texts = []
        for sample in samples:
            formatted = self.format_training_sample(
                sample['keywords'],
                sample['story']
            )
            formatted_texts.append(formatted)
        
        # Tokenize
        print(f"🔄 Tokenizing...")
        encodings = tokenizer(
            formatted_texts,
            truncation=True,
            max_length=256,  # Further reduced for low VRAM
            padding='max_length',
            return_tensors='pt'
        )
        
        # Create dataset
        dataset_dict = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        
        print(f"✅ Dataset prepared: {len(dataset)} samples")
        return dataset
    
    def train_model(self, language: str, epochs: int = 3, batch_size: int = 4):
        """
        Train story generation model for a specific language
        
        Args:
            language: 'en', 'hi', or 'ta'
            epochs: Number of training epochs (3-5 recommended)
            batch_size: Training batch size (4-8 for 4GB GPU)
        """
        lang_names = {"en": "English", "hi": "Hindi", "ta": "Tamil"}
        print(f"\n{'='*70}")
        print(f"  TRAINING {lang_names.get(language, language.upper())} STORY GENERATOR")
        print(f"{'='*70}")
        
        if not TRANSFORMERS_AVAILABLE:
            print("❌ Cannot train without transformers library")
            return None
        
        # Check for GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️  Device: {device}")
        if device == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Load dataset
        samples = self.load_dataset(language)
        
        # Split train/val (90/10)
        split_idx = int(len(samples) * 0.9)
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]
        
        print(f"📊 Train samples: {len(train_samples)}")
        print(f"📊 Val samples:   {len(val_samples)}")
        
        # Prepare tokenizer
        print(f"\n🔧 Initializing tokenizer...")
        tokenizer = self.prepare_tokenizer()
        
        # Prepare datasets
        train_dataset = self.prepare_training_data(train_samples, tokenizer)
        val_dataset = self.prepare_training_data(val_samples, tokenizer)
        
        # Load base model
        print(f"\n🔧 Loading base model (distilgpt2)...")
        model = GPT2LMHeadModel.from_pretrained("distilgpt2")
        
        # Resize token embeddings for special tokens
        model.resize_token_embeddings(len(tokenizer))
        
        print(f"✅ Model loaded. Parameters: {model.num_parameters():,}")
        
        # Training arguments
        output_dir = self.models_dir / f"story_model_{language}_10k"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=8,  # Simulate batch_size=8 with less memory
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_steps=100,
            logging_steps=50,
            logging_dir=str(output_dir / "logs"),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            fp16=False,  # Disable mixed precision for small GPU
            dataloader_num_workers=0,
            report_to="none",  # Disable wandb/tensorboard
            max_grad_norm=1.0,  # Gradient clipping for stability
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # Causal LM, not masked LM
        )
        
        # Initialize trainer
        print(f"\n🎯 Starting training...")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {training_args.learning_rate}")
        print(f"\n{'─'*70}")
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        # Train
        start_time = datetime.now()
        trainer.train()
        end_time = datetime.now()
        
        training_time = (end_time - start_time).total_seconds() / 60
        
        print(f"\n{'─'*70}")
        print(f"✅ Training complete! Time: {training_time:.1f} minutes")
        
        # Evaluate
        print(f"\n📊 Evaluating model...")
        eval_results = trainer.evaluate()
        
        print(f"\n📈 Evaluation Results:")
        print(f"   Loss: {eval_results['eval_loss']:.4f}")
        print(f"   Perplexity: {torch.exp(torch.tensor(eval_results['eval_loss'])):.2f}")
        
        # Save final model and tokenizer
        final_model_dir = self.models_dir / f"story_model_{language}"
        print(f"\n💾 Saving model to: {final_model_dir}")
        
        trainer.save_model(str(final_model_dir))
        tokenizer.save_pretrained(str(final_model_dir))
        
        # Save training info
        training_info = {
            "language": language,
            "base_model": "distilgpt2",
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "epochs": epochs,
            "batch_size": batch_size,
            "final_loss": eval_results['eval_loss'],
            "perplexity": float(torch.exp(torch.tensor(eval_results['eval_loss']))),
            "training_time_minutes": training_time,
            "trained_on": str(datetime.now()),
        }
        
        with open(final_model_dir / "training_info.json", 'w') as f:
            json.dump(training_info, f, indent=2)
        
        print(f"✅ Model saved successfully!")
        print(f"\n{'='*70}\n")
        
        return trainer, eval_results
    
    def test_generation(self, language: str, test_keywords: str):
        """Test the trained model with sample keywords"""
        print(f"\n🧪 Testing model with keywords: {test_keywords}")
        
        model_dir = self.models_dir / f"story_model_{language}"
        
        if not model_dir.exists():
            print(f"❌ Model not found: {model_dir}")
            return
        
        # Load model and tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(str(model_dir))
        model = GPT2LMHeadModel.from_pretrained(str(model_dir))
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        
        # Format input
        prompt = f"{self.special_tokens['keywords_token']} {test_keywords} {self.special_tokens['story_token']}"
        
        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_length=400,
                temperature=0.8,
                top_p=0.9,
                top_k=50,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
            )
        
        # Decode - skip special tokens to avoid decoding issues
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up text
        story = generated_text.strip()
        
        print(f"\n📖 Generated Story ({len(story.split())} words):")
        print(f"{'─'*70}")
        print(story[:500] + "..." if len(story) > 500 else story)
        print(f"{'─'*70}\n")
    
    def train_all_languages(self, epochs: int = 3, batch_size: int = 4):
        """Train models for all languages"""
        print(f"\n{'#'*70}")
        print(f"  TRAINING STORY GENERATORS FOR ALL LANGUAGES")
        print(f"{'#'*70}")
        
        results = {}
        
        # Train English
        print(f"\n[1/3] Training English model...")
        en_trainer, en_results = self.train_model('en', epochs=epochs, batch_size=batch_size)
        results['en'] = en_results
        
        # Test English
        self.test_generation('en', 
            "genre: sci-fi, hero: astronaut, setting: Mars, plot: survival mission")
        
        # Clear GPU memory before next language
        print(f"\n🧹 Clearing GPU memory...")
        import gc
        del en_trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Train Hindi
        print(f"\n[2/3] Training Hindi model...")
        hi_trainer, hi_results = self.train_model('hi', epochs=epochs, batch_size=batch_size)
        results['hi'] = hi_results
        
        # Test Hindi
        self.test_generation('hi',
            "शैली: रोमांच, नायक: योद्धा, कथानक: राक्षस से युद्ध")
        
        # Clear GPU memory before next language
        print(f"\n🧹 Clearing GPU memory...")
        del hi_trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Train Tamil
        print(f"\n[3/3] Training Tamil model...")
        ta_trainer, ta_results = self.train_model('ta', epochs=epochs, batch_size=batch_size)
        results['ta'] = ta_results
        
        # Test Tamil
        self.test_generation('ta',
            "வகை: அறிவியல், கதாநாயகன்: விஞ்ஞானி, கதைக்களம்: எதிர்காலம்")
        
        print(f"\n{'#'*70}")
        print(f"  ALL MODELS TRAINED SUCCESSFULLY!")
        print(f"{'#'*70}")
        print(f"\n📊 Summary:")
        for lang, result in results.items():
            perplexity = torch.exp(torch.tensor(result['eval_loss']))
            print(f"  {lang.upper()}: Loss={result['eval_loss']:.4f}, Perplexity={perplexity:.2f}")
        
        print(f"\n💾 Models saved in: {self.models_dir.absolute()}")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Story Generator Models")
    parser.add_argument('--language', type=str, default='all', 
                       choices=['all', 'en', 'hi', 'ta'],
                       help='Language to train (default: all)')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs (default: 3)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Training batch size (default: 1)')
    
    args = parser.parse_args()
    
    trainer = StoryGeneratorTrainer()
    
    if args.language == 'all':
        trainer.train_all_languages(epochs=args.epochs, batch_size=args.batch_size)
    else:
        trainer.train_model(args.language, epochs=args.epochs, batch_size=args.batch_size)
        
        # Test generation
        test_keywords = {
            'en': "genre: fantasy, hero: wizard, plot: quest for magic stone",
            'hi': "शैली: रोमांच, नायक: योद्धा, कथानक: राक्षस से युद्ध",
            'ta': "வகை: சாகசம், கதாநாயகன்: வீரன், கதைக்களம்: போர்"
        }
        trainer.test_generation(args.language, test_keywords.get(args.language, "test keywords"))


if __name__ == "__main__":
    main()
