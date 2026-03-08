"""
Complete Training Pipeline for VisualVerse
Trains all NLP models using the loaded datasets
"""

import os
import sys
import pickle
import asyncio
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

# Import data loader
from data.dataset_loader import DatasetLoader


def print_header(title: str):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60 + "\n")


def print_metrics(metrics: dict, indent: int = 2):
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(" " * indent + f"{key}:")
            print_metrics(value, indent + 2)
        elif isinstance(value, float):
            print(" " * indent + f"{key}: {value:.4f}")
        elif isinstance(value, list) and len(value) > 5:
            print(" " * indent + f"{key}: [{value[0]}, {value[1]}, ... ({len(value)} items)]")
        else:
            print(" " * indent + f"{key}: {value}")


async def train_text_classifier(texts, labels):
    """Train text classifier with real data"""
    print_header("Training Text Classifier (BiLSTM + Attention)")
    print("Architecture: Embedding → BiLSTM → Attention → FC")
    print("Syllabus: Unit 3 (LSTM, Attention)")
    print("-" * 40)
    
    try:
        from nlp.classification.lstm_classifier import AdvancedTextClassifier
        
        classifier = AdvancedTextClassifier()
        
        # Inject training data
        classifier._get_training_data = lambda: (texts, labels)
        
        print(f"Training on {len(texts)} examples...")
        print(f"  Narratives: {sum(labels)}")
        print(f"  Informational: {len(labels) - sum(labels)}")
        
        metrics = await classifier.train()
        
        print("\n✅ Training Complete!")
        print_metrics(metrics)
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
        # Fallback to simpler classifier
        print("\nFalling back to sklearn classifier...")
        from nlp.classification.classifier import TextClassifier
        
        classifier = TextClassifier()
        classifier._get_sample_training_data = lambda: (texts, labels)
        
        metrics = await classifier.train()
        print_metrics(metrics)
        
        return metrics


async def train_keyphrase_extractor(texts, keyphrases):
    """Train keyphrase extractor with real data"""
    print_header("Training Keyphrase Extractor")
    print("Architecture: Candidate Generation → Feature Extraction → Classification")
    print("Syllabus: Unit 2 (TF-IDF), Unit 3 (Seq2Seq possible)")
    print("-" * 40)
    
    try:
        from nlp.keyphrase.extractor import KeyphraseExtractor
        
        extractor = KeyphraseExtractor()
        
        # Create custom data loader
        def custom_loader(dataset_name=None):
            return texts, keyphrases
        
        extractor._load_training_data = custom_loader
        
        print(f"Training on {len(texts)} documents...")
        print(f"  Total keyphrases: {sum(len(kp) for kp in keyphrases)}")
        print(f"  Avg per doc: {sum(len(kp) for kp in keyphrases) / len(texts):.1f}")
        
        metrics = await extractor.train()
        
        print("\n✅ Training Complete!")
        print_metrics(metrics)
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


async def train_topic_modeler(texts):
    """Train topic model with real data"""
    print_header("Training Topic Model (LDA)")
    print("Architecture: CountVectorizer → LDA → Topic-Word Distributions")
    print("Syllabus: Unit 2 (BoW), Unit 3 (Topic Modeling)")
    print("-" * 40)
    
    try:
        from nlp.topic_model.topic_modeler import TopicModeler
        
        modeler = TopicModeler(n_topics=10)
        
        # Custom data loader
        modeler._load_training_data = lambda dataset_name=None: texts
        
        print(f"Training on {len(texts)} documents...")
        
        metrics = await modeler.train()
        
        print("\n✅ Training Complete!")
        print_metrics(metrics)
        
        # Print discovered topics
        if "topic_words" in metrics:
            print("\n📊 Discovered Topics:")
            for topic_id, words in metrics["topic_words"].items():
                print(f"  Topic {topic_id}: {', '.join(words[:5])}")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


async def train_relation_extractor():
    """Train relation extractor"""
    print_header("Training Relation Extractor")
    print("Architecture: Context Encoding → Entity Marking → Classification")
    print("Syllabus: Unit 3 (BERT/Transformer or MLP)")
    print("-" * 40)
    
    try:
        from nlp.relation.relation_extractor import RelationExtractor
        
        extractor = RelationExtractor()
        
        print("Training on built-in relation data...")
        
        metrics = await extractor.train()
        
        print("\n✅ Training Complete!")
        print_metrics(metrics)
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


async def main():
    """Main training pipeline"""
    print("\n" + "=" * 60)
    print(" VisualVerse - Complete NLP Training Pipeline")
    print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Load datasets
    print("\n📂 Loading datasets...")
    loader = DatasetLoader()
    loader.load_all()
    
    all_metrics = {}
    
    # 1. Train Text Classifier
    try:
        texts, labels = loader.prepare_classification_data()
        metrics = await train_text_classifier(texts, labels)
        all_metrics["classifier"] = metrics
    except Exception as e:
        print(f"❌ Classifier failed: {e}")
        all_metrics["classifier"] = {"error": str(e)}
    
    # 2. Train Keyphrase Extractor
    try:
        texts, keyphrases = loader.prepare_keyphrase_data()
        metrics = await train_keyphrase_extractor(texts, keyphrases)
        all_metrics["keyphrase"] = metrics
    except Exception as e:
        print(f"❌ Keyphrase extractor failed: {e}")
        all_metrics["keyphrase"] = {"error": str(e)}
    
    # 3. Train Topic Modeler
    try:
        texts = loader.prepare_topic_data()
        metrics = await train_topic_modeler(texts)
        all_metrics["topic_model"] = metrics
    except Exception as e:
        print(f"❌ Topic modeler failed: {e}")
        all_metrics["topic_model"] = {"error": str(e)}
    
    # 4. Train Relation Extractor
    try:
        metrics = await train_relation_extractor()
        all_metrics["relation"] = metrics
    except Exception as e:
        print(f"❌ Relation extractor failed: {e}")
        all_metrics["relation"] = {"error": str(e)}
    
    # Summary
    print_header("Training Summary")
    
    for model_name, metrics in all_metrics.items():
        if "error" in metrics:
            print(f"  ❌ {model_name}: FAILED - {metrics['error']}")
        else:
            acc = metrics.get('accuracy', metrics.get('f1_score', metrics.get('coherence', 'N/A')))
            print(f"  ✅ {model_name}: SUCCESS (score: {acc})")
    
    # Save metrics
    metrics_path = Path("training/training_results.pkl")
    with open(metrics_path, 'wb') as f:
        pickle.dump(all_metrics, f)
    print(f"\n📁 Results saved to {metrics_path}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_metrics


if __name__ == "__main__":
    asyncio.run(main())
