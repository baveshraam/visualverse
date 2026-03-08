"""
Keyphrase Extraction Model Training
Detailed training script with dataset integration and evaluation
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    precision_recall_fscore_support, 
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))


class KeyphraseTrainer:
    """
    Comprehensive Keyphrase Extraction Model Trainer
    
    Datasets supported:
    - Inspec (scientific abstracts)
    - SemEval-2010 (scientific papers)  
    - WikiHow (procedural text)
    - Custom datasets
    
    Training approach:
    1. Load and preprocess dataset
    2. Generate candidate phrases (n-grams)
    3. Create feature vectors for candidates
    4. Train binary classifier (keyphrase vs non-keyphrase)
    5. Evaluate on held-out test set
    6. Save trained model
    """
    
    def __init__(self, output_dir: str = "backend/models"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.model = None
        self.tfidf = None
        self.results = {}
        
    def load_inspec_dataset(self, data_path: str = None) -> Tuple[List[str], List[List[str]]]:
        """
        Load Inspec dataset (abstracts with keyphrases)
        """
        # If no path provided, use sample data
        if data_path and os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            return data['texts'], data['keyphrases']
        
        # Sample Inspec-style data
        texts = [
            "We present a novel approach to automatic text summarization using neural networks. Our method combines extractive and abstractive techniques to generate coherent summaries. Experiments on standard benchmarks demonstrate significant improvements over baseline methods.",
            
            "This paper introduces a new algorithm for keyphrase extraction from scientific documents. The approach uses graph-based ranking combined with semantic embeddings. Results show improved precision and recall compared to existing methods.",
            
            "Machine translation has made significant progress with the introduction of transformer architectures. We propose modifications to improve handling of rare vocabulary and long-range dependencies. Evaluation on multiple language pairs confirms the effectiveness of our approach.",
            
            "Sentiment analysis of social media content presents unique challenges including informal language and short texts. We develop a hybrid model combining rule-based and machine learning approaches. Testing on Twitter data shows robust performance.",
            
            "Named entity recognition is fundamental to many NLP applications. This work explores the use of pre-trained language models for entity extraction in specialized domains. We achieve state-of-the-art results on biomedical and legal entity recognition.",
            
            "Question answering systems require understanding of both questions and relevant context. We present a retrieval-augmented model that improves answer accuracy. Experiments on open-domain QA benchmarks demonstrate competitive performance.",
            
            "Topic modeling enables automatic discovery of themes in document collections. We extend LDA with neural variational inference for improved topic coherence. Analysis of news corpora validates the quality of discovered topics.",
            
            "Dialogue systems for customer service need to handle diverse user intents. Our multi-task learning approach jointly optimizes intent detection and response generation. Deployed system shows reduced human agent escalation.",
            
            "Information extraction from unstructured text enables knowledge base population. We develop a pipeline combining relation extraction and entity linking. Evaluation on newswire text demonstrates high extraction accuracy.",
            
            "Text classification is essential for content moderation and organization. Transfer learning from pre-trained models significantly improves classification accuracy on limited labeled data. We analyze the effect of domain adaptation."
        ]
        
        keyphrases = [
            ["text summarization", "neural networks", "extractive", "abstractive", "benchmarks"],
            ["keyphrase extraction", "scientific documents", "graph-based ranking", "semantic embeddings", "precision", "recall"],
            ["machine translation", "transformer", "vocabulary", "long-range dependencies", "language pairs"],
            ["sentiment analysis", "social media", "informal language", "machine learning", "twitter"],
            ["named entity recognition", "nlp", "pre-trained language models", "entity extraction", "biomedical"],
            ["question answering", "retrieval-augmented", "answer accuracy", "open-domain"],
            ["topic modeling", "lda", "neural variational inference", "topic coherence", "news"],
            ["dialogue systems", "customer service", "intent detection", "response generation", "multi-task learning"],
            ["information extraction", "knowledge base", "relation extraction", "entity linking"],
            ["text classification", "content moderation", "transfer learning", "domain adaptation"]
        ]
        
        return texts, keyphrases
    
    def extract_candidates(self, text: str, max_ngram: int = 3) -> List[str]:
        """Extract candidate keyphrases"""
        import re
        
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        
        stopwords = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by',
                     'from', 'as', 'into', 'through', 'during', 'before', 'after',
                     'above', 'below', 'between', 'and', 'but', 'or', 'this', 'that'}
        
        candidates = []
        for n in range(1, max_ngram + 1):
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])
                # Skip if starts/ends with stopword
                if words[i] not in stopwords and words[i+n-1] not in stopwords:
                    if len(ngram) > 2:
                        candidates.append(ngram)
        
        return list(set(candidates))
    
    def extract_features(self, candidate: str, text: str) -> np.ndarray:
        """Extract features for a candidate phrase"""
        text_lower = text.lower()
        
        # Position feature (earlier = more important)
        first_pos = text_lower.find(candidate)
        position_score = 1 - (first_pos / max(len(text_lower), 1)) if first_pos >= 0 else 0
        
        # Frequency
        frequency = text_lower.count(candidate)
        
        # Length features
        word_count = len(candidate.split())
        char_count = len(candidate)
        
        # Capitalization in original text
        original_count = text.count(candidate.title()) + text.count(candidate.upper())
        
        # Spread (last - first occurrence)
        last_pos = text_lower.rfind(candidate)
        spread = (last_pos - first_pos) / max(len(text_lower), 1) if last_pos > first_pos else 0
        
        return np.array([
            position_score,
            frequency / 10.0,  # Normalize
            word_count / 3.0,
            char_count / 30.0,
            original_count / 5.0,
            spread
        ])
    
    def prepare_training_data(self, texts: List[str], 
                             keyphrases_list: List[List[str]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from texts and their keyphrases"""
        X = []
        y = []
        
        # Build TF-IDF on all texts
        self.tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=3000)
        self.tfidf.fit(texts)
        
        for text, true_keyphrases in zip(texts, keyphrases_list):
            true_kp_set = {kp.lower().strip() for kp in true_keyphrases}
            candidates = self.extract_candidates(text)
            
            for candidate in candidates:
                # Get basic features
                features = self.extract_features(candidate, text)
                
                # Get TF-IDF features for candidate
                try:
                    tfidf_vec = self.tfidf.transform([candidate]).toarray().flatten()[:10]
                except:
                    tfidf_vec = np.zeros(10)
                
                # Pad if necessary
                if len(tfidf_vec) < 10:
                    tfidf_vec = np.pad(tfidf_vec, (0, 10 - len(tfidf_vec)))
                
                combined = np.concatenate([features, tfidf_vec])
                X.append(combined)
                
                # Label: 1 if matches a keyphrase, 0 otherwise
                label = 1 if candidate in true_kp_set else 0
                y.append(label)
        
        return np.array(X), np.array(y)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train the keyphrase extraction model"""
        # Handle class imbalance
        class_weights = {0: 1.0, 1: sum(y == 0) / max(sum(y == 1), 1)}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train multiple models and compare
        models = {
            "Logistic Regression": LogisticRegression(
                class_weight=class_weights, max_iter=1000, random_state=42
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, class_weight=class_weights, random_state=42
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100, random_state=42
            )
        }
        
        best_model = None
        best_f1 = 0
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='binary', zero_division=0
            )
            
            results[name] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
            
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                self.results = results[name]
        
        self.model = best_model
        
        # Cross-validation for best model
        cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='f1')
        self.results["cv_f1_mean"] = cv_scores.mean()
        self.results["cv_f1_std"] = cv_scores.std()
        
        print(f"\nBest Model CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        return results
    
    def save_model(self, filename: str = "keyphrase_model.pkl"):
        """Save the trained model"""
        model_path = os.path.join(self.output_dir, filename)
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        tfidf_path = os.path.join(self.output_dir, "keyphrase_tfidf.pkl")
        with open(tfidf_path, 'wb') as f:
            pickle.dump(self.tfidf, f)
        
        print(f"\nModel saved to {model_path}")
        
    def generate_report(self, output_path: str = "training/keyphrase_training/report.txt"):
        """Generate training report"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        report = [
            "=" * 60,
            "KEYPHRASE EXTRACTION MODEL TRAINING REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
            "MODEL PERFORMANCE",
            "-" * 40,
            f"Precision: {self.results.get('precision', 0):.4f}",
            f"Recall: {self.results.get('recall', 0):.4f}",
            f"F1 Score: {self.results.get('f1_score', 0):.4f}",
            f"CV F1 (mean): {self.results.get('cv_f1_mean', 0):.4f}",
            f"CV F1 (std): {self.results.get('cv_f1_std', 0):.4f}",
            "",
            "TRAINING DETAILS",
            "-" * 40,
            f"Model Type: {type(self.model).__name__}",
            f"Features: Position, Frequency, Length, TF-IDF, Spread",
            "",
            "=" * 60
        ]
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Report saved to {output_path}")


def main():
    """Main training function"""
    print("\n" + "=" * 60)
    print(" KEYPHRASE EXTRACTION MODEL TRAINING")
    print("=" * 60)
    
    trainer = KeyphraseTrainer()
    
    # Load dataset
    print("\n1. Loading dataset...")
    texts, keyphrases = trainer.load_inspec_dataset()
    print(f"   Loaded {len(texts)} documents")
    
    # Prepare training data
    print("\n2. Preparing training data...")
    X, y = trainer.prepare_training_data(texts, keyphrases)
    print(f"   Generated {len(X)} training examples")
    print(f"   Positive examples: {sum(y)}")
    print(f"   Negative examples: {len(y) - sum(y)}")
    
    # Train models
    print("\n3. Training models...")
    results = trainer.train(X, y)
    
    # Save model
    print("\n4. Saving model...")
    trainer.save_model()
    
    # Generate report
    print("\n5. Generating report...")
    trainer.generate_report()
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
