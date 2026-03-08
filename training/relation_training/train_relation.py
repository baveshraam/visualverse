"""
Relation Extraction Model Training
Trains a model to classify relationships between concepts
"""

import os
import sys
import pickle
import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))


class RelationTrainer:
    """
    Relation Extraction Model Trainer
    
    Trains a classifier to identify relationships between concept pairs.
    
    Relation types:
    - IS_A: Taxonomy (X is a Y)
    - PART_OF: Composition
    - CAUSES: Causation
    - REQUIRES: Prerequisite
    - RELATES_TO: General relation
    - CONTRASTS: Opposition
    - NONE: No relation
    """
    
    RELATION_TYPES = [
        "IS_A", "PART_OF", "CAUSES", "REQUIRES", 
        "RELATES_TO", "CONTRASTS", "NONE"
    ]
    
    def __init__(self, output_dir: str = "backend/models"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.vectorizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.results = {}
    
    def load_training_data(self, data_path: str = None) -> List[Dict]:
        """
        Load relation training data
        Format: {sentence, entity1, entity2, relation}
        """
        # Comprehensive training data
        training_data = [
            # IS_A relations (Taxonomy)
            {"sentence": "A neural network is a computing system inspired by biological brains.", 
             "entity1": "neural network", "entity2": "computing system", "relation": "IS_A"},
            {"sentence": "Python is a high-level programming language known for readability.", 
             "entity1": "python", "entity2": "programming language", "relation": "IS_A"},
            {"sentence": "Machine learning is a subset of artificial intelligence.", 
             "entity1": "machine learning", "entity2": "artificial intelligence", "relation": "IS_A"},
            {"sentence": "A smartphone is a type of mobile device with computing capabilities.", 
             "entity1": "smartphone", "entity2": "mobile device", "relation": "IS_A"},
            {"sentence": "Diabetes is a chronic disease affecting insulin production.", 
             "entity1": "diabetes", "entity2": "disease", "relation": "IS_A"},
            {"sentence": "SQL is a query language used for database management.", 
             "entity1": "sql", "entity2": "query language", "relation": "IS_A"},
            {"sentence": "A router is a networking device that forwards data packets.", 
             "entity1": "router", "entity2": "networking device", "relation": "IS_A"},
            {"sentence": "Photosynthesis is a biological process in plants.", 
             "entity1": "photosynthesis", "entity2": "process", "relation": "IS_A"},
            
            # PART_OF relations (Composition)
            {"sentence": "The CPU is a core component of every computer system.", 
             "entity1": "cpu", "entity2": "computer", "relation": "PART_OF"},
            {"sentence": "Grammar is an essential part of language learning.", 
             "entity1": "grammar", "entity2": "language learning", "relation": "PART_OF"},
            {"sentence": "Data preprocessing is included in the machine learning pipeline.", 
             "entity1": "data preprocessing", "entity2": "machine learning pipeline", "relation": "PART_OF"},
            {"sentence": "The nucleus is contained within every cell.", 
             "entity1": "nucleus", "entity2": "cell", "relation": "PART_OF"},
            {"sentence": "Keywords are part of search engine optimization.", 
             "entity1": "keywords", "entity2": "search engine optimization", "relation": "PART_OF"},
            {"sentence": "Unit testing is a phase of software development.", 
             "entity1": "unit testing", "entity2": "software development", "relation": "PART_OF"},
            {"sentence": "Variables are fundamental elements of programming.", 
             "entity1": "variables", "entity2": "programming", "relation": "PART_OF"},
            {"sentence": "Authentication is a component of security systems.", 
             "entity1": "authentication", "entity2": "security systems", "relation": "PART_OF"},
            
            # CAUSES relations (Causation)
            {"sentence": "Overfitting causes poor generalization in machine learning models.", 
             "entity1": "overfitting", "entity2": "poor generalization", "relation": "CAUSES"},
            {"sentence": "Global warming leads to rising sea levels worldwide.", 
             "entity1": "global warming", "entity2": "rising sea levels", "relation": "CAUSES"},
            {"sentence": "Lack of sleep results in decreased cognitive function.", 
             "entity1": "lack of sleep", "entity2": "decreased cognitive function", "relation": "CAUSES"},
            {"sentence": "Inflation causes reduction in purchasing power.", 
             "entity1": "inflation", "entity2": "reduction in purchasing power", "relation": "CAUSES"},
            {"sentence": "Poor diet leads to various health problems.", 
             "entity1": "poor diet", "entity2": "health problems", "relation": "CAUSES"},
            {"sentence": "Deforestation results in habitat destruction.", 
             "entity1": "deforestation", "entity2": "habitat destruction", "relation": "CAUSES"},
            {"sentence": "Stress causes numerous physical symptoms.", 
             "entity1": "stress", "entity2": "physical symptoms", "relation": "CAUSES"},
            {"sentence": "Bugs in code lead to software failures.", 
             "entity1": "bugs", "entity2": "software failures", "relation": "CAUSES"},
            
            # REQUIRES relations (Prerequisite)
            {"sentence": "Deep learning requires massive computational resources.", 
             "entity1": "deep learning", "entity2": "computational resources", "relation": "REQUIRES"},
            {"sentence": "Web development needs knowledge of HTML and CSS.", 
             "entity1": "web development", "entity2": "html", "relation": "REQUIRES"},
            {"sentence": "Machine learning depends on quality training data.", 
             "entity1": "machine learning", "entity2": "training data", "relation": "REQUIRES"},
            {"sentence": "Photography requires understanding of light and composition.", 
             "entity1": "photography", "entity2": "light", "relation": "REQUIRES"},
            {"sentence": "Database design needs careful planning.", 
             "entity1": "database design", "entity2": "planning", "relation": "REQUIRES"},
            {"sentence": "Statistical analysis requires mathematical knowledge.", 
             "entity1": "statistical analysis", "entity2": "mathematical knowledge", "relation": "REQUIRES"},
            {"sentence": "Research depends on proper methodology.", 
             "entity1": "research", "entity2": "methodology", "relation": "REQUIRES"},
            {"sentence": "Running APIs needs server infrastructure.", 
             "entity1": "apis", "entity2": "server infrastructure", "relation": "REQUIRES"},
            
            # RELATES_TO relations (General association)
            {"sentence": "Statistics and machine learning are closely connected fields.", 
             "entity1": "statistics", "entity2": "machine learning", "relation": "RELATES_TO"},
            {"sentence": "Art relates to creativity in many ways.", 
             "entity1": "art", "entity2": "creativity", "relation": "RELATES_TO"},
            {"sentence": "Mathematics connects with physics fundamentally.", 
             "entity1": "mathematics", "entity2": "physics", "relation": "RELATES_TO"},
            {"sentence": "Psychology and behavior are closely linked.", 
             "entity1": "psychology", "entity2": "behavior", "relation": "RELATES_TO"},
            {"sentence": "Design and user experience go hand in hand.", 
             "entity1": "design", "entity2": "user experience", "relation": "RELATES_TO"},
            {"sentence": "Security and privacy are interconnected concerns.", 
             "entity1": "security", "entity2": "privacy", "relation": "RELATES_TO"},
            {"sentence": "Marketing and sales work together closely.", 
             "entity1": "marketing", "entity2": "sales", "relation": "RELATES_TO"},
            {"sentence": "Data and insights are strongly connected.", 
             "entity1": "data", "entity2": "insights", "relation": "RELATES_TO"},
            
            # CONTRASTS relations (Opposition)
            {"sentence": "Supervised learning versus unsupervised learning approaches.", 
             "entity1": "supervised learning", "entity2": "unsupervised learning", "relation": "CONTRASTS"},
            {"sentence": "Unlike Python, Java is statically typed.", 
             "entity1": "python", "entity2": "java", "relation": "CONTRASTS"},
            {"sentence": "Classical computing differs from quantum computing.", 
             "entity1": "classical computing", "entity2": "quantum computing", "relation": "CONTRASTS"},
            {"sentence": "Hardware contrasts with software in computers.", 
             "entity1": "hardware", "entity2": "software", "relation": "CONTRASTS"},
            {"sentence": "Theory versus practice in education.", 
             "entity1": "theory", "entity2": "practice", "relation": "CONTRASTS"},
            {"sentence": "Agile differs from waterfall methodology.", 
             "entity1": "agile", "entity2": "waterfall", "relation": "CONTRASTS"},
            {"sentence": "Frontend vs backend development.", 
             "entity1": "frontend", "entity2": "backend", "relation": "CONTRASTS"},
            {"sentence": "Centralized systems versus decentralized systems.", 
             "entity1": "centralized", "entity2": "decentralized", "relation": "CONTRASTS"},
            
            # NONE relations (Negative examples)
            {"sentence": "The meeting was scheduled for Monday morning.", 
             "entity1": "meeting", "entity2": "monday", "relation": "NONE"},
            {"sentence": "She bought a new laptop yesterday.", 
             "entity1": "she", "entity2": "laptop", "relation": "NONE"},
            {"sentence": "The conference will be held in New York.", 
             "entity1": "conference", "entity2": "new york", "relation": "NONE"},
            {"sentence": "He finished reading the book.", 
             "entity1": "he", "entity2": "book", "relation": "NONE"},
            {"sentence": "The project deadline is next week.", 
             "entity1": "project", "entity2": "week", "relation": "NONE"},
            {"sentence": "They discussed the weather briefly.", 
             "entity1": "they", "entity2": "weather", "relation": "NONE"},
            {"sentence": "The restaurant serves Italian food.", 
             "entity1": "restaurant", "entity2": "food", "relation": "NONE"},
            {"sentence": "The train arrives at noon.", 
             "entity1": "train", "entity2": "noon", "relation": "NONE"},
        ]
        
        return training_data
    
    def extract_features(self, sentence: str, entity1: str, entity2: str) -> np.ndarray:
        """Extract features for entity pair classification"""
        sent_lower = sentence.lower()
        
        # Position features
        pos1 = sent_lower.find(entity1.lower())
        pos2 = sent_lower.find(entity2.lower())
        
        distance = abs(pos2 - pos1) / max(len(sent_lower), 1) if pos1 >= 0 and pos2 >= 0 else 1.0
        order = 1.0 if pos1 < pos2 else 0.0
        
        # Lexical features
        words1 = set(entity1.lower().split())
        words2 = set(entity2.lower().split())
        shared_ratio = len(words1 & words2) / max(len(words1 | words2), 1)
        
        # Pattern indicators
        patterns = {
            "is_a": r'\b(is a|is an|are|type of|kind of)\b',
            "part_of": r'\b(part of|component|contains|includes|within)\b',
            "causes": r'\b(causes|leads to|results in|makes|produces)\b',
            "requires": r'\b(requires|needs|depends|necessary|essential)\b',
            "contrasts": r'\b(vs|versus|unlike|differs|contrasts|but|however)\b',
            "relates": r'\b(relates|connected|linked|associated|together)\b'
        }
        
        pattern_features = [
            1.0 if re.search(p, sent_lower) else 0.0 
            for p in patterns.values()
        ]
        
        return np.array([distance, order, shared_ratio] + pattern_features)
    
    def prepare_training_data(self, data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature matrix and labels"""
        sentences = [d["sentence"] for d in data]
        
        # Build TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        tfidf_features = self.vectorizer.fit_transform(sentences).toarray()
        
        # Extract pair features
        pair_features = []
        labels = []
        
        for d in data:
            features = self.extract_features(d["sentence"], d["entity1"], d["entity2"])
            pair_features.append(features)
            labels.append(d["relation"])
        
        pair_features = np.array(pair_features)
        
        # Combine features
        X = np.hstack([tfidf_features, pair_features])
        
        # Encode labels
        self.label_encoder.fit(self.RELATION_TYPES)
        y = self.label_encoder.transform(labels)
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train relation extraction models"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train multiple models
        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=42, multi_class='multinomial'
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, random_state=42
            ),
            "MLP": MLPClassifier(
                hidden_layer_sizes=(128, 64), max_iter=500, 
                random_state=42, early_stopping=True
            )
        }
        
        best_model = None
        best_accuracy = 0
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"  Accuracy: {accuracy:.4f}")
            
            results[name] = {
                "accuracy": accuracy,
                "report": classification_report(
                    y_test, y_pred, 
                    target_names=self.RELATION_TYPES,
                    output_dict=True,
                    zero_division=0
                )
            }
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
        
        self.model = best_model
        self.results = {
            "accuracy": best_accuracy,
            "model_comparison": results
        }
        
        # Print best model report
        y_pred = best_model.predict(X_test)
        print("\nBest Model Classification Report:")
        print(classification_report(
            y_test, y_pred, 
            target_names=self.RELATION_TYPES,
            zero_division=0
        ))
        
        return results
    
    def save_model(self):
        """Save trained model"""
        model_path = os.path.join(self.output_dir, "relation_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        vectorizer_path = os.path.join(self.output_dir, "relation_vectorizer.pkl")
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        encoder_path = os.path.join(self.output_dir, "relation_encoder.pkl")
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"\nModels saved to {self.output_dir}")
    
    def generate_report(self, output_path: str = "training/relation_training/report.txt"):
        """Generate training report"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        report = [
            "=" * 60,
            "RELATION EXTRACTION MODEL TRAINING REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
            "RELATION TYPES",
            "-" * 40,
        ]
        
        for rt in self.RELATION_TYPES:
            report.append(f"  - {rt}")
        
        report.extend([
            "",
            "MODEL PERFORMANCE",
            "-" * 40,
            f"Best Accuracy: {self.results.get('accuracy', 0):.4f}",
            f"Best Model: {type(self.model).__name__}",
            "",
            "MODEL COMPARISON",
            "-" * 40,
        ])
        
        for name, result in self.results.get("model_comparison", {}).items():
            report.append(f"{name}: {result['accuracy']:.4f}")
        
        report.extend([
            "",
            "=" * 60
        ])
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Report saved to {output_path}")


def main():
    """Main training function"""
    print("\n" + "=" * 60)
    print(" RELATION EXTRACTION MODEL TRAINING")
    print("=" * 60)
    
    trainer = RelationTrainer()
    
    # Load data
    print("\n1. Loading training data...")
    data = trainer.load_training_data()
    print(f"   Loaded {len(data)} examples")
    
    # Show distribution
    relation_counts = {}
    for d in data:
        rel = d["relation"]
        relation_counts[rel] = relation_counts.get(rel, 0) + 1
    
    print("   Relation distribution:")
    for rel, count in sorted(relation_counts.items()):
        print(f"     {rel}: {count}")
    
    # Prepare data
    print("\n2. Preparing training data...")
    X, y = trainer.prepare_training_data(data)
    print(f"   Feature matrix shape: {X.shape}")
    
    # Train
    print("\n3. Training models...")
    results = trainer.train(X, y)
    
    # Save
    print("\n4. Saving model...")
    trainer.save_model()
    
    # Report
    print("\n5. Generating report...")
    trainer.generate_report()
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
