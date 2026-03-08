"""
Topic Modeling Training Script
Trains LDA and evaluates topic quality
"""

import os
import sys
import pickle
import numpy as np
from typing import List, Dict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))


class TopicModelTrainer:
    """
    Topic Modeling Trainer using LDA
    
    Training approach:
    1. Preprocess documents
    2. Build vocabulary (CountVectorizer)
    3. Train LDA with different n_topics
    4. Evaluate using perplexity and coherence
    5. Select best model
    6. Save for inference
    """
    
    def __init__(self, output_dir: str = "backend/models"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.vectorizer = None
        self.lda_model = None
        self.topic_words = {}
        self.best_n_topics = 5
        self.results = {}
    
    def load_wikihow_dataset(self, data_path: str = None) -> List[str]:
        """
        Load WikiHow-style dataset for topic modeling
        WikiHow articles are perfect for learning hierarchical topics
        """
        # Sample WikiHow-style documents
        documents = [
            # Technology
            "How to learn programming. Start with basic concepts like variables and loops. Choose a beginner-friendly language like Python. Practice coding every day. Build small projects to apply your knowledge. Join online communities for support.",
            
            "How to build a website. Learn HTML for structure and CSS for styling. Use JavaScript for interactivity. Choose a hosting provider. Register a domain name. Deploy your website and test on different browsers.",
            
            "How to protect your computer. Install antivirus software. Keep your operating system updated. Use strong passwords. Be careful with email attachments. Back up your data regularly.",
            
            # Health
            "How to exercise effectively. Set realistic fitness goals. Create a workout schedule. Warm up before exercise. Mix cardio and strength training. Stay hydrated and eat properly. Rest and recover.",
            
            "How to eat healthy. Include fruits and vegetables. Choose whole grains. Limit processed foods. Drink plenty of water. Control portion sizes. Plan meals ahead.",
            
            "How to manage stress. Identify stress triggers. Practice deep breathing. Exercise regularly. Get enough sleep. Talk to friends or family. Consider meditation.",
            
            # Education
            "How to study effectively. Create a study schedule. Find a quiet place. Take notes while reading. Use active recall techniques. Take regular breaks. Review material periodically.",
            
            "How to write an essay. Choose your topic. Research thoroughly. Create an outline. Write clear introduction. Develop body paragraphs. Write a strong conclusion. Edit and proofread.",
            
            "How to give a presentation. Know your audience. Prepare your content. Create visual aids. Practice your delivery. Manage nervousness. Engage with your audience.",
            
            # Finance
            "How to save money. Track your expenses. Create a budget. Cut unnecessary costs. Automate savings. Find cheaper alternatives. Set financial goals.",
            
            "How to invest wisely. Understand different investment types. Diversify your portfolio. Start early. Consider risk tolerance. Monitor your investments. Seek professional advice.",
            
            "How to manage debt. List all your debts. Prioritize high-interest debt. Create a repayment plan. Cut expenses. Consider debt consolidation. Avoid new debt.",
            
            # Career
            "How to find a job. Update your resume. Build your network. Search job boards. Prepare for interviews. Follow up after applying. Negotiate salary offers.",
            
            "How to advance your career. Set career goals. Develop new skills. Seek mentorship. Take on challenges. Build professional relationships. Stay current in your field.",
            
            "How to start a business. Develop a business idea. Create a business plan. Secure funding. Register your business. Build a team. Market your products."
        ]
        
        return documents
    
    def preprocess_documents(self, documents: List[str]) -> np.ndarray:
        """Preprocess documents for topic modeling"""
        self.vectorizer = CountVectorizer(
            max_df=0.95,
            min_df=2,
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        doc_term_matrix = self.vectorizer.fit_transform(documents)
        
        return doc_term_matrix
    
    def train_lda(self, doc_term_matrix: np.ndarray, n_topics: int = 5) -> LatentDirichletAllocation:
        """Train LDA model"""
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=20,
            learning_method='online',
            learning_offset=50.0,
            random_state=42,
            n_jobs=-1
        )
        
        lda.fit(doc_term_matrix)
        
        return lda
    
    def calculate_coherence(self, lda: LatentDirichletAllocation, 
                           doc_term_matrix: np.ndarray) -> float:
        """
        Calculate topic coherence score
        Uses PMI-based coherence approximation
        """
        feature_names = self.vectorizer.get_feature_names_out()
        doc_term_dense = doc_term_matrix.toarray()
        n_docs = doc_term_dense.shape[0]
        
        coherence_scores = []
        
        for topic_idx, topic in enumerate(lda.components_):
            # Get top 10 words for this topic
            top_indices = topic.argsort()[:-11:-1]
            
            topic_coherence = 0
            pairs = 0
            
            for i in range(len(top_indices)):
                for j in range(i + 1, len(top_indices)):
                    wi, wj = top_indices[i], top_indices[j]
                    
                    # Document frequency
                    df_i = np.sum(doc_term_dense[:, wi] > 0)
                    df_j = np.sum(doc_term_dense[:, wj] > 0)
                    df_ij = np.sum((doc_term_dense[:, wi] > 0) & (doc_term_dense[:, wj] > 0))
                    
                    if df_i > 0 and df_ij > 0:
                        # PMI-like score
                        pmi = np.log((df_ij * n_docs) / (df_i * df_j + 1) + 1)
                        topic_coherence += pmi
                        pairs += 1
            
            if pairs > 0:
                coherence_scores.append(topic_coherence / pairs)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def find_optimal_topics(self, doc_term_matrix: np.ndarray, 
                           topic_range: range = range(3, 10)) -> Dict:
        """
        Find optimal number of topics using perplexity and coherence
        """
        print("\nFinding optimal number of topics...")
        
        results = {}
        
        for n_topics in topic_range:
            print(f"  Testing n_topics={n_topics}...", end=" ")
            
            lda = self.train_lda(doc_term_matrix, n_topics)
            
            perplexity = lda.perplexity(doc_term_matrix)
            coherence = self.calculate_coherence(lda, doc_term_matrix)
            
            results[n_topics] = {
                "perplexity": perplexity,
                "coherence": coherence,
                "model": lda
            }
            
            print(f"Perplexity: {perplexity:.2f}, Coherence: {coherence:.4f}")
        
        # Select best model (highest coherence)
        best_n = max(results, key=lambda x: results[x]["coherence"])
        
        print(f"\nOptimal n_topics: {best_n}")
        
        self.best_n_topics = best_n
        self.lda_model = results[best_n]["model"]
        
        return results
    
    def extract_topic_words(self, n_words: int = 10) -> Dict[int, List[str]]:
        """Extract top words for each topic"""
        feature_names = self.vectorizer.get_feature_names_out()
        
        self.topic_words = {}
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_indices = topic.argsort()[:-n_words-1:-1]
            top_words = [feature_names[i] for i in top_indices]
            self.topic_words[topic_idx] = top_words
        
        return self.topic_words
    
    def print_topics(self):
        """Print discovered topics"""
        print("\nDiscovered Topics:")
        print("-" * 50)
        
        for topic_idx, words in self.topic_words.items():
            print(f"\nTopic {topic_idx}:")
            print(f"  Keywords: {', '.join(words[:7])}")
    
    def save_model(self):
        """Save trained model"""
        model_path = os.path.join(self.output_dir, "topic_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump({
                'lda': self.lda_model,
                'topic_words': self.topic_words,
                'n_topics': self.best_n_topics
            }, f)
        
        vectorizer_path = os.path.join(self.output_dir, "topic_vectorizer.pkl")
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        print(f"\nModel saved to {model_path}")
    
    def generate_report(self, search_results: Dict, 
                       output_path: str = "training/topic_training/report.txt"):
        """Generate training report"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        report = [
            "=" * 60,
            "TOPIC MODEL TRAINING REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
            "MODEL SELECTION",
            "-" * 40,
        ]
        
        for n_topics, result in search_results.items():
            report.append(f"n_topics={n_topics}: perplexity={result['perplexity']:.2f}, coherence={result['coherence']:.4f}")
        
        report.extend([
            "",
            f"Selected n_topics: {self.best_n_topics}",
            "",
            "DISCOVERED TOPICS",
            "-" * 40,
        ])
        
        for topic_idx, words in self.topic_words.items():
            report.append(f"Topic {topic_idx}: {', '.join(words[:7])}")
        
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
    print(" TOPIC MODEL TRAINING (LDA)")
    print("=" * 60)
    
    trainer = TopicModelTrainer()
    
    # Load documents
    print("\n1. Loading documents...")
    documents = trainer.load_wikihow_dataset()
    print(f"   Loaded {len(documents)} documents")
    
    # Preprocess
    print("\n2. Preprocessing documents...")
    doc_term_matrix = trainer.preprocess_documents(documents)
    print(f"   Vocabulary size: {len(trainer.vectorizer.get_feature_names_out())}")
    print(f"   Document-term matrix shape: {doc_term_matrix.shape}")
    
    # Find optimal topics
    print("\n3. Training and evaluating models...")
    search_results = trainer.find_optimal_topics(doc_term_matrix, range(3, 8))
    
    # Extract topic words
    print("\n4. Extracting topic words...")
    trainer.extract_topic_words()
    trainer.print_topics()
    
    # Save model
    print("\n5. Saving model...")
    trainer.save_model()
    
    # Generate report
    print("\n6. Generating report...")
    trainer.generate_report(search_results)
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
