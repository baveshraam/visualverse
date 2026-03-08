"""
Advanced Text Classifier using LSTM/BiLSTM
Aligned with NLP Syllabus Unit 3: RNN, LSTM, Attention
"""

import os
import pickle
import numpy as np
from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import re


class LSTMClassifier(nn.Module):
    """
    BiLSTM Text Classifier
    
    Architecture (from Unit 3):
    - Embedding Layer (Word embeddings - Unit 2)
    - Bidirectional LSTM (Unit 3)
    - Attention Mechanism (Unit 3)
    - Fully Connected Layer
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 100, 
                 hidden_dim: int = 128, num_layers: int = 2, 
                 num_classes: int = 2, dropout: float = 0.3):
        super(LSTMClassifier, self).__init__()
        
        # Embedding layer (Unit 2: Neural word embeddings)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Bidirectional LSTM (Unit 3: LSTM, Bidirectional)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention layer (Unit 3: Attention mechanism)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def attention_mechanism(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        Attention Mechanism (Unit 3)
        Computes attention weights over LSTM outputs
        """
        # lstm_output: (batch, seq_len, hidden*2)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # Weighted sum
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        
        # BiLSTM
        lstm_out, _ = self.lstm(embedded)  # (batch, seq_len, hidden*2)
        
        # Attention
        attended = self.attention_mechanism(lstm_out)  # (batch, hidden*2)
        
        # Dropout and FC
        dropped = self.dropout(attended)
        output = self.fc(dropped)  # (batch, num_classes)
        
        return output


class AdvancedTextClassifier:
    """
    Advanced Text Classifier using BiLSTM + Attention
    
    Demonstrates concepts from syllabus:
    - Unit 2: TF-IDF, Word embeddings
    - Unit 3: LSTM, Bidirectional RNN, Attention mechanism
    - Unit 4: Text classification
    """
    
    MODEL_PATH = "backend/models/lstm_classifier.pt"
    VOCAB_PATH = "backend/models/vocab.pkl"
    
    def __init__(self, max_vocab: int = 5000, max_seq_len: int = 100):
        self.max_vocab = max_vocab
        self.max_seq_len = max_seq_len
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.model = None
        self._trained = False
        self._ready = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._load_model()
    
    def is_ready(self) -> bool:
        return self._ready
    
    def is_trained(self) -> bool:
        return self._trained
    
    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics if hasattr(self, 'metrics') else {}
    
    def _load_model(self):
        """Load trained model"""
        try:
            if os.path.exists(self.MODEL_PATH) and os.path.exists(self.VOCAB_PATH):
                with open(self.VOCAB_PATH, 'rb') as f:
                    self.vocab = pickle.load(f)
                
                self.model = LSTMClassifier(len(self.vocab))
                self.model.load_state_dict(torch.load(self.MODEL_PATH, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                self._trained = True
        except Exception as e:
            print(f"Could not load LSTM model: {e}")
    
    def _save_model(self):
        """Save trained model"""
        os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
        torch.save(self.model.state_dict(), self.MODEL_PATH)
        with open(self.VOCAB_PATH, 'wb') as f:
            pickle.dump(self.vocab, f)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    def _build_vocab(self, texts: List[str]):
        """Build vocabulary from texts (Unit 2: Word representation)"""
        word_counts = Counter()
        for text in texts:
            tokens = self._tokenize(text)
            word_counts.update(tokens)
        
        # Keep top words
        most_common = word_counts.most_common(self.max_vocab - 2)
        for word, _ in most_common:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
    
    def _text_to_sequence(self, text: str) -> List[int]:
        """Convert text to sequence of indices"""
        tokens = self._tokenize(text)
        sequence = [self.vocab.get(t, 1) for t in tokens]  # 1 = <UNK>
        
        # Pad or truncate
        if len(sequence) < self.max_seq_len:
            sequence = sequence + [0] * (self.max_seq_len - len(sequence))
        else:
            sequence = sequence[:self.max_seq_len]
        
        return sequence
    
    def classify(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """Classify text as narrative or informational"""
        text = preprocessed.get("original_text", "")
        
        if self._trained and self.model is not None:
            return self._classify_with_model(text)
        
        # Fallback to rule-based
        return self._classify_rule_based(text)
    
    def _classify_with_model(self, text: str) -> Dict[str, Any]:
        """Classify using trained LSTM model"""
        self.model.eval()
        
        sequence = self._text_to_sequence(text)
        tensor = torch.LongTensor([sequence]).to(self.device)
        
        with torch.no_grad():
            output = self.model(tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        text_type = "narrative" if prediction == 1 else "informational"
        
        return {
            "type": text_type,
            "confidence": float(confidence),
            "model": "BiLSTM + Attention"
        }
    
    def _classify_rule_based(self, text: str) -> Dict[str, Any]:
        """Rule-based fallback classification"""
        # Same as before - narrative indicators
        narrative_score = len(re.findall(r'\b(he|she|they|said|walked|ran|looked)\b', text, re.I))
        info_score = len(re.findall(r'\b(is|are|means|refers|therefore|however)\b', text, re.I))
        
        if narrative_score > info_score:
            return {"type": "narrative", "confidence": 0.7, "model": "rule-based"}
        else:
            return {"type": "informational", "confidence": 0.7, "model": "rule-based"}
    
    async def train(self, dataset_name: str = None) -> Dict[str, Any]:
        """
        Train BiLSTM classifier
        
        Training process:
        1. Build vocabulary (Unit 2)
        2. Convert texts to sequences
        3. Train BiLSTM with Attention (Unit 3)
        4. Evaluate and save
        """
        print("Training BiLSTM + Attention Classifier...")
        print("(Demonstrates: LSTM, Bidirectional RNN, Attention - Unit 3)")
        
        # Load training data
        texts, labels = self._get_training_data()
        
        # Build vocabulary
        print("Building vocabulary...")
        self._build_vocab(texts)
        
        # Convert to sequences
        sequences = [self._text_to_sequence(t) for t in texts]
        X = np.array(sequences)
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create DataLoaders
        train_dataset = TensorDataset(
            torch.LongTensor(X_train), 
            torch.LongTensor(y_train)
        )
        test_dataset = TensorDataset(
            torch.LongTensor(X_test), 
            torch.LongTensor(y_test)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16)
        
        # Initialize model
        self.model = LSTMClassifier(len(self.vocab)).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training loop
        num_epochs = 20
        print(f"Training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")
        
        # Evaluate
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_y.numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        
        self.metrics = {
            "accuracy": float(accuracy),
            "model": "BiLSTM + Attention",
            "vocab_size": len(self.vocab),
            "epochs": num_epochs,
            "architecture": {
                "embedding_dim": 100,
                "hidden_dim": 128,
                "num_layers": 2,
                "attention": True
            }
        }
        
        print(f"\n✅ Training complete! Accuracy: {accuracy:.4f}")
        
        # Save model
        self._save_model()
        self._trained = True
        
        return self.metrics
    
    def _get_training_data(self) -> Tuple[List[str], List[int]]:
        """Get training data (expanded)"""
        narratives = [
            "Once upon a time, there was a young princess who lived in a tall tower. She dreamed of adventure beyond the castle walls.",
            "John walked into the dark room, his heart pounding. He could hear footsteps behind him, getting closer.",
            "The old wizard raised his staff and spoke ancient words. Lightning crackled through the sky.",
            "She ran through the forest, branches scratching her face. They were following her.",
            "The spaceship landed on the alien planet. Captain Sarah stepped out, amazed by the purple sky.",
            "He found the treasure map in his grandfather's attic. This was the beginning of his great adventure.",
            "The dragon roared and breathed fire. The knight raised his shield just in time.",
            "Marie looked at the letter in her hands. After all these years, he had finally written back.",
            "The detective examined the crime scene carefully. Something wasn't right about this case.",
            "They sailed across the stormy sea, hoping to reach the island before nightfall.",
            "The robot opened its eyes for the first time. It wondered about the meaning of consciousness.",
            "She said goodbye to her family and boarded the train. A new life awaited her in the city.",
        ]
        
        informational = [
            "Machine learning is a subset of artificial intelligence that enables systems to learn from data without being explicitly programmed.",
            "The water cycle consists of evaporation, condensation, precipitation, and collection. This process is essential for life on Earth.",
            "Python is a high-level programming language known for its simple syntax and readability. It supports multiple programming paradigms.",
            "Climate change refers to long-term shifts in global temperatures. The main causes include greenhouse gas emissions and deforestation.",
            "Neural networks are computing systems inspired by biological brains. They consist of layers of interconnected nodes.",
            "The periodic table organizes chemical elements by atomic number. Elements in the same group share similar properties.",
            "Photosynthesis is the process by which plants convert sunlight into chemical energy. The equation is 6CO2 + 6H2O → C6H12O6 + 6O2.",
            "Database management systems organize and store data efficiently. Common types include relational and NoSQL databases.",
            "The Internet uses a protocol called TCP/IP for data transmission. This ensures reliable communication between devices.",
            "Object-oriented programming organizes code into classes and objects. Key principles include encapsulation, inheritance, and polymorphism.",
            "The human brain contains approximately 86 billion neurons. These cells communicate through electrical and chemical signals.",
            "Blockchain is a distributed ledger technology that provides transparency and security. It is the foundation of cryptocurrencies.",
        ]
        
        texts = narratives + informational
        labels = [1] * len(narratives) + [0] * len(informational)
        
        return texts, labels
