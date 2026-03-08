"""
BERT-based Relation Extractor
Aligned with NLP Syllabus Unit 3: Transformer Networks, BERT
"""

import os
import pickle
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import re

# Check if transformers is available
try:
    from transformers import BertTokenizer, BertModel
    import torch
    import torch.nn as nn
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("Note: transformers library not installed. Using fallback model.")


class BERTRelationClassifier(nn.Module):
    """
    BERT-based Relation Classifier (Unit 3: Transformer, BERT)
    
    Architecture:
    - BERT Encoder (pre-trained)
    - Entity marker tokens for highlighting entities
    - Classification head
    """
    
    def __init__(self, num_classes: int = 7, hidden_dim: int = 768):
        super(BERTRelationClassifier, self).__init__()
        
        if BERT_AVAILABLE:
            # Use DistilBERT for efficiency
            from transformers import DistilBertModel
            self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        logits = self.classifier(cls_output)
        return logits


class TransformerRelationExtractor:
    """
    Transformer-based Relation Extractor
    
    Demonstrates concepts from syllabus Unit 3:
    - Transformer Networks
    - BERT (Bidirectional Encoder Representations from Transformers)
    - Pre-training and Fine-tuning
    
    Relation types:
    - IS_A, PART_OF, CAUSES, REQUIRES, RELATES_TO, CONTRASTS, NONE
    """
    
    RELATION_TYPES = ["IS_A", "PART_OF", "CAUSES", "REQUIRES", "RELATES_TO", "CONTRASTS", "NONE"]
    MODEL_PATH = "backend/models/bert_relation.pt"
    
    def __init__(self):
        self._ready = True
        self._trained = False
        self.model = None
        self.tokenizer = None
        self.metrics = {}
        
        if BERT_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            try:
                from transformers import DistilBertTokenizer
                self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            except:
                pass
    
    def is_ready(self) -> bool:
        return self._ready
    
    def is_trained(self) -> bool:
        return self._trained
    
    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics
    
    def extract(self, preprocessed: Dict[str, Any], 
                keyphrases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relations between keyphrases"""
        text = preprocessed.get("original_text", "")
        sentences = preprocessed.get("sentences", [])
        
        if len(keyphrases) < 2:
            return []
        
        keyphrase_texts = [kp["phrase"] for kp in keyphrases]
        
        if self._trained and self.model is not None and BERT_AVAILABLE:
            return self._extract_with_bert(text, sentences, keyphrase_texts)
        
        return self._extract_pattern_based(text, sentences, keyphrase_texts)
    
    def _extract_with_bert(self, text: str, sentences: List[str], 
                          keyphrases: List[str]) -> List[Dict[str, Any]]:
        """Extract relations using BERT model"""
        relations = []
        
        self.model.eval()
        
        for i, kp1 in enumerate(keyphrases):
            for j, kp2 in enumerate(keyphrases):
                if i >= j:
                    continue
                
                # Find context sentence
                context = self._find_context(kp1, kp2, sentences) or f"{kp1} and {kp2}"
                
                # Mark entities in context
                marked_text = context.replace(kp1, f"[E1]{kp1}[/E1]")
                marked_text = marked_text.replace(kp2, f"[E2]{kp2}[/E2]")
                
                # Tokenize
                encoding = self.tokenizer(
                    marked_text,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(input_ids, attention_mask)
                    probs = torch.softmax(outputs, dim=1)
                    pred = torch.argmax(probs, dim=1).item()
                    confidence = probs[0][pred].item()
                
                if pred != self.RELATION_TYPES.index("NONE") and confidence > 0.3:
                    relations.append({
                        "source": kp1,
                        "target": kp2,
                        "relation": self.RELATION_TYPES[pred],
                        "confidence": float(confidence),
                        "context": context[:100]
                    })
        
        return relations[:15]
    
    def _find_context(self, kp1: str, kp2: str, sentences: List[str]) -> str:
        """Find sentence containing both keyphrases"""
        for sent in sentences:
            if kp1.lower() in sent.lower() and kp2.lower() in sent.lower():
                return sent
        return None
    
    def _extract_pattern_based(self, text: str, sentences: List[str], 
                               keyphrases: List[str]) -> List[Dict[str, Any]]:
        """Pattern-based extraction fallback"""
        relations = []
        
        patterns = {
            "IS_A": [r"(\w+) is a (\w+)", r"(\w+) are (\w+)"],
            "PART_OF": [r"(\w+) is part of (\w+)", r"(\w+) contains (\w+)"],
            "CAUSES": [r"(\w+) causes (\w+)", r"(\w+) leads to (\w+)"],
            "REQUIRES": [r"(\w+) requires (\w+)", r"(\w+) needs (\w+)"],
            "CONTRASTS": [r"(\w+) vs (\w+)", r"unlike (\w+), (\w+)"]
        }
        
        text_lower = text.lower()
        
        for rel_type, pats in patterns.items():
            for pattern in pats:
                for match in re.finditer(pattern, text_lower, re.I):
                    e1, e2 = match.group(1), match.group(2)
                    
                    src = self._match_keyphrase(e1, keyphrases)
                    tgt = self._match_keyphrase(e2, keyphrases)
                    
                    if src and tgt and src != tgt:
                        relations.append({
                            "source": src,
                            "target": tgt,
                            "relation": rel_type,
                            "confidence": 0.7,
                            "context": match.group(0)
                        })
        
        # Add co-occurrence relations
        for sent in sentences:
            sent_lower = sent.lower()
            present = [kp for kp in keyphrases if kp.lower() in sent_lower]
            
            for i, kp1 in enumerate(present):
                for kp2 in present[i+1:]:
                    if not any(r["source"] == kp1 and r["target"] == kp2 for r in relations):
                        relations.append({
                            "source": kp1,
                            "target": kp2,
                            "relation": "RELATES_TO",
                            "confidence": 0.5,
                            "context": sent[:100]
                        })
        
        return relations[:15]
    
    def _match_keyphrase(self, text: str, keyphrases: List[str]) -> str:
        """Match text to a keyphrase"""
        text_lower = text.lower()
        for kp in keyphrases:
            if kp.lower() in text_lower or text_lower in kp.lower():
                return kp
        return None
    
    async def train(self, dataset_name: str = None) -> Dict[str, Any]:
        """
        Train BERT relation classifier
        
        Demonstrates:
        - Transformer architecture (Unit 3)
        - BERT pre-training and fine-tuning (Unit 3)
        """
        print("Training Transformer-based Relation Extractor...")
        print("(Demonstrates: Transformer Networks, BERT - Unit 3)")
        
        if not BERT_AVAILABLE:
            print("Note: transformers library not installed.")
            print("Using pattern-based extraction as fallback.")
            self.metrics = {
                "model": "Pattern-based (BERT not available)",
                "note": "Install transformers library for BERT-based extraction"
            }
            return self.metrics
        
        # Load training data
        training_data = self._get_training_data()
        
        print(f"Training on {len(training_data)} examples...")
        
        # Prepare data
        texts = []
        labels = []
        
        for d in training_data:
            # Mark entities
            sentence = d["sentence"]
            marked = sentence.replace(d["entity1"], f"[E1]{d['entity1']}[/E1]")
            marked = marked.replace(d["entity2"], f"[E2]{d['entity2']}[/E2]")
            texts.append(marked)
            labels.append(self.RELATION_TYPES.index(d["relation"]))
        
        # Tokenize
        encodings = self.tokenizer(
            texts,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids']
        attention_masks = encodings['attention_mask']
        labels = torch.LongTensor(labels)
        
        # Split
        train_idx, test_idx = train_test_split(
            range(len(texts)), test_size=0.2, random_state=42
        )
        
        # Initialize model
        self.model = BERTRelationClassifier(num_classes=len(self.RELATION_TYPES))
        self.model.to(self.device)
        
        # Training (simplified - full training would need more epochs/data)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        
        for epoch in range(3):  # Short training for demo
            total_loss = 0
            
            for i in train_idx:
                optimizer.zero_grad()
                
                ids = input_ids[i:i+1].to(self.device)
                mask = attention_masks[i:i+1].to(self.device)
                label = labels[i:i+1].to(self.device)
                
                outputs = self.model(ids, mask)
                loss = criterion(outputs, label)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"  Epoch {epoch+1}/3, Loss: {total_loss/len(train_idx):.4f}")
        
        # Evaluate
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for i in test_idx:
                ids = input_ids[i:i+1].to(self.device)
                mask = attention_masks[i:i+1].to(self.device)
                
                outputs = self.model(ids, mask)
                pred = torch.argmax(outputs, dim=1).item()
                
                all_preds.append(pred)
                all_labels.append(labels[i].item())
        
        accuracy = accuracy_score(all_labels, all_preds)
        
        self.metrics = {
            "model": "DistilBERT + Classification Head",
            "accuracy": float(accuracy),
            "num_classes": len(self.RELATION_TYPES),
            "architecture": {
                "encoder": "DistilBERT (pre-trained)",
                "classifier": "2-layer MLP",
                "entity_marking": "[E1]...[/E1], [E2]...[/E2]"
            },
            "syllabus_concepts": [
                "Transformer Networks",
                "BERT (Bidirectional Encoder)",
                "Pre-training and Fine-tuning",
                "Attention mechanism (self-attention)"
            ]
        }
        
        self._trained = True
        print(f"\n✅ Training complete! Accuracy: {accuracy:.4f}")
        
        return self.metrics
    
    def _get_training_data(self) -> List[Dict]:
        """Get training data"""
        return [
            {"sentence": "A neural network is a computing system.", "entity1": "neural network", "entity2": "computing system", "relation": "IS_A"},
            {"sentence": "Python is a programming language.", "entity1": "python", "entity2": "programming language", "relation": "IS_A"},
            {"sentence": "The CPU is part of the computer.", "entity1": "cpu", "entity2": "computer", "relation": "PART_OF"},
            {"sentence": "Functions are building blocks of programs.", "entity1": "functions", "entity2": "programs", "relation": "PART_OF"},
            {"sentence": "Overfitting causes poor generalization.", "entity1": "overfitting", "entity2": "poor generalization", "relation": "CAUSES"},
            {"sentence": "Climate change leads to rising sea levels.", "entity1": "climate change", "entity2": "sea levels", "relation": "CAUSES"},
            {"sentence": "Machine learning requires data.", "entity1": "machine learning", "entity2": "data", "relation": "REQUIRES"},
            {"sentence": "Web development needs HTML.", "entity1": "web development", "entity2": "html", "relation": "REQUIRES"},
            {"sentence": "Statistics and ML are connected.", "entity1": "statistics", "entity2": "ml", "relation": "RELATES_TO"},
            {"sentence": "Design and UX go together.", "entity1": "design", "entity2": "ux", "relation": "RELATES_TO"},
            {"sentence": "Supervised vs unsupervised learning.", "entity1": "supervised", "entity2": "unsupervised", "relation": "CONTRASTS"},
            {"sentence": "Python unlike Java is dynamic.", "entity1": "python", "entity2": "java", "relation": "CONTRASTS"},
            {"sentence": "The meeting is on Monday.", "entity1": "meeting", "entity2": "monday", "relation": "NONE"},
            {"sentence": "She wrote the code.", "entity1": "she", "entity2": "code", "relation": "NONE"},
        ]
