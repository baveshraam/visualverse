---
description: Complete build workflow for VisualVerse project
---

# VisualVerse Build Workflow

## Project Structure
```
VisualVerse/
├── frontend/               # React + Vite frontend
├── backend/                # Python FastAPI backend
│   ├── api/               # API endpoints
│   ├── nlp/               # NLP modules
│   │   ├── preprocessing/ # Tokenization, POS, NER
│   │   ├── classification/# Narrative vs Informational classifier
│   │   ├── keyphrase/     # Keyphrase extraction (TRAINED)
│   │   ├── topic_model/   # Topic modeling (TRAINED)
│   │   └── relation/      # Relationship extraction (TRAINED)
│   ├── comic_gen/         # Comic generation pipeline
│   ├── mindmap_gen/       # Mind map generation pipeline
│   └── models/            # Trained model weights
├── training/               # Training scripts & notebooks
│   ├── data/              # Datasets
│   ├── keyphrase_training/
│   ├── topic_training/
│   └── relation_training/
└── evaluation/             # Evaluation metrics
```

## Phase 1: Project Setup
// turbo
1. Create backend directory structure
// turbo
2. Install Python dependencies
// turbo
3. Setup FastAPI server

## Phase 2: NLP Training (Mind Map Components)
4. Download and prepare datasets (WikiHow, BBC News, arXiv)
5. Train keyphrase extraction model
6. Train topic classification model
7. Train relationship extraction model
8. Save trained models

## Phase 3: Backend Development
9. Build NLP preprocessing pipeline
10. Build text classification module
11. Build comic generation pipeline
12. Build mind map generation pipeline
13. Create API endpoints

## Phase 4: Frontend Development
14. Update React frontend with new UI
15. Integrate with backend API
16. Add visualization components

## Phase 5: Testing & Evaluation
17. Test full pipeline
18. Calculate evaluation metrics
19. Final polish

