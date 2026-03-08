# VisualVerse Story Generator - Quick Start Guide

## 🚀 Quick Start (5 Steps)

### Step 1: Install Backend Dependencies

```bash
cd backend
pip install transformers datasets torch sentencepiece
```

### Step 2: Train the Models

**Option A: Automated (Recommended)**
```bash
cd training
./train_story_models.sh
```

**Option B: Manual**
```bash
# Prepare datasets
cd training/story_training
python prepare_dataset.py

# Train models (will take 1-2 hours with GPU, 4-6 hours with CPU)
python train_story_generator.py --language all --epochs 3 --batch-size 4
```

### Step 3: Verify Models are Saved

Check that these directories exist:
```
backend/models/story_model_en/
backend/models/story_model_hi/
backend/models/story_model_ta/
```

Each should contain:
- `config.json`
- `pytorch_model.bin`
- `tokenizer.json`
- `training_info.json`

### Step 4: Start Backend Server

```bash
cd backend
python main.py
```

You should see:
```
✅ Loaded story model for en
✅ Loaded story model for hi
✅ Loaded story model for ta
```

### Step 5: Start Frontend

In a new terminal:
```bash
npm run dev
```

Open http://localhost:5173

---

## 🎮 Using the Feature

1. **Click "Launch Studio"**
2. **Select "Story Generator"** (4th option with sparkle icon)
3. **Enter keywords**, for example:
   - English: `genre: sci-fi, hero: astronaut, setting: Mars, plot: survival mission`
   - Hindi: `शैली: रोमांच, नायक: योद्धा, कथानक: राक्षस से युद्ध`
   - Tamil: `வகை: சாகசம், கதாநாயகன்: வீரன், கதைக்களம்: போர்`
4. **Select language**
5. **Click "Generate Story"**
6. **Review generated story**
7. **Click "Generate Comic Strip from Story"** to visualize!

---

## ⏱️ Training Time Estimates

| Hardware | Time (All 3 Languages) |
|----------|----------------------|
| GTX 3050ti (4GB) | 1-2 hours |
| GTX 1660 (6GB) | 1.5-2.5 hours |
| CPU only | 4-6 hours |
| Google Colab (Free T4) | 45-90 minutes |

---

## 🐛 Common Issues

### Issue: "Model not found"
**Cause**: Models not trained yet  
**Solution**: Run `training/train_story_models.sh`

### Issue: "CUDA out of memory"
**Cause**: GPU doesn't have enough memory  
**Solution**: Reduce batch size to 2:
```bash
python train_story_generator.py --batch-size 2
```

### Issue: "transformers not found"
**Cause**: Package not installed  
**Solution**: 
```bash
pip install transformers datasets torch
```

### Issue: Generated stories are low quality
**Cause**: Need more training  
**Solution**: Train for more epochs:
```bash
python train_story_generator.py --epochs 5
```

---

## 📊 What Gets Trained?

| Language | Samples | Model Size | Training Time (GPU) |
|----------|---------|------------|---------------------|
| English | 5,000 | 80MB | 30-45 min |
| Hindi | 2,000 | 80MB | 20-30 min |
| Tamil | 2,000 | 80MB | 20-30 min |
| **Total** | **9,000** | **~250MB** | **~1-2 hours** |

---

## 🎯 Expected Results

After training, you should be able to:
- Generate 500-800 word stories from keywords
- Support English, Hindi, and Tamil
- Create stories in various genres (sci-fi, fantasy, mystery, etc.)
- Convert generated stories into comic strips

**Sample Output:**
```
Input: "genre: mystery, detective, foggy London, murder case"

Output (excerpt):
"Detective Sarah Morrison walked through the fog-covered streets of London.
The murder case had left the city in shock. A prominent businessman found dead
in his study, a glass of wine beside him. Six suspects, each with a motive..."
```

---

## 🎓 For Academic Projects

This feature demonstrates:
- ✅ **Fine-tuning** pre-trained language models
- ✅ **Conditional text generation** from structured input
- ✅ **Multilingual NLP** across 3 languages
- ✅ **Transfer learning** from English to other languages
- ✅ **End-to-end pipeline** from data prep to deployment

**Perfect for NLP course projects!**

---

## 📚 Further Reading

- [training/story_training/README.md](training/story_training/README.md) - Detailed documentation
- [backend/story_gen/story_generator.py](backend/story_gen/story_generator.py) - Implementation details
- [Hugging Face Transformers Docs](https://huggingface.co/docs/transformers/) - GPT-2 documentation

---

**Ready to train? Run `cd training && ./train_story_models.sh` to get started!** 🚀
