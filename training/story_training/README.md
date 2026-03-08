# Story Generator Feature - Training & Usage Guide

## 📖 Overview

The **Story Generator** is a new feature in VisualVerse that generates narrative stories from keywords using trained GPT-2 models. It supports English, Hindi, and Tamil languages.

## 🎯 Feature Workflow

1. **User enters keywords** (e.g., "genre: sci-fi, hero: astronaut, plot: Mars mission")
2. **AI generates story** using trained distilgpt2 model (~500-800 words)
3. **User reviews story** in a preview interface
4. **Optional: Generate comic strip** from the generated story

---

## 🏋️ Training the Models

### Prerequisites

1. **Python Environment**: Activated virtual environment
2. **GPU Recommended**: NVIDIA GPU with 4GB+ VRAM (CPU training works but slower)
3. **Disk Space**: ~5GB for datasets and models

### Quick Training (Recommended)

Run the automated training script:

```bash
# From project root
cd training
./train_story_models.sh
```

This script will:
- Install required packages (transformers, datasets, torch)
- Download WritingPrompts dataset
- Prepare datasets for all 3 languages
- Train models sequentially
- Save trained models to `backend/models/`

**Training Time:**
- **With GPU (GTX 3050ti)**: ~1-2 hours
- **Without GPU (CPU)**: ~4-6 hours

### Manual Training (Advanced)

#### Step 1: Prepare Datasets

```bash
cd training/story_training
python prepare_dataset.py
```

This creates:
- `data/stories_en.json` (5,000 English samples)
- `data/stories_hi.json` (2,000 Hindi samples)
- `data/stories_ta.json` (2,000 Tamil samples)

#### Step 2: Train Models

Train all languages:
```bash
python train_story_generator.py --language all --epochs 3 --batch-size 4
```

Train specific language:
```bash
python train_story_generator.py --language en --epochs 3 --batch-size 4
```

**Training Arguments:**
- `--language`: Language to train (`en`, `hi`, `ta`, or `all`)
- `--epochs`: Number of training epochs (default: 3)
- `--batch-size`: Batch size (default: 4, increase if you have more VRAM)

---

## 🚀 Using the Feature

### 1. Start the Backend

```bash
cd backend
python main.py
```

The backend will automatically load the trained models from `backend/models/`.

### 2. Start the Frontend

```bash
# From project root
npm run dev
```

### 3. Use the Story Generator

1. Open http://localhost:5173
2. Click "Launch Studio"
3. Select **"Story Generator"** mode (4th option)
4. Enter keywords in the text area

**Example Keywords:**

**English:**
```
genre: mystery, hero: detective, setting: foggy London, plot: murder investigation
```

**Hindi:**
```
शैली: रोमांच, नायक: योद्धा, कथानक: राक्षस से युद्ध
```

**Tamil:**
```
வகை: சாகசம், கதாநாயகன்: வீரன், கதைக்களம்: போர்
```

5. Select language (English/Hindi/Tamil)
6. Click **"Generate Story"**
7. Review the generated story
8. Click **"Generate Comic Strip from Story"** to visualize it!

---

## 📊 Model Details

### Architecture
- **Base Model**: distilgpt2 (80MB)
- **Fine-tuning**: Causal language modeling with special tokens
- **Special Tokens**:
  - `<|keywords|>` - Marks keyword input
  - `<|story|>` - Marks story output
  - `<|endoftext|>` - End of generation

### Dataset
- **English**: WritingPrompts dataset (5,000 samples)
- **Hindi**: Synthetic samples (2,000 samples)
- **Tamil**: Synthetic samples (2,000 samples)

### Training Configuration
- **Epochs**: 3
- **Batch Size**: 4
- **Learning Rate**: 5e-5
- **Max Length**: 512 tokens
- **Mixed Precision**: FP16 (GPU only)

---

## 🧪 Testing Models

Test generation after training:

```bash
cd training/story_training
python train_story_generator.py --language en --epochs 0  # Just test, no training
```

Or use Python directly:

```python
from backend.story_gen.story_generator import StoryGenerator
import asyncio

generator = StoryGenerator()
result = asyncio.run(generator.generate(
    keywords="genre: sci-fi, hero: astronaut, plot: Mars mission",
    language="en"
))
print(result['story'])
```

---

## 📁 File Structure

```
training/story_training/
├── __init__.py
├── prepare_dataset.py       # Dataset preparation script
├── train_story_generator.py # Training script
└── data/                     # Generated datasets
    ├── stories_en.json
    ├── stories_hi.json
    └── stories_ta.json

backend/story_gen/
├── __init__.py
├── story_generator.py       # Story generation inference

backend/models/
├── story_model_en/          # Trained English model
├── story_model_hi/          # Trained Hindi model
└── story_model_ta/          # Trained Tamil model
```

---

## 🔧 Troubleshooting

### "Model not found" error
**Solution**: Train the models first using `training/train_story_models.sh`

### "CUDA out of memory" error
**Solution**: Reduce batch size:
```bash
python train_story_generator.py --batch-size 2
```

### Slow training on CPU
**Solution**: This is normal. CPU training takes 4-6 hours. Consider using Google Colab with free GPU:
1. Upload project to Colab
2. Run training script with GPU enabled
3. Download trained models

### Poor quality stories
**Solution**: Train for more epochs:
```bash
python train_story_generator.py --epochs 5
```

---

## 📈 Evaluation Metrics

After training, check `backend/models/story_model_XX/training_info.json`:

```json
{
  "language": "en",
  "base_model": "distilgpt2",
  "train_samples": 4500,
  "val_samples": 500,
  "final_loss": 2.34,
  "perplexity": 10.38,
  "training_time_minutes": 45.2
}
```

**Good metrics:**
- **Perplexity**: < 30 (lower is better)
- **Loss**: < 3.0

---

## 🎓 Academic Notes

This feature demonstrates:
- **Conditional Text Generation** using transformer models
- **Fine-tuning** pre-trained language models
- **Multilingual NLP** across three languages
- **Zero-shot learning** adaptation from English to other languages

**Papers to cite:**
- GPT-2: "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
- DistilGPT-2: "DistilBERT, a distilled version of BERT" (Sanh et al., 2019)
- WritingPrompts: "A Corpus and Evaluation Framework for Deeper Understanding of Commonsense Stories" (Fan et al., 2018)

---

## 💡 Tips for Best Results

1. **Keyword Quality**: Provide specific keywords with genre, characters, setting, and plot
2. **Language Selection**: Choose correct language for better results
3. **Temperature**: Default 0.8 is good. Lower (0.6) for more focused, higher (1.0) for more creative
4. **Story Length**: Models generate 500-800 words optimally

---

## 🚀 Future Improvements

- [ ] Add more training data for Hindi/Tamil
- [ ] Support for mixed-language stories
- [ ] Fine-tune on specific genres (fantasy, sci-fi, mystery)
- [ ] Add story quality scoring
- [ ] Support for custom story length control

---

## 📞 Support

If you encounter issues:
1. Check the error messages in terminal
2. Verify GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`
3. Ensure virtual environment is activated
4. Check that all packages are installed: `pip install -r backend/requirements.txt`

---

**Enjoy creating stories with AI! 🎨📚**
