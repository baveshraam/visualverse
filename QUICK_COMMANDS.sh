#!/bin/bash
# Quick Commands for Story Generator Feature

echo "╔════════════════════════════════════════════════════════╗"
echo "║  VisualVerse Story Generator - Quick Commands          ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

cat << 'EOF'

📦 SETUP & TRAINING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Install dependencies
pip install transformers datasets torch sentencepiece

# Train all models (one command - recommended)
cd training && ./train_story_models.sh

# Or train manually
cd training/story_training
python prepare_dataset.py                              # Step 1: Prepare data
python train_story_generator.py --language all         # Step 2: Train models


🚀 RUNNING THE APPLICATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Terminal 1: Start backend
cd backend
python main.py

# Terminal 2: Start frontend
npm run dev

# Open browser
http://localhost:5173


🧪 TESTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Test trained model
cd training/story_training
python train_story_generator.py --language en --epochs 0

# Check GPU availability
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"

# Verify models exist
ls -lh backend/models/story_model_*/


📊 TRAINING OPTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Train single language
python train_story_generator.py --language en          # English only
python train_story_generator.py --language hi          # Hindi only
python train_story_generator.py --language ta          # Tamil only

# Custom training parameters
python train_story_generator.py --language all --epochs 5 --batch-size 2

# With more epochs for better quality
python train_story_generator.py --language all --epochs 5


🔧 TROUBLESHOOTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# If "CUDA out of memory"
python train_story_generator.py --batch-size 2

# If models not loading
ls backend/models/           # Check if models exist
cd training && ./train_story_models.sh    # Retrain if needed

# Check backend logs
cd backend && python main.py             # Look for model loading messages

# Reinstall dependencies
pip install --upgrade transformers datasets torch


📁 FILE LOCATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Training Scripts:
  training/story_training/prepare_dataset.py
  training/story_training/train_story_generator.py
  training/train_story_models.sh

Backend:
  backend/story_gen/story_generator.py
  backend/main.py (see /api/generate-story endpoint)

Frontend:
  App.tsx (see WorkspacePage and ResultsPage)
  services/geminiService.ts (see generateStory function)

Models (after training):
  backend/models/story_model_en/
  backend/models/story_model_hi/
  backend/models/story_model_ta/


📖 DOCUMENTATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Quick Start:
  STORY_GENERATOR_QUICKSTART.md

Detailed Guide:
  training/story_training/README.md

Implementation Details:
  IMPLEMENTATION_SUMMARY.md


💡 EXAMPLE KEYWORDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

English:
  genre: mystery, hero: detective, setting: foggy London, plot: murder investigation
  genre: sci-fi, characters: rebel pilot & android, setting: space station, conflict: alien invasion
  genre: fantasy, hero: young wizard, mentor: old sage, quest: ancient artifact, danger: dark lord

Hindi:
  शैली: रोमांच, नायक: योद्धा, स्थान: पहाड़, कथानक: राक्षस से युद्ध
  शैली: विज्ञान कथा, पात्र: वैज्ञानिक, स्थान: प्रयोगशाला, कथानक: आविष्कार

Tamil:
  வகை: சாகசம், கதாநாயகன்: வீரன், இடம்: காடு, கதைக்களம்: போர்
  வகை: அறிவியல், பாத்திரம்: விஞ்ஞானி, இடம்: ஆய்வகம், கதை: கண்டுபிடிப்பு


⏱️ TRAINING TIME
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GTX 3050ti (4GB): ~1-2 hours
GTX 1660 (6GB):   ~1.5-2.5 hours
CPU only:         ~4-6 hours
Colab T4 GPU:     ~45-90 minutes


🎯 USAGE FLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Launch Studio
2. Select "Story Generator" (4th option with sparkle icon ✨)
3. Enter keywords
4. Select language (English/Hindi/Tamil)
5. Click "Generate Story"
6. Review generated story
7. Click "Generate Comic Strip from Story"
8. View comic panels!


✅ QUICK VALIDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Check everything is working
python -c "import torch, transformers, datasets; print('✅ All packages installed')"

# Verify models
[ -d "backend/models/story_model_en" ] && echo "✅ English model exists"
[ -d "backend/models/story_model_hi" ] && echo "✅ Hindi model exists"  
[ -d "backend/models/story_model_ta" ] && echo "✅ Tamil model exists"

# Check model size
du -sh backend/models/story_model_*


📞 HELP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If you need help:
1. Check STORY_GENERATOR_QUICKSTART.md
2. Check training/story_training/README.md
3. Check backend/story_gen/story_generator.py for code details

EOF

echo ""
echo "Ready to start? Run: cd training && ./train_story_models.sh"
echo ""
