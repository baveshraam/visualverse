#!/bin/bash

###############################################################################
# VisualVerse Story Generator - Complete Training Pipeline
# This script prepares datasets and trains models for all three languages
###############################################################################

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║     VISUALVERSE STORY GENERATOR - TRAINING PIPELINE                ║"
echo "║     Training keyword-to-story models for English, Hindi, Tamil     ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${YELLOW}⚠️  Virtual environment not detected${NC}"
    echo "   Please activate your virtual environment first:"
    echo "   source /path/to/venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting..."
        exit 1
    fi
fi

# Navigate to project root
cd "$(dirname "$0")/.." || exit

echo -e "${BLUE}📂 Current directory: $(pwd)${NC}"
echo ""

# Check GPU availability
echo -e "${BLUE}🔍 Checking GPU availability...${NC}"
python3 -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "PyTorch not installed yet"
echo ""

# Install dependencies
echo -e "${BLUE}📦 Installing required packages...${NC}"
echo "   This may take a few minutes..."
pip install -q transformers datasets torch sentencepiece 2>&1 | grep -v "already satisfied" || true
echo -e "${GREEN}✅ Dependencies installed${NC}"
echo ""

# Step 1: Prepare Datasets
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  STEP 1: DATASET PREPARATION                                       ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${BLUE}Preparing training datasets...${NC}"
echo "   - English: 5,000 samples (WritingPrompts + synthetic)"
echo "   - Hindi:   2,000 samples (synthetic templates)"
echo "   - Tamil:   2,000 samples (synthetic templates)"
echo ""

python3 training/story_training/prepare_dataset.py

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Dataset preparation failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Datasets prepared successfully!${NC}"
echo ""

# Ask user if they want to proceed with training
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  STEP 2: MODEL TRAINING                                            ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${YELLOW}Training will now begin.${NC}"
echo ""
echo "Training Configuration:"
echo "  • Base Model: distilgpt2 (80MB)"
echo "  • Epochs: 3"
echo "  • Batch Size: 1 (with gradient accumulation=8)"
echo "  • Max Length: 256 tokens"
echo "  • Languages: English, Hindi, Tamil"
echo ""
echo "Estimated Time:"
echo "  • With GPU (3050ti):  1-2 hours total"
echo "  • Without GPU (CPU):  4-6 hours total"
echo ""

read -p "Continue with training? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled. You can run training later with:"
    echo "  python3 training/story_training/train_story_generator.py"
    exit 0
fi

echo ""
echo -e "${BLUE}🎯 Starting model training...${NC}"
echo ""

# Train all models
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
python3 training/story_training/train_story_generator.py --language all --epochs 3 --batch-size 1

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}❌ Training failed${NC}"
    echo "   Check the error messages above"
    exit 1
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  TRAINING COMPLETE! 🎉                                             ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}✅ All models trained successfully!${NC}"
echo ""
echo "📁 Models saved in: backend/models/"
echo "   • backend/models/story_model_en/"
echo "   • backend/models/story_model_hi/"
echo "   • backend/models/story_model_ta/"
echo ""
echo "🚀 Next steps:"
echo "   1. Start the backend server:"
echo "      cd backend && python main.py"
echo ""
echo "   2. Start the frontend in another terminal:"
echo "      npm run dev"
echo ""
echo "   3. Open http://localhost:5173 and test the Story Generator!"
echo ""
