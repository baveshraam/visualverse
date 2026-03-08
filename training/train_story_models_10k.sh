#!/bin/bash

###############################################################################
# VisualVerse Story Generator - 10K+ Training Pipeline
# Training with larger datasets for better quality models
###############################################################################

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║   VISUALVERSE STORY GENERATOR - 10K+ TRAINING PIPELINE             ║"
echo "║   Training with 15K English + 5K Hindi + 5K Tamil = 25K total      ║"
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
echo "║  STEP 1: DATASET PREPARATION (10K+ VERSION)                        ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${BLUE}Preparing large training datasets...${NC}"
echo "   - English: 15,000 samples (WritingPrompts + synthetic)"
echo "   - Hindi:   5,000 samples (synthetic templates)"
echo "   - Tamil:   5,000 samples (synthetic templates)"
echo "   - Total:   25,000 samples"
echo ""

python3 training/story_training/prepare_dataset_10k.py

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Dataset preparation failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Datasets prepared successfully!${NC}"
echo ""

# Ask user if they want to proceed with training
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  STEP 2: MODEL TRAINING (10K+ VERSION)                             ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${YELLOW}Training will now begin with optimized GPU settings.${NC}"
echo ""
echo "Training Configuration:"
echo "  • Base Model: distilgpt2 (80MB)"
echo "  • Epochs: 3"
echo "  • Batch Size: 1 (with gradient accumulation=8)"
echo "  • Max Length: 256 tokens (GPU-optimized)"
echo "  • FP16: Disabled (better for 4GB GPU)"
echo "  • Languages: English (15K), Hindi (5K), Tamil (5K)"
echo "  • Output: backend/models/story_model_{lang}_10k/"
echo ""
echo "Estimated Time:"
echo "  • With GPU (3050ti):  3-4 hours total"
echo "  • Without GPU (CPU):  12-15 hours total"
echo ""
echo -e "${BLUE}💡 GPU Memory Optimization Applied:${NC}"
echo "  • Reduced batch size from 4 to 1"
echo "  • Reduced max_length from 512 to 256"
echo "  • Added gradient accumulation (8 steps)"
echo "  • Disabled FP16 mixed precision"
echo "  • GPU memory cleanup between languages"
echo ""

read -p "Continue with 10K+ training? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled. You can run training later with:"
    echo "  python3 training/story_training/train_story_generator_10k.py"
    exit 0
fi

echo ""
echo -e "${BLUE}🎯 Starting model training with 10K+ datasets...${NC}"
echo ""

# Set CUDA memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

echo -e "${BLUE}[1/3] Training English (15K samples)...${NC}"
python3 training/story_training/train_story_generator_10k.py --language en --epochs 3 --batch-size 1
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ English training failed${NC}"
    exit 1
fi

echo -e "${BLUE}🧹 Clearing GPU memory...${NC}"
python3 -c "import gc, torch; gc.collect(); torch.cuda.empty_cache()" 2>/dev/null || true

echo -e "${BLUE}[2/3] Training Hindi (5K samples)...${NC}"
python3 training/story_training/train_story_generator_10k.py --language hi --epochs 3 --batch-size 1
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Hindi training failed${NC}"
    exit 1
fi

echo -e "${BLUE}🧹 Clearing GPU memory...${NC}"
python3 -c "import gc, torch; gc.collect(); torch.cuda.empty_cache()" 2>/dev/null || true

echo -e "${BLUE}[3/3] Training Tamil (5K samples)...${NC}"
python3 training/story_training/train_story_generator_10k.py --language ta --epochs 3 --batch-size 1
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Tamil training failed${NC}"
    exit 1
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  TRAINING COMPLETE! 🎉                                             ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}✅ All 10K+ models trained successfully!${NC}"
echo ""
echo "📁 Models saved in: backend/models/"
echo "   NEW (10K+ samples):"
echo "   • backend/models/story_model_en_10k/"
echo "   • backend/models/story_model_hi_10k/"
echo "   • backend/models/story_model_ta_10k/"
echo ""
echo "   ORIGINAL (5K samples):"
echo "   • backend/models/story_model_en/"
echo "   • backend/models/story_model_hi/"
echo "   • backend/models/story_model_ta/"
echo ""
echo "🔬 Model Comparison:"
echo "   You now have two versions of each model for comparison!"
echo "   - 5K version: Faster training, good baseline"
echo "   - 10K version: Better quality, more diverse outputs"
echo ""
echo "🚀 Next steps:"
echo "   1. Update backend/story_gen/story_generator.py to use 10K models"
echo "   2. Start the backend server: cd backend && python main.py"
echo "   3. Start the frontend: npm run dev"
echo "   4. Test and compare both versions!"
echo ""
