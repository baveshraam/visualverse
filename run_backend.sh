#!/bin/bash
# VisualVerse - Complete Setup and Run Script

echo "============================================================"
echo " VisualVerse - Dual-Mode NLP System Setup"
echo "============================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 is not installed"
    echo "Please install Python 3.9+"
    exit 1
fi

echo "Step 1: Creating virtual environment..."
cd backend
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

echo "Step 2: Activating virtual environment..."
source venv/bin/activate

echo "Step 3: Installing dependencies..."
pip install -r requirements.txt

echo "Step 4: Downloading SpaCy model..."
python -m spacy download en_core_web_sm

echo "Step 5: Preparing datasets..."
cd ..
python training/data/prepare_datasets.py

echo "Step 6: Training NLP models..."
python training/train_all.py

echo ""
echo "============================================================"
echo " Setup Complete! Starting Backend Server..."
echo "============================================================"
echo ""

cd backend
python main.py
