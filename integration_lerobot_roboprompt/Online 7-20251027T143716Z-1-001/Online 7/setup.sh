#!/bin/bash
# Quick start script for RoboPrompt SO101

echo "========================================="
echo "RoboPrompt for SO101 - Quick Start"
echo "========================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment (optional but recommended)
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate
echo "✓ Virtual environment activated"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -U pip
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Create output directories
echo ""
echo "Creating output directories..."
mkdir -p output/demonstrations
mkdir -p output/results
mkdir -p output/logs
echo "✓ Output directories created"

# Check for URDF file
echo ""
if [ -f "config/so100.urdf" ]; then
    echo "✓ URDF file found: config/so100.urdf"
else
    echo "⚠ WARNING: URDF file not found!"
    echo "  Please copy your SO101 URDF file to config/so100.urdf"
fi

# Check for OpenAI API key
echo ""
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠ WARNING: OPENAI_API_KEY not set!"
    echo "  Set it with: export OPENAI_API_KEY=your_key_here"
else
    echo "✓ OpenAI API key found"
fi

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Copy SO101 URDF to config/so100.urdf (if not done)"
echo "2. Set OpenAI API key: export OPENAI_API_KEY=your_key"
echo "3. Run the pipeline:"
echo "   python src/main_pipeline.py --mode offline --num-episodes 10"
echo ""
echo "For more information, see README.md"
echo ""
