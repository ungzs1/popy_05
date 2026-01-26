#!/bin/bash
# Quick script to train RNN agent and open analysis notebook

echo "=========================================="
echo "RNN Agent Training Pipeline"
echo "=========================================="
echo ""

# Check if results exist
if [ -f "results/rnn_agent_trained.pth" ]; then
    echo "⚠️  Results already exist!"
    echo ""
    read -p "Do you want to retrain? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping training..."
        echo ""
        echo "Opening analysis notebook..."
        jupyter notebook results.ipynb
        exit 0
    fi
    echo "Removing old results..."
    rm -rf results/
fi

# Train agent
echo "Starting training..."
echo ""
python train_rnn_agent.py

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Training complete!"
    echo "=========================================="
    echo ""
    read -p "Open analysis notebook? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        jupyter notebook results.ipynb
    else
        echo "You can analyze results later with:"
        echo "  jupyter notebook results.ipynb"
    fi
else
    echo ""
    echo "=========================================="
    echo "❌ Training failed!"
    echo "=========================================="
    exit 1
fi
