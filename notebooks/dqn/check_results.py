#!/usr/bin/env python
"""
Check if RNN agent needs to be trained.
"""

import os

def check_training_status():
    """Check if training results exist."""
    results_dir = 'results'
    required_files = [
        'rnn_agent_trained.pth',
        'training_history.csv',
        'simulation_recording.csv',
        'agent_parameters.csv'
    ]
    
    print("="*60)
    print("RNN AGENT TRAINING STATUS")
    print("="*60)
    
    if not os.path.exists(results_dir):
        print("❌ Results directory not found")
        print("\n➡️  Please run: python train_rnn_agent.py")
        return False
    
    missing = []
    for file in required_files:
        path = os.path.join(results_dir, file)
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024  # KB
            print(f"✓ {file:<30} ({size:.1f} KB)")
        else:
            print(f"❌ {file:<30} (missing)")
            missing.append(file)
    
    print("="*60)
    
    if missing:
        print(f"\n❌ Missing {len(missing)} file(s)")
        print("➡️  Please run: python train_rnn_agent.py")
        return False
    else:
        print("\n✅ All training results found!")
        print("➡️  You can now run: jupyter notebook results.ipynb")
        return True

if __name__ == "__main__":
    check_training_status()
