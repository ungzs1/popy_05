# RNN Agent: Clean Code Organization

## Summary of Changes

The code has been reorganized for clarity and best practices:

### ✅ What Changed

1. **Training moved to separate script** (`train_rnn_agent.py`)
   - Cleaner separation of concerns
   - Can be run independently
   - Saves all results automatically

2. **Notebook now focuses on analysis** (`results.ipynb`)
   - Loads pre-trained results
   - Visualizes training curves
   - Computes performance metrics
   - Compares agents

3. **Results saved to disk** (`results/` directory)
   - Trained model weights
   - Training history (losses, rewards)
   - Full simulation recording (100k trials)
   - Agent parameters

## New Workflow

### Step 1: Train the Agent
```bash
cd notebooks/dqn
python train_rnn_agent.py
```

**Output:**
```
RNN AGENT TRAINING
Training steps: 50,000
...
✓ Training complete!
✓ Simulation complete!
✓ All results saved to: results/
```

**Time:** ~2-5 minutes

### Step 2: Analyze Results
```bash
jupyter notebook results.ipynb
```

**Or in VS Code:**
1. Open `results.ipynb`
2. Run all cells

**The notebook will:**
- Load training history
- Plot loss curves
- Display simulation statistics
- Compare with ForagingAgent

## File Structure

```
notebooks/dqn/
├── train_rnn_agent.py       # NEW: Training script
├── check_results.py          # NEW: Check if results exist
├── results.ipynb             # UPDATED: Analysis only
├── rnn_agent.py             # Core implementation
├── rnn_analysis_tools.py    # Advanced analysis
├── agents.py                # ForagingAgent
└── results/                 # NEW: Output directory
    ├── rnn_agent_trained.pth
    ├── training_history.csv
    ├── simulation_recording.csv
    ├── agent_parameters.csv
    └── README.md
```

## Key Features

### Training Script (`train_rnn_agent.py`)

**Functions:**
- `train_agent()` - Train the RNN agent
- `simulate_with_trained_agent()` - Run full simulation
- `save_results()` - Save all outputs
- `main()` - Complete pipeline

**Configurable parameters:**
```python
n_training_steps = 50_000
n_simulation_steps = 100_000
hidden_size = 32
learning_rate = 0.001
```

**Saved outputs:**
1. **Model weights** - Trained network parameters
2. **Training history** - Loss and reward per episode
3. **Simulation recording** - Complete behavioral data
4. **Agent parameters** - Configuration for reproducibility

### Analysis Notebook (`results.ipynb`)

**Sections:**
1. **Load Data** - Read saved results
2. **Training Curves** - Visualize learning progress
3. **Performance Metrics** - Compute statistics
4. **Comparison** - RNN vs ForagingAgent

**Key Metrics Displayed:**
- Mean reward
- Best target selection rate
- Switch probability by outcome
- Learning curves
- Performance comparison

## Quick Reference

### Check Training Status
```bash
python check_results.py
```

### Train Agent
```bash
python train_rnn_agent.py
```

### Load Results in Python
```python
import pandas as pd

# Load training history
training = pd.read_csv('results/training_history.csv')

# Load simulation
simulation = pd.read_csv('results/simulation_recording.csv')

# Get statistics
print(f"Mean reward: {simulation['reward'].mean():.4f}")
print(f"Best target selection: {(simulation['action'] == simulation['best_arm']).mean():.4f}")
```

### Load Trained Model
```python
import torch
from rnn_agent import RNNAgent

# Create agent
agent = RNNAgent(n_arms=3, hidden_size=32)

# Load trained weights
agent.network.load_state_dict(torch.load('results/rnn_agent_trained.pth'))

# Use for inference
agent.exploration_noise = 0.0  # Disable exploration
action, switched = agent.act(reward=1)
```

## Benefits of New Organization

### ✅ Clarity
- Training and analysis are separate
- Each file has a single, clear purpose
- Easier to understand workflow

### ✅ Reproducibility
- All parameters saved automatically
- Results can be regenerated
- Easy to share trained models

### ✅ Efficiency
- Train once, analyze many times
- No need to retrain in notebook
- Faster iteration on analysis

### ✅ Modularity
- Can train on different machines
- Can analyze results offline
- Easy to integrate into pipelines

### ✅ Best Practices
- Follows standard ML workflow
- Separation of training/evaluation
- Version control friendly

## Output Files Explained

### `rnn_agent_trained.pth`
- PyTorch state dict
- Contains all network weights
- Size: ~50 KB
- Load with: `torch.load()`

### `training_history.csv`
Columns:
- `episode`: Training update number
- `loss`: Policy gradient loss
- `mean_reward`: Average reward over last 100 steps

Use for: Plotting learning curves

### `simulation_recording.csv`
Columns:
- `trial_id`: Trial number
- `block_id`: Block number
- `best_arm`: Optimal action
- `action`: Agent's choice
- `reward`: Received reward
- `switched`: Whether agent switched
- `switch_prob`: RNN output probability
- `previous_switch`: Previous switch indicator

Use for: Detailed behavioral analysis

### `agent_parameters.csv`
Columns:
- `n_arms`: Number of actions
- `hidden_size`: RNN hidden state size
- `learning_rate`: Optimizer step size
- `gamma`: Discount factor
- `exploration_noise`: Decision noise
- `feedback_duration`: Temporal processing
- `decision_delay`: Deliberation time

Use for: Reproducing results

## Migration Guide

If you have old notebook-based code:

### Before (Old Way)
```python
# All in notebook
agent = RNNAgent(...)
# ... train for 50k steps in notebook ...
# ... analyze results in notebook ...
```

### After (New Way)
```bash
# In terminal
python train_rnn_agent.py
```

```python
# In notebook
training = pd.read_csv('results/training_history.csv')
simulation = pd.read_csv('results/simulation_recording.csv')
# ... analyze ...
```

## Advanced Usage

### Custom Training Parameters
Edit `train_rnn_agent.py`:
```python
agent_params = {
    'hidden_size': 64,        # Larger network
    'learning_rate': 0.003,   # Faster learning
    'feedback_duration': 5,   # Longer feedback
}
```

### Multiple Training Runs
```bash
# Run 1
python train_rnn_agent.py
mv results results_run1

# Run 2
python train_rnn_agent.py
mv results results_run2

# Compare
python
>>> import pandas as pd
>>> run1 = pd.read_csv('results_run1/simulation_recording.csv')
>>> run2 = pd.read_csv('results_run2/simulation_recording.csv')
>>> print(run1['reward'].mean(), run2['reward'].mean())
```

### Batch Processing
```python
# train_multiple.py
from train_rnn_agent import train_agent, simulate_with_trained_agent, save_results

for hidden_size in [16, 32, 64]:
    agent, losses, rewards = train_agent(hidden_size=hidden_size)
    recording = simulate_with_trained_agent(agent)
    save_results(agent, losses, rewards, recording, 
                 save_dir=f'results_h{hidden_size}')
```

## Troubleshooting

### Results Not Found
```bash
python check_results.py  # Check status
python train_rnn_agent.py  # Train if needed
```

### Retrain from Scratch
```bash
rm -rf results/  # Delete old results
python train_rnn_agent.py  # Train again
```

### Analysis Without Training
If someone shared results with you:
```bash
# Just copy their results/ folder
cp -r /path/to/shared/results .

# Then analyze
jupyter notebook results.ipynb
```

## Next Steps

1. ✅ Train agent: `python train_rnn_agent.py`
2. ✅ Check results: `python check_results.py`
3. ✅ Analyze: Open `results.ipynb`
4. ✅ Experiment: Modify parameters and retrain
5. ✅ Share: Send `results/` folder to collaborators

---

**Questions?** See:
- `RNN_AGENT_README.md` - Detailed documentation
- `TROUBLESHOOTING.md` - Common issues
- `GETTING_STARTED.md` - Step-by-step guide
